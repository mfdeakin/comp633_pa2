
#include <cuda.h>
#include <cublas.h>
#include <sys/time.h>

#include "pa2.h"

__global__ void AATrans(mtxel *mtx, mtxel *dest, int dim, int blksize, int smsize)
{
	int t = blockDim.x * blockIdx.x + threadIdx.x;
	/* Calculate the column the thread is working in.
	 * We are only computing half the matrix,
	 * since the matrix is symmetric along the diagonal.
	 */
	int c = floor((1+2*dim-sqrtf(1+4*dim+4*dim*dim-8*t))/2);
	/* The row follows from the column */
	int r = t - c * dim + c * (c - 1) / 2 + c;
	
	DBGPRINT("Dim: %d, Thread: %d, Row: %d, Column: %d, "
					 "Block Size: %d\n", dim, t, r, c, blksize);

	/* Will be treated as mtxel rows[2 * blksize][dim] 
	 * The first blksize arrays are for the rows of the matrix at r
	 * The second blksize arrays are for the rows of the matrix at c
	 */
	__shared__ mtxel rowmem[0xc000 / sizeof(mtxel) - 1];
	
	/* Copy our rows into fast local memory */
	for(int i = 0; i < blksize; i++)
		for(int k = 0; k < dim; k++)
			rowmem[i * dim + k] = mtx[(r + i) * dim + k];
	for(int i = 0; i < blksize; i++)
		for(int k = 0; k < dim; k++)
			rowmem[(i + blksize) * dim + k] = mtx[(c + i) * dim + k];

	/* Compute A A^T */
	for(int i = 0; i < blksize; i++) {
		for(int j = 0; j < blksize; j++) {
			if((c + i) >= 0 && (c + i) < dim &&
				 (r + j) >= 0 && (r + j) < dim) {
				dest[(c + i) * dim + (r + j)] = 0.0;
				for(int k = 0; k < dim; k++)
					dest[(c + i) * dim + (r + j)] += rowmem[j * dim + k] *
						rowmem[(i + blksize) * dim + k];
				dest[(r + j) * dim + (c + i)] = dest[(c + i) * dim + (r + j)];
			}
		}
	}
}

__global__ void AATransSmall(mtxel *mtx, mtxel *dest, int dim)
{
	/* Naive implementation. Rather slow, even with small matrices */
	int t = blockDim.x * blockIdx.x + threadIdx.x;
	int c = t / dim;
	int r = t % dim;
	if(c >= 0 && c < dim && r >= 0 && r < dim) {
		dest[c * dim + r] = 0.0;
		for(int k = 0; k < dim; k++)
			dest[c * dim + r] += mtx[r * dim + k] * mtx[c * dim + k];
	}
}

void computeCUDA(mtxel *hostmtx, mtxel *dest, int dim)
{
	if(dim == 1)
		return;
	mtxel *devmtx, *devdest;

	cudaMalloc(&devmtx, sizeof(mtxel[dim * dim]));
	cudaMalloc(&devdest, sizeof(mtxel[dim * dim]));
	if(!devmtx || !devdest)
		return;
	cudaMemcpy(devmtx, hostmtx, sizeof(mtxel[dim * dim]), cudaMemcpyHostToDevice);

	/* blksize is the number of rows and columns a thread works with */
	int blksize = 1;
	/* maxdim * (maxdim + 1) / 2 < 2^16, while anything greater is above 2^16
	 * This constraint exists because CUDA only supports up to 2^16 blocks
	 */
	const int maxdim = 361;
	int curdim = dim;
	/* Now calculate the size of the blocks each thread works with,
	 * and add one extra thread, just in case
	 */
	while(curdim > maxdim) {
		blksize++;
		curdim /= 2;
	}
	curdim++;

	/* The threads shared memory will consist of blksize rows
	 * So the total shared memory is dim * blksize
	 */

	AATrans <<<
		curdim * (curdim + 1) / 2, 1
					>>> (devmtx, devdest, dim, blksize, dim * blksize);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("CUDA Error %d: %s\n", err, cudaGetErrorString(err));
	}

	cudaMemcpy(dest, devdest, sizeof(mtxel[dim * dim]), cudaMemcpyDeviceToHost);
	cudaFree(devmtx);
	cudaFree(devdest);
}

void checkCUBLAS(cublasStatus_t err, char *event)
{
	switch(err) {
	case CUBLAS_STATUS_SUCCESS:
		break;
	default:
		printf("Unknown error %d! %s\n", err, event);
	}
}

void computeCUBLAS(mtxel *mtx, mtxel *dest, int dim)
{
	cublasStatus_t err;
	mtxel *devmtx1, *devdest;
	err = cublasAlloc(dim * dim, sizeof(mtxel), (void **)&devmtx1);
	checkCUBLAS(err, "Allocated dev matrix 1");
	err = cublasAlloc(dim * dim, sizeof(mtxel), (void **)&devdest);
	checkCUBLAS(err, "Allocated dev dest matrix");
	err = cublasSetMatrix(dim, dim, sizeof(mtxel), (void *)mtx, dim, (void *)devmtx1, dim);
	checkCUBLAS(err, "Set dev matrix 1");

	cublasDgemm('T', 'N', dim, dim, dim, 1.0,
		    devmtx1, dim, devmtx1, dim, 0.0, devdest, dim);

	err = cublasGetError();
	checkCUBLAS(err, "Multiplied matrix");
	err = cublasGetMatrix(dim, dim, sizeof(mtxel), (void *)devdest, dim, dest, dim);
	checkCUBLAS(err, "Got matrix");
	cublasFree(devmtx1);
	cublasFree(devdest);
}

int initCUDA()
{
	/* Make certain we have a CUDA capable machine */
	int count = 0;
	cudaGetDeviceCount(&count);
	if(count == 0) {
		return 1;
	}
	/* Find out some information about it.
	 * Require at least compute 2.0
	 */
	cudaSetDevice(0);
	cudaDeviceProp dev;
	cudaGetDeviceProperties(&dev, 0);
	if(dev.major < 2) {
		return 2;
	}
	/* Make a call to a CUDA function so initialization time
	 * isn't included in our computeCUDA time calculation
	 */
	void *mem = NULL;
	cudaMalloc(&mem, 0);
	if(mem)
		cudaFree(mem);

	/* Similarly for CUBLAS */
	cublasInit();
	return 0;
}

void shutdownCUDA()
{
	cublasShutdown();
	
}
