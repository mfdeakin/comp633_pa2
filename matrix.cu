
#include <cuda.h>
#include <cublas.h>
#include <sys/time.h>

#include "pa2.h"

/* For accurate performance metrics, run with CUDA_LAUNCH_BLOCKING="1" */

__global__ void AATrans(mtxel *mtx, mtxel *dest, int dim,
												int blksize, int maxcache)
{
	int t = (blockDim.x * blockIdx.x + threadIdx.x) * blksize;
	/* Calculate the column the thread is working in.
	 * We are only computing half the matrix,
	 * since the matrix is symmetric along the diagonal.
	 * For a 4x4 matrix, the thread assignment looks as follows:
	 * 1 2 3 4
	 * 2 5 6 7
	 * 3 6 8 9
	 * 4 7 9 A
	 */
	int c = floor((1+2*dim-sqrtf(1+4*dim+4*dim*dim-8*t))/2);
	/* The row follows from the column */
	int r = t - c * dim + c * (c - 1) / 2 + c;
	
	DBGPRINT("Thread %d Initial Position: (%d, %d) with "
					 "dim %d and blocksize %d\n", t, r, c, dim, blksize);

	/* Will be treated as mtxel rows[2 * blksize][dim] 
	 * The first blksize arrays are for the rows of the matrix at r
	 * The second blksize arrays are for the rows of the matrix at c
	 */
	extern __shared__ mtxel rowmem[];
	
	int currentcol = -1;;
	/* Compute A A^T */
	for(int i = 0; i < blksize; i++) {
		if(c >= 0 && c < dim &&
			 r >= 0 && r < dim) {
			dest[c * dim + r] = 0.0;
			if(currentcol != c) {
				/* Move our current column into fast shared memory */
				for(int k = 0; k < dim && k < maxcache; k++) {
					rowmem[k] = mtx[c * dim + k];
					dest[c * dim + r] += mtx[r * dim + k] * rowmem[k];
				}
				currentcol = c;
			}
			else {
				for(int k = 0; k < dim && k < maxcache; k++)
					dest[c * dim + r] += mtx[r * dim + k] * rowmem[k];
			}
			for(int k = maxcache; k < dim; k++) {
				dest[c * dim + r] += mtx[r * dim + k] * mtx[c * dim + k];
			}
			DBGPRINT("t: %d, Pos: (%d, %d), value: %f\n", t, blksize, r, c, dest[c * dim + r]);
			dest[r * dim + c] = dest[c * dim + r];
			/* Move to the next element to compute */

			if(r >= dim) {
				c++;
				r = c;
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

int mpcount = 0;
int maxsharedmem = 0;

void computeCUDA(mtxel *hostmtx, mtxel *dest, int dim)
{
	if(dim == 1) {
		printf("0.000000, ");
		return;
	}
	mtxel *devmtx, *devdest;

	cudaMalloc(&devmtx, sizeof(mtxel[dim * dim]));
	cudaMalloc(&devdest, sizeof(mtxel[dim * dim]));
	if(!devmtx || !devdest)
		return;
	cudaMemset(devdest, 0, sizeof(mtxel[dim * dim]));
	cudaMemcpy(devmtx, hostmtx, sizeof(mtxel[dim * dim]), cudaMemcpyHostToDevice);

	/* blksize is the number of rows and columns a thread works with */
	int blksize = 1;
	/* We want to keep all processors busy, so use some number of
	 * blocks higher than the number of processors.
	 * 4 seems to be the magic number, after which performance doesn't
	 * significantly improve.
	 */
	const int maxblocks = mpcount * 4;
	int blocks = dim * (dim + 1) / 2;
	/* Now calculate the size of the blocks each thread works with,
	 * and add one extra block for the common case
	 */
	while(blocks > maxblocks) {
		blksize *= 2;
		blocks /= 2;
	}
	blocks++;
	/* There are issues with using all the shared memory (not unexpected),
	 * so use a large fraction of it instead
	 */
	maxsharedmem *= 3 / 4;

	struct timeval t1, t2, elapsed;
	gettimeofday(&t1, NULL);
	/* Note that performance metrics must be collected with CUDA_LAUNCH_BLOCKING set
	 */
	AATrans <<< blocks, 1, maxsharedmem >>>
		(devmtx, devdest, dim, blksize, maxsharedmem / sizeof(mtxel));
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("CUDA Error %d: %s\n", err, cudaGetErrorString(err));
	}
	gettimeofday(&t2, NULL);
	timersub(&t2, &t1, &elapsed);
	printf("%d.%06d, ",
				 elapsed.tv_sec, elapsed.tv_usec);

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
	mtxel *devmtx1, *devmtx2, *devdest;
	err = cublasAlloc(dim * dim, sizeof(mtxel), (void **)&devmtx1);
	checkCUBLAS(err, "Allocated dev matrix 1");
	err = cublasAlloc(dim * dim, sizeof(mtxel), (void **)&devdest);
	checkCUBLAS(err, "Allocated dev dest matrix");
	err = cublasSetMatrix(dim, dim, sizeof(mtxel), (void *)mtx, dim, (void *)devmtx1, dim);
	checkCUBLAS(err, "Set dev matrix 1");

	struct timeval t1, t2, elapsed;
	gettimeofday(&t1, NULL);

	cublasDgemm('T', 'N', dim, dim, dim, 1.0,
		    devmtx1, dim, devmtx1, dim, 0.0, devdest, dim);

	gettimeofday(&t2, NULL);
	timersub(&t2, &t1, &elapsed);
	printf("%d.%06d, ",
				 elapsed.tv_sec, elapsed.tv_usec);

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
	mpcount = dev.multiProcessorCount;
	maxsharedmem = dev.multiProcessorCount;
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
