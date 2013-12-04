
#include <cuda.h>
#include <cublas.h>

#include "pa2.h"

__global__ void AATrans(mtxel *mtx, mtxel *dest, int dim)
{
	int t = blockDim.x * blockIdx.x + threadIdx.x;
	/* Calculate the column the thread is working in.
	 * We are only computing half the matrix,
	 * since the matrix is symmetric along the diagonal.
	 */
	int c = floor((1+2*dim-sqrtf(1+4*dim+4*dim*dim-8*t))/2);
	/* The row follows from the column */
	int r = t - c * dim + c * (c - 1) / 2 + c;
	DBGPRINT("Dim: %d, Thread: %d, Row: %d, Column: %d\n", dim, t, r, c);
	if(c >= 0 && c < dim && r >= 0 && r < dim) {
		dest[c * dim + r] = 0.0;
		for(int k = 0; k < dim; k++)
			dest[c * dim + r] += mtx[r * dim + k] * mtx[c * dim + k];
		dest[r * dim + c] = dest[c * dim + r];
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
	cudaMemset(devdest, 0.0, dim * dim);

	AATrans <<<dim * (dim + 1) / 2, 1>>> (devmtx, devdest, dim);
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

	cublasSgemm('T', 'N', dim, dim, dim, 1.0,
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
	cublasInit();
	return 0;
}

void shutdownCUDA()
{
	cublasShutdown();
	
}
