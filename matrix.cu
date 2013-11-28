
#include <cuda.h>

#include "pa2.h"

__global__ void AATrans(mtxel *mtx, mtxel *dest, int dim)
{
	int t = threadIdx.x;
	/* Calculate the column the thread is working in.
	 * We are only computing half the matrix,
	 * since the matrix is symmetric along the diagonal.
	 */
	int c = floor((1+2*dim-sqrtf(1+4*dim+4*dim*dim-8*t))/2);
	/* The row follows from the column */
	int r = t - c * dim + c * (c - 1) / 2 + c;
	/* printf("Dim: %d, Thread: %d, Row: %d, Column: %d, sqrt(%d): %f -> %f\n", dim, t, r, c, */
	/* 			 1 + 4 * dim + 4 * dim * dim - 8 * t, sqrtf(1+4*dim+4*dim*dim-8*t), */
	/* 			 1 + 2 * dim - sqrtf(1+4*dim+4*dim*dim-8*t)); */
	if(c >= 0 && c < dim && r >= 0 && r < dim) {
		dest[c * dim + r] = 0.0;
		for(int k = 0; k < dim; k++)
			dest[c * dim + r] += mtx[r * dim + k] * mtx[c * dim + k];
		dest[r * dim + c] = dest[c * dim + r];
	}
}

void computeCUDA(mtxel *hostmtx, mtxel *dest, int dim)
{
	mtxel *devmtx, *devdest;
	cudaMalloc(&devmtx, sizeof(mtxel[dim * dim]));
	cudaMalloc(&devdest, sizeof(mtxel[dim * dim]));
	if(!devmtx)
		return;
	cudaMemcpy(devmtx, hostmtx, sizeof(mtxel[dim * dim]), cudaMemcpyHostToDevice);
	
	AATrans <<<1, dim * (dim + 1) / 2>>> (devmtx, devdest, dim);
	
	cudaMemcpy(dest, devdest, sizeof(mtxel[dim * dim]), cudaMemcpyDeviceToHost);
	cudaFree(devmtx);
	cudaFree(devdest);
}

void computeCUBLAS(mtxel *mtx, mtxel *dest, int dim)
{
}
