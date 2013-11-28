
#ifndef PA2_H
#define PA2_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef float mtxel;

#ifdef __cplusplus
extern "C" {
#endif

	void computeCUDA(mtxel *mtx, mtxel *dest, int dim);
	void computeCUBLAS(mtxel *mtx, mtxel *dest, int dim);

#ifdef __cplusplus
}
#endif

#endif
