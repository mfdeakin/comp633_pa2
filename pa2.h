
#ifndef PA2_H
#define PA2_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef double mtxel;


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __DEBUG__
#define DBGPRINT(...) printf(__VA_ARGS__);
#else
#define DBGPRINT(...) while(false)
#endif

	void computeCUDA(mtxel *mtx, mtxel *dest, int dim);
	void computeCUBLAS(mtxel *mtx, mtxel *dest, int dim);

	int initCUDA();
	void shutdownCUDA();

#ifdef __cplusplus
}
#endif

#endif
