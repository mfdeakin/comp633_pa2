
#define _BSD_SOURCE

#include <getopt.h>
#include <sys/time.h>
#include <time.h>

#include "pa2.h"

void printMatrix(mtxel *mtx, int dim);
void fillMatrix(mtxel *mtx, int dim);
void computeStandard(mtxel *mtx, mtxel *dest, int dim);
int compareMtx(mtxel *rhs, mtxel *lhs, int dim, float tolerance);

int main(int argc, char **argv)
{
	/* Default Values */
	int dim = 0;
	float tolerance = 0.0001;
	for(;;) {
		static struct option longOpts[] =
			{
				{"dim", required_argument, 0, 'd'},
				{"tolerance", required_argument, 0, 't'},
				{0, 0, 0, 0}
			};
		int index = 0;
		int curopt = getopt_long(argc, argv, "d:t:", longOpts, &index);
		if(curopt == -1)
			break;
		switch(curopt) {
		case 'd':
			sscanf(optarg, "%d", &dim);
			break;
		case 't':
			sscanf(optarg, "%f", &tolerance);
			break;
		}
	}
	if(dim < 1) {
		printf("Minimum matrix size of 1x1. Exiting...");
		return 1;
	}
	srand(time(0));
	size_t size = dim * dim;
	mtxel *mtx = malloc(sizeof(mtxel[size])),
		*standard = malloc(sizeof(mtxel[size])),
		*cuda = malloc(sizeof(mtxel[size])),
		*cublas = malloc(sizeof(mtxel[size]));
	if(!mtx || !standard || !cuda || !cublas) {
		printf("Could not allocate memory for an nxn matrix. Exiting...");
		return 2;
	}
	fillMatrix(mtx, dim);

	struct {
		mtxel *dest;
		void (* compute)(mtxel *src, mtxel *dest, int dim);
		mtxel *compare;
		char *name;
	} computes[] = {{standard, &computeStandard, NULL, "Standard"},
									{cuda, &computeCUDA, standard, "CUDA"},
									{cublas, &computeCUBLAS, standard, "CUBLAS"}};
	for(int i = 0; i < 3; i++) {
		struct timeval t1, t2, elapsed;
		gettimeofday(&t1, NULL);
		computes[i].compute(mtx, computes[i].dest, dim);
		gettimeofday(&t2, NULL);
		timersub(&t2, &t1, &elapsed);
		printf("%s calculation took %d.%06d seconds\n",
					 computes[i].name, elapsed.tv_sec, elapsed.tv_usec);
		if(computes[i].compare &&
			 compareMtx(computes[i].compare, computes[i].dest, dim, tolerance)) {
			printf("%s Matrix differs from standard!\n", computes[i].name);
			printf("M M^T =\n");
			printMatrix(computes[i].dest, dim);
			printf("Standard result:\n");
			printMatrix(computes[i].compare, dim);
		}
	}	
	free(mtx);
	free(cuda);
	free(standard);
	return 0;
}

int compareMtx(mtxel *rhs, mtxel *lhs, int dim, float tolerance)
{
	for(int i = 0; i < dim * dim; i++) {
		if(abs(rhs[i] - lhs[i]) > tolerance)
			return 1;
	}
	return 0;
}

void fillMatrix(mtxel *mtx, int dim)
{
	for(int i = 0; i < dim * dim; i++)
		mtx[i] = (rand() - 1.0) / RAND_MAX;
}

void computeStandard(mtxel *mtx, mtxel *dest, int dim)
{
	for(int i = 0; i < dim; i++) {
		for(int j = i; j < dim; j++) {
			dest[i * dim + j] = 0;
			for(int k = 0; k < dim; k++)
				dest[i * dim + j] += mtx[i * dim + k] * mtx[j * dim + k];
			dest[i * dim + j] = dest[i * dim + j];
			dest[j * dim + i] = dest[i * dim + j];
		}
	}
}

void printMatrix(mtxel *mtx, int dim)
{
	printf("[\n");
	for(int i = 0; i < dim; i++) {
		for(int j = 0; j < dim; j++)
			printf("%.4f   ", mtx[i * dim + j]);
		printf("\n");
	}
	printf("]\n");
}
