
CC=gcc
CFLAGS=-g -std=c99 -fms-extensions
NVCC=nvcc
NVCFLAGS=-g -m64 -gencode arch=compute_20,code=sm_20
LIBS=-L/usr/lib/nvidia-current -lcudart -lcublas -lm

pa2: main.o matrix.o
	$(NVCC) -o pa2 main.o matrix.o $(LIBS)

matrix.o: matrix.cu
	$(NVCC) $(NVCFLAGS) -o matrix.o -c matrix.cu

clean:
	rm -f *.o pa2