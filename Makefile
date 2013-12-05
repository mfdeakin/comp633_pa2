
CC=gcc
CFLAGS=-g -std=c99 -fms-extensions
NVCC=nvcc
NVCFLAGS=-g -m64 -gencode arch=compute_20,code=sm_20 -prec-sqrt=true
LIBS=-L/usr/lib/nvidia-current -lcudart -lcublas -lm

pa2: main.o matrix.o
	$(NVCC) -o pa2 main.o matrix.o $(LIBS)

main.o: main.c pa2.h
	$(CC) $(CFLAGS) -o main.o -c main.c

matrix.o: matrix.cu pa2.h
	$(NVCC) $(NVCFLAGS) -o matrix.o -c matrix.cu

clean:
	rm -f *.o pa2
