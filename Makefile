all:
	mpicc -std=c99 -fopenmp fem.c new.c old.c -o fem