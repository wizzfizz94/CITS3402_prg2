all: fem old new

fem: fem.c
	gcc -std=c99 fem.c -o fem

old: old.c fem.c
	gcc -std=c99 -fopenmp old.c -o old

new: new.c fem.c
	mpicc -std=c99 -fopenmp new.c -o new