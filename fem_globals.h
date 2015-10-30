#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include "mpi.h"

//--------------- FUNCTIONS -------------------------------

double runNew();
double runOld();

//--------------- VARIBLES --------------------------------

long int NSUB;
long int NL;
int THREADS;
int TASKS;
FILE *fp_sol;
FILE *fp_out;
int Argc;
char **Argv;


