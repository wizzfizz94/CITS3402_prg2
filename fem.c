#include "fem_globals.h"

/**
*			Computes and returns average of an array of doubles
*
*/
double Average(double ary[]){
	double ave = 0;
	int i;
	int length = (int)(sizeof(ary)/sizeof(double));
	for(i=0;i < length;i++){
		ave += ary[i];
	}
	return ave / (double)length;
}

/******************************************************************************/

/**
*     checks output files for equality to determind correctness
*     of new version. 
*/
int check(){

  FILE *fp1 = fopen("old_sol.txt","r");
  FILE *fp2 = fopen("new_sol.txt","r");

  int ch1, ch2;

   if (fp1 == NULL) {
      printf("Cannot open for reading ld_out.txt");
      exit(1);
   } else if (fp2 == NULL) {
      printf("Cannot open for reading new_out.txt");
      exit(1);
   } else {
      ch1 = getc(fp1);
      ch2 = getc(fp2);
      //printf("%d, %d\n",ch1,ch2);
 
      while ((ch1 != EOF) && (ch2 != EOF) && (ch1 == ch2)) {
         ch1 = getc(fp1);
         ch2 = getc(fp2);
      }
 
      if (ch1 == ch2){
        fclose(fp1);
        fclose(fp2);
        return 0;
      } else if (ch1 != ch2) {
        fclose(fp1);
        fclose(fp2);
        return 1;
      }
   }
   return 1;
}

/******************************************************************************/


int main(int argc, char const **argv)
{
	int TRIALS;
	bool error = false;

	//get NSUB, threads, tasks and trails from argument
	if(argc != 6){
		error = true;
	} else if((NSUB = atoi(argv[1])) == 0) {
		printf("Invalid subdivison size.\n");
		error = true;
    } else if ((NL = atoi(argv[2])) == 0){
        printf("Invalid base function degree.\n");
        error = true;
	} else if ((THREADS = atoi(argv[3])) == 0){
		printf("Invalid number of threads.\n");
		error = true;
	} else if ((TASKS = atoi(argv[4])) == 0){
		printf("Invalid number of tasks.\n");
		error = true;
	} else if ((TRIALS = atoi(argv[5])) == 0){
		printf("Invalid number of trails.\n");
		error = true;
	}

	if(error){
		printf("Usage: ./fem [SUB_SIZE] [NL] [NUM_THREADS] [NUM_TASKS] [TRIALS]\n");
		exit(EXIT_FAILURE);
	}

  	printf("NSUB = %ld, NL = %ld, threads = %d,tasks = %d, trails = %d\n",
    NSUB,NL,THREADS,TASKS,TRIALS);


	//set number of threads
	//omp_set_num_threads(THREADS);

	//wipe solutions files
	fp_sol = fopen("old_sol.txt","w");
	fclose(fp_sol);
	fp_sol = fopen("new_sol.txt","w");
	fclose(fp_sol);

	//time spent on execution for each trail
	double time_spent[TRIALS];

	int ver = 0;
	double old_time[TRIALS];
	double new_time[TRIALS];
	int oc = 0;
	int nc = 0;

	//toggle between versions and perform TRIALS
	for(int i=0;i<(TRIALS*2);i++){
		if(ver == 0){
			printf("Running old version: trail %d...\n",oc+1);
			old_time[oc]= runNew(argc, argv);
			printf("Succesfully complete: Time taken: %fsec\n",old_time[oc]);
			oc++;
			ver++;
		} else if (ver == 1){
			printf("Running new version trail %d...\n",nc+1);
			/*fork and disguard parent, so MPI can be 
			called multiple times */ 
			if(fork() != 0){
				exit(EXIT_SUCCESS);
			}
			new_time[nc]= runOld(argc, argv);
			printf("Succesfully complete: Time taken: %fsec\n",new_time[nc]);
			nc++;
			ver = ver - 1;
		}
	}

	//print results
	printf("********************************** RESULTS ************************************\n");
	double o_time = Average(old_time);
	double n_time = Average(new_time);
  	printf("NSUB = %ld, NL = %ld, threads = %d,tasks = %d, trails = %d\n",
    NSUB,NL,THREADS,TASKS,TRIALS);
    printf("New version Average Time: %fsec\n",n_time);
	printf("Old version Average Time: %fsec\n",o_time);
	printf("Speed up: %fsec\n", (o_time - n_time));

	//CHECK FOR CORRCTNESS
	if(check()==0){
		printf("CORRECT: Outputs are identical.\n");
	} else {
		printf("INCORRECT: Outputs are not identical.\n");
	}

	return 0;
}