#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include "mpi.h"

//FUNCTON DECLARTIONS
/******************************************************************************/
int main(int argc, char **argv);
void init ();
void output ();
void assemble ();
double ff ( double x );
void geometry ();
void phi ( int il, double x, double *phii, double *phiix, double xleft, 
          double xrite );
double pp ( double x );
void prsys ();
double qq ( double x );
void solve ();
void timestamp ( void );

//GLOBALS
/******************************************************************************/
  #define MASTER 0

  int numprocs, rank;
  MPI_Status status;

  /* NSUB + 1 */
  int slaveSize1;
  int masterSize1;
  /* NSUB */
  int slaveSize2;
  int masterSize2;

  long int NSUB;
  long int NL;
  int THREADS;
  FILE *fp_sol;
  FILE *fp_out;

  double *adiag;
  double *aleft;
  double *arite;
  double *f;
  double *h;
  int ibc;
  int *indx;
  int *node;
  int nquad;
  int nu;
  double ul;
  double ur;
  double xl;
  double *xn;
  double *xquad;
  double xr;

/******************************************************************************/
void printState(){
  printf("\n");
  printf("ibc %d\n",ibc);
  printf("nquad %d\n",nquad);
  printf("nu %d\n",nu);
  printf("ul %f\n", ul);
  printf("ur %f\n", ur);
  printf("xl %f\n", ul);
  printf("xr %f\n", xr);

  printf("adiag        aleft     arite       f          h         indx    node xn         xquad\n");
  for (int i = 0; i < NSUB; ++i)
  {
    printf("%10f %10f %10f %10f %10f %5d %5d %10f %10f\n",adiag[i],aleft[i],arite[i], f[i], h[i], indx[i], node[i], xn[i], xquad[i]);
  }
}


/**
*
*/
int main(int argc, char **argv){

  bool error = false;

  //get NSUB, threads, tasks and trails from argument
  if(argc != 4){
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
  }

  if(error){
    printf("Usage: mpirun -np [TASKS] new [SUB_SIZE] [NL] [NUM_THREADS]\n");
    exit(EXIT_FAILURE);
  }

    if((fp_out = fopen("new_out.txt", "a")) == NULL || 
        (fp_sol = fopen("new_sol.txt", "a")) == NULL){
            printf("New Version files not found.\n");
            exit(EXIT_FAILURE);
    }

    //Allocate array memory
    adiag = (double *)malloc(sizeof(double)*(double)(NSUB+1));
    aleft = (double *)malloc(sizeof(double)*(double)(NSUB+1));
    arite = (double *)malloc(sizeof(double)*(double)(NSUB+1));
    f = (double *)malloc(sizeof(double)*(double)(NSUB+1));
    h = (double *)malloc(sizeof(double)*(double)(NSUB));
    indx = (int *)malloc(sizeof(int)*(int)(NSUB+1));
    node = (int *)malloc(sizeof(int)*((int)NL*(int)NSUB));
    xn = (double *)malloc(sizeof(double)*(double)(NSUB+1));
    xquad = (double *)malloc(sizeof(double)*(double)(NSUB));

    //START TIMER//
    double begin, end, time_spent;
    begin = omp_get_wtime();

    //set number of threads
    omp_set_num_threads(THREADS);

    /****************** MPI Initialisations ***************/
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* set up block sizes for MPI work */
    slaveSize1 = (NSUB+1) / numprocs;
    masterSize1 = slaveSize1 + ((NSUB+1) % numprocs);
    slaveSize2 = NSUB / numprocs;
    masterSize2 = slaveSize2 + (NSUB % numprocs);

    printf("MPI: Process %d of %d\n", rank, numprocs);

    /*  If we are the master process
        Master coordinates the slaves */
    if (rank == MASTER){
      printf("MASTER: Number of processes is: %d\n",numprocs);


      timestamp ();

      fprintf (fp_out, "\n" );
      fprintf (fp_out, "FEM1D\n" );
      fprintf (fp_out, "  C version\n" );
      fprintf (fp_out, "\n" );
      fprintf (fp_out, "  Solve the two-point boundary value problem\n" );
      fprintf (fp_out, "\n" );
      fprintf (fp_out, "  - d/dX (P dU/dX) + Q U  =  F\n" );
      fprintf (fp_out, "\n" );
      fprintf (fp_out, "  on the interval [XL,XR], specifying\n" );
      fprintf (fp_out,"  the value of U or U' at each end.\n" );
      fprintf (fp_out, "\n" );
      fprintf (fp_out,"  The interval [XL,XR] is broken into NSUB = %ld subintervals\n", NSUB );
      fprintf (fp_out, "  Number of basis functions per element is NL = %ld\n", NL );
    }

    //Initialize the data.
    init ();

    //Compute the geometric quantities.
    geometry ();
    
    //Assemble the linear system.
    assemble ();

    if(rank == MASTER){
      //Print out the linear system.
      prsys ();

      //Solve the linear system.
      solve ();

      //Print out the solution.
      output ();
    }

    //Terminate.
    fprintf (fp_out, "\n" );
    fprintf (fp_out,"FEM1D:\n" );
    fprintf (fp_out, "  Normal end of execution.\n" );

    fprintf ( fp_out,"\n" );

    //END TIMER//
    end = omp_get_wtime();
    time_spent = end - begin;
    timestamp ( );

    //CLOSE STREAMS
    fclose(fp_out);
    fclose(fp_sol);

    //FREE MEMORY
    free(adiag); 
    free(aleft);
    free(arite); 
    free(f); 
    free(h); 
    free(indx); 
    free(node); 
    free(xn); 
    free(xquad);


  MPI_Finalize();

  if(rank == MASTER){
    FILE *fp_time = fopen("times.txt","a");
    fprintf(fp_time, "%f\n", time_spent);
  }

  return 0;
}

/******************************************************************************/
  void init (){
  /*
    IBC declares what the boundary conditions are.
  */
    ibc = 1;
  /*
    NQUAD is the number of quadrature points per subinterval.
    The program as currently written cannot handle any value for
    NQUAD except 1.
  */
    nquad = 1;
  /*
    Set the values of U or U' at the endpoints.
  */
    ul = 0.0;
    ur = 1.0;
  /*
    Define the location of the endpoints of the interval.
  */
    xl = 0.0;
    xr = 1.0;
  /*
    Print out the values that have been set.
  */
    if(rank == MASTER){
      fprintf (fp_out, "\n" );
      fprintf ( fp_out,"  The equation is to be solved for\n" );
      fprintf ( fp_out,"  X greater than XL = %f\n", xl );
      fprintf ( fp_out,"  and less than XR = %f\n", xr );
      fprintf ( fp_out,"\n" );
      fprintf (fp_out, "  The boundary conditions are:\n" );
      fprintf (fp_out, "\n" );

      if ( ibc == 1 || ibc == 3 )
      {
        fprintf (fp_out, "  At X = XL, U = %f\n", ul );
      }
      else
      {
        fprintf ( fp_out,"  At X = XL, U' = %f\n", ul );
      }

      if ( ibc == 2 || ibc == 3 )
      {
        fprintf (fp_out, "  At X = XR, U = %f\n", ur );
      }
      else
      {
        fprintf (fp_out, "  At X = XR, U' = %f\n", ur );
      }

      fprintf (fp_out, "\n" );
      fprintf (fp_out, "  Number of quadrature points per element is %d\n", nquad );
    }

    return;
  }
/******************************************************************************/
void geometry (){
  
  long int i;
  int offset = 0;

  /* MASTER WORK */
  if(rank == MASTER){

    /* move offset to end of master block */
    offset = masterSize1;

    /* send offset to slaves */
    for (int i = 1; i < numprocs; i++)
    {
      MPI_Send(&offset,1,MPI_INT,i,100,MPI_COMM_WORLD);
      offset += slaveSize1;
    }

    /* master does it work on XN with openmp */
    int end = masterSize1;
    if(end < NSUB+1){
      end++;
    }
    #pragma omp parallel for
    for ( i = 0; i < end; i++ )
    {
      xn[i]  =  ( ( double ) ( NSUB - i ) * xl 
      + ( double )          i   * xr ) 
      / ( double ) ( NSUB );
    }

    /* move offset to end of master block */
    offset = masterSize1;

    /* receive data from slaves */
    for (int i = 1; i < numprocs; i++)
    {
      MPI_Recv(&xn[offset],slaveSize1,MPI_DOUBLE,i,101,MPI_COMM_WORLD,&status);
      offset += slaveSize1;
    }

    /* update offset to end of new master block */
    offset = masterSize2;

    /* send next offset to slaves */
    for (int i = 1; i < numprocs; i++)
    {
      MPI_Send(&offset,1,MPI_INT,i,102,MPI_COMM_WORLD);
      offset += slaveSize2;
    }

    /* master does its work with openmp */
    #pragma omp parallel for
    for ( i = 0; i < masterSize2; i++ )
    {
      // printf("%d\n",omp_get_num_threads());
      h[i] = xn[i+1] - xn[i];
      xquad[i] = 0.5 * ( xn[i] + xn[i+1] );
      node[0+i*2] = i;
      node[1+i*2] = i + 1;
    }

    /* update offset to end of new master block */
    offset = masterSize2;

    /* master gets data */
    for (int i = 1; i < numprocs; i++)
    {
      MPI_Recv(&h[offset],slaveSize2,MPI_DOUBLE,i,103,MPI_COMM_WORLD,&status);
      MPI_Recv(&xquad[offset],slaveSize2,MPI_DOUBLE,i,104,MPI_COMM_WORLD,&status);
      MPI_Recv(&node[offset*2],slaveSize2*2,MPI_INT,i,105,MPI_COMM_WORLD,&status);
      offset += slaveSize2;
    }

    /* perform prints sequentially */
    fprintf ( fp_out,"\n" );
    fprintf (fp_out, "  Node      Location\n" );
    fprintf (fp_out, "\n" );
    for (i=0;i<=NSUB;i++)
    {
      fprintf (fp_out, "  %8ld  %14f \n", i, xn[i] );
    }
    fprintf (fp_out, "\n" );
    fprintf (fp_out, "Subint    Length\n" );
    fprintf ( fp_out,"\n" );
    for (i=0;i<=NSUB;i++)
    {
      fprintf (fp_out, "  %8ld  %14f\n", i+1, h[i] );
    }
    fprintf (fp_out, "\n" );
    fprintf (fp_out, "Subint    Quadrature point\n" );
    fprintf ( fp_out,"\n" );
    for (i=0;i<=NSUB;i++)
    {
      fprintf ( fp_out,"  %8ld  %14f\n", i+1, xquad[i] );
    }
    fprintf ( fp_out,"\n" );
    fprintf ( fp_out,"Subint  Left Node  Right Node\n" );
    fprintf (fp_out, "\n" );
    for (i=0;i<=NSUB;i++)
    {
      fprintf (fp_out, "  %8ld  %8d  %8d\n", i+1, node[0+i*2], node[1+i*2] );
    }

    /*
    Starting with node 0, see if an unknown is associated with
    the node.  If so, give it an index.
    */
    nu = 0;
    /*
    Handle first node.
    */
    i = 0;
    if ( ibc == 1 || ibc == 3 )
    {
      indx[i] = -1;
    }
    else
    {
      nu = nu + 1;
      indx[i] = nu;
    }
    /*
    Handle nodes 1 through nsub-1
    */

    /* cannot parallelize due to nu */
    for ( i = 1; i < NSUB; i++ )
    {
      nu = nu + 1;
      indx[i] = nu;
    }
    /*
    Handle the last node.
    /*/
    i = NSUB;

    if ( ibc == 2 || ibc == 3 )
    {
      indx[i] = -1;
    }
    else
    {
      nu = nu + 1;
      indx[i] = nu;
    }

    fprintf ( fp_out,"\n" );
    fprintf ( fp_out,"  Number of unknowns NU = %8d\n", nu );
    fprintf (fp_out, "\n" );
    fprintf (fp_out, "  Node  Unknown\n" );
    fprintf (fp_out, "\n" );
    for ( i = 0; i <= NSUB; i++ )
    {
      fprintf (fp_out, "  %8ld  %8d\n", i, indx[i] );
    }
  }

  /* SLAVE WORK */
  if(rank != MASTER){

    /* receive offset from master */
    MPI_Recv(&offset,1,MPI_INT,MASTER,100,MPI_COMM_WORLD,&status);

    /* slave does work with openmp */
    int end = offset+slaveSize1;
    if(end < NSUB+1){
      end++;
    }
    #pragma omp parallel for
    for ( i = offset-1; i < end; i++ )
    {
      xn[i]  =  ( ( double ) ( NSUB - i ) * xl 
      + ( double )          i   * xr ) 
      / ( double ) ( NSUB );
    }

    /* send data to master */
    MPI_Send(&xn[offset],slaveSize1,MPI_DOUBLE,MASTER,101,MPI_COMM_WORLD);
  
    /* receive next offset from master */
    MPI_Recv(&offset,1,MPI_INT,MASTER,102,MPI_COMM_WORLD,&status);

    /* slave does more openmp work */
    #pragma omp parallel for
    for ( i = offset; i < (offset + slaveSize2); i++ )
    {
      // printf("%d\n",omp_get_num_threads());
      h[i] = xn[i+1] - xn[i];
      xquad[i] = 0.5 * ( xn[i] + xn[i+1] );
      node[0+i*2] = i;
      node[1+i*2] = i + 1;
    }

    /* send data to master for h, xquad and node */
    MPI_Send(&h[offset],slaveSize2,MPI_DOUBLE,MASTER,103,MPI_COMM_WORLD);
    MPI_Send(&xquad[offset],slaveSize2,MPI_DOUBLE,MASTER,104,MPI_COMM_WORLD);
    MPI_Send(&node[offset*2],slaveSize2*2,MPI_INT,MASTER,105,MPI_COMM_WORLD);
  }
}

/******************************************************************************/

/**
*
*/
void assemble (){

  int offset = 0;
  int slaveSizeNU = nu / numprocs;
  
  /* MASTER WORK */
  if(rank == MASTER){
    int masterSizeNU = slaveSizeNU + (nu % numprocs);
    double aij;
    double he;
    int i;
    int ie;
    int ig;
    int il;
    int iq;
    int iu;
    int jg;
    int jl;
    int ju;
    double phii;
    double phiix;
    double phij;
    double phijx;
    double x;
    double xleft;
    double xquade;
    double xrite;

    /* set offset to end of master block */
    offset = masterSizeNU;

    /* send offsets to slaves */
    for (int i = 1; i < numprocs; i++)
    {
      MPI_Send(&offset,1,MPI_INT,i,110,MPI_COMM_WORLD);
      offset += slaveSizeNU;
    }

    /* master does its work with openmp */
    #pragma omp parallel for
    for ( i = 0; i < masterSizeNU; i++ )
    {
      f[i] = 0.0;
      adiag[i] = 0.0;
      aleft[i] = 0.0;
      arite[i] = 0.0;
    }

    /* set offset to end of master block */
    offset = masterSizeNU;

    /* master receives data from slaves */
    for (int i = 1; i < numprocs; i++)
    {
      MPI_Recv(&f[offset],slaveSizeNU,MPI_DOUBLE,i,111,MPI_COMM_WORLD,&status);
      MPI_Recv(&adiag[offset],slaveSizeNU,MPI_DOUBLE,i,112,MPI_COMM_WORLD,&status);
      MPI_Recv(&aleft[offset],slaveSizeNU,MPI_DOUBLE,i,113,MPI_COMM_WORLD,&status);
      MPI_Recv(&arite[offset],slaveSizeNU,MPI_DOUBLE,i,114,MPI_COMM_WORLD,&status);
      offset += masterSizeNU;
    }

    /*
      For interval number IE,
    */
    for ( ie = 0; ie < NSUB; ie++ )
    {
      he = h[ie];
      xleft = xn[node[0+ie*2]];
      xrite = xn[node[1+ie*2]];
      /*
      consider each quadrature point IQ,
      */
      for ( iq = 0; iq < nquad; iq++ )
      {
        xquade = xquad[ie];
        /*
        and evaluate the integrals associated with the basis functions
        for the left, and for the right nodes.
        */
        for ( il = 1; il <= NL; il++ )
        {
          ig = node[il-1+ie*2];
          iu = indx[ig] - 1;

          if ( 0 <= iu )
          {
            phi ( il, xquade, &phii, &phiix, xleft, xrite );
            f[iu] = f[iu] + he * ff ( xquade ) * phii;
            /*
            Take care of boundary nodes at which U' was specified.
            */
            if ( ig == 0 )
            {
              x = 0.0;
              f[iu] = f[iu] - pp ( x ) * ul;
            }
            else if ( ig == NSUB )
            {
              x = 1.0;
              f[iu] = f[iu] + pp ( x ) * ur;
            }



            /*
            Evaluate the integrals that take a product of the basis
            function times itself, or times the other basis function
            that is nonzero in this interval.
            */
            for ( jl = 1; jl <= NL; jl++ )
            {

              jg = node[jl-1+ie*2];

              ju = indx[jg] - 1;

              phi ( jl, xquade, &phij, &phijx, xleft, xrite );

              aij = he * ( pp ( xquade ) * phiix * phijx 
               + qq ( xquade ) * phii  * phij   );
              /*
              If there is no variable associated with the node, then it's
              a specified boundary value, so we multiply the coefficient
              times the specified boundary value and subtract it from the
              right hand side.
              */

              if ( ju < 0 )
              {
                if ( jg == 0 )
                {
                  f[iu] = f[iu] - aij * ul;
                }
                else if ( jg == NSUB )
                {               
                  f[iu] = f[iu] - aij * ur;
                }
              }

              /*
              Otherwise, we add the coefficient we've just computed to the
              diagonal, or left or right entries of row IU of the matrix.
              */
              else
              {
                if ( iu == ju )
                {
                  adiag[iu] = adiag[iu] + aij;
                }
                else if ( ju < iu )
                {
                  aleft[iu] = aleft[iu] + aij;
                }
                else
                {
                  arite[iu] = arite[iu] + aij;
                }
              }
            }
            // printf("%d\n",jl );
          }
        }
      }
    }
  }

  /* SLAVE WORK */
  if(rank != MASTER){

    /* receive offset */
    MPI_Recv(&offset,1,MPI_INT,MASTER,110,MPI_COMM_WORLD,&status);

    /* slave does its work with openmp */
    #pragma omp parallel for
    for (int i = offset; i < (offset + slaveSizeNU); i++ )
    {
      f[i] = 0.0;
      adiag[i] = 0.0;
      aleft[i] = 0.0;
      arite[i] = 0.0;
    }

    /* slave sends data to master */
    MPI_Send(&f[offset],slaveSizeNU,MPI_DOUBLE,MASTER,111,MPI_COMM_WORLD);
    MPI_Send(&adiag[offset],slaveSizeNU,MPI_DOUBLE,MASTER,112,MPI_COMM_WORLD);
    MPI_Send(&aleft[offset],slaveSizeNU,MPI_DOUBLE,MASTER,113,MPI_COMM_WORLD);
    MPI_Send(&arite[offset],slaveSizeNU,MPI_DOUBLE,MASTER,114,MPI_COMM_WORLD);

  }
}
  
/******************************************************************************/

   double ff ( double x ){
    double value;

    value = 0.0;

    return value;
  }
/******************************************************************************/

void output (){

  int i;
  double u[NSUB+1];
  int offsetF = 0;
  int offsetIndx = 0;

  if(rank == MASTER){

    fprintf (fp_sol,"\n" );
    fprintf (fp_sol,"  Computed solution coefficients:\n" );
    fprintf (fp_sol, "\n" );
    fprintf (fp_sol,"  Node    X(I)        U(X(I))\n" );
    fprintf (fp_sol,"\n" );

    /* set indx offset to end of master block *
    offset = masterSize1;*/
    /* set f offset one back */

    /* send data and offsets to slaves *
    for (int i = 1; i < numprocs; i++)
    {
      MPI_Send(&offset,1,MPI_INT,i,110,MPI_COMM_WORLD);
      MPI_Send(&)
    }*/


    #pragma omp parallel for
    for ( i = 0; i <= NSUB; i++ )
    {
      /*
      If we're at the first node, check the boundary condition.
      */
      if ( i == 0 )
      {
        if ( ibc == 1 || ibc == 3 )
        {
          u[i] = ul;
        }
        else
        {
          u[i] = f[indx[i]-1];
        }
      }
      /*
      If we're at the last node, check the boundary condition.
      */
      else if ( i == NSUB )
      {
        if ( ibc == 2 || ibc == 3 )
        {
          u[i] = ur;
        }
        else
        {
          u[i] = f[indx[i]-1];
        }
      }
      /*
      Any other node, we're sure the value is stored in F.
      */
      else
      {
        u[i] = f[indx[i]-1];
      }

    }

    for ( i = 0; i <= NSUB; i++ )
    {
        fprintf ( fp_sol,"  %8d  %8f  %14f\n", i, xn[i], u[i] );
    }

  }
}
/******************************************************************************/

  void phi ( int il, double x, double *phii, double *phiix, double xleft, 
    double xrite ){

    if ( xleft <= x && x <= xrite )
    {
      if ( il == 1 )
      {
        *phii = ( xrite - x ) / ( xrite - xleft );
        *phiix =         -1.0 / ( xrite - xleft );
      }
      else
      {
        *phii = ( x - xleft ) / ( xrite - xleft );
        *phiix = 1.0          / ( xrite - xleft );
      }
    }
  /*
    If X is outside of the interval, just set everything to 0.
  */
    else
    {
      *phii  = 0.0;
      *phiix = 0.0;
    }

    return;
  }
/******************************************************************************/

 double pp ( double x ){
    double value;

    value = 1.0;

    return value;
  }
/******************************************************************************/

   void prsys (){

    int i;

    fprintf (fp_out, "\n" );
    fprintf (fp_out,"Printout of tridiagonal linear system:\n" );
    fprintf (fp_out,"\n" );
    fprintf (fp_out,"Equation  ALEFT  ADIAG  ARITE  RHS\n" );
    fprintf (fp_out,"\n" );

    /* print statments left unparallelized for speed up */
    for ( i = 0; i < nu; i++ )
    {
      fprintf (fp_out, "  %8d  %14f  %14f  %14f  %14f\n",
        i + 1, aleft[i], adiag[i], arite[i], f[i] );
    }

    return;
  }
/******************************************************************************/


   double qq ( double x ){
    double value;

    value = 0.0;

    return value;
  }
/******************************************************************************/

   void solve (){

    int i;
  /*
    Carry out Gauss elimination on the matrix, saving information
    needed for the backsolve.
  */
    arite[0] = arite[0] / adiag[0];

    for ( i = 1; i < nu - 1; i++ )
    {
      adiag[i] = adiag[i] - aleft[i] * arite[i-1];
      arite[i] = arite[i] / adiag[i];
    }
    adiag[nu-1] = adiag[nu-1] - aleft[nu-1] * arite[nu-2];
  /*
    Carry out the same elimination steps on F that were done to the
    matrix.
  */
    f[0] = f[0] / adiag[0];

    for ( i = 1; i < nu; i++ )
    {
            //printf("%f\n",f[i]);

      f[i] = ( f[i] - aleft[i] * f[i-1] ) / adiag[i];
    }
    
  /*
    And now carry out the steps of "back substitution".
  */
    for ( i = nu - 2; 0 <= i; i-- )
    {
      f[i] = f[i] - arite[i] * f[i+1];
    }

    return;
  }
/******************************************************************************/

  void timestamp ( void ){
    
  # define TIME_SIZE 40

     char time_buffer[TIME_SIZE];
    const struct tm *tm;
  //  size_t len;
    time_t now;

    now = time ( NULL );
    tm = localtime ( &now );

    /*len =*/ strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

    fprintf ( fp_out,"%s\n", time_buffer );

    return;
  # undef TIME_SIZE
  }