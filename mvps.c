#include<stdio.h>
#include<stdlib.h>
#include "mpi.h"

void printMatrixA(unsigned long int M, unsigned long int K,double matrix[][K]){
/*------------------
 * prints the product
 */
int m,k;

printf("Matrix: \n");

for (m=0;m<M;m++)
{ 
 for (k=0;k<K;k++)
  {
  printf(" %8.4f\t", matrix[m][k]);
  }
  printf("\n");
}
}

void printMatrixB(unsigned long int K,double matrix[K]){
/*------------------
 * prints the product
 */
int m,k;

printf("Matrix: \n");
 
 for (k=0;k<K;k++)
  {
  printf(" %8.4f\t\n", matrix[k]);
  }
}

void getIdentity(unsigned long int M, unsigned long int N, double matrix[][N]){
/*------------------
 * generates the identity matrix
 */
int m,n;

for (m=0;m<M;m++)
{
 for (n=0;n<N;n++)
 {
  if (m==n)
   {
   matrix[m][n]=1.;    
   }
  else
   {
   matrix[m][n]=0.; 
   } 
 }
 }

}

void getReverseIdentity(unsigned long int M, unsigned long int N, double matrix[][N]){
/*------------------
 * generates the reverse of the identity matrix
 */
int m,n;

for (m=0;m<M;m++)
{
 for (n=0;n<N;n++)
 {
  if (m==N-n-1)
   {
   matrix[m][n]=1.;    
   }
  else
   {
   matrix[m][n]=0.; 
   }
 }
}

}

void getRandomA(unsigned long M, unsigned long N, double matrix[][N]){ 

/*------------------
 * generates a matrix filled with random numbers (A)
 */
static const int seed=12345;
static const double upper_limit=1.;
int m,n;

srand(seed); //set a seed 

for (m=0;m<M;m++) 
{
 for (n=0;n<N;n++) 
 {
  matrix[m][n]=((double)rand()/(double)(RAND_MAX))*upper_limit;
 }
}
}

void getRandomB(unsigned long N, double matrix[N]){ 

/*------------------
 * generates a matrix filled with random numbers (A)
 */
static const int seed=6789;
static const double upper_limit=1.;
int m,n;

srand(seed); //set a seed 

 for (n=0;n<N;n++) 
 {
  matrix[n]=((double)rand()/(double)(RAND_MAX))*upper_limit;
 }
}

//////////////////////////////////////////////////////////
main(int argc, char *argv[]) {
/*------------------
 * computes the product A*b=c, where
 *          A: MxN;
 *          b: Nx1;    
 *          c: Mx1.  
 */
static const unsigned long int M=5, N=5;
double A[M][N], b[N], c[M], buff[N], ans;
double tStart,tInit,t0calc,t1calc,tEnd,tcalc,tcomm,t0comm,t1comm,t2comm,t3comm;
int noprocs, nid;
long int noworkers,rowsent;
int i,n,m,sender,anstype,crow;
static const int MASTER=0;

tStart=MPI_Wtime();

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &noprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &nid);
MPI_Status status;

tInit=MPI_Wtime();

noworkers=noprocs-1;

// MASTER   

if (nid==MASTER) { 

/*
 * generate matrices
 */
// getIdentity(M,N,A);    
// getReverseIdentity(M,N,A);
 getRandomA(M,N,A);
 getRandomB(N,b);
 printMatrixA(M,N,A);
 printMatrixB(N,b);

 MPI_Bcast(b, N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

  if (M < noworkers ) 
  {
   printf("ERROR: [row = %lu] > [noworkers = %lu]\n",M,noworkers);
   MPI_Abort(MPI_COMM_WORLD,-1); // crash the calculation if norows<noworkers 
  } 
 
/*
 * send rows of A to slaves
 */

 rowsent=0;
 for (i=1; i<=noworkers; i++) {
  MPI_Send(&A[rowsent][0], N, MPI_DOUBLE,i,rowsent+1,MPI_COMM_WORLD);
  rowsent=rowsent+1;
  }

 
  for (m=0; m<M; m++) {
   MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG,MPI_COMM_WORLD, &status);
   sender = status.MPI_SOURCE;
   anstype = status.MPI_TAG;            
   c[anstype-1] = ans;
/*
 * make sure all rows are sent
 */
   if (rowsent < M) {          
    MPI_Send(&A[rowsent][0],N,MPI_DOUBLE,sender,rowsent+1,MPI_COMM_WORLD);
    rowsent=rowsent+1;
    }
   else        
    MPI_Send(MPI_BOTTOM,0,MPI_DOUBLE,sender,0,MPI_COMM_WORLD);

   } 

/*
 * print the resulting matrix
 */

 printMatrixB(M,c); 
 
} 
else 
{

// SLAVES

 t0comm=MPI_Wtime();
 
 MPI_Bcast(b, N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
 
 MPI_Recv(&buff[0],N,MPI_DOUBLE,MASTER,MPI_ANY_TAG,MPI_COMM_WORLD,&status); 

 t1comm=MPI_Wtime();
  
 tcalc=0.; //counters for time
 tcomm=t1comm-t0comm;

 while(status.MPI_TAG != 0) {
  crow = status.MPI_TAG;
  ans=0.;
// evalutation of dot product

  t0calc=MPI_Wtime();
 
  for (n=0; n< N; n++){
   ans+=buff[n]*b[n];
  }

  t1calc=MPI_Wtime();
  if (nid==1) tcalc=tcalc+(t1calc-t0calc);  

  t2comm=MPI_Wtime(); 

  MPI_Send(&ans,1,MPI_DOUBLE,0, crow, MPI_COMM_WORLD);
  MPI_Recv(&buff[0],N,MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status); 

  t3comm=MPI_Wtime();
  tcomm=tcomm+t3comm-t2comm; 

  }

}


if (nid==1) printf("init: %8.4f\n",tInit-tStart); 
if (nid==1) printf("calc: %8.4f\n",tcalc); 
if (nid==1) printf("comm: %8.4f\n",tcomm); 

MPI_Finalize();

tEnd=MPI_Wtime();

if (nid==1) printf("total: %8.4f\n",tEnd-tStart); 

free(buff);

return 0;

}
 
