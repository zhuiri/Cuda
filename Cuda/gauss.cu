
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <math.h>
#include <stdio.h>
//#include <stdlib.h>
#include <string.h>
#include <time.h>

//__shared__ int ipiv[3];
__shared__ int indxc[3],indxr[3];


template<typename Typeval>
__device__ void Swap(Typeval &a,Typeval &b)
//void Swap(Typeval &a,Typeval &b)
{
	Typeval temp;
	temp=a;
	a=b;
	b=temp;
}

__global__ void kernel(double *a)
{
	/*b[0]=2*b[0];
	a[10]=a[10]+b[0];*/

	int x=threadIdx.x+blockIdx.x*blockDim.x;
	int y=threadIdx.y+blockIdx.y*blockDim.y;

	int offset=x+y*blockDim.x*gridDim.x;

	__shared__ float shared[32][32];

	const float period=128.0f;

	
	shared[threadIdx.x][threadIdx.y]=x+y;
	__syncthreads();

	
	a[offset]=255*shared[threadIdx.x][threadIdx.y];
	
	__syncthreads();
	

	


	int aa=0;



}

__global__ void fixRow(double*matrix,double*b,int size,int rowId)
{
	__shared__ double Ri[512];
	__shared__ double Bi[100];
	__shared__ double Aii;
	int colId=threadIdx.x;
	Ri[colId]=matrix[size*rowId+colId];//matrix[size*rowId+colId];

	Bi[colId]=b[size*rowId+0];

	Aii=matrix[size*rowId+rowId];//the diagonal element for ith row
	__syncthreads();

	Ri[colId]=Ri[colId]/Aii;
	matrix[size*rowId+colId]=Ri[colId];

	Bi[colId]=Bi[colId]/Aii;
	b[size*rowId+0]=Bi[colId];
}

__global__ void fixColumn(double *matrix,double *b,int size,int colId)
{
	int i=threadIdx.x;
	int j=blockIdx.x;

	__shared__ double col[512];
	
	__shared__ double AcolIdj;
	__shared__ double BcolIdj;
	__shared__ double colj[512];
	__shared__ double Bj[100];

	col[i]=matrix[i*size+colId];

	if(col[i]!=0)
	{
		colj[i]=matrix[i*size+j];
		Bj[i]=b[i*size+j];

		AcolIdj=matrix[colId*size+j];
		BcolIdj=b[colId*size+j];
		if(i!=colId)
		{
			colj[i]=colj[i]-AcolIdj*col[i];
			Bj[i]=Bj[i]-BcolIdj*col[i];
		}
		matrix[i*size+j]=colj[i];
		b[i*size+j]=Bj[i];

	}
}





extern "C" int
 runGauss(int MatrixSize,double *b,double**a )
{

	int vectorsize=30;
	//int MatrixSize=100;

	double *a_new,*b_new;
	
	a_new=new double[vectorsize];
	b_new=new double[vectorsize];
	//b=new float[vectorsize];
	//ipiv=new int[3];
	for(int i=0;i<MatrixSize;i++)
	{
		b_new[i*MatrixSize]=b[i];
		for(int j=0;j<MatrixSize;j++)
			a_new[i*MatrixSize+j]=a[i][j];
	}
	

	
	 
	 double *a_device,*b_device;
	
	 cudaMalloc((void**)&a_device,vectorsize*sizeof(double));
	 cudaMalloc((void**)&b_device,vectorsize*sizeof(double));
	 
	
	
	 cudaEvent_t start,stop;
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start,0);
	 /*clock_t start,end;
	 start = clock();*/
	
		 cudaMemcpy(a_device,a_new,vectorsize*sizeof(double),cudaMemcpyHostToDevice);
	     cudaMemcpy(b_device,b_new,vectorsize*sizeof(double),cudaMemcpyHostToDevice);
		 for(int i=0;i<MatrixSize;i++)
		 {
			 fixRow<<<1,MatrixSize>>>(a_device,b_device,MatrixSize,i);	
			 fixColumn<<<MatrixSize,MatrixSize>>>(a_device,b_device,MatrixSize,i);
		 }
		 
	
	

	 cudaEventRecord(stop,0);
	 cudaEventSynchronize(stop);
	 float elapseTime;
	 cudaEventElapsedTime(&elapseTime,start,stop);
	
	 cudaMemcpy(a_new,a_device,vectorsize*sizeof(double),cudaMemcpyDeviceToHost);
	 cudaMemcpy(b_new,b_device,vectorsize*sizeof(double),cudaMemcpyDeviceToHost);
	 for(int i=0;i<MatrixSize*MatrixSize;i++)
		 printf("%d-%5.3f\n",i+1,a_new[i]);


	 for(int i=0;i<MatrixSize;i++)
	 {
		 b[i]=b_new[i*MatrixSize];
		for(int j=0;j<MatrixSize;j++)
			a[i][j]=a_new[i*MatrixSize+j];
	 }

	 cudaFree((void*)a_device);
	 cudaFree((void*)b_device);

	 free(a_new);
	 free(b_new);
	 
	return 0;
}