#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include "fp16_conversion.h"
#define vect_len 16
using namespace std;
const int blocksize = 16;
__global__ void vect_add(half *a, half *b){
	int start=threadIdx.x+blockDim.x*blockIdx.x;
	int stride=blockDim.x*gridDim.x;
	
	//a[threadIdx.x] += b[threadIdx.x];
}
int main(){
	const int vect_size = vect_len*sizeof(half);
	half* vect1=(half*)malloc(vect_size);
	half* vect2=(half*)malloc(vect_size);
	half* result=(half*)malloc(vect_size);
 	bool flag;
	for(int i = 0; i < vect_len; i++){
		vect1[i] = i;
		vect2[i] = 2 * i;
	}
	int *ad, *bc;
	cudaMalloc((void**)&ad, vect_size);
	cudaMalloc((void**)&bc, vect_size);
	cudaMemcpy(ad, vect1, vect_size, cudaMemcpyHostToDevice);
	cudaMemcpy(bc, vect2, vect_size, cudaMemcpyHostToDevice);
	dim3 dimBlock(blocksize, 1, 1);
	dim3 dimGrid(vect_len/blocksize, 1 , 1);
	vect_add<<<dimGrid, dimBlock>>>(ad, bc);
	cudaMemcpy(result, ad, vect_size, cudaMemcpyDeviceToHost);
	flag = true;
	for(int i = 0; i < vect_len; i++){
		if(result[i] != vect1[i] + vect2[i]){
			cout << "Verification fail at " << i << endl;
			flag = false;
			break;
		}
	}
	if(flag)
		cout << "Verification passes." <<endl;
	// free device memory
	cudaFree(ad);	cudaFree(bc);	free(vect1);	free(vect2);	free(result);
	return EXIT_SUCCESS;
}


