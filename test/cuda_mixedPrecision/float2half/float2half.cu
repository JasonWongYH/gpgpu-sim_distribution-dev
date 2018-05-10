#include <cstdio>
#include <cuda_fp16.h>
#include <assert.h>
#include "fp16_conversion.h"   // host function for half conversion

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__
void myTest(int n, float a, const float *x, half *y)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < n; i+= stride) {
	    //y[i] = __float2half(a) * __float2half(x[i]) + y[i]; // error : __half * __half is not supported
	    y[i] = __hfma(__float2half(a), __float2half(x[i]), y[i]);
	}
}

int main(int argc, char** argv) {

  int devid = atoi(argv[1]);
  cudaSetDevice(devid);

  cudaDeviceProp prop;                                                    
  cudaGetDeviceProperties(&prop, devid);                                 
  printf("device %d : %s\n", devid, prop.name);

  const int n = 10;

  const float a = 2.11f; 
  printf("a = %f\n", a);

  float *x;
  checkCuda(cudaMallocManaged(&x, n * sizeof(float)));

  half *y;
  checkCuda(cudaMallocManaged(&y, n * sizeof(half)));
  
  for (int i = 0; i < n; i++) {
    x[i] = 1.0f;
    y[i] = approx_float_to_half(2.f);
  }


  const int blockSize = 256;
  const int nBlocks = (n + blockSize - 1) / blockSize;

  myTest<<<nBlocks, blockSize>>>(n, a, x, y);

  // must wait for kernel to finish before CPU accesses
  checkCuda(cudaDeviceSynchronize());
  
  for (int i = 0; i < n; i++)
  	printf("%f\n", half_to_float(y[i]));


  return 0;
}
