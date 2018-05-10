#include <cstdio>
#include <cuda_fp16.h>
#include <assert.h>
#include <math.h>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "fp16_conversion.h"
#define vector_size 1000000
#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))
using namespace std;
class CSVRow{
  public:
        std::string const& operator[](std::size_t index) const{
            return m_data[index];
        }
        std::size_t size() const{
            return m_data.size();
        }
        void readNextRow(std::istream& str)        {
            std::string         line;
            std::getline(str, line);
            std::stringstream   lineStream(line);
            std::string         cell;
            m_data.clear();
            while(std::getline(lineStream, cell, ','))            {
                m_data.push_back(cell);
            }
            // This checks for a trailing comma with no data after it.
            if (!lineStream && cell.empty())            {
                // If there was a trailing comma then add an empty element.
                m_data.push_back("");
            }
        }
    private:
      std::vector<std::string>    m_data;
};
std::istream& operator>>(std::istream& str, CSVRow& data){
  data.readNextRow(str);
    return str;
} 
inline cudaError_t checkCuda(cudaError_t result){
//#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
  return result;
}

__global__ void haxpy(int n, half a, const half *x, half *y){
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
#if __CUDA_ARCH__ >= 530
  int n2 = n/2;
  half2 *x2 = (half2*)x, *y2 = (half2*)y;
  for (int i = start; i < n2; i+= stride) 
    y2[i] = __hfma2(__halves2half2(a, a), x2[i], y2[i]);
	// first thread handles singleton for odd arrays
  if (start == 0 && (n%2))
  	y[n-1] = __hfma(a, x[n-1], y[n-1]);   
#else
  for (int i = start; i < n; i+= stride) {
    y[i] = __float2half(__half2float(a) * __half2float(x[i]) + __half2float(y[i]));
  }
#endif
}
int main(void) {
  int vector_size_bytes=vector_size*sizeof(half);
  const half pi = approx_float_to_half(M_PI);
  half* A_CPU=(half*)malloc(vector_size_bytes);
  half* B_CPU=(half*)malloc(vector_size_bytes);
  half* C_CPU=(half*)malloc(vector_size_bytes);
  half *x, *y;
  //checkCuda(cudaMallocManaged(&x, n * sizeof(half)));
  //checkCuda(cudaMallocManaged(&y, n * sizeof(half)));
  std::ifstream file("col_vectors.csv");
  CSVRow row; int i=0;
  while(file >> row){
    A_CPU[i]=approx_float_to_half(::atof(row[0].c_str()));B_CPU[i++]=approx_float_to_half(::atof(row[1].c_str()));
  }
  cudaMalloc((void**)&x,vector_size_bytes);
  cudaMalloc((void**)&y,vector_size_bytes);
  cudaMemcpy(x,A_CPU,vector_size_bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(y,B_CPU,vector_size_bytes,cudaMemcpyHostToDevice);
  /*for (int i = 0; i < n; i++) {
    x[i] = approx_float_to_half(1.0f);
    y[i] = approx_float_to_half((float)i);
  }*/
  const int blockSize = 256;
  const int nBlocks = (vector_size + blockSize - 1) / blockSize;
  haxpy<<<nBlocks, blockSize>>>(vector_size, pi, x, y);
  // must wait for kernel to finish before CPU accesses
  checkCuda(cudaDeviceSynchronize());  
  cudaMemcpy(A_CPU,x,vector_size_bytes,cudaMemcpyDeviceToHost);
  cudaMemcpy(B_CPU,y,vector_size_bytes,cudaMemcpyDeviceToHost);
  for (int i = 0; i < NELEMS(B_CPU); i++)
  	printf("%f\n", half_to_float(B_CPU[i]));

  return 0;
}

