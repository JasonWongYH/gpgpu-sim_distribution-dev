#include <iostream>
#include <cuda.h>
#include "CSVRow.cpp"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <cuda_fp16.h>
#define vector_size 1000000
using namespace std;
const int blocksize = 16;
typedef unsigned int uint;
union FP32{
    uint u;
    float f; 
    struct{
      uint Mantissa : 23;
      uint Exponent : 8;
      uint Sign : 1;
    };
};
union FP16{
    uint i; 
    unsigned short u;
    struct{
      uint Mantissa : 10;
      uint Exponent : 5;
      uint Sign : 1;
    };
};
__global__ void dot_product(float s,float* A, float* B){
  A[threadIdx.x] = s*A[threadIdx.x]+B[threadIdx.x];
}
// Original ISPC reference version; this always rounds ties up. TODO 5-bit exponent in [-14,15], 10-bit fraction (with hidden 1 bit)
static FP16 float_to_half_full(FP32 f,int& flag){
  FP16 o = { 0 };
  // Based on ISPC reference code (with minor modifications)
  if (f.Exponent == 0){ // Signed zero/denormal (which will underflow)
    o.Exponent = 0;
    flag=1;
  }
  else if (f.Exponent == 255){ // Inf or NaN (all exponent bits set)
    o.Exponent = 31;
    o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
    flag=1;
  }
  else{ 
    int newexp = f.Exponent - 127 + 15;
    if (newexp >= 31) {// Overflow, return signed infinity
      o.Exponent = 31;
      flag=1;
    }else if (newexp <= 0){ // Underflow 
      flag=1;
      if((14 - newexp) <= 24){ // Mantissa might be non-zero
        uint mant = f.Mantissa | 0x800000; // Hidden 1 bit
        o.Mantissa = mant >> (14 - newexp);//TODO why do I right shift
        if ((mant >> (13 - newexp)) & 1) // Check for rounding
          o.i++; // Round, might overflow into exp bit, but this is OK
        printf("(14 - newexp) <= 24, newexp==%i, o.i==%i, o.Mantissa==%i, f.Mantissa==%i\n",newexp,o.i,o.Mantissa,f.Mantissa);
      }
    }else{
        o.Exponent = newexp;
        o.Mantissa = f.Mantissa >> 13;
        if (f.Mantissa & 0x1000) // Check for rounding
          o.i++; // Round, might overflow to inf, this is OK
    }

  }
  o.Sign = f.Sign;
  return o;
}

/**/

// Same as above, but with full round-to-nearest-even.
static FP16 float_to_half_full_rtne(FP32 f)
{
    FP16 o = { 0 };

    // Based on ISPC reference code (with minor modifications)
    if (f.Exponent == 0) // Signed zero/denormal (which will underflow)
        o.Exponent = 0;
    else if (f.Exponent == 255) // Inf or NaN (all exponent bits set)
    {
        o.Exponent = 31;
        o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
    }
    else // Normalized number
    {
        // Exponent unbias the single, then bias the halfp
        int newexp = f.Exponent - 127 + 15;
        if (newexp >= 31) // Overflow, return signed infinity
            o.Exponent = 31;
        else if (newexp <= 0) // Underflow
        {
            if ((14 - newexp) <= 24) // Mantissa might be non-zero
            {
                uint mant = f.Mantissa | 0x800000; // Hidden 1 bit
                uint shift = 14 - newexp;
                o.Mantissa = mant >> shift;

                uint lowmant = mant & ((1 << shift) - 1);
                uint halfway = 1 << (shift - 1);

                if (lowmant >= halfway) // Check for rounding
                {
                    if (lowmant > halfway || (o.Mantissa & 1)) // if above halfway point or unrounded result is odd
                        o.u++; // Round, might overflow into exp bit, but this is OK
                }
            }
        }
        else
        {
            o.Exponent = newexp;
            o.Mantissa = f.Mantissa >> 13;
            if (f.Mantissa & 0x1000) // Check for rounding
            {
                if (((f.Mantissa & 0x1fff) > 0x1000) || (o.Mantissa & 1)) // if above halfway point or unrounded result is odd
                    o.u++; // Round, might overflow to inf, this is OK
            }
        }
    }

    o.Sign = f.Sign;
    return o;
}

static FP16 float_to_half_fast(FP32 f)
{
    FP16 o = { 0 };

    // Based on ISPC reference code (with minor modifications)
    if (f.Exponent == 255) // Inf or NaN (all exponent bits set)
    {
        o.Exponent = 31;
        o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
    }
    else // Normalized number
    {
        // Exponent unbias the single, then bias the halfp
        int newexp = f.Exponent - 127 + 15;
        if (newexp >= 31) // Overflow, return signed infinity
            o.Exponent = 31;
        else if (newexp <= 0) // Underflow
        {
            if ((14 - newexp) <= 24) // Mantissa might be non-zero
            {
                uint mant = f.Mantissa | 0x800000; // Hidden 1 bit
                o.Mantissa = mant >> (14 - newexp);
                if ((mant >> (13 - newexp)) & 1) // Check for rounding
                    o.u++; // Round, might overflow into exp bit, but this is OK
            }
        }
        else
        {
            o.Exponent = newexp;
            o.Mantissa = f.Mantissa >> 13;
            if (f.Mantissa & 0x1000) // Check for rounding
                o.u++; // Round, might overflow to inf, this is OK
        }
    }

    o.Sign = f.Sign;
    return o;
}

static FP16 float_to_half_fast2(FP32 f)
{
    FP32 infty = { 31 << 23 };
    FP32 magic = { 15 << 23 };
    FP16 o = { 0 };

    uint sign = f.Sign;
    f.Sign = 0;

    // Based on ISPC reference code (with minor modifications)
    if (f.Exponent == 255) // Inf or NaN (all exponent bits set)
    {
        o.Exponent = 31;
        o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
    }
    else // (De)normalized number or zero
    {
        f.u &= ~0xfff; // Make sure we don't get sticky bits
        // Not necessarily the best move in terms of accuracy, but matches behavior
        // of other versions.

        // Shift exponent down, denormalize if necessary.
        // NOTE This represents half-float denormals using single precision denormals.
        // The main reason to do this is that there's no shift with per-lane variable
        // shifts in SSE*, which we'd otherwise need. It has some funky side effects
        // though:
        // - This conversion will actually respect the FTZ (Flush To Zero) flag in
        //   MXCSR - if it's set, no half-float denormals will be generated. I'm
        //   honestly not sure whether this is good or bad. It's definitely interesting.
        // - If the underlying HW doesn't support denormals (not an issue with Intel
        //   CPUs, but might be a problem on GPUs or PS3 SPUs), you will always get
        //   flush-to-zero behavior. This is bad, unless you're on a CPU where you don't
        //   care.
        // - Denormals tend to be slow. FP32 denormals are rare in practice outside of things
        //   like recursive filters in DSP - not a typical half-float application. Whether
        //   FP16 denormals are rare in practice, I don't know. Whatever slow path your HW
        //   may or may not have for denormals, this may well hit it.
        f.f *= magic.f;

        f.u += 0x1000; // Rounding bias
        if (f.u > infty.u) f.u = infty.u; // Clamp to signed infinity if overflowed

        o.u = f.u >> 13; // Take the bits!
    }

    o.Sign = sign;
    return o;
}

static FP16 float_to_half_fast3(FP32 f)
{
    FP32 f32infty = { 255 << 23 };
    FP32 f16infty = { 31 << 23 };
    FP32 magic = { 15 << 23 };
    uint sign_mask = 0x80000000u;
    uint round_mask = ~0xfffu; 
    FP16 o = { 0 };

    uint sign = f.u & sign_mask;
    f.u ^= sign;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).

    if (f.u >= f32infty.u) // Inf or NaN (all exponent bits set)
        o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
    else // (De)normalized number or zero
    {
        f.u &= round_mask;
        f.f *= magic.f;
        f.u -= round_mask;
        if (f.u > f16infty.u) f.u = f16infty.u; // Clamp to signed infinity if overflowed

        o.u = f.u >> 13; // Take the bits!
    }

    o.u |= sign >> 16;
    return o;
}

// Same, but rounding ties to nearest even instead of towards +inf
static FP16 float_to_half_fast3_rtne(FP32 f)
{
    FP32 f32infty = { 255 << 23 };
    FP32 f16max   = { (127 + 16) << 23 };
    FP32 denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
    uint sign_mask = 0x80000000u;
    FP16 o = { 0 };

    uint sign = f.u & sign_mask;
    f.u ^= sign;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).

    if (f.u >= f16max.u) // result is Inf or NaN (all exponent bits set)
        o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
    else // (De)normalized number or zero
    {
        if (f.u < (113 << 23)) // resulting FP16 is subnormal or zero
        {
            // use a magic value to align our 10 mantissa bits at the bottom of
            // the float. as long as FP addition is round-to-nearest-even this
            // just works.
            f.f += denorm_magic.f;

            // and one integer subtract of the bias later, we have our final float!
            o.u = f.u - denorm_magic.u;
        }
        else
        {
            uint mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd

            // update exponent, rounding bias part 1
            f.u += ((15 - 127) << 23) + 0xfff;
            // rounding bias part 2
            f.u += mant_odd;
            // take the bits!
            o.u = f.u >> 13;
        }
    }

    o.u |= sign >> 16;
    return o;
}

// Approximate solution. This is faster but converts some sNaNs to
// infinity and doesn't round correctly. Handle with care.
static FP16 approx_float_to_half(FP32 f)
{
    FP32 f32infty = { 255 << 23 };
    FP32 f16max = { (127 + 16) << 23 };
    FP32 magic = { 15 << 23 };
    FP32 expinf = { (255 ^ 31) << 23 };
    uint sign_mask = 0x80000000u;
    FP16 o = { 0 };

    uint sign = f.u & sign_mask;
    f.u ^= sign;

    if (!(f.f < f32infty.u)) // Inf or NaN
        o.u = f.u ^ expinf.u;
    else
    {
        if (f.f > f16max.f) f.f = f16max.f;
        f.f *= magic.f;
    }

    o.u = f.u >> 13; // Take the mantissa bits
    o.u |= sign >> 16;
    return o;
}

/**/
int PrintBinary (int x, int start, int end){
  unsigned int mask    = 1 << start;
  unsigned int endmask = 1 << end;
  int k=start-end; int partial_sum=0;
  while (mask >= endmask){   
    //putchar ((mask & x) ? '1' : '0');
    int term=(int)pow((double)2,(double)k);
    partial_sum+=(mask & x) ? term : 0;
    mask >>= 1; k--;
  }
  //printf("\npartial sum for %i start and %i end = %i\n", start,end,partial_sum);
  return partial_sum;
}
float PrintVal(FP16 x){
	int sign = PrintBinary(x.i, 15, 15)==0 ? 1 : -1;
	int exponent = PrintBinary (x.i, 14, 10); 
	exponent-=15;
	int mantissa = PrintBinary (x.i, 9,  0);
	int inverse_scale=exponent < 0 ? 1 : 0;
	if (inverse_scale==1) exponent=-1*exponent;
	float scale = 1 << exponent; 
	if (inverse_scale==1) scale=1/scale;
	float real=(float)sign*scale*(1+((float)mantissa/(float)1024));
	return real;
}
int main(int argc, char** argv){	
	const int vector_size_bytes = vector_size*sizeof(float);
	float* A_CPU=(float*)malloc(vector_size_bytes);
	float* B_CPU=(float*)malloc(vector_size_bytes);
	float* C_CPU=(float*)malloc(vector_size_bytes);
 	std::ifstream file("col_vectors.csv");
    CSVRow row; int i=0;
    while(file >> row){
    	A_CPU[i]=::atof(row[0].c_str());B_CPU[i++]=::atof(row[1].c_str());
    }
 	float* A_GPU;
	float* B_GPU;
	cudaMalloc((void**)&A_GPU,vector_size_bytes);
	cudaMalloc((void**)&B_GPU,vector_size_bytes);
	cudaMemcpy(A_GPU,A_CPU,vector_size_bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU,B_CPU,vector_size_bytes,cudaMemcpyHostToDevice);
	dim3 dimBlock(blocksize, 1, 1);
	dim3 dimGrid(vector_size/blocksize, 1 , 1);
	dot_product<<<dimGrid, dimBlock>>>(M_PI,A_GPU,B_GPU);
	//TODO __syncthreads, convert to FP16, dump out real numbers 
	//__float2half 
	cudaMemcpy(C_CPU,A_GPU,vector_size,cudaMemcpyDeviceToHost);
	std::fstream log_stream("real_numbers.csv", std::fstream::out);
	for(int i=0;i<vector_size;i++){
		int denormal=0;
		FP32 f=*((FP32*)&C_CPU[i]);
		FP16 h=float_to_half_full(f,denormal);
		if(denormal==1){
			log_stream << f.f << " out of range\n";
		}else{
			float real=PrintVal(h); //TODO byte aligned and keeping withing FP16 representable range for now
			log_stream << f.f << " , " << real << "\n";  //float , half , relative error
		}
	}	
	cudaFree(A_GPU);cudaFree(B_GPU);free(A_CPU);free(B_CPU);free(C_CPU);
	return EXIT_SUCCESS;
}


