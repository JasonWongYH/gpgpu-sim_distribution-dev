#include <random>
#include <iostream>
#include <cfloat> // DBL_MAX
#include <cmath>
#include "../src/cuda-sim/cuda-math.h" 
#include <cuda_fp16.h>
#include "../src/cuda-sim/fp16_conversion.h"
// This happens because you call f 4000 times in a loop, which probably takes less than a mili second, so at each call time(0) returns the same value, hence initializes the pseudo-random generator with the same seed. The correct way is to initialize the seed once and for all, preferably via a std::random_device, like so:
static std::random_device rd; // random device engine, usually based on /dev/random on UNIX-like systems
// initialize Mersennes' twister using rd to generate the seed
static std::mt19937 rng(rd()); 
int dice(){
    static std::uniform_int_distribution<int> uid(1,6); // random dice
    return uid(rng); // use rng as a generator
}
int main() {
    //std::random_device rd;
    //std::mt19937 mt(rd());
    //std::uniform_real_distribution<double> dist(1.0, 10.0);
 	std::uniform_real_distribution<float> dist(-2^(-24)/*1 00000 0000000001*/, std::nextafter(65504/*0 11110 1111111111*/, FLT_MAX));
    for (int i=0; i<1000000; ++i)
        std::cout << dist(rng) << "," << dist(rng) << "\n";
}


