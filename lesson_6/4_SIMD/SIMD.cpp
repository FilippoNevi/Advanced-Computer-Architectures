#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>
#include "Timer.hpp"
using namespace timer;

//const int N = (1 << 28);

const int N = (1 << 18);

int main(int argc, char ** argv) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    int* A = new int[N];
    int* B = new int[N];
    int* C = new int[N];
    
    for (int i = 0; i < N; ++i) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }
    //--------------------------------------------------------------------------
    Timer<HOST> TM;
    //to get data into cache
    for (int i = 0; i < N; ++i)
        C[i] = A[i] + B[i];
    
    
    TM.start();

    for (int i = 0; i < N; ++i)
        C[i] = A[i] + B[i];

    TM.stop();
    TM.print("Sequential vector sum");

    
    //--------------------------------------------------------------------------
    TM.start();
    #pragma omp parallel for
        for (int i = 0; i < N; ++i)
            C[i] = A[i] + B[i];
    TM.stop();
    TM.print("Parallel vector sum");

    
    //--------------------------------------------------------------------------
    TM.start();

    #pragma omp parallel for simd
    for (int i = 0; i < N; ++i)
        C[i] = A[i] + B[i];

    TM.stop();
    TM.print("Parallel SIMD vector sum");

    
    //--------------------------------------------------------------------------
    TM.start();

    int sum = 0;
    for (int i = 0; i < N; ++i)
        sum = sum + A[i];

    TM.stop();
    TM.print("Sequential sum of integers of an array");
    //std::cout << sum << std::endl;

    //--------------------------------------------------------------------------
    TM.start();

    sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < N; ++i)
        sum = sum + A[i];

    TM.stop();
    TM.print("Parallel Reduction (sum) of integers of an array");

    //--------------------------------------------------------------------------
    TM.start();

    sum = 0;
    #pragma omp parallel for simd reduction(+ : sum)
    for (int i = 0; i < N; ++i)
        sum = sum + A[i];

    TM.stop();
    TM.print("Parallel Reduction SIMD (sum) of integers of an array");
     std::cout << std::endl << std::endl;
     
}

//less /proc/cpuinfo
