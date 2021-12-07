#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>
#include "Timer.hpp"

int main() {
    using namespace timer;
    const int N = (1 << 27);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    omp_set_dynamic(0);
    std::cout << "Available Threads: " << omp_get_max_threads() << std::endl;

    int* A = new int[N];
    for (int i = 0; i < N; ++i)
        A[i] = distribution(generator);

    // --------------- TEST SEQUENTIAL -----------------------------------------
    Timer<HOST> TM;
    TM.start();

    int sum = 0;
    for (int i = 0; i < N; ++i)
        sum += A[i];

    TM.stop();
    TM.print("Sequential");

    // --------------- TEST CRITICAL ------------------------------------------
    TM.start();
    
    sum = 0;
    #pragma parallel for
    for (int i = 0; i < N; ++i) {
    #pragma omp critical
        sum += A[i];
    }
    
    TM.stop();
    TM.print("Critical");
    
    
    // --------------- TEST ATOMIC ---------------------------------------------
    TM.start();
    
    sum = 0;
    #pragma parallel for
    for (int i = 0; i < N; ++i)
    #pragma omp atomic
    sum += A[i];
    
    TM.stop();
    TM.print("Atomic");

    
    // --------------- TEST LOCK -----------------------------------------------
    TM.start();
    
    omp_lock_t lock;
    omp_init_lock(&lock);
    
    sum = 0;
    #pragma parallel for
    for (int i = 0; i < N; ++i) {
        omp_set_lock(&lock);
        sum += A[i];
        omp_unset_lock(&lock);
    }
    
    TM.stop();
    TM.print("Lock");
    
    
    // --------------- TEST REDUCTION ------------------------------------------
    TM.start();

    sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < N; ++i)
        sum = sum + A[i];

    TM.stop();
    TM.print("Reduction");

    

   
    
}
