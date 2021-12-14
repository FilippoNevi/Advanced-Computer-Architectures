#include <chrono>
#include <iostream>
#include <iomanip>      // std::setprecision
#include <limits>       // std::numeric_limits
#include <random>
#include <omp.h>
#include "include/Timer.hpp"

int main() {
    using namespace timer;
    int N = (1 << 25);

    int* array = new int[N];
    const int to_find1 = 18;        //search [ .. ][ .. ][ 18 ][ 64 ][ .. ]
    const int to_find2 = 64;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 1000);
        // for lower values the sequential code wins  e.g. 1000

    for (int i = 0; i < N; ++i)
        array[i] = distribution(generator);

    // ------------ SEQUENTIAL SEARCH ------------------------------------------
    Timer<HOST> TM;
    TM.start();

    int index = std::numeric_limits<int>::max();
    for (int i = 0; i < N - 1; ++i) {
        if (array[i] == to_find1 && array[i + 1] == to_find2) {
            index = i;
            break;            // !!! error in OPENMP
        }
    }

    TM.stop();
    TM.print("Sequential");

    // ------------ PARALLEL SEARCH --------------------------------------------
    TM.start();

    index = std::numeric_limits<int>::max();
    #pragma omp parallel reduction(min : index)  // <-- require OpenMP 3.1
    for (int i = 0; i < N - 1; ++i) {
        if (array[i] == to_find1 && array[i + 1] == to_find2)
            index = i;   // <-- error: this is not a reduction on array values
    }

    TM.stop();
    TM.print("Parallel");

    std::cout << index << std::endl;

    // ------------ PARALLEL SEARCH --------------------------------------------
    TM.start();

    index = std::numeric_limits<int>::max();
    int local_index = std::numeric_limits<int>::max();
                                 //  IMPORTANT:  outside the parallel
                                 //  region otherwise the "local_index" variable
                                 // is PRIVATE!!
    #pragma omp parallel shared(index) firstprivate(local_index)
    {
        bool flag = true;        // <-- "flag" is private
        #pragma omp for
        for (int i = 0; i < N - 1; ++i) {
            if (flag && array[i] == to_find1 && array[i + 1] == to_find2) {
                local_index = std::min(i, local_index);
                flag = false;
            }
        }
        #pragma omp critical
        index = std::min(index, local_index);
        //#pragma omp atomic
        //index = std::min(index, local_index); // <-- atomic support only simple
    }                                          // assignments ERROR

    TM.stop();
    TM.print("Parallel");
    std::cout << index << std::endl;
}

//@author Federico Busato
