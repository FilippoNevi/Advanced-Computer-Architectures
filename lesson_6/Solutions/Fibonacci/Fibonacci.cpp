#include <iostream>
#include <omp.h>
#include "Timer.hpp"
//#define DEBUG

int fibonacci(int value, int level) {
    if (value <= 1)
        return 1;

    long long int fib_left, fib_right;
    fib_left  = fibonacci(value - 1, level + 1);
    fib_right = fibonacci(value - 2, level + 1);

    return fib_left + fib_right;
}


int fibonacci_par(int value, int level) {
	if (value <= 1)
		return 1;
    
    if (value < 20)
    {
        return fibonacci_par(value-1, level + 1)+fibonacci_par(value-2, level + 1);
    }
    else{
        int fib_left, fib_right;
        #pragma omp task shared(fib_left) firstprivate(value, level)
        {
            #ifdef DEBUG
            std::printf("Left. Thread: %d \t task: %d at level %d\n",
                         omp_get_thread_num(), value - 1, level);
            #endif
            fib_left = fibonacci_par(value - 1, level + 1);
        }
        #pragma omp task shared(fib_right) firstprivate(value, level)
        {
            #ifdef DEBUG
            std::printf("Right. Thread: %d \t task: %d at level %d\n",
                        omp_get_thread_num(), value - 2, level);
            #endif
            fib_right = fibonacci_par(value - 2, level + 1);
        }
        #pragma omp taskwait
        #ifdef DEBUG
        std::printf("\nThread: %d   join \t   fib-Left %d \t fib-Right %d"
                    "\t at level %d\n",
                    omp_get_thread_num(), fib_left, fib_right, level);
        #endif
        return fib_left + fib_right;
    }
}

int main() {
	using namespace timer;
    //  ------------------------- TEST FIBONACCI ----------------------
	omp_set_dynamic(0);
	int value = 44;
    //sequential:
    Timer<HOST> TM_seq;
    Timer<HOST> TM_par;
    
    std::cout << std::endl << "Starting Sequential Fibonacci....";
    
    TM_seq.start();
    int result_seq = fibonacci(value, 1);
    std::cout << "\nresult: " << result_seq << std::endl;
    TM_seq.stop();
    
    std::cout << "done!" << std::endl<< std::endl;
    TM_seq.print("Sequential Fibonacci");

    std::cout << std::endl << "Starting Parallel Fibonacci....";
    TM_par.start();
	#pragma omp parallel firstprivate(value)
	{
		#pragma omp single// <-- the starting point is a single threads (level0)
		{
			int result = fibonacci_par(value, 1);
			std::cout << "\nresult: " << result << std::endl;
		}
	}
    TM_par.stop();
    std::cout << "done!" << std::endl<< std::endl;
    TM_par.print("PARALLEL Fibonacci");
    
    // ------------ Speedup: --------------------------------------------
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    std::cout << std::setprecision(1)
    << "Speedup: " << TM_seq.duration() / TM_par.duration()
    << "x\n\n";
    
    
}
