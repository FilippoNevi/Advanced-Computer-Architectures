#include <iostream>
#include <omp.h>
#include "Timer.hpp"

long long int fibonacci(long long int value, int level) {
    if (value <= 1)
        return 1;

    long long int fib_left, fib_right;
    fib_left  = fibonacci(value - 1, level + 1);
    fib_right = fibonacci(value - 2, level + 1);

    return fib_left + fib_right;
}

long long int fibonacci_par(long long int value, int level) {
    if (value <= 1)
        return 1;

    long long int fib_left, fib_right;
    #pragma omp parallel sections
    {
        #pragma omp section
        { fib_left  = fibonacci(value - 1, level + 1); }
        #pragma omp section
        { fib_right = fibonacci(value - 2, level + 1); }
    }

    return fib_left + fib_right;
}

int main() {
    using namespace timer;
    //  ------------------------- TEST FIBONACCI ----------------------
    omp_set_dynamic(0);
    int value = 44;

    Timer<HOST> TM;

    TM.start();
    long long int result = fibonacci(value, 1);
    TM.stop();
    TM.print("Sequential: ");

    TM.start();
    long long int result_par = fibonacci_par(value, 1);
    TM.stop();
    TM.print("Parallel: ");

    std::cout << "\nresult: " << result << "\n" << std::endl;
}
