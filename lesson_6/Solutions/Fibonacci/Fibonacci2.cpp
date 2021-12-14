    #include <iostream>
#include <omp.h>
#include "Timer.hpp"

long long int sequencial_fibonacci(long long int value, int level)
{
    if (value <= 1)
        return 1;

    long long int fib_left, fib_right;
    fib_left = sequencial_fibonacci(value - 1, level + 1);
    fib_right = sequencial_fibonacci(value - 2, level + 1);

    return fib_left + fib_right;
}

long long int parallel_fibonacci(long long int value, int level)
{
    if (value <= 1)
        return 1;

    long long int fib_left, fib_right;
#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp task shared(fib_left)
            {
                fib_left = sequencial_fibonacci(value - 1, level + 1);
            }

#pragma omp task shared(fib_right)
            fib_right = sequencial_fibonacci(value - 2, level + 1);
        }
    }
#pragma omp taskwait
    return fib_left + fib_right;
}

int main()
{
    //  ------------------------- TEST FIBONACCI ----------------------
    using namespace timer;
    omp_set_dynamic(0);
    int value = 43;
    Timer<HOST> TM;

    TM.start();
    long long int result_sequential = sequencial_fibonacci(value, 1);
    TM.stop();
    std::cout << "\nresult sequential: " << result_sequential << "\n"
              << std::endl;
    TM.print("Sequential result: ");

    TM.start();
    long long int result_parallel;

    result_parallel = parallel_fibonacci(value, 1);

    TM.stop();
    std::cout << "\nresult parallel: " << result_parallel << "\n"
              << std::endl;
    TM.print("Parallel result: ");
}
