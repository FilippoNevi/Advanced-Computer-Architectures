#include <iostream>
#include <omp.h>

int main() {
    // -------------------------- TEST SCHEDULING ------------------------------
    //omp_set_dynamic(0);
    omp_set_num_threads(4);

    // -------------------------------------------------------------------------
    std::cout << std::endl << "Static Scheduling" << std::endl;

    #pragma omp parallel
    {
        #pragma omp    for schedule(static)
        for (int i = 0; i < 12; i++)
            std::printf("Thread: %d\n", omp_get_thread_num());
    }

    // -------------------------------------------------------------------------
    std::cout << std::endl << "Static Scheduling (5)" << std::endl;

    #pragma omp parallel
    {
        #pragma omp    for schedule(static, 5)
        for (int i = 0; i < 12; i++)
            std::printf("Thread: %d\n", omp_get_thread_num());
    }

    // -------------------------------------------------------------------------
    std::cout << std::endl << "Dynamic Scheduling" << std::endl;

    #pragma omp parallel
    {
        #pragma omp    for schedule(dynamic)
        for (int i = 0; i < 12; i++)
            std::printf("Thread: %d\n", omp_get_thread_num());
    }

    // -------------------------------------------------------------------------
    std::cout << std::endl << "Dynamic Scheduling (5)" << std::endl;

    #pragma omp parallel
    {
        #pragma omp    for schedule(dynamic, 5)
        for (int i = 0; i < 12; i++)
            std::printf("Thread: %d\n", omp_get_thread_num());
    }

    // -------------------------------------------------------------------------
    std::cout << std::endl << "Guided Scheduling" << std::endl;

    #pragma omp parallel
    {
        #pragma omp    for schedule(guided)
        for (int i = 0; i < 12; i++)
            std::printf("Thread: %d\n", omp_get_thread_num());
    }

    // -------------------------------------------------------------------------
    std::cout << std::endl << "Guided Scheduling (5)" << std::endl;

    #pragma omp parallel
    {
        #pragma omp    for schedule(guided, 5)
        for (int i = 0; i < 12; i++)
            std::printf("Thread: %d\n", omp_get_thread_num());
    }

    // -------------------------------------------------------------------------
    std::cout << std::endl << "Runtime Scheduling" << std::endl;

    
    //e.g., export OMP_SCHEDULE=static
    //e.g. omp_set_schedule(static);
    
    
    
    #pragma omp parallel
    {
        #pragma omp    for schedule(runtime)
        for (int i = 0; i < 12; i++)
            std::printf("Thread: %d\n", omp_get_thread_num());
    }

    // -------------------------------------------------------------------------
    std::cout << std::endl << "Auto Scheduling" << std::endl;

    #pragma omp parallel
    {
        #pragma omp    for schedule(auto)
        for (int i = 0; i < 12; i++)
            std::printf("Thread: %d\n", omp_get_thread_num());
    }
}
