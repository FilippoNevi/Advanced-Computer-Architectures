#include <iostream>
#include <omp.h>

int main() {
    
    //  ------------------------- TEST SINGLE/MASTER ---------------------------
    
    #pragma omp parallel
    {
    #pragma omp master
        std::printf("Master Section -> Thread: %d\n", omp_get_thread_num());
        
    #pragma omp single
        std::printf("Single Section -> Thread: %d\n", omp_get_thread_num());
        
    #pragma omp master
        std::printf("Master Section -> Thread: %d\n", omp_get_thread_num());
        
    #pragma omp single
        std::printf("Single Section -> Thread: %d\n", omp_get_thread_num());
        
        
        std::printf("after single section  Thread: %d\n", omp_get_thread_num());
    }
    
    std::cout << std::endl << std::endl;
    
    //  ------------------------- TEST PARALLEL SECTIONS -----------------------

    #pragma omp parallel sections
    {
        #pragma omp section
        std::printf("Section 1 -> Thread: %d\n", omp_get_thread_num());

        #pragma omp section
        {
            std::printf(" Section 2 (1) -> Thread: %d\n", omp_get_thread_num());
            std::printf(" Section 2 (2) -> Thread: %d\n", omp_get_thread_num());
        }
        #pragma omp section
        std::printf("Section 3 -> Thread: %d\n", omp_get_thread_num());
    }

    
    
}
