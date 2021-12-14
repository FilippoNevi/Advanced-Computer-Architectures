#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>      // std::setprecision
#include <random>
#include <omp.h>
#include "Timer.hpp"
//#define DEBUG

int partition(int array[], int start, int end) {
	int p = start;
	int pivotElement = array[end];
	for(int i = start; i < end; i++) {
		if(array[i] < pivotElement) {
			std::swap(array[i], array[p]);
			++p;
		}
	}
	std::swap(array[p], array[end]);
	return p;
}

void quickSort(int array[], int start, int end) {
    if(start < end) {
        int pivot = partition(array, start, end);
		quickSort(array, start, pivot - 1);
		quickSort(array, pivot + 1, end);
    }
}


void quickSortOMP(int array[], int start, int end) {
    if (start < end)
        {
            int pivot = partition(array, start, end);
    #pragma omp task shared(array)
            quickSortOMP(array, start, pivot - 1);
    #pragma omp task shared(array)
            quickSortOMP(array, pivot + 1, end);
        }
}

template<typename T>
void printArray(T* array, int size, const char* str) {
	std::cout << str << std::endl;
	for (int i = 0; i < size; ++i)
		std::cout << array[i] << ' ';
	std::cout << std::endl << std::endl;
}


int main() {
    using namespace timer;
    const int N = (1 << 20);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::uniform_int_distribution<int> distribution(1, 100);

	int* input = new int[N];
	for (int i = 0; i < N; ++i)
		input[i] =  distribution(generator);
    
	int* inputTMP = new int[N];
    Timer<HOST> TM_seq;
    Timer<HOST> TM_par;
	// --------------------- SEQUENTIAL QUICKSORT -------------------------------
	std::copy(input, input + N, inputTMP);
    
    #ifdef DEBUG
        printArray(inputTMP, N, "\nInput:");
    #endif
	TM_seq.start();

	quickSort(inputTMP, 0, N - 1);

    TM_seq.stop();
    #ifdef DEBUG
        printArray(inputTMP, N, "\nInput:");
    #endif
    
    TM_seq.print("\nSeq. Elapsed time: ");

	// --------------------- OpenMP QUICKSORT -------------------------------
	std::copy(input, input + N, inputTMP);
    
    #ifdef DEBUG
        printArray(inputTMP, N, "\nInput:");
    #endif
    
    TM_par.start();

	#pragma omp parallel //num_threads(1)
	{
		#pragma omp single
		{
			quickSortOMP(inputTMP, 0, N - 1);
		}
	}
	TM_par.stop();
    #ifdef DEBUG
        printArray(inputTMP, N, "\nInput:");
    #endif
    
    TM_par.print("\nOmp. Elapsed time: ");
	//printArray(input, N, "Sorted:");
    // ------------ Speedup: --------------------------------------------
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    std::cout << std::setprecision(1)
    << "Speedup: " << TM_seq.duration() / TM_par.duration()
    << "x\n\n";
    
}
