#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "Timer.hpp"

void test_producer_consumer(int Buffer[32]) {
	int i = 0;
	int count = 0;

	while (i < 35000) {					// number of test

		// PRODUCER
		if ((rand() % 50) == 0) {		// some random computations

			if (count < 31) {
				++count;
				std::cout << "Thread:\t" << omp_get_thread_num()
                          << "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) {		// some random computations

			if (count >= 1) {
				int var = Buffer[count];
				std::cout << "Thread:\t" << omp_get_thread_num()
                          << "\tConsume on index: " << count
                          << "\tvalue: " << var << std::endl;
				--count;
			}
		}
		i++;
	}
}

void test_producer_consumer_cr(int Buffer[32]) {
	int i = 0;
	int count = 0;
	#pragma omp parallel shared(i, count, Buffer)
	while (i < 35000) {					// number of test

		// PRODUCER
		if ((rand() % 50) == 0) {		// some random computations

			#pragma omp critical(prod_cons)
			{
				if (count < 31) {
					++count;
					std::cout << "Thread:\t" << omp_get_thread_num()
                        << "\tProduce on index: " << count << std::endl;
					Buffer[count] = omp_get_thread_num();
				}
			}
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) {		// some random computations

			#pragma omp critical(prod_cons)
			{
				if (count >= 1) {
					int var = Buffer[count];
					std::cout << "Thread:\t" << omp_get_thread_num()
														<< "\tConsume on index: " << count
														<< "\tvalue: " << var << std::endl;
					--count;
				}
			}
		}
		#pragma omp critical(index)
		{ i++; }
	}
}

void test_producer_consumer_locks(int Buffer[32]) {
	int i = 0;
	int count = 0;

	omp_lock_t buf_lock;
	omp_init_lock(&buf_lock);

	omp_lock_t index_lock;
	omp_init_lock(&index_lock);

	#pragma omp parallel shared(i, count, Buffer)
	while (i < 35000) {					// number of test

		// PRODUCER
		if ((rand() % 50) == 0) {		// some random computations
			omp_set_lock(&buf_lock);
			if (count < 31) {
				++count;
				std::cout << "Thread:\t" << omp_get_thread_num()
											<< "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
			omp_unset_lock(&buf_lock);
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) {		// some random computations
			omp_set_lock(&buf_lock);
			if (count >= 1) {
				int var = Buffer[count];
				std::cout << "Thread:\t" << omp_get_thread_num()
													<< "\tConsume on index: " << count
													<< "\tvalue: " << var << std::endl;
				--count;
			}
			omp_unset_lock(&buf_lock);
		}

		omp_set_lock(&index_lock);
		i++;
		omp_unset_lock(&index_lock);
	}

	omp_destroy_lock(&buf_lock);
	omp_destroy_lock(&index_lock);
}

int main() {
	using namespace timer;
	Timer<HOST> TM;
	int Buffer[32];
	std::srand(time(NULL));

	omp_set_num_threads(2);
	TM.start();
	test_producer_consumer(Buffer);
	TM.stop();
	float no_cr_time = TM.duration();

	TM.start();
	test_producer_consumer_cr(Buffer);
	TM.stop();
	float cr_time = TM.duration();

	TM.start();
	test_producer_consumer_locks(Buffer);
	TM.stop();
	float lock_time = TM.duration();
	
	std::cout << std::endl << "Without CR: " << no_cr_time;
	std::cout << std::endl << "With CR: " << cr_time;
	std::cout << std::endl << "With locks: " << lock_time;
	std::cout << std::endl;
}
