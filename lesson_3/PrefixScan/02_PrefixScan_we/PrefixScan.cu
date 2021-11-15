#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"

using namespace timer;
using namespace timer_cuda;

const int BLOCK_SIZE = 512;
__device__ int block_counter;

__global__ void PrefixScan(int* VectorIN, int N) {
	int step, limit;
	int valueRight, valueLeft;

	step = 1;
	for (limit = blockDim.x / 2; limit > 0; limit /= 2) {
		if (threadIdx.x < limit) {
			valueRight = (threadIdx.x + 1) * (step * 2) - 1;
			valueLeft = valueRight - step;
			VectorIN[valueRight] = VectorIN[valueRight] + VectorIN[valueLeft];
		}
		step *= 2;
		__syncthreads();
	}
	
	if (threadIdx.x == 0)
		VectorIN[blockDim.x - 1] = 0;
	__syncthreads();

	limit = 1;	
	for (step = blockDim.x / 2; step > 0; step /= 2) {
		if (threadIdx.x < limit) {
			valueRight = (threadIdx.x * 2 + 1) * step - 1;
			valueLeft = valueRight - step;
			int tmp = VectorIN[valueLeft];
			VectorIN[valueLeft] = VectorIN[valueRight];
			VectorIN[valueRight] = tmp + VectorIN[valueRight];
		}
		limit *= 2;
		__syncthreads();
	}
}

void printArray(int* Array, int N, const char str[] = "") {
	std::cout << str;
	for (int i = 0; i < N; ++i)
		std::cout << std::setw(5) << Array[i] << ' ';
	std::cout << std::endl << std::endl;
}

void printArray(int* Array, int start, int end, const char str[] = "") {
	std::cout << str;
	for (int i = start; i < end; ++i)
		std::cout << std::setw(5) << Array[i] << ' ';
	std::cout << std::endl << std::endl;
}

#define DIV(a,b)	(((a) + (b) - 1) / (b))

int main() {
	const int blockDim = BLOCK_SIZE;
	const int N = BLOCK_SIZE * 131072;
	
    // ------------------- INIT ------------------------------------------------

    // Random Engine Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    timer::Timer<HOST> host_TM;
    timer_cuda::Timer<DEVICE> dev_TM;

	// ------------------ HOST INIT --------------------------------------------

	int* VectorIN = new int[N];
	for (int i = 0; i < N; ++i)
		VectorIN[i] = distribution(generator);

	// ------------------- CUDA INIT -------------------------------------------

	int* devVectorIN;
	__SAFE_CALL( cudaMalloc(&devVectorIN, N * sizeof(int)) );
  __SAFE_CALL( cudaMemcpy(devVectorIN, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );

	int* prefixScan = new int[N];
	float dev_time;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------

	dev_TM.start();
	PrefixScan<<<DIV(N, blockDim), blockDim>>>(devVectorIN, N);
	dev_TM.stop();
	dev_time = dev_TM.duration();

	printArray(devVectorIN, 10);
	__SAFE_CALL(cudaMemcpy(prefixScan, devVectorIN, N * sizeof(int), cudaMemcpyDeviceToHost) );

	// ------------------- CUDA ENDING -----------------------------------------

	std::cout << std::fixed << std::setprecision(1) << "KernelTime Naive  : " << dev_time << std::endl << std::endl;

	// ------------------- VERIFY ----------------------------------------------

    host_TM.start();

	int* host_result = new int[N];
	std::partial_sum(VectorIN, VectorIN + N, host_result);

    host_TM.stop();

	if (!std::equal(host_result, host_result + blockDim - 1, prefixScan + 1)) {
		std::cerr << " Error! :  prefixScan" << std::endl << std::endl;
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}

    // ----------------------- SPEEDUP -----------------------------------------

    float speedup1 = host_TM.duration() / dev_time;
	std::cout << "Correct result" << std::endl
              << "(1) Speedup achieved: " << speedup1 << " x" << std::endl
              << std::endl << std::endl;

    std::cout << host_TM.duration() << ";" << dev_TM.duration() << ";" << host_TM.duration() / dev_TM.duration() << std::endl;
	
	delete[] host_result;
    delete[] prefixScan;
    
    __SAFE_CALL( cudaFree(devVectorIN) );
    
    cudaDeviceReset();
}
