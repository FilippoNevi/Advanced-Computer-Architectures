#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

const int RADIUS = 7;
const int BLOCK_SIZE = 256;
const int N = 10000000;

__global__
void stencilKernel(const int* d_input, int N,int* d_output) {
    __shared__ int ds_input[BLOCK_SIZE + 2*RADIUS];

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    /**
      Each thread loads in memory the corresponding item that has index = [global_thread_index + RADIUS]
      */
    // ds_input[threadIdx.x + RADIUS] = global_id < N ? d_input[global_id + RADIUS] : 0;
    if(global_id < N)
        ds_input[threadIdx.x + RADIUS] = d_input[global_id + RADIUS];
    else
        ds_input[threadIdx.x + RADIUS] = 0;
    __syncthreads();

    /**
      If the thread id is less than the radius, the global id is equivalent to the thread id
      */
    if (threadIdx.x < RADIUS)
        ds_input[threadIdx.x] = d_input[global_id];

    /**
      If the thread id is greater than the dimension of the block minus the radius,
      then the global id + radius*2 is equivalent to the thread id + radius*2
      */
    if (threadIdx.x >= blockDim.x - RADIUS)
        ds_input[threadIdx.x + RADIUS*2] = d_input[global_id + RADIUS*2];
    __syncthreads();
    
    /**
      Instead of computing the span between (index - radius) and (index + radius),
      it computes it between (index) and (index + radius*2)
      */
    int sum = 0;
    for (int j = threadIdx.x; j < threadIdx.x + RADIUS*2; ++j)
        sum += ds_input[j];
    d_output[global_id + RADIUS] = sum;

    if (global_id >= N - RADIUS && global_id < N)
        d_output[global_id] = 0;
}

int main() {
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_input      = new int[N];
    int* h_output_tmp = new int[N]; // <-- used for device result
    int* h_output     = new int[N](); // initilization to zero

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N; i++)
        h_input[i] = distribution(generator);

    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    TM_host.start();

    for (int i = RADIUS; i < N - RADIUS; i++) {
        for (int j = i - RADIUS; j <= i + RADIUS; j++)
            h_output[i] += h_input[j];
    }

    TM_host.stop();
    TM_host.print("1DStencil host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int *d_input, *d_output;
    SAFE_CALL( cudaMalloc( &d_input, N * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_output, N * sizeof(int) ) );

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    SAFE_CALL( cudaMemcpy( d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice ) );

    // DEVICE INIT
    dim3 dim_grid(N/BLOCK_SIZE, 1, 1);
    if (N % BLOCK_SIZE) dim_grid.x++;
    dim3 dim_blocks(BLOCK_SIZE, 1, 1);

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    stencilKernel<<< dim_grid, dim_blocks >>>(d_input, N, d_output);

    TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("1DStencil device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL( cudaMemcpy( h_output_tmp, d_output, N * sizeof(int), cudaMemcpyDeviceToHost ) );

    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N; i++) {
        if (h_output[i] != h_output_tmp[i]) {
            std::cerr << "wrong result at: " << i
                      << "\nhost:   " << h_output[i]
                      << "\ndevice: " << h_output_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_tmp;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL( cudaFree( d_input ) );
    SAFE_CALL( cudaFree( d_output ) );

    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
