#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

#define HEIGHT 2000
#define WIDTH 1000
#define CHANNELS 3
#define N 5
#define BLOCK_SIZE 32 

__global__ void GaussianBlur(const unsigned char* matrix_in, unsigned char* matrix_out, float* mask) {
    int global_id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_id_y = threadIdx.y + blockIdx.y * blockDim.y;

    for (int channel = 0; channel < CHANNELS; ++channel) {
        float pixel_value = 0;
        for (int u = 0; u < N; ++u) {
            for (int v = 0; v < N; ++v) {
                int new_x = min(WIDTH, max(0, global_id_x+u-N/2));
                int new_y = min(HEIGHT, max(0, global_id_y+v-N/2));
                pixel_value += mask[v*N+u] * matrix_in[(new_y*WIDTH+new_x)*4+channel];
            }
        }
        matrix_out[(global_id_y*WIDTH+global_id_x)*4+channel] = (unsigned char)pixel_value;
    }
    // no transparency
    matrix_out[(global_id_y*WIDTH+global_id_x)*4+CHANNELS] = (unsigned char)255;
}

int main() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);
    
    Timer<HOST> host_TM;
    Timer<DEVICE> dev_TM;
    
    float mask[] = { 0.0030, 0.0133, 0.0219, 0.0133, 0.0030, 0.0133, 0.0596, 0.0983, 0.0596, 0.0133, 0.0219, 0.0983, 0.1621, 0.0983, 0.0219, 0.0133, 0.0596, 0.0983, 0.0596, 0.0133, 0.0030, 0.0133, 0.0219, 0.0133, 0.0030 };
   
    unsigned char* h_matrix_in = new unsigned char[WIDTH * HEIGHT * CHANNELS];
    unsigned char* h_matrix_out = new unsigned char[WIDTH * HEIGHT * CHANNELS];
    unsigned char* d_matrix_out_tmp = new unsigned char[WIDTH * HEIGHT * CHANNELS];
    unsigned char *d_matrix_in, *d_matrix_out;
    float* d_mask;

    SAFE_CALL(cudaMalloc(&d_matrix_in, WIDTH * HEIGHT * CHANNELS *  sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&d_matrix_out, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&d_mask, N * N * sizeof(float)));
    
    SAFE_CALL(cudaMemcpy(d_matrix_in, h_matrix_in, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_mask, &mask, N * N * sizeof(float), cudaMemcpyHostToDevice)); 

    for (int i = 0; i < WIDTH; ++i)
    for (int j = 0; j < HEIGHT; ++j)
        h_matrix_in[i*WIDTH + j] = distribution(generator);

    std::cout<<"Starting computation on DEVICE "<<std::endl;

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 numBlocks(WIDTH/BLOCK_SIZE - 1, HEIGHT/BLOCK_SIZE - 1, 1);

    dev_TM.start();

    GaussianBlur<<<blockSize, numBlocks>>>(d_matrix_in, d_matrix_out, d_mask);
    
    dev_TM.stop();

    CHECK_CUDA_ERROR;

    SAFE_CALL(cudaMemcpy(d_matrix_out_tmp, d_matrix_out, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    host_TM.start();

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float pixel_value = 0;
                for (int u = 0; u < N; ++u) {
                    for (int v = 0; v < N; ++v) {
                        int new_x = min(WIDTH, max(0, x+u-N/2));
                        int new_y = min(HEIGHT, max(0, y+v-N/2));
                        pixel_value += mask[v*N+u] * h_matrix_in[(new_y * WIDTH + new_x) * 4 + channel];
                    }
                }
                h_matrix_out[(y * WIDTH + x) * 4 + channel] = (unsigned char)pixel_value;
            }
            // no transparency
            h_matrix_out[(y * WIDTH + x) * 4 + CHANNELS] = (unsigned char)255;
        }
    }

    host_TM.stop();

    for (int i = 0; i < HEIGHT; ++i) {
        for(int j = 0; j < WIDTH; ++j) {
            if(h_matrix_out[i * WIDTH + j] < d_matrix_out_tmp[i * WIDTH + j]-1 || h_matrix_out[i * WIDTH + j] > d_matrix_out[i * WIDTH + j]+1) {
                std::cerr << "wrong result at indexes [" << i << "][ " << j << "]" << std::endl;
                std::cerr << "Host's value = " << (short)h_matrix_out[i * WIDTH + j] << std::endl;
                std::cerr << "Device's value = " << (short)d_matrix_out_tmp[i * WIDTH + j] << std::endl;
                cudaDeviceReset();
                std::exit(EXIT_FAILURE);
            }
        }
    }
    std::cout << "<> Correct!\n\n";

    delete[] h_matrix_in;
    delete[] h_matrix_out;
    delete[] d_matrix_out_tmp;
   
    SAFE_CALL( cudaFree(d_matrix_in) );
    SAFE_CALL( cudaFree(d_matrix_out) );
    SAFE_CALL( cudaFree(mask) );

    std::cout << host_TM.duration() << std::endl;
    cudaDeviceReset();
}
