#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

#define DIV(a, b)   (((a) + (b) - 1) (b))

#define HEIGHT 4000
#define WIDTH 2000
#define CHANNELS 3
#define N 5
#define BLOCK_SIZE 256

__global__ void GaussianBlur(const unsigned char* matrix_in, int* matrix_out, float* mask) {
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
        matrix_out[(y*WIDTH+x)*4+channel] = (unsigned char)pixel_value;
    }
    // no transparency
    matrix_out[(y*WIDTH+x)*4+CHANNELS] = (unsigned char)255;
}

int main() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);
    
    Timer<HOST> host_TM;
    Timer<DEVICE> dev_TM;
    
    float mask[] = { 0.0030, 0.0133, 0.0219, 0.0133, 0.0030, 0.0133, 0.0596, 0.0983, 0.0596, 0.0133, 0.0219, 0.0983, 0.1621, 0.0983, 0.0219, 0.0133, 0.0596, 0.0983, 0.0596, 0.0133, 0.0030, 0.0133, 0.0219, 0.0133, 0.0030 };
   
    int* h_matrix_in = new int[WIDTH * HEIGHT * CHANNELS];
    int* h_matrix_out = new int[WIDTH * HEIGHT * CHANNELS];

    int *d_matrix_in, *d_matrix_out, *d_mask;

    SAFE_CALL(cudaMalloc(&d_matrix_in, WIDTH * HEIGHT * CHANNELS *  sizeof(unsigned char)));
    SAFE_CALL(cudaMalloc(&d_matrix_out, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char)));
    SAFE_CALL(mudaMalloc(&d_mask, N * N * sizeof(float)));
    
    SAFE_CALL(cudaMemcpy(d_matrix_out, h_Matrix_in, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_mask, mask, N * N * sizeof(float), cudaMemcpyHostToDevice)); 

    for (int i = 0; i < WIDTH; ++i)
    for (int j = 0; j < HEIGHT; ++j)
        h_matrix_in[i*WIDTH + j] = distribution(generator);

    std::cout<<"Starting computation on DEVICE "<<std::endl;

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 numBlocks(WIDTH/BLOCK_SIZE - 1, HEIGHT/BLOCK_SIZE - 1, 1);

    dev_TM.start();

    GaussianBlur<<<blockSize, numBlocks>>>(d_matrix_in, d_mask, d_matrix_out);
    
    dev_TM.stop();

    CHECK_CUDA_ERROR;

    SAFE_CALL(cudaMemcpy(h_image_out, d_image_out
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
            if(h_matrix_out[i * WIDTH + j] < d_matrix_out
    
    std::cout << host_TM.duration() << std::endl;
    cudaDeviceReset();
}
