#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

#define DIV(a, b)   (((a) + (b) - 1) (b))

const int height = 4000;
const int width = 2000;
const int N = 2000;
#define BLOCK_SIZE 256

__global__ void GaussianBlur(int* MatrixA, int* MatrixB, int N, int height, int width) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float pixel_value = 0;
                for (int u = 0; u < N; ++u) {
                    for (int v = 0; v < N; ++v) {
                        int new_x = min(width, max(0, x+u-N/2));
                        int new_y = min(height, max(0, y+v-N/2));
                        pixel_value += mask[v*N+u]*MatrixA[(new_y*width+new_x)*4+channel];
                    }
                }
                MatrixB[(y*width+x)*4+channel] = (unsigned char)pixel_value;
            }
            // no transparency
            MatrixB[(y*width+x)*4+3] = (unsigned char)255;
        }
    }
}

int main() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);
    
    Timer<HOST> host_TM;
    Timer<DEVICE> dev_TM;

    int* h_MatrixA = new int[N][N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            h_MatrixA[i][j] = distribution(generator);

    int* d_matrixA, d_matrixB;
    SAFE_CALL(cudaMalloc(&d_matrixA, N * N * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_matrixB, N * N * sizeof(int)));
    SAFE_CALL(cudaMemcpy(d_matrixA, h_matrixA, N * N * sizeof(int), cudaMemcpyHostToDevice));

    std::cout<<"Starting computation on DEVICE "<<std::endl;

    dev_TM.start();
    GaussianBlur<<<DIV(N, BLOCK_SIZE), BLOCK_SIZE>>>(d_matrixA, d_matrixB, N, height, width);

    dev_TM.stop();
    dev_time = dev_TM.duration();
    CHECK_CUDA_ERROR;
    
}
