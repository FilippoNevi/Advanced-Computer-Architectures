#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

#define DIV(a, b)   (((a) + (b) - 1) (b))

const int height = 4000;
const int width = 2000;
const int N = 5;
#define BLOCK_SIZE 256
const float mask[N][N] = {
    {0.0030, 0.0133, 0.0219, 0.0133, 0.0030},
    {0.0133, 0.0596, 0.0983, 0.0596, 0.0133},
    {0.0219, 0.0983, 0.1621, 0.0983, 0.0219},
    {0.0133, 0.0596, 0.0983, 0.0596, 0.0133}, 
    {0.0030, 0.0133, 0.0219, 0.0133, 0.0030}
};
__global__ void GaussianBlur(int* MatrixA, int* MatrixB, float mask[][], int N, int height, int width) {
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

    int* h_MatrixA = new int[width * height];
    int* h_MatrixB = new int[width * height];

    int *d_MatrixA, *d_MatrixB;
    SAFE_CALL(cudaMalloc(&d_MatrixA, width * height * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_MatrixB, width * height * sizeof(int)));
    SAFE_CALL(cudaMemcpy(d_MatrixA, h_MatrixA, width * height * sizeof(int), cudaMemcpyHostToDevice));


    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
            h_MatrixA[i*width + j] = distribution(generator);


    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float pixel_value = 0;
                for (int u = 0; u < N; ++u) {
                    for (int v = 0; v < N; ++v) {
                        int new_x = min(width, max(0, x+u-N/2));
                        int new_y = min(height, max(0, y+v-N/2));
                        pixel_value += mask[v*N+u]*h_MatrixA[(new_y * width + new_x) * 4 + channel];
                    }
                }
                h_MatrixB[(y * width + x) * 4 + channel] = (unsigned char)pixel_value;
            }
            // no transparency
            h_MatrixB[(y * width + x) * 4 + 3] = (unsigned char)255;
        }
    }

    std::cout<<"Starting computation on DEVICE "<<std::endl;

    dev_TM.start();
    GaussianBlur<<<DIV(N, BLOCK_SIZE), BLOCK_SIZE>>>(d_MatrixA, d_MatrixB, mask, N, height, width);

    dev_TM.stop();
    dev_time = dev_TM.duration();
    CHECK_CUDA_ERROR;
    
}
