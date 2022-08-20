#include "stdio.h"

#define N 10000000


__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = threadIdx.x; i < n; i += blockDim.x){
        out[i] = a[i] + b[i];
    }
}

__global__ void vector_add_block(float *out, float *a, float *b, int n) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    out[idx] = a[idx] + b[idx];
}


int main(){
    float *a, *b, *out;
    
    float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }
    
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Main function
    //vector_add<<<1,256>>>(d_out, d_a, d_b, N);
    
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    vector_add_block<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
    
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    printf("val: %f, val[255]: %f, val[256]: %f\n", *out, *(out+255), *(out+256));
}
