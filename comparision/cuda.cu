// CUDA code for only forward propagation 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

#define INPUT_SIZE 4800
#define HIDDEN_SIZE 4800
#define OUTPUT_SIZE 1
#define NUM_EPOCHS 1
#define numTrain 455
#define NUM_THREADS 1

// Sigmoid activation function
__device__ double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Forward propagation kernel
__global__ void forward_kernel(double* X, double* W1, double* W2, double* W3, double* b1, double* b2, double* b3, double* hidden1, double* hidden2, double* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += X[j] * W1[j * HIDDEN_SIZE + tid];
        }
        hidden1[tid] = sigmoid(sum + b1[tid]);
    }

    __syncthreads();

    if (tid < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += hidden1[i] * W2[i * HIDDEN_SIZE + tid];
        }
        hidden2[tid] = sigmoid(sum + b2[tid]);
    }

    __syncthreads();

    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += hidden2[i] * W3[i];
        }
        *output = sigmoid(sum + b3[0]);
    }
}


int main(int argc, char *argv[]) {

    double X[numTrain][INPUT_SIZE];

    // Initialize input data
    for (int i = 0; i < numTrain; i++) {
        for (int j = 0; j < INPUT_SIZE ; j++) {
            X[i][j] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }

    double y[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE ; i++) {
            y[i] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
    }

    double *h_W1 = (double *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double *h_W2 = (double *)malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double));
    double *h_W3 = (double *)malloc(HIDDEN_SIZE * sizeof(double));

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W1[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < HIDDEN_SIZE * HIDDEN_SIZE; i++) {
        h_W2[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_W3[i] = (double)rand() / RAND_MAX;
    }

    double *h_b1 = (double *)malloc(HIDDEN_SIZE * sizeof(double));
    double *h_b2 = (double *)malloc(HIDDEN_SIZE * sizeof(double));
    double *h_b3 = (double *)malloc(OUTPUT_SIZE * sizeof(double));


    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_b1[i] = (double)rand() / RAND_MAX;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_b2[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_b3[i] = (double)rand() / RAND_MAX;
    }

    struct timeval t1, t2;
    gettimeofday(&t1, 0);

    // Initialize weights
    double *d_W1, *d_W2, *d_W3;
    cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_W3, HIDDEN_SIZE * sizeof(double));

    // Initialize weights
    double *d_b1, *d_b2, *d_b3;
    cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_b3, OUTPUT_SIZE * sizeof(double));

    cudaMemcpy(d_W1, h_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, h_W3, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice); 

    // Training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        for(int rowIdx = 0; rowIdx < numTrain; rowIdx ++){

            double hidden[HIDDEN_SIZE];
            double hidden2[HIDDEN_SIZE];
            double output;

            // Forward propagation
            forward_kernel<<<1, NUM_THREADS>>>(X[rowIdx], d_W1, d_W2, d_W3, d_b1, d_b2, d_b3, hidden, hidden2, &output);
            cudaDeviceSynchronize();
        }
    }


    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to generate:  %10.3f ms \n", time);

    // Free memory
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_W3);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_b3);
    free(h_W1);
    free(h_W2);
    free(h_W3);
    free(h_b1);
    free(h_b2);
    free(h_b3);

    return 0;
}
