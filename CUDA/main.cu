// Deep Neural Network parallelized using CUDA

// include necesary libraries 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define INPUT_SIZE 4800
#define HIDDEN_SIZE 4800
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.0001
#define NUM_EPOCHS 10
#define numTrain 455

// error handler defintion
#define HANDLE_ERROR( err ) ( HandleError( err , __FILE__, __LINE__ ) )
static void HandleError(cudaError_t err , const char *file , int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),file , line);
        exit (EXIT_FAILURE) ;
    } 
}

#define MAX(a,b) (((a)>(b))?(a):(b))           // Macro to find maximum of two numbers

// Sigmoid activation function
__device__ double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Relu activation function
__device__ double relu(double x) { return MAX(x,0) ;}

// Derivative of sigmoid for backpropagation
__device__ double dSigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));     
}

// Derivative of sigmoid for backpropagation
__device__ double dRelu(double x) {
    if(x<0)
    {
        return 0;                              
    }
    else
    {
        return 1;
    }
}

// Forward propagation kernel
__global__ void forward_kernel(double* X, double* W1, double* W2, double* W3, double* b1, double* b2, double* b3, double* hidden1, double* hidden2, double* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += X[j] * W1[j * HIDDEN_SIZE + tid];
        }
        hidden1[tid] = relu(sum + b1[tid]);
    }

    __syncthreads();

    if (tid < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += hidden1[i] * W2[i * HIDDEN_SIZE + tid];
        }
        hidden2[tid] = relu(sum + b2[tid]);
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

__global__ void backward_kernel(double* X, double* W1, double* W2, double* W3, double* b1, double* b2, double* b3, double* hidden1, double* hidden2, double* output, double target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        double d_output = (*output - target) * dSigmoid(*output);

        double d_hidden2[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            d_hidden2[i] = dRelu(hidden2[i]) * W3[i] * d_output;
        }

        double d_hidden1[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = 0.0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += W3[j * HIDDEN_SIZE + i] * d_hidden2[j];
            }
            d_hidden1[i] = dRelu(hidden1[i]) * sum;
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                W1[j * HIDDEN_SIZE + i] -= LEARNING_RATE * X[j] * d_hidden1[i];
            }
            b1[i] -= LEARNING_RATE * d_hidden1[i];
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                W2[j * HIDDEN_SIZE + i] -= LEARNING_RATE * hidden1[i] * d_hidden2[j];
            }
            b2[i] -= LEARNING_RATE * d_hidden2[i];
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            W3[i] -= LEARNING_RATE * hidden2[i] * d_output;
        }
        b3[0] -= LEARNING_RATE * d_output;
    }
}


int main(int argc, char *argv[]) {

    double inputTrain[numTrain][INPUT_SIZE+1];

    char buffer[160000];
    char *record, *line;
    int i = 0;
    int j = 0;
    // Read Train data from train_data.csv
    FILE *fstream = fopen("train_data4801.csv", "r");
    if (fstream == NULL) {
        printf("\n file opening failed train ");
        return -1;
    }
    while ((line = fgets(buffer, sizeof(buffer), fstream)) != NULL) {
        record = strtok(line, ",");
        while (record != NULL) {
            inputTrain[i][j++] = strtod(record, NULL);
            record = strtok(NULL, ",");
        }
        if (j == INPUT_SIZE)
            i += 1;
    }

    double X[numTrain][INPUT_SIZE];

    // Initialize input data
    for (int i = 0; i < numTrain; i++) {
        for (int j = 1; j < INPUT_SIZE+1 ; j++) {
            X[i][j-1] = inputTrain[i][j];
        }
    }

    double y[numTrain];
    for (int i = 0; i < numTrain ; i++) {
            y[i] = inputTrain[i][0];
    }

    // Initialize weights
    double *d_W1, *d_W2, *d_W3;
    cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_W2, HIDDEN_SIZE * HIDDEN_SIZE *  sizeof(double));
    cudaMalloc(&d_W3, HIDDEN_SIZE * sizeof(double));

    // Initialize weights
    double *d_b1, *d_b2, *d_b3;
    cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_b2, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_b3, OUTPUT_SIZE * sizeof(double));

    double *h_W1 = (double *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double *h_W2 = (double *)malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double));
    double *h_W3 = (double *)malloc(HIDDEN_SIZE * sizeof(double));

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W1[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
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
            forward_kernel<<<1, 2>>>(X[rowIdx], d_W1, d_W2, d_W3, d_b1, d_b2, d_b3, hidden, hidden2, &output);
            cudaDeviceSynchronize();

            // Backward propagation
            backward_kernel<<<1, 2>>>(X[rowIdx], d_W1, d_W2, d_W3, d_b1, d_b2, d_b3, hidden, hidden2, &output, y[epoch]);
            cudaDeviceSynchronize();
        }
    }

    // Copy final weights back to host
    cudaMemcpy(h_W1, d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W2, d_W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W3, d_W3, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b1, d_b1, INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2, d_b2, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b3, d_b3, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);


    gettimeofday(&t2, 0);

    // Print final weights
    printf("Final weights W1: \n");
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        printf("%.4f ", h_W1[i]);
    }
    printf("\n");
    printf("Final weights W2: \n");
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        printf("%.4f ", h_W2[i]);
    }
    printf("\n");


    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to generate:  %3.1f ms \n", time);

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
