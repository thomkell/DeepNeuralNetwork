// Deep Neural Network parallelized using CUDA
// include necesary libraries 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TIME_DIVIDER 1000000

long long get_time() {
    return TIME_DIVIDER * clock() / CLOCKS_PER_SEC;
}

long long t_start;
void timerStart() {
    t_start = get_time();
}

double timerStop(const char* info) {
    double time = ((double)(get_time() - t_start)) / TIME_DIVIDER;
    printf("Timing - %s. \t\tElasped %.12f seconds \n", info, time);
    return time;
}

#define INPUT_SIZE 570
#define HIDDEN_SIZE 570
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.0001
#define NUM_EPOCHS 1
#define numTrain 455
#define MAX_LINE_BUFFER_SIZE 160000

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
        // calculate output weight
        double d_output = (*output - target) * dSigmoid(*output);

        // calculate hidden 2 weight
        double d_hidden2[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            d_hidden2[i] = dRelu(hidden2[i]) * W3[i] * d_output;
        }
        
        // calculate hidden 1 weight
        double d_hidden1[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = 0.0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += W3[j * HIDDEN_SIZE + i] * d_hidden2[j];
            }
            d_hidden1[i] = dRelu(hidden1[i]) * sum;
        }

        // update hidden weights 1 & bias
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                W1[j * HIDDEN_SIZE + i] -= LEARNING_RATE * X[j] * d_hidden1[i];
            }
            b1[i] -= LEARNING_RATE * d_hidden1[i];
        }

        // update hidden weights 2 & bias
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                W2[j * HIDDEN_SIZE + i] -= LEARNING_RATE * hidden1[i] * d_hidden2[j];
            }
            b2[i] -= LEARNING_RATE * d_hidden2[i];
        }

        // update output weights & bias
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            W3[i] -= LEARNING_RATE * hidden2[i] * d_output;
        }
        b3[0] -= LEARNING_RATE * d_output;
    }
}


int main(int argc, char *argv[]) {

    double** inputTrain = new double* [numTrain];
    double** X = new double* [numTrain];

    int numThread = atoi(argv[1]);
   
    for (int i = 0; i < numTrain; ++i) {
        inputTrain[i] = new double[INPUT_SIZE + 1];
        X[i] = new double[INPUT_SIZE];
    }

    char* buffer = new char[MAX_LINE_BUFFER_SIZE];
    char *record, *line;
    int i = 0;
    int j = 0;
    // Read Train data from train_data.csv
    FILE *fstream = fopen("train_data4801.csv", "r");
    if (fstream == NULL) {
        printf("\n file opening failed train ");
        return -1;
    }
    while ((line = fgets(buffer, MAX_LINE_BUFFER_SIZE, fstream)) != NULL) {
        record = strtok(line, ",");
        while (record != NULL) {
            inputTrain[i][j++] = strtod(record, NULL);
            record = strtok(NULL, ",");
            if (j == INPUT_SIZE) {
                j = 0;
                i += 1;
                break;
            }
        }
    }    

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

    double* dX;
    cudaMalloc(&dX, numTrain * INPUT_SIZE * sizeof(double));

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

    clock_t startTime;
    clock_t endTime;


    startTime = clock();


    for (int i = 0; i < numTrain; ++i) {
        cudaMemcpy(dX + i * INPUT_SIZE, X[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_W1, h_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, h_W3, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);   

    double* hidden;
    cudaMalloc(&hidden, HIDDEN_SIZE * sizeof(double));
    double* hidden2;
    cudaMalloc(&hidden2, HIDDEN_SIZE * sizeof(double));

    double* output;
    cudaMalloc(&output, sizeof(double));

    char szInfo[0x100];

    // Training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        for(int rowIdx = 0; rowIdx < numTrain; rowIdx ++){
            timerStart();

            // Forward propagation
            forward_kernel<<<1, numThread>>>(dX + rowIdx * INPUT_SIZE, d_W1, d_W2, d_W3, d_b1, d_b2, d_b3, hidden, hidden2, output);
            HANDLE_ERROR(cudaDeviceSynchronize());
            // Backward propagation
            backward_kernel<<<1, numThread>>>(dX + rowIdx * INPUT_SIZE, d_W1, d_W2, d_W3, d_b1, d_b2, d_b3, hidden, hidden2, output, y[epoch]);
            HANDLE_ERROR(cudaDeviceSynchronize());

            // sprintf(szInfo, "[%d th-epoch][%d th-train] time elapsed : ", epoch, rowIdx);
            // timerStop(szInfo);
        }
    }

    // Copy final weights back to host
    cudaMemcpy(h_W1, d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W2, d_W2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W3, d_W3, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b1, d_b1, INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2, d_b2, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b3, d_b3, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    endTime = clock();
    double total_t;
    total_t = (double)(endTime - startTime)/ CLOCKS_PER_SEC;
    printf("Total time taken = %fs \n", total_t);



/*
    // Print final weights
    printf("Final weights W1: \n");
    for (int i = 0; i < 4 * HIDDEN_SIZE; i++) {
        printf("%.4f ", h_W1[i]);
    }
    printf("\n");
    printf("Final weights W2: \n");
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        printf("%.4f ", h_W2[i]);
    }
    printf("\n");
*/
    // Free memory
    cudaFree(dX);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_W3);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_b3);
    cudaFree(hidden);
    cudaFree(hidden2);
    cudaFree(output);
    free(h_W1);
    free(h_W2);
    free(h_W3);
    free(h_b1);
    free(h_b2);
    free(h_b3);

    for (int i = 0; i < numTrain; ++i) {
       delete inputTrain[i];
       delete X[i];
    }
    delete inputTrain;
    delete X;
    delete buffer;

    return 0;
}
