// Neural Network serial code, v2.0
// Anu, Thomas, Zack

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>


#define learningRate 0.0001f                    // defining a constant for learning rate
#define numberOfEpochs 1                        // number of epochs 1500

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double initWeights() {
    return ((double)rand()) / ((double)RAND_MAX);  // function to initialize weights
}

// random shuffle data
void shuffle(int *array, size_t n){
    // Initializes random number generator
    srand(44);

    if (n > 1){
        size_t i;
        for(i = 0; i < n-1; i++){
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);  // generate a random index to swap with
            int temp = array[j];                               // creating a temporary variable
            array[j] = array[i];                               // updating the values
            array[i] = temp;                                   // swapping the elements
        }
    }
}

#define numInputs 4800               // number of columns
#define numHiddenNodes 4800          // number of nodes in the first hidden layer
#define numHiddenNodes2 4800         // number of nodes in the second hidden layer
#define numOutputs 1               // number of outputs
#define numTrain 455
#define numTest 114
#define numTrainingSets 569        // number of instances of total data

int main(int argc, char *argv[]) {

    // learning rate
    const double lr = learningRate;
    int thread_count = atoi(argv[1]);

    // Declare pointers for dynamically allocated arrays
    double* hiddenLayer;
    double* hiddenLayer2;
    double* outputLayer;
    double* hiddenLayerBias;
    double* hiddenLayerBias2;
    double* outputLayerBias;
    double** hiddenWeights;
    double** hiddenWeights2;
    double** outputWeights;

    // Allocate memory for hidden layer nodes vectors
    hiddenLayer = (double*) malloc(numHiddenNodes * sizeof(double));
    hiddenLayer2 = (double*) malloc(numHiddenNodes2 * sizeof(double));
    outputLayer = (double*) malloc(numOutputs * sizeof(double));

    // Allocate memory for hidden layer bias vectors
    hiddenLayerBias = (double*) malloc(numHiddenNodes * sizeof(double));
    hiddenLayerBias2 = (double*) malloc(numHiddenNodes2 * sizeof(double));
    outputLayerBias = (double*) malloc(numOutputs * sizeof(double));

    // Allocate memory for hidden and output weights matrices
    hiddenWeights = (double**) malloc(numInputs * sizeof(double*));
    hiddenWeights2 = (double**) malloc(numInputs * sizeof(double*));
    outputWeights = (double**) malloc(numHiddenNodes2 * sizeof(double*));

    for (int i = 0; i < numInputs; i++) {
        hiddenWeights[i] = (double*) malloc(numHiddenNodes * sizeof(double));
        hiddenWeights2[i] = (double*) malloc(numHiddenNodes2 * sizeof(double));
    }

    for (int i = 0; i < numHiddenNodes2; i++) {
        outputWeights[i] = (double*) malloc(numOutputs * sizeof(double));
    }

    // Dynamically allocate memory for training and testing inputs and outputs
    double **trainingInputs = (double **)malloc(numTrain * sizeof(double *));
    for (int i = 0; i < numTrain; i++) {
        trainingInputs[i] = (double *)malloc(numInputs * sizeof(double));
    }


    double **trainingOutputs = (double **)malloc(numTrain * sizeof(double *));
    for (int i = 0; i < numTrain; i++) {
        trainingOutputs[i] = (double *)malloc(numOutputs * sizeof(double));
    }



    double inputTrain[numTrain][numInputs+1];

    for (int i = 0; i < numTrain; i++) {
        for (int j = 0; j < numInputs+1; j++) {
            inputTrain[i][j] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
 


    // training data (inputs)
    for (int ro=0; ro<numTrain; ro++)
    {
        for(int columns=1; columns<numInputs+1; columns++)
        {
            trainingInputs[ro][columns-1] = inputTrain[ro][columns];
        }
    }
    // training data (inputs)
    // double trainingInputs[numTrain][numInputs];

    #pragma omp parallel for num_threads(thread_count) collapse(2) shared(trainingInputs, inputTrain)
    for (int ro=0; ro<numTrain; ro++)
    {
        for(int columns=1; columns<numInputs+1; columns++)
        {
            trainingInputs[ro][columns-1] = inputTrain[ro][columns];
        }
    }


    // training data (outputs)

    #pragma omp parallel for num_threads(thread_count) collapse(2) shared(trainingOutputs, inputTrain)
    for (int ro=0; ro<numTrain; ro++)
    {
        for(int columns=0; columns<1; columns++)
        {
            //int row = newTrainingSetOrder[ro];
            trainingOutputs[ro][columns] = inputTrain[ro][columns];
        }
    }


    // initialize bias and weight terms to random
    // hidden layer 1 weights
    for(int i = 0; i < numHiddenNodes; i++){
        hiddenLayerBias[i] = initWeights();
    }
    // hidden layer 2 weights
    for(int i = 0; i < numHiddenNodes2; i++){
        hiddenLayerBias2[i] = initWeights();
    }
    // output layer weights
    for(int i = 0; i < numOutputs; i++){
        outputLayerBias[i] = initWeights();
    }
    // hidden layer 1 bias
    for(int i = 0; i < numInputs; i++){
        for(int j = 0; j < numHiddenNodes; j++){
            hiddenWeights[i][j] = initWeights();
        }
    }
    // hidden layer 2 bias
    for(int i = 0; i < numHiddenNodes; i++){
        for(int j = 0; j < numHiddenNodes2; j++){
            hiddenWeights2[i][j] = initWeights();
        }
    }
    // output layer bias
    for(int i = 0 ; i < numHiddenNodes; i++){
        for(int j = 0; j < numOutputs; j++){
            outputWeights[i][j] = initWeights();
        }
    }

    // specify training set
    int trainingSetOrder[numTrain];
    for(int i = 0 ; i < numTrain ; i++)
    {
        trainingSetOrder[i] = i;
    }

    // shuffling training set
    // shuffle(trainingSetOrder, cutOffTrain);


    // int numberOfEpochs = 1500;                            // number of epochs


    // start time measurement
    double time1, time2;
    time1 = omp_get_wtime();

    //training loop
    for(int epoch = 0; epoch < numberOfEpochs; epoch++){

        shuffle(trainingSetOrder, numTrain);
	//#pragma omp parallel for num_threads(thread_count)
        for(int x = 0; x < numTrain; x ++){
            int i = trainingSetOrder[x];

            // forward pass
            // compute hidden layer activation

            // hidden layer 1
            int k;
            //#pragma omp parallel for num_threads(thread_count) shared(trainingInputs, hiddenWeights, hiddenLayer, hiddenLayerBias) private(j, k)
            #pragma omp parallel for num_threads(thread_count)
	    for(int j =0; j < numHiddenNodes; j++){
                double activation = hiddenLayerBias[j];

                for(int k = 0; k < numInputs; k++){
                    activation += trainingInputs[i][k] * hiddenWeights[k][j];
                }

                hiddenLayer[j] = sigmoid(activation);
            }

            // hidden layer 2
            double activation = 0;
	    //#pragma omp parallel for num_threads(thread_count) shared(hiddenLayer, hiddenWeights2, hiddenLayer2, hiddenLayerBias2) private(j, k)
	    //double activation = 0;
	    #pragma omp parallel for reduction(+:activation) num_threads(thread_count)
            for(int j =0; j < numHiddenNodes2; j++){
                activation = hiddenLayerBias2[j];

                for(int k = 0; k < numHiddenNodes; k++){
                    activation += hiddenLayer[k] * hiddenWeights2[k][j];
                }

                hiddenLayer2[j] = sigmoid(activation);
            }

            // compute output layer activation
            //#pragma omp parallel for num_threads(thread_count) shared(hiddenLayer2, outputWeights, outputLayer, outputLayerBias) private(j, k)
	    #pragma omp parallel for reduction(+:activation) num_threads(thread_count)
            for(int j =0; j < numOutputs; j++){
                activation = outputLayerBias[j];

                for(int k = 0; k < numHiddenNodes2; k++){
                    activation += hiddenLayer2[k] * outputWeights[k][j];
                }

                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: %g | %g | %g | %g | %g | %g |      Output: %g      Expected Output: %g \n",
                   trainingInputs[i][1], trainingInputs[i][2], trainingInputs[i][3], trainingInputs[i][4], trainingInputs[i][5], trainingInputs[i][6],
                   outputLayer[0], trainingOutputs[i][0]);
        }
    }

    // barrier for time measurements
    #pragma omp barrier

    time2 = omp_get_wtime();
/*
    // print final weights after done training
    fputs ("\nFinal Hidden Weights\n[ ", stdout);
    for(int j = 0; j < numHiddenNodes; j++){
        fputs ("[ ", stdout);
        for(int k = 0; k < numInputs; k++){
            printf("%f ", hiddenWeights[k][j]);
        }
        fputs("] ", stdout);
    }
    fputs ("\nFinal Hidden2 Weights\n[ ", stdout);
    for(int j = 0; j < numHiddenNodes2; j++){
        fputs ("[ ", stdout);
        for(int k = 0; k < numHiddenNodes; k++){
            printf("%f ", hiddenWeights2[k][j]);
        }
        fputs("] ", stdout);
    }

    fputs ("]\nFinal Hidden Biases\n[ ", stdout);
    for(int j = 0; j < numHiddenNodes; j++){
        printf("%f ", hiddenLayerBias[j]);
    }

    fputs ("]\nFinal Hidden2 Biases\n[ ", stdout);
    for(int j = 0; j < numHiddenNodes2; j++){
        printf("%f ", hiddenLayerBias2[j]);
    }

    fputs ("]\nFinal Output Weights\n", stdout);
    for(int j = 0; j < numOutputs; j++){
        fputs ("[ ", stdout);
        for(int k = 0; k < numHiddenNodes2; k++){
            printf("%f ", outputWeights[k][j]);
        }
        fputs("] \n", stdout);
    }

    fputs ("\nFinal Output Biases\n[ ", stdout);
    for(int j = 0; j < numOutputs; j++){
        printf("%f ", outputLayerBias[j]);
    }

    fputs("] \n", stdout);
*/
// calculate total time in s
    double totalTime;
    totalTime = time2 - time1;
    printf("Total time: %fs \n", totalTime);  // time


    free(hiddenLayer);
    free(hiddenLayer2);
    free(outputLayer);
    free(hiddenLayerBias);
    free(hiddenLayerBias2);
    free(outputLayerBias);
    free(hiddenWeights);
    free(hiddenWeights2);
    free(outputWeights);


    // Free dynamically allocated memory for training and testing inputs and outputs
    for (int i = 0; i < numTrain; i++) {
        free(trainingInputs[i]);
    }
    free(trainingInputs);

    for (int i = 0; i < numTrain; i++) {
        free(trainingOutputs[i]);
    }
    free(trainingOutputs);

    return 0;

}
