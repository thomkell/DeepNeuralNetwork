// Deep Neural Network serial code with 2 layers 
// Names: Anuradha Agarwal, Thomas Keller, Zack Humphries 
// Final Project

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "evaluation.h"

int numInputs;
int numHiddenNodes;
int numHiddenNodes2;
double learningRate;
int numberOfEpochs; 

#define numOutputs 1                        // Number of outputs
#define numTrain 455                        // Number of rows in train set 
#define numTest 114                         // Number of rows in test set
#define numTrainingSets 569                 // Number of instances of total data

// The main starts here 
int main(int argc, char *argv[]) {

    // Checking for command line arguments 
    if (argc != 4){
        printf("Please provide number of inputs as your first argument, learning rate as your second argument and number of epochs as your third argument. \n");
        exit(1);
    }

    numInputs = atoi(argv[1]);             // Number of columns
    numHiddenNodes = atoi(argv[1]);        // Number of nodes in the first hidden layer
    numHiddenNodes2 = atoi(argv[1]);       // Number of nodes in the second hidden layer
    learningRate = atof(argv[2]);          // Constant for learning rate
    numberOfEpochs = atoi(argv[3]);        // Number of epochs

    char training_set[100];
    char testing_set[100];
    int characters;

    // Updating variables depending on the command line arguments 
    if (numInputs == 30){
        strncpy(training_set, "datasets/train_data.csv", sizeof(training_set));
        strncpy(testing_set, "datasets/test_data.csv", sizeof(testing_set));
        characters = 1024;
    }
    else if (numInputs == 4800){
        strncpy(training_set, "datasets/train_data4801.csv", sizeof(training_set));
        strncpy(testing_set, "datasets/test_data4801.csv", sizeof(testing_set));
        characters = 1602400;
    }
    else{
        printf("Please make sure your first argument is either 30 or 4800. \n");
        exit(2);
    }

    // Learning rate
    const double lr = learningRate;

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
    
    // Dynamically allocate memory for hiddenWeights and hiddenWeights2
    for (int i = 0; i < numInputs; i++) {
        hiddenWeights[i] = (double*) malloc(numHiddenNodes * sizeof(double));
        hiddenWeights2[i] = (double*) malloc(numHiddenNodes2 * sizeof(double));
    }

    // Dynamically allocate memory for outputWeights
    for (int i = 0; i < numHiddenNodes2; i++) {
        outputWeights[i] = (double*) malloc(numOutputs * sizeof(double));
    }

    // Dynamically allocate memory for training and testing inputs and outputs
    double **trainingInputs = (double **)malloc(numTrain * sizeof(double *));
    for (int i = 0; i < numTrain; i++) {
        trainingInputs[i] = (double *)malloc(numInputs * sizeof(double));
    }
    
    // Dynamically allocate memory for testingInputs
    double **testingInputs = (double **)malloc(numTest * sizeof(double *));
    for (int i = 0; i < numTest; i++) {
        testingInputs[i] = (double *)malloc(numInputs * sizeof(double));
    }

    // Dynamically allocate memory for trainingInputs
    double **trainingOutputs = (double **)malloc(numTrain * sizeof(double *));
    for (int i = 0; i < numTrain; i++) {
        trainingOutputs[i] = (double *)malloc(numOutputs * sizeof(double));
    }

    // Allocate memory for testingOutputs 
    double *testingOutputs = (double *)malloc(numTest * sizeof(double));

   
    // Reading csv files 
    char buffer[characters];
    char buffer2[characters];
    char *record, *line;
    char *record2, *line2;
    int i = 0, j = 0;
    double inputTrain[numTrain][numInputs+1];
    double inputTest[numTrain][numInputs+1];

    // Read Train data from train_data.csv
    FILE *fstream = fopen(training_set, "r");
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
        if (j == numInputs)
            i += 1;
    }

    fclose(fstream);

    i = 0, j = 0;

    // Read Test data from test_data.csv
    FILE *gstream = fopen(testing_set, "r");
    if (gstream == NULL) {
        printf("\n file opening failed test ");
        return -1;
    }
    while ((line2 = fgets(buffer2, sizeof(buffer2), gstream)) != NULL) {
        record2 = strtok(line2, ",");
        while (record2 != NULL) {
            inputTest[i][j++] = strtod(record2, NULL);
            record2 = strtok(NULL, ",");
        }
        if (j == numInputs)
            i += 1;
    }

    fclose(gstream);

    // Training data (inputs)
    for (int ro=0; ro<numTrain; ro++)
    {
        for(int columns=1; columns<numInputs+1; columns++)
        {
            trainingInputs[ro][columns-1] = inputTrain[ro][columns];
        }
    }

    // Testing data (inputs)
    for (int ro=0; ro<numTest; ro++)
    {
        for(int columns=1; columns<numInputs+1; columns++)
        {
            testingInputs[ro][columns-1] = inputTest[ro][columns];
        }
    }

    // Training data (outputs)
    for (int ro=0; ro<numTrain; ro++)
    {
        for(int columns=0; columns<1; columns++)
        {
            trainingOutputs[ro][columns] = inputTrain[ro][columns];
        }
    }

    // Testing data (outputs)
    for (int ro=0; ro<numTest; ro++)
    {
        for(int columns=0; columns<1; columns++)
        {
            testingOutputs[ro] = inputTest[ro][columns];
        }
    }

    // Initialize bias and weight terms to random
    // Hidden layer 1 weights
    for(int i = 0; i < numHiddenNodes; i++){
        hiddenLayerBias[i] = initWeights();
    }
    // Hidden layer 2 weights
    for(int i = 0; i < numHiddenNodes2; i++){
        hiddenLayerBias2[i] = initWeights();
    }
    // Output layer weights
    for(int i = 0; i < numOutputs; i++){
        outputLayerBias[i] = initWeights();
    }
    // Hidden layer 1 bias
    for(int i = 0; i < numInputs; i++){
        for(int j = 0; j < numHiddenNodes; j++){
            hiddenWeights[i][j] = initWeights();
        }
    }
    // Hidden layer 2 bias
    for(int i = 0; i < numHiddenNodes; i++){
        for(int j = 0; j < numHiddenNodes2; j++){
            hiddenWeights2[i][j] = initWeights();
        }
    }
    // Output layer bias
    for(int i = 0 ; i < numHiddenNodes; i++){
        for(int j = 0; j < numOutputs; j++){
            outputWeights[i][j] = initWeights();
        }
    }

    // Specify training set
    int trainingSetOrder[numTrain];
    for(int i = 0 ; i < numTrain ; i++)
    {
        trainingSetOrder[i] = i;
    }

    // Start time measurement
    clock_t start, end;
    start = clock();

    // Training loop
    for(int epoch = 0; epoch < numberOfEpochs; epoch++){

        shuffle(trainingSetOrder, numTrain);

        for(int x = 0; x < numTrain; x ++){
            int i = trainingSetOrder[x];

            // forward pass: compute hidden layer activation
            // hidden layer 1
            for(int j =0; j < numHiddenNodes; j++){
                double activation = hiddenLayerBias[j];

                for(int k = 0; k < numInputs; k++){
                    activation += trainingInputs[i][k] * hiddenWeights[k][j];
                }

                hiddenLayer[j] = relu(activation);
            }

            // hidden layer 2
            for(int j =0; j < numHiddenNodes2; j++){
                double activation = hiddenLayerBias2[j];

                for(int k = 0; k < numHiddenNodes; k++){
                    activation += hiddenLayer[k] * hiddenWeights2[k][j];
                }

                hiddenLayer2[j] = relu(activation);
            }

            // compute output layer activation
            for(int j =0; j < numOutputs; j++){
                double activation = outputLayerBias[j];

                for(int k = 0; k < numHiddenNodes2; k++){
                    activation += hiddenLayer2[k] * outputWeights[k][j];
                }

                outputLayer[j] = sigmoid(activation);
            }

            // Print training output (only first 6 inputs for readability)
            printf("Input: %g | %g | %g | %g | %g | %g |      Output: %g      Expected Output: %g \n",
                   trainingInputs[i][1], trainingInputs[i][2], trainingInputs[i][3], trainingInputs[i][4], trainingInputs[i][5], trainingInputs[i][6],
                   outputLayer[0], trainingOutputs[i][0]);

            // Backpropagation
            // Compute change in output weights
            double deltaOutput[numOutputs];
            for(int j = 0; j < numOutputs; j++){
                double error = (trainingOutputs[i][j] - outputLayer[j]); // L1
                deltaOutput[j] = error * dSigmoid(outputLayer[j]) ;
            }

            // Compute change in hidden weights (second layer)
            double deltaHidden2[numHiddenNodes2];
            for(int j = 0; j < numHiddenNodes2; j++){
                double error = 0.0f;
                for(int k = 0; k < numOutputs; k++){
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden2[j] = error * dRelu(hiddenLayer[j]);
            }

            // Compute change in hidden weights (first layer)
            double deltaHidden[numHiddenNodes];
            for(int j = 0; j < numHiddenNodes; j++){
                double error = 0.0f;
                for(int k = 0; k < numHiddenNodes2; k++){
                    error += deltaHidden2[k] * hiddenWeights2[j][k];
                }
                deltaHidden[j] = error * dRelu(hiddenLayer2[j]);
            }

            // Apply change in output weights
            for(int j = 0; j < numOutputs; j++){
                outputLayerBias[j] += deltaOutput[j] * lr;
                for(int k = 0; k < numHiddenNodes2; k++){
                    outputWeights[k][j] += hiddenLayer2[k] * deltaOutput[j] * lr;
                }
            }

            // Apply change in second hidden layer weights
            for(int j = 0; j < numHiddenNodes2; j++){
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for(int k = 0; k < numHiddenNodes; k++){
                    hiddenWeights2[k][j] += hiddenLayer[k] * deltaHidden2[j] * lr;
                }
            }

            // Apply change in first hidden layer weights
            for(int j = 0; j < numHiddenNodes; j++){
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for(int k = 0; k < numInputs; k++){
                    hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

    end = clock();

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

    // Building neural network with the trained weights and bias
    // initialize testInput and testResults
    double testInput[numTest];
    double testResults[numTest];

    // looping through the matrix and sending in one vector at a time to evaluate
    for(int i = 0; i < numTest; i++) {
        for (int j = 0; j < numInputs; j++) {
            testInput[j] = testingInputs[i][j];
        }
        // predicted solution
        testResults[i] = evaluation(numInputs, numHiddenNodes, numHiddenNodes2, numOutputs,
                                    testInput,hiddenWeights,hiddenWeights2,outputWeights,hiddenLayerBias,hiddenLayerBias2,outputLayerBias);
        printf("predicted results: %f   actual result: %f \n", testResults[i], testingOutputs[i]);
    }

    accuracy(testResults,testingOutputs,numTest);             // accuracy, precision, fscore

    // calculate total time in ms
    double duration = ((double)end - start)/CLOCKS_PER_SEC;

    printf("Total time: %fs \n", duration);  // time

    for (int i = 0; i < numInputs; i++) {
        free(hiddenWeights[i]);
        free(hiddenWeights2[i]);
    }

    for (int i = 0; i < numHiddenNodes2; i++) {
        free(outputWeights[i]);
    }

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

    for (int i = 0; i < numTest; i++) {
        free(testingInputs[i]);
    }
    free(testingInputs);

    for (int i = 0; i < numTrain; i++) {
        free(trainingOutputs[i]);
    }
    free(trainingOutputs);

    free(testingOutputs);
    return 0;

}
