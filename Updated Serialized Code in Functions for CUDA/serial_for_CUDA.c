// Neural Network serial code 
// Thomas, Zack, Anu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "evaluation.h"


//#define learningRate 0.0001f                    // defining a constant for learning rate
//#define numberOfEpochs 1500                            // number of epochs


//#define numInputs 30               // number of columns
//#define numHiddenNodes 30          // number of nodes in the first hidden layer
//#define numHiddenNodes2 30         // number of nodes in the second hidden layer
#define numOutputs 1               // number of outputs
#define numTrain 455
#define numTest 114
#define numTrainingSets 569        // number of instances of total data

#define srand_num 44


int numInputs;
int numHiddenNodes;
int numHiddenNodes2;

// learning rate
double learningRate;
double lr;
int numberOfEpochs;


// initializing hidden and output layer nodes vectors
double * hiddenLayer;
double * hiddenLayer2;
double * outputLayer;

// initializing hidden and output layer bias vectors
double * hiddenLayerBias;
double * hiddenLayerBias2;
double * outputLayerBias;

// initializing hidden and output weights matrices 
double ** hiddenWeights;
double ** hiddenWeights2;
double ** outputWeights;

//intitialize training and testing matricies
double ** trainingInputs;
double ** testingInputs;
double trainingOutputs[numTrain][numOutputs];
double testingOutputs[numTest];


double dSigmoid(double x);
double dRelu(double x);
double initWeights();
void initialize_bias(double * hidden, int num);
void initialize_weights(int num1, int num2, double (*hidden)[num2]);
void shuffle(int *array, size_t n);
//void train_and_test_allocate_memory();
void forward_propegation_hidden_layer1(int i, int num1, int num2, double * hidden_layer, double * bias, double (*weights)[num1], double (*inputs)[num2]);
void forward_propegation_hidden_layer2(int i, int num1, int num2, double * hidden_layer, double * bias, double (*weights)[num1], double *inputs);
void forward_propegation_output_layer(int i, int num1, int num2, double * output_layer, double * bias, double (*weights)[num1], double *last_layer);
void forward_propegation(int i);
void backward_propegation(int i);


int main(int argc, char *argv[]) {

    if (argc != 4){
        printf("Please provide number of inputs as your first argument, learning rate as your second argument and number of epochs as your third argument. \n");
        exit(1);
    }

    numInputs = atoi(argv[1]);             // Number of columns
    numHiddenNodes = atoi(argv[1]);        // Number of nodes in the first hidden layer
    numHiddenNodes2 = atoi(argv[1]);       // Number of nodes in the second hidden layer
    learningRate = atof(argv[2]);          // Constant for learning rate
    numberOfEpochs = atoi(argv[3]);        // Number of epochs

    char training_set[30];
    char testing_set[30];
    int characters;

    if (numInputs == 30){
        strncpy(training_set, "train_data.csv", sizeof(training_set));
        strncpy(testing_set, "test_data.csv", sizeof(testing_set));
        characters = 1024;
    }
    else if (numInputs == 4800){
        strncpy(training_set, "train_data4801.csv", sizeof(training_set));
        strncpy(testing_set, "test_data4801.csv", sizeof(testing_set));
        characters = 1602400;
    }
    else{
        printf("Please make sure your first argument is either 30 or 4800. \n");
        exit(2);
    }

    // Learning rate
    lr = learningRate;


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

    double **testingInputs = (double **)malloc(numTest * sizeof(double *));
    for (int i = 0; i < numTest; i++) {
        testingInputs[i] = (double *)malloc(numInputs * sizeof(double));
    }

    double **trainingOutputs = (double **)malloc(numTrain * sizeof(double *));
    for (int i = 0; i < numTrain; i++) {
        trainingOutputs[i] = (double *)malloc(numOutputs * sizeof(double));
    }

    double *testingOutputs = (double *)malloc(numTest * sizeof(double));



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


    // initialize bias and weight terms to random
    // hidden layer 1 bias
    initialize_bias(hiddenLayerBias, numHiddenNodes);

    // hidden layer 2 bias
    initialize_bias(hiddenLayerBias2, numHiddenNodes2);

    // output layer bias
    initialize_bias(outputLayerBias, numOutputs);

    // hidden layer 1 weights
    initialize_weights(numInputs, numHiddenNodes, hiddenWeights);
    
    // hidden layer 2 weights
    initialize_weights(numHiddenNodes, numHiddenNodes2, hiddenWeights2);

    // output layer weights
    initialize_weights(numHiddenNodes2, numOutputs, outputWeights);


    // specify training set
    int trainingSetOrder[numTrain];
    for(int i = 0 ; i < numTrain ; i++)
    {
        trainingSetOrder[i] = i;
    }

    
    // Start time measurement
    clock_t start, end;
    start = clock();

    //training loop
    for(int epoch = 0; epoch < numberOfEpochs; epoch++){

        shuffle(trainingSetOrder, numTrain);

        for(int x = 0; x < numTrain; x ++){
            int i = trainingSetOrder[x];

            // forward pass
            // compute hidden layer activation

            forward_propegation_hidden_layer1(i, numHiddenNodes, numInputs, hiddenLayer, hiddenLayerBias, hiddenWeights, trainingInputs);
            forward_propegation_hidden_layer2(i, numHiddenNodes2, numHiddenNodes, hiddenLayer2, hiddenLayerBias2, hiddenWeights2, hiddenLayer);
            forward_propegation_output_layer(i, numOutputs, numHiddenNodes2, outputLayer, outputLayerBias, outputWeights, hiddenLayer2);

            // Print training output
            printf("Input: %g | %g | %g | %g | %g | %g |      Output: %g      Expected Output: %g \n",
                   trainingInputs[i][1], trainingInputs[i][2], trainingInputs[i][3], trainingInputs[i][4], trainingInputs[i][5], trainingInputs[i][6],
                   outputLayer[0], trainingOutputs[i][0]);
            
           backward_propegation(i);

        }

    }


    end = clock();


    
    // // print final weights after done training
    // fputs ("\nFinal Hidden Weights\n[ ", stdout);
    // for(int j = 0; j < numHiddenNodes; j++){
    //     fputs ("[ ", stdout);
    //     for(int k = 0; k < numInputs; k++){
    //         printf("%f ", hiddenWeights[k][j]);
    //     }
    //     fputs("] ", stdout);
    // }
    // fputs ("\nFinal Hidden2 Weights\n[ ", stdout);
    // for(int j = 0; j < numHiddenNodes2; j++){
    //     fputs ("[ ", stdout);
    //     for(int k = 0; k < numHiddenNodes; k++){
    //         printf("%f ", hiddenWeights2[k][j]);
    //     }
    //     fputs("] ", stdout);
    // }

    // fputs ("]\nFinal Hidden Biases\n[ ", stdout);
    // for(int j = 0; j < numHiddenNodes; j++){
    //     printf("%f ", hiddenLayerBias[j]);
    // }

    // fputs ("]\nFinal Hidden2 Biases\n[ ", stdout);
    // for(int j = 0; j < numHiddenNodes2; j++){
    //     printf("%f ", hiddenLayerBias2[j]);
    // }

    // fputs ("]\nFinal Output Weights\n", stdout);
    // for(int j = 0; j < numOutputs; j++){
    //     fputs ("[ ", stdout);
    //     for(int k = 0; k < numHiddenNodes2; k++){
    //         printf("%f ", outputWeights[k][j]);
    //     }
    //     fputs("] \n", stdout);
    // }

    // fputs ("\nFinal Output Biases\n[ ", stdout);
    // for(int j = 0; j < numOutputs; j++){
    //     printf("%f ", outputLayerBias[j]);
    // }

    // fputs("] \n", stdout);


    // Building neural network with the trained weights and bias 
    // initialize testInput and testResults 
    double testInput[numTest];
    double testResults[numTest];

    // looping through the matrix and sending in one vector at a time to evaluate
    for(int i = 0; i < numTest; i++)
    {
        for(int j = 0; j < numInputs; j++) {
            testInput[j] = testingInputs[i][j];
        }
        // predicted solution
        testResults[i] = evaluation(numInputs, numHiddenNodes, numHiddenNodes2, numOutputs,
                   testInput,hiddenWeights,hiddenWeights2,outputWeights,hiddenLayerBias,hiddenLayerBias2,outputLayerBias);
        printf("predicted results: %f actual result: %f \n", testResults[i], testingOutputs[i]);
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

//double sigmoid(double x) { return 1/(1+exp(-x)); } //forward propagation
double dSigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));      // derivative of sigmoid for backpropagation
}

//double relu(double x) { return MAX(x,0) ;}
double dRelu(double x) {
    if(x<0)
    {
        return 0;                              // derivative of ReLU for backpropagation
    }
    else
    {
        return 1;
    }
}

double initWeights() {
    return ((double)rand()) / ((double)RAND_MAX);  // function to initialize weights
}

// random shuffle data
void shuffle(int *array, size_t n){
    // Initializes random number generator
    srand(srand_num);

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

// void train_and_test_allocate_memory(){
//     trainingInputs = malloc(numTrain * sizeof(trainingInputs[0]));
//     for(int i = 0; i < numTrain; i++)
//     {
//         trainingInputs[i] = malloc(numInputs * sizeof(trainingInputs[0][0]));
//     }

//     testingInputs =  malloc(numTest* sizeof(testingInputs[0]));
//     for(int i = 0; i < numTest; i++)
//     {
//         testingInputs[i] = malloc(numInputs * sizeof(testingInputs[0][0]));
//     }

//     trainingOutputs = malloc(numTrain * sizeof(trainingOutputs[0]));
//     for(int i = 0; i < numTrain; i++)
//     {
//         trainingOutputs[i] = malloc(numOutputs * sizeof(trainingOutputs[0][0]));
//     }

//     testingOutputs = malloc(numTest * sizeof(testingOutputs[0]));

// }

void initialize_bias(double * hidden, int num){
    for(int i = 0; i < num; i++){
        hidden[i] = initWeights();
    }
}

void initialize_weights(int num1, int num2, double (*hidden)[num2]){
    for(int i = 0; i < num1; i++){
        for(int j = 0; j < num2; j++) hidden[i][j] = initWeights();
    }
}

void forward_propegation_hidden_layer1(int i, int num1, int num2, double * hidden_layer, double * bias, double (*weights)[num1], double (*inputs)[num2]){
    for(int j = 0; j < num1; j++){
        double activation = bias[j];

        for(int k = 0; k < num2; k++){
            activation += inputs[i][k] * weights[k][j];
        }
        hidden_layer[j] = relu(activation);
    }
}

void forward_propegation_hidden_layer2(int i, int num1, int num2, double * hidden_layer, double * bias, double (*weights)[num1], double *inputs){
    for(int j = 0; j < num1; j++){
        double activation = bias[j];

        for(int k = 0; k < num2; k++){
            activation += inputs[k] * weights[k][j];
        }
        hidden_layer[j] = relu(activation);
    }
}

void forward_propegation_output_layer(int i, int num1, int num2, double * output_layer, double * bias, double (*weights)[num1], double *last_layer){
    for(int j = 0; j < num1; j++){
        double activation = bias[j];

        for(int k = 0; k < num2; k++){
            activation += last_layer[k] * weights[k][j];
        }
        output_layer[j] = sigmoid(activation);
    }
}


// void backward_propegation_output_layer(int i, int num1, double (*training_output)[num1], double * output_layer){

//     // Compute change in output weights
//     double deltaOutput[num1];
//     double error;
//     for(int j = 0; j < num1; j++){
//         error = (training_output[i][j] - output_layer[j]);
//         deltaOutput[j] = error * dSigmoid(output_layer[j]) ;
//     }

// }


void backward_propegation(int i){

    // Backpropagation
    // Compute change in output weights
    double deltaOutput[numOutputs];
    for(int j = 0; j < numOutputs; j++){
        double error = (trainingOutputs[i][j] - outputLayer[j]);
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


void forward_propegation(int i){
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
            activation += trainingInputs[i][k] * hiddenWeights2[k][j];
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

}