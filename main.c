// Neural Network serial code
// Thomas, Zack, Anu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "evaluation.h"
#include <time.h>
#include <sys/time.h>

#define MAX(a,b) (((a)>(b))?(a):(b))           // macro to find maximum of two numbers

#define learningRate 0.001f                    // defining a constant for learning rate

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
    srand(45);

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


#define numInputs 30               // number of columns
#define numHiddenNodes 30          // number of nodes in the first hidden layer
#define numHiddenNodes2 30         // number of nodes in the second hidden layer
#define numOutputs 1               // number of outputs
#define numTrainingSets 569        // number of instances of total data

int main() {

    // learning rate
    const double lr = learningRate;

    // initializing hidden and output layer nodes vectors
    double hiddenLayer[numHiddenNodes];
    double hiddenLayer2[numHiddenNodes2];
    double outputLayer[numOutputs];

    // initializing hidden and output layer bias vectors
    double hiddenLayerBias[numHiddenNodes];
    double hiddenLayerBias2[numHiddenNodes2];
    double outputLayerBias[numOutputs];

    // initializing hidden and output weights matrices
    double hiddenWeights[numInputs][numHiddenNodes];
    double hiddenWeights2[numInputs][numHiddenNodes2];
    double outputWeights[numHiddenNodes2][numOutputs];

    // read data from data.csv
    char buffer[1024] ;
    char *record,*line;
    int i=0,j=0;
    double inputCSV[1000][31];

    // read input data from a csv
    FILE *fstream = fopen("/Users/thomaskeller/CLionProjects/COMP605/dataALL_N1.csv", "r");
    if(fstream == NULL)
    {
        printf("\n file opening failed ");
        return -1 ;
    }
    while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL)
    {
        record = strtok(line,",");
        while(record != NULL)
        {
            inputCSV[i][j++] = strtod(record,NULL) ;
            record = strtok(NULL,",");
        }
        if(j == 30)
            i+=1;
    }

    // split data
    double percentTrain = 0.8f;                     // 80 percent
    int trainingSetOrderSplit[569];

    // creating a vector to shuffle the data
    for(int i = 0 ; i < 569 ; i++)
    {
        trainingSetOrderSplit[i] = i;
    }

    // shuffle complete data
    shuffle(trainingSetOrderSplit, numTrainingSets);

    // splitting into training and test data
    int cutOffTrain = (int) ceil(numTrainingSets * percentTrain);
    int newTrainingSetOrder[cutOffTrain];
    int cutOffTest = numTrainingSets - cutOffTrain;
    int newTestingSetOrder[cutOffTest];

    // finding the indices to populate the order for training
    for (int i = 0; i < cutOffTrain; i++){
        newTrainingSetOrder[i] = trainingSetOrderSplit[i];
    }

    // finding the indices to populate the order for testing
    for (int i = cutOffTrain; i < numTrainingSets; i++){
        newTestingSetOrder[i-cutOffTrain] = trainingSetOrderSplit[i];
    }

    // training data (inputs)
    double trainingInputs[cutOffTrain][numInputs];

    for (int ro=0; ro<cutOffTrain; ro++)
    {
        for(int columns=1; columns<31; columns++)
        {
            int row = newTrainingSetOrder[ro];
            trainingInputs[ro][columns-1] = inputCSV[row][columns];
        }
    }

    // testing data (inputs)
    double testingInputs[cutOffTest][numInputs];

    for (int ro=0; ro<cutOffTest; ro++)
    {
        for(int columns=1; columns<31; columns++)
        {
            int rowX = newTestingSetOrder[ro];
            testingInputs[ro][columns-1] = inputCSV[rowX][columns];
        }
    }

    // training data (outputs)
    double trainingOutputs[cutOffTrain][numOutputs];
    for (int ro=0; ro<cutOffTrain; ro++)
    {
        for(int columns=0; columns<1; columns++)
        {
            int row = newTrainingSetOrder[ro];
            trainingOutputs[ro][columns] = inputCSV[row][columns];
        }
    }

    // testing data (outputs)
    double testingOutputs[cutOffTest];
    for (int ro=0; ro<cutOffTest; ro++)
    {
        for(int columns=0; columns<1; columns++)
        {
            int row = newTestingSetOrder[ro];
            testingOutputs[ro] = inputCSV[row][columns];
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
    int trainingSetOrder[cutOffTrain];
    for(int i = 0 ; i < cutOffTrain ; i++)
    {
        trainingSetOrder[i] = i;
    }

    // shuffling training set
    shuffle(trainingSetOrder, cutOffTrain);
    int numberOfEpochs = 1500;                            // number of epochs

    // start time measurement
    struct timeval time1, time2;
    gettimeofday(&time1, NULL);

    //training loop
    for(int epoch = 0; epoch < numberOfEpochs; epoch++){

        shuffle(trainingSetOrder, cutOffTrain);

        for(int x = 0; x < cutOffTrain; x ++){
            int i = trainingSetOrder[x];

            // forward pass
            // compute hidden layer activation

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
            /*
            printf("Input: %g | %g | %g | %g | %g | %g |      Output: %g      Expected Output: %g \n",
                   trainingInputs[i][1], trainingInputs[i][2], trainingInputs[i][3], trainingInputs[i][4], trainingInputs[i][5], trainingInputs[i][6],
                   outputLayer[0], trainingOutputs[i][0]);
            */
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

    }

    gettimeofday(&time2, NULL);
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
    double testInput[cutOffTest];
    double testResults[cutOffTest];

    // looping through the matrix and sending in one vector at a time to evaluate
    for(int i = 0; i < cutOffTest; i++)
    {
        for(int j = 0; j < numInputs; j++) {
            testInput[j] = testingInputs[i][j];
        }
        // predicted solution
        testResults[i] = evaluation(numInputs, numHiddenNodes, numHiddenNodes2, numOutputs,
                                    testInput,hiddenWeights,hiddenWeights2,outputWeights,hiddenLayerBias,hiddenLayerBias2,outputLayerBias);
        printf("predicted results: %f actual result: %f \n", testResults[i], testingOutputs[i]);
    }

    accuracy(testResults,testingOutputs,cutOffTest);             // accuracy, precision, fscore

    double totalTime;
    // calculate total time in ms
    totalTime = (time2.tv_sec - time1.tv_sec);      // s
    totalTime += (time2.tv_usec - time1.tv_usec)/1E6;

    printf("Total time: %fs \n", totalTime);                      // time

    return 0;
}
