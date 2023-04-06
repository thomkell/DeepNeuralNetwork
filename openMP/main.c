// Neural Network serial code
// Anu, Thomas, Zack

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define MAX(a,b) (((a)>(b))?(a):(b))           // macro to find maximum of two numbers

#define learningRate 0.0001f                   // defining a constant for learning rate

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// relu activation function
double relu(double x) { return MAX(x,0) ;}

double evaluation(int numInputs, int numHiddenNodes, int numHiddenNodes2, int numOutputs, double inputs[numInputs], double hiddenWeights[numInputs][numHiddenNodes], double hiddenWeights2[numHiddenNodes][numHiddenNodes2], double outputWeights[numHiddenNodes2][numOutputs], double hiddenLayerBias[numHiddenNodes], double hiddenLayerBias2[numHiddenNodes2], double outputLayerBias[numOutputs]){

    // Compute output of first hidden layer
    double hiddenLayer[numHiddenNodes];
    for(int j = 0; j < numHiddenNodes; j++){
        double activation = hiddenLayerBias[j];
        for(int k = 0; k < numInputs; k++){
            activation += inputs[k] * hiddenWeights[k][j];
        }
        hiddenLayer[j] = relu(activation);
    }

    // Compute output of second hidden layer
    double hiddenLayer2[numHiddenNodes2];
    for(int j = 0; j < numHiddenNodes2; j++){
        double activation = hiddenLayerBias2[j];
        for(int k = 0; k < numHiddenNodes; k++){
            activation += hiddenLayer[k] * hiddenWeights2[k][j];
        }
        hiddenLayer2[j] = relu(activation);
    }

    // Compute output of output layer
    double outputs[numOutputs];
    for(int j = 0; j < numOutputs; j++){
        double activation = outputLayerBias[j];
        for(int k = 0; k < numHiddenNodes2; k++){
            activation += hiddenLayer2[k] * outputWeights[k][j];
        }
        outputs[j] = sigmoid(activation);
        if (outputs[j] <= 0.5){
            outputs[j] = 0;
        }
        else{
            outputs[j] = 1;
        }
    }

    return outputs[0];
}


// checking accuracy
double accuracy(double predicted_labels[], double true_labels[], int num_samples){
        // Initialize confusion matrix
        int confusion_matrix[2][2] = {{0, 0}, {0, 0}};

        // Populate confusion matrix
        for (int i = 0; i < num_samples; i++) {
            int predicted = (int)predicted_labels[i];
            int true_label = (int)true_labels[i];
            confusion_matrix[true_label][predicted]++;
        }

        // Calculate evaluation metrics
        int true_positives = confusion_matrix[1][1];
        int false_positives = confusion_matrix[0][1];
        int false_negatives = confusion_matrix[1][0];
        int true_negatives = confusion_matrix[0][0];

        printf("True positives: %i\n", true_positives);
        printf("false positives: %i\n", false_positives);
        printf("false negatives: %i\n", false_negatives);
        printf("True negatives: %i\n", true_negatives);


        double accuracy = (double)(true_positives + true_negatives) / num_samples;
        double precision = (double)true_positives / (true_positives + false_positives);
        double recall = (double)true_positives / (true_positives + false_negatives);
        double f1_score = 2 * ((precision * recall) / (precision + recall));

        // Print evaluation metrics
        printf("Accuracy: %.2f%%\n", accuracy * 100);
        printf("Precision: %.2f\n", precision);
        printf("Recall: %.2f\n", recall);
        printf("F1 Score: %.2f\n", f1_score);

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

#define numInputs 30               // number of columns
#define numHiddenNodes 30          // number of nodes in the first hidden layer
#define numHiddenNodes2 30         // number of nodes in the second hidden layer
#define numOutputs 1               // number of outputs
#define numTrain 455
#define numTest 114
#define numTrainingSets 569        // number of instances of total data

int main() {

    // learning rate
    const double lr = learningRate;
    int thread_count = 1;

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


    // read data from inputTrain.csv
    char buffer[1024];
    char buffer2[1024];
    char *record, *line;
    char *record2, *line2;
    int i = 0, j = 0;
    double inputTrain[numTrain][31];
    double inputTest[numTrain][31];

    // read input data from a csv
    FILE *fstream = fopen("train_data.csv", "r");
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
        if (j == 30)
            i += 1;
    }

    fclose(fstream);

    i = 0, j = 0;

    // read data from inputTrain.csv
    // read input data from a csv

    FILE *gstream = fopen("test_data.csv", "r");
    if (gstream == NULL) {
        printf("\n file opening failed test ");
        return -1;
    }
    while ((line2 = fgets(buffer2, sizeof(buffer2), gstream)) != NULL) {
        record2 = strtok(line2, ",");
        //printf("%s ", record2);
        while (record2 != NULL) {
            inputTest[i][j++] = strtod(record2, NULL);
            record2 = strtok(NULL, ",");
        }
        if (j == 30)
            i += 1;
    }

    // training data (inputs)
    double trainingInputs[numTrain][numInputs];

    #pragma omp parallel for num_threads(thread_count) collapse(2) shared(trainingInputs, inputTrain)
    for (int ro=0; ro<numTrain; ro++)
    {
        for(int columns=1; columns<31; columns++)
        {
            trainingInputs[ro][columns-1] = inputTrain[ro][columns];
        }
    }

    // testing data (inputs)
    double testingInputs[numTest][numInputs];

    #pragma omp parallel for num_threads(thread_count) collapse(2) shared(testingInputs, inputTest)
    for (int ro=0; ro<numTest; ro++)
    {
        for(int columns=1; columns<31; columns++)
        {
            //int rowX = newTestingSetOrder[ro];
            testingInputs[ro][columns-1] = inputTest[ro][columns];
            //printf("%f ", testingInputs[ro][columns-1]);

        }
    }

    // training data (outputs)
    double trainingOutputs[numTrain][numOutputs];

    #pragma omp parallel for num_threads(thread_count) collapse(2) shared(trainingOutputs, inputTrain)
    for (int ro=0; ro<numTrain; ro++)
    {
        for(int columns=0; columns<1; columns++)
        {
            //int row = newTrainingSetOrder[ro];
            trainingOutputs[ro][columns] = inputTrain[ro][columns];
        }
    }

    // testing data (outputs)
    double testingOutputs[numTest];

    #pragma omp parallel for num_threads(thread_count) shared(testingOutputs, inputTest)
    for (int ro=0; ro<numTest; ro++)
    {
        for(int columns=0; columns<1; columns++)
        {
            //int row = newTestingSetOrder[ro];
            testingOutputs[ro] = inputTest[ro][columns];
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


    int numberOfEpochs = 1500;                            // number of epochs

    // start time measurement
    struct timeval time1, time2;
    gettimeofday(&time1, NULL);

    //training loop
    for(int epoch = 0; epoch < numberOfEpochs; epoch++){

        shuffle(trainingSetOrder, numTrain);

        for(int x = 0; x < numTrain; x ++){
            int i = trainingSetOrder[x];

            // forward pass
            // compute hidden layer activation

            // hidden layer 1
            int k;
            #pragma omp parallel for num_threads(thread_count) shared(trainingInputs, hiddenWeights, hiddenLayer, hiddenLayerBias) private(j, k)
            for(int j =0; j < numHiddenNodes; j++){
                double activation = hiddenLayerBias[j];

                for(int k = 0; k < numInputs; k++){
                    activation += trainingInputs[i][k] * hiddenWeights[k][j];
                }

                hiddenLayer[j] = relu(activation);
            }

            // hidden layer 2
            #pragma omp parallel for num_threads(thread_count) shared(hiddenLayer, hiddenWeights2, hiddenLayer2, hiddenLayerBias2) private(j, k)
            for(int j =0; j < numHiddenNodes2; j++){
                double activation = hiddenLayerBias2[j];

                for(int k = 0; k < numHiddenNodes; k++){
                    activation += hiddenLayer[k] * hiddenWeights2[k][j];
                }

                hiddenLayer2[j] = relu(activation);
            }

            // compute output layer activation
            #pragma omp parallel for num_threads(thread_count) shared(hiddenLayer2, outputWeights, outputLayer, outputLayerBias) private(j, k)
            for(int j =0; j < numOutputs; j++){
                double activation = outputLayerBias[j];

                for(int k = 0; k < numHiddenNodes2; k++){
                    activation += hiddenLayer2[k] * outputWeights[k][j];
                }

                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: %g | %g | %g | %g | %g | %g |      Output: %g      Expected Output: %g \n",
                   trainingInputs[i][1], trainingInputs[i][2], trainingInputs[i][3], trainingInputs[i][4], trainingInputs[i][5], trainingInputs[i][6],
                   outputLayer[0], trainingOutputs[i][0]);

            // Backpropagation
            // Compute change in output weights
            double deltaOutput[numOutputs];
            #pragma omp parallel for num_threads(thread_count) shared(deltaOutput, trainingOutputs, outputLayer) private(j)
            for(int j = 0; j < numOutputs; j++){
                double error = (trainingOutputs[i][j] - outputLayer[j]); // L1
                //double error = -((outputLayer[j] * log(trainingOutputs[i][j])) + ((1-outputLayer[j]) * log(trainingOutputs[i][j]))); // cross entropy
                //double error = (1/numOutputs)*(pow(trainingOutputs[i][j] - outputLayer[j],2)); // MSE
                deltaOutput[j] = error * dSigmoid(outputLayer[j]) ;
            }

            // Compute change in hidden weights (second layer)
            double deltaHidden2[numHiddenNodes2];
            #pragma omp parallel for num_threads(thread_count) shared(deltaHidden2, deltaOutput, outputWeights, hiddenLayer, hiddenLayer2) private(j, k)
            for(int j = 0; j < numHiddenNodes2; j++){
                double error = 0.0f;
                for(int k = 0; k < numOutputs; k++){
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden2[j] = error * dRelu(hiddenLayer[j]);
            }

            // Compute change in hidden weights (first layer)
            double deltaHidden[numHiddenNodes];
            #pragma omp parallel for num_threads(thread_count) shared(deltaHidden, deltaHidden2, hiddenWeights2, hiddenLayer, hiddenLayerBias2) private(j, k)
            for(int j = 0; j < numHiddenNodes; j++){
                double error = 0.0f;
                for(int k = 0; k < numHiddenNodes2; k++){
                    error += deltaHidden2[k] * hiddenWeights2[j][k];
                }
                deltaHidden[j] = error * dRelu(hiddenLayer2[j]);
            }

            // Apply change in output weights
            #pragma omp parallel for num_threads(thread_count) shared(outputLayerBias, outputWeights, deltaOutput) private(j, k)
            for(int j = 0; j < numOutputs; j++){
                outputLayerBias[j] += deltaOutput[j] * lr;
                for(int k = 0; k < numHiddenNodes2; k++){
                    outputWeights[k][j] += hiddenLayer2[k] * deltaOutput[j] * lr;
                }
            }

            // Apply change in second hidden layer weights
            #pragma omp parallel for num_threads(thread_count) shared(hiddenLayerBias, hiddenWeights2, deltaHidden) private(j, k)
            for(int j = 0; j < numHiddenNodes2; j++){
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for(int k = 0; k < numHiddenNodes; k++){
                    hiddenWeights2[k][j] += hiddenLayer[k] * deltaHidden2[j] * lr;
                }
            }

            // Apply change in first hidden layer weights
            #pragma omp parallel for num_threads(thread_count) shared(hiddenLayerBias, hiddenWeights, deltaHidden) private(j, k)
            for(int j = 0; j < numHiddenNodes; j++){
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for(int k = 0; k < numInputs; k++){
                    hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

    gettimeofday(&time2, NULL);

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


    // Building neural network with the trained weights and bias
    // initialize testInput and testResults
    double testInput[numTest];
    double testResults[numTest];

    // looping through the matrix and sending in one vector at a time to evaluate
    #pragma omp parallel for num_threads(thread_count) shared(testInput, testResults) private(j)
    for(int i = 0; i < numTest; i++) {
        for (int j = 0; j < numInputs; j++) {
            testInput[j] = testingInputs[i][j];
            // printf("%f ", testInput[j]);
        }
        // printf("\n");


        // predicted solution
        testResults[i] = evaluation(numInputs, numHiddenNodes, numHiddenNodes2, numOutputs,
                                    testInput,hiddenWeights,hiddenWeights2,outputWeights,hiddenLayerBias,hiddenLayerBias2,outputLayerBias);
        printf("predicted results: %f   actual result: %f \n", testResults[i], testingOutputs[i]);
    }

    accuracy(testResults,testingOutputs,numTest);             // accuracy, precision, fscore

    double totalTime;
    // calculate total time in ms
    totalTime = (time2.tv_sec - time1.tv_sec);      // s
    totalTime += (time2.tv_usec - time1.tv_usec)/1E6;

    printf("Total time: %fs \n", totalTime);                      // time

    return 0;

}

