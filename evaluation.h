// evaluation for main.c

#include <math.h>

#define MAX(a,b) (((a)>(b))?(a):(b))           // macro to find maximum of two numbers

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