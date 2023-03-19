#include <stdio.h>
#include <math.h>

// max function 
#define MAX(a,b) (((a)>(b))?(a):(b))

// sigmoid activation function
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// relu activation function 
double relu(double x) { return MAX(x,0) ;}

double evaluation(int INPUT_SIZE,int HIDDEN_SIZE,int HIDDEN_SIZE2, int OUTPUT_SIZE, double input[INPUT_SIZE],
                double hiddenWeights[INPUT_SIZE][HIDDEN_SIZE], double hiddenWeights2[HIDDEN_SIZE][HIDDEN_SIZE2],
                double outputWeights[HIDDEN_SIZE2][OUTPUT_SIZE], double hiddenLayerBias[HIDDEN_SIZE], double hiddenLayerBias2[HIDDEN_SIZE2],
                double outputLayerBias[OUTPUT_SIZE]) {
    // creating the neural network with predefined weights and repeating the forward pass
    // allocating space 
    double *hidden = malloc(HIDDEN_SIZE * sizeof(double));
    double *hidden2 = malloc(HIDDEN_SIZE2 * sizeof(double));
    double *output = malloc(OUTPUT_SIZE * sizeof(double));

    // Hidden layer 1
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            hidden[j] += input[i] * hiddenWeights[i][j];
        }
        hidden[j] += hiddenLayerBias[j];
        hidden[j] = relu(hidden[j]);
    }

    // Hidden layer 2
    for (int j = 0; j < HIDDEN_SIZE2; j++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            hidden2[j] += hidden[i] * hiddenWeights2[i][j];
        }
        hidden2[j] += hiddenLayerBias2[j];
        hidden2[j] = relu(hidden2[j]);
    }

    // Output layer
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            output[k] += hidden2[j] * outputWeights[j][k];

        }
        output[k] += outputLayerBias[k];
        output[k] = sigmoid(output[k]);

        if (output[k] <= 0.5){
            output[k] = 0;
        }
        else{
            output[k] = 1;
        }
    }

    free(hidden);
    free(hidden2);
    return output[0];
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