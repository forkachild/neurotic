#include "nn.h"

#include "training.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// These are hard coded for the "data_banknote_authentication.txt" file
#define LAYER_COUNT 3
#define INPUT_FEATURES 4
#define HIDDEN_LAYER 2
#define OUTPUTS 1

#define TRAINING_SET_FRACTION 0.6
#define LEARNING_RATE 0.6

static int LAYER_SIZES[LAYER_COUNT] = {
    INPUT_FEATURES,
    HIDDEN_LAYER,
    OUTPUTS,
};

// Mean squared error
static double error_fn(double expected, double predicted) {
    return (expected - predicted) * (expected - predicted);
}

int main() {
    Activation activation;
    Cost cost;
    NeuralNetwork nn;
    TrainingSet set;

    activation_init_sigmoid(&activation);
    cost_init_bin_cross_entropy(&cost);

    neural_network_init(&nn, LAYER_COUNT, LAYER_SIZES, LEARNING_RATE,
                        activation, cost);
    training_set_init_from_csv(&set, "../data_banknote_authentication.txt");

    int training_samples_count = (int)(set.rows * TRAINING_SET_FRACTION);
    int test_samples_count = set.rows - training_samples_count;

    // Can you train it? Yes you can!
    for (int i = 0; i < training_samples_count; i++) {
        int offset = i * set.cols;
        neural_network_train(&nn, &set.data[offset], &set.data[offset + 4]);
    }

    double accuracy_sum = 0.0;

    for (int i = 0; i < test_samples_count; i++) {
        int offset = (i + training_samples_count) * set.cols;
        const DenseLayer *output = neural_network_last_layer(&nn);

        neural_network_predict(&nn, &set.data[offset]);

        // Safely extract the value
        double prediction = output->neurons[0].value;
        double expected = set.data[offset + 4];
        double error = error_fn(expected, prediction);
        double accuracy = 1.0 - error;

        accuracy_sum += accuracy;
    }

    double accuracy = accuracy_sum / test_samples_count;

    printf("Average accuracy: %.02f%%\n", accuracy * 100);

    training_set_deinit(&set);
    neural_network_deinit(&nn);

    return EXIT_SUCCESS;
}