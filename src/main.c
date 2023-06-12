#include "nn.h"

#include "training.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define LAYER_COUNT 3

#define TRAINING_SET_FRACTION 0.6
#define LEARNING_RATE 0.6

int main() {
    Activation sigmoid, relu;
    Cost bin_cross_entropy, mean_squared;
    TrainingSet training_set;
    DenseLayer layers[LAYER_COUNT];
    NeuralNetwork nn;

    // Initialize the activation functions
    activation_init_sigmoid(&sigmoid);
    activation_init_relu(&relu);

    // Initialize the cost functions
    cost_init_bin_cross_entropy(&bin_cross_entropy);
    cost_init_mean_squared(&mean_squared);

    // Read some CSV rows from file
    training_set_init_from_csv(&training_set,
                               "../data_banknote_authentication.txt");

    // Calculate the input and output vector size
    const int input_size = training_set_cols(&training_set) - 1;
    const int output_size = 1;

    // Initialize the dense layers with the correct sizes
    dense_layer_init(&layers[0], input_size, &sigmoid);
    dense_layer_init(&layers[1], input_size / 2, &sigmoid);
    dense_layer_init(&layers[2], output_size, &sigmoid);

    // Initialize the neural network with the dense layers
    neural_network_init(&nn, LAYER_COUNT, layers, LEARNING_RATE,
                        &bin_cross_entropy);

    // Divide samples between training and test
    // TRAINING_SET_FRACTION determines what part of the dataset is used for
    // training
    int training_samples_count =
        (int)(training_set.rows * TRAINING_SET_FRACTION);
    int test_samples_count = training_set.rows - training_samples_count;

    // Train the neural network with the training samples
    for (int i = 0; i < training_samples_count; i++) {
        // First go into the corresponding training row start
        int offset = i * training_set.cols;

        // Train the neural network with input and output vectors
        // Each row of set is [I0, ..., IN, O0]
        //                     ^_________^  ^
        //                        Input     Output
        neural_network_train(&nn, &training_set.data[offset],
                             &training_set.data[offset + input_size]);
    }

    // An accumulation of the total error
    double error_sum = 0.0;

    for (int i = 0; i < test_samples_count; i++) {
        const int offset = (i + training_samples_count) * training_set.cols;
        const DenseLayer *output = neural_network_last_layer(&nn);

        neural_network_predict(&nn, &training_set.data[offset]);

        // Safely extract the value
        const double prediction = output->neurons[0].value;
        const double expected = training_set.data[offset + input_size];

        // Calculate error
        double error = mean_squared.fn(prediction, expected);

        // Keep adding the errors
        error_sum += error;
    }

    // Lastly find the overall accuracy based on the total errors
    double accuracy = 1.0 - error_sum / test_samples_count;

    printf("Average accuracy: %.02f%%\n", accuracy * 100);

    // Time to bid goodbye to the memory we hogged
    neural_network_deinit(&nn);

    for (int i = 0; i < LAYER_COUNT; i++)
        dense_layer_deinit(&layers[i]);

    training_set_deinit(&training_set);

    cost_deinit(&mean_squared);
    cost_deinit(&bin_cross_entropy);

    activation_deinit(&relu);
    activation_deinit(&sigmoid);

    // All's well that ends well
    return EXIT_SUCCESS;
}