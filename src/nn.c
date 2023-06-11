#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double sigmoid(double input) { return 1.0 / (1.0 + exp(-input)); }

static double d_sigmoid(double input) { return input * (1.0 - input); }

void activation_init_sigmoid(Activation *activation) {
    activation->fn = &sigmoid;
    activation->d_fn = &d_sigmoid;
}

void activation_deinit(Activation *activation) {
    activation->fn = NULL;
    activation->d_fn = NULL;
}

static double binary_cross_entropy(double predicted, double expected) {
    return -(expected * log2(predicted)) -
           ((1.0 - expected) * log2(1.0 - predicted));
}

static double d_binary_cross_entropy(double predicted, double expected) {
    return -(expected / predicted) + ((1.0 - expected) / (1.0 - predicted));
}

void cost_init_bin_cross_entropy(Cost *cost) {
    cost->fn = &binary_cross_entropy;
    cost->d_fn = &d_binary_cross_entropy;
}

void cost_deinit(Cost *cost) {
    cost->fn = NULL;
    cost->d_fn = NULL;
}

void neuron_init(Neuron *neuron) { neuron_deinit(neuron); }

void neuron_deinit(Neuron *neuron) {
    neuron->value = 0.0;
    neuron->bias = 0.0;
    neuron->loss_gradient = 0.0;
}

void dense_layer_init(DenseLayer *layer, int count, Activation activation) {
    Neuron *neurons = (Neuron *)malloc(count * sizeof(Neuron));

    for (int i = 0; i < count; i++)
        neuron_init(&neurons[i]);

    layer->count = count;
    layer->neurons = neurons;
    layer->activation = activation;
}

void dense_layer_fill_values(DenseLayer *layer, const double *input) {
    for (int i = 0; i < layer->count; i++)
        layer->neurons[i].value = input[i];
}

void dense_layer_fill_loss_gradients(DenseLayer *layer, Cost cost,
                                     const double *expected) {
    Neuron *neurons = layer->neurons;

    for (int i = 0; i < layer->count; i++) {
        Neuron *neuron = &neurons[i];

        neuron->loss_gradient = cost.d_fn(neuron->value, expected[i]) *
                                layer->activation.d_fn(neuron->value);
    }
}

void dense_layer_print(const DenseLayer *layer) {
    for (int i = 0; i < layer->count; i++)
        printf("%.02f\t", layer->neurons[i].value);
    printf("\n");
}

void dense_layer_deinit(DenseLayer *layer) {
    if (layer->neurons != NULL) {
        free(layer->neurons);
        layer->neurons = NULL;
    }
}

void layer_join_init(LayerJoin *join, int count) {
    if (count < 1)
        return;

    double *weights = (double *)malloc(count * sizeof(double));

    for (int i = 0; i < count; i++)
        weights[i] = (double)rand() / RAND_MAX;

    join->count = count;
    join->weights = weights;
}

void layer_join_forward(LayerJoin *join, DenseLayer *input,
                        DenseLayer *output) {
    for (int i = 0; i < output->count; i++) {
        // Initialize sum to the bias
        double sum = output->neurons[i].bias;

        // Weighted sum
        for (int j = 0; j < input->count; j++)
            sum +=
                join->weights[i * input->count + j] * input->neurons[j].value;

        // Assign the activated value
        output->neurons[i].value = output->activation.fn(sum);
    }
}

void layer_join_backward(LayerJoin *join, DenseLayer *input, DenseLayer *output,
                         double learning_rate) {
    // Init the loss-gradient to zero
    // This loss gradient will be useful to adjust the bias and connected
    // weights as well as evaluate the error of the previous layer
    for (int j = 0; j < input->count; j++)
        input->neurons[j].loss_gradient = 0.0;

    for (int i = 0; i < output->count; i++) {
        Neuron *output_neuron = &output->neurons[i];

        output_neuron->bias -= learning_rate * output_neuron->loss_gradient;

        for (int j = 0; j < input->count; j++) {
            Neuron *input_neuron = &input->neurons[j];
            double *weight = &join->weights[i * input->count + j];

            // TODO: Swap the following two statements and see
            input_neuron->loss_gradient +=
                output_neuron->loss_gradient * *weight *
                input->activation.d_fn(input_neuron->value);

            // Adjust the weight
            *weight -= learning_rate * output_neuron->loss_gradient *
                       input_neuron->value;
        }
    }

    for (int j = 0; j < input->count; j++)
        input->neurons[j].loss_gradient /= output->count;
}

void layer_join_deinit(LayerJoin *join) {
    free(join->weights);
    join->weights = NULL;
}

void neural_network_init(NeuralNetwork *nn, int count, int *layer_counts,
                         double learning_rate, Activation activation,
                         Cost cost) {
    if (count < 2)
        return;

    DenseLayer *layers = (DenseLayer *)malloc(count * sizeof(DenseLayer));

    for (int i = 0; i < count; i++)
        dense_layer_init(&layers[i], layer_counts[i], activation);

    LayerJoin *joins = (LayerJoin *)malloc((count - 1) * sizeof(LayerJoin));

    for (int i = 0; i < count - 1; i++)
        layer_join_init(&joins[i], layers[i].count * layers[i + 1].count);

    nn->count = count;
    nn->layers = layers;
    nn->joins = joins;
    nn->learning_rate = learning_rate;
    nn->cost = cost;
}

const DenseLayer *neural_network_first_layer(const NeuralNetwork *nn) {
    return &nn->layers[0];
}

const DenseLayer *neural_network_last_layer(const NeuralNetwork *nn) {
    return &nn->layers[nn->count - 1];
}

static void neural_network_forward(NeuralNetwork *nn) {
    for (int l = 0; l <= nn->count - 2; l++)
        layer_join_forward(&nn->joins[l], &nn->layers[l], &nn->layers[l + 1]);
}

static void neural_network_error_backpropagate(NeuralNetwork *nn) {
    for (int l = nn->count - 1; l >= 1; l--)
        layer_join_backward(&nn->joins[l - 1], &nn->layers[l - 1],
                            &nn->layers[l], nn->learning_rate);
}

void neural_network_train(NeuralNetwork *nn, const double *input,
                          const double *expected) {
    neural_network_predict(nn, input);
    dense_layer_fill_loss_gradients(&nn->layers[nn->count - 1], nn->cost,
                                    expected);
    neural_network_error_backpropagate(nn);
}

void neural_network_predict(NeuralNetwork *nn, const double *input) {
    dense_layer_fill_values(&nn->layers[0], input);
    neural_network_forward(nn);
}

void neural_network_deinit(NeuralNetwork *nn) {
    for (int i = 0; i < nn->count; i++)
        dense_layer_deinit(&nn->layers[i]);

    for (int i = 0; i < nn->count - 1; i++)
        layer_join_deinit(&nn->joins[i]);

    free(nn->joins);
    free(nn->layers);
}