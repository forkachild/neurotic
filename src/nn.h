#pragma once

typedef double FnActivation(double input);
typedef double FnCost(double predicted, double expected);

typedef struct Activation {
    FnActivation *fn;
    FnActivation *d_fn;
} Activation;

void activation_init_sigmoid(Activation *activation);
void activation_deinit(Activation *activation);

typedef struct Cost {
    FnCost *fn;
    FnCost *d_fn;
} Cost;

void cost_init_bin_cross_entropy(Cost *cost);
void cost_deinit(Cost *cost);

typedef struct {
    double bias;
    double value;
    double loss_gradient;
} Neuron;

void neuron_init(Neuron *neuron);
void neuron_deinit(Neuron *neuron);

typedef struct DenseLayer {
    int count;
    Neuron *neurons;
    const Activation *activation;
} DenseLayer;

void dense_layer_init(DenseLayer *layer, int count, Activation *activation);
void dense_layer_fill_values(DenseLayer *layer, const double *input);
void dense_layer_fill_loss_gradients(DenseLayer *layer, Cost *cost,
                                     const double *expected);
void dense_layer_print(const DenseLayer *layer);
void dense_layer_deinit(DenseLayer *layer);

typedef struct LayerJoin {
    int count;
    double *weights;
} LayerJoin;

void layer_join_init(LayerJoin *join, int count);
void layer_join_forward(LayerJoin *join, DenseLayer *input, DenseLayer *output);
void layer_join_backward(LayerJoin *join, DenseLayer *input, DenseLayer *output,
                         double learning_rate);
void layer_join_deinit(LayerJoin *join);

// Define type NeuralNetwork
typedef struct NeuralNetwork {
    int count;
    DenseLayer *layers;
    LayerJoin *joins;
    double learning_rate;
    Cost cost;
} NeuralNetwork;

void neural_network_init(NeuralNetwork *nn, int count, int *layer_counts,
                         double learning_rate, Cost cost);
const DenseLayer *neural_network_first_layer(const NeuralNetwork *nn);
const DenseLayer *neural_network_last_layer(const NeuralNetwork *nn);
void neural_network_train(NeuralNetwork *nn, const double *input,
                          const double *expected);
void neural_network_predict(NeuralNetwork *nn, const double *input);
void neural_network_deinit(NeuralNetwork *nn);