# Neurotic: A Neural Network in C

A very basic neural network in C. It acts as a POC as well as to showcase how simple the base concept is.

# Features

- Activation functions:
  - Sigmoid
  - ReLU
- Cost functions:
  - Binary Cross Entropy
  - Mean Squared
- Minimal allocation
- Row-major access all-over
- Embedded friendly

# Build

You'll need CMake to build it. Easiest way is to install CMake extension from the Visual Studio Code marketplace.

```shell
cmake -S . -B build
cd build
make neurotic
./neurotic
```

# Usage

```C
Activation sigmoid;
Cost bin_cross_entropy;
DenseLayer layers[3];
NeuralNetwork nn;
```

## Initialize the activation and cost functions

You'll get multiple variants of initializers for each based on the types available.

```C
activation_init_sigmoid(&sigmoid);
cost_init_bin_cross_entropy(&bin_cross_entropy);
```

## Initialize the dense layers

Here you can mention the neuron size and the activation function to be used.

```C
dense_layer_init(&layers[0], 4, &sigmoid);
dense_layer_init(&layers[1], 2, &sigmoid);
dense_layer_init(&layers[2], 1, &sigmoid);
```

## Initialize the neural network with the dense layers

You must pass the array of dense layers, the learning rate and also the cost function used for backpropagation.

```C
neural_network_init(&nn, 3, layers, learning_rate, &bin_cross_entropy);
```

## Train the neural network with samples

The input and output vector size are implicitly assumed to be same.

```C
neural_network_train(&nn, &input_data, &output_data);
```

## Use your neural network to make awesome predictions

This is the actual final use of the neural network.

```C
neural_network_predict(&nn, &input_data);
const DenseLayer *output = neural_network_last_layer(&nn);
double prediction = output->neurons[0].value;

printf("Predicted value: %d\n", prediction);
```

## Free the precious memory

Always clean-up before you leave. That's good manners!

```C
neural_network_deinit(&nn);

dense_layer_deinit(&layers[0]);
dense_layer_deinit(&layers[1]);
dense_layer_deinit(&layers[2]);
```

Remember to check the `main.c` file for a more real-life example of using this library.

# TODO

## Essentials

- [ ] Ability to import any CSV and re-scale/re-map column(s).
- [ ] R/W of weights & biases from/to file(s) respectively.

## Good-to-be

- [ ] Print the entire network horizontally

# LICENSE

```
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
```
