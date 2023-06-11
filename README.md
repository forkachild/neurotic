# Neurotic: A Neural Network in C

A very basic but working neural network. Only `sigmoid` activation and `binary cross entropy` cost available for now. This is a highly experimental project and the APIs will definitely change in the future.

# Build

You'll need CMake to build it. Easiest way is to install CMake extension from the Visual Studio Code marketplace.

```shell
cmake -S . -B build
cd build
make nn
./nn
```

# Usage

Check out the example in `main.c`
