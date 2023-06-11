cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_STANDARD 17)

project(neurotic C)

add_executable(neurotic
    src/main.c
    src/nn.c
    src/training.c
)

target_include_directories(neurotic PRIVATE src)