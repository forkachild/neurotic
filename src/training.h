#pragma once

typedef struct TrainingSet {
    int cols;
    int rows;
    double *data;
} TrainingSet;

void training_set_init_from_csv(TrainingSet *set, const char *filename);
void training_set_deinit(TrainingSet *set);
