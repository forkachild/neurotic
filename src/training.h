#pragma once

typedef struct TrainingSet {
    int cols;
    int rows;
    double *data;
} TrainingSet;

void training_set_init_from_csv(TrainingSet *set, const char *filename);
int training_set_rows(const TrainingSet *set);
int training_set_cols(const TrainingSet *set);
void training_set_deinit(TrainingSet *set);
