#include "training.h"

#include <stdio.h>
#include <stdlib.h>

void training_set_init_from_csv(TrainingSet *set, const char *filename) {
    FILE *fp;

    fp = fopen(filename, "r");

    if (fp == NULL) {
        printf("Could not read file!\n");
        exit(EXIT_FAILURE);
    }

    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buffer = (char *)malloc(len * sizeof(char));
    fread(buffer, sizeof(char), len, fp);
    fclose(fp);

    int rows = 0, cols = 0;

    for (long i = 0; i < len; i++) {
        if (buffer[i] == '\n')
            rows++;

        if (rows == 0 && buffer[i] == ',')
            cols++;
    }

    if (rows == 0 || cols == 0)
        return;

    rows++;
    cols++;

    double *data = (double *)malloc(rows * cols * sizeof(double));

    int row = 0, col = 0, start = 0;

    for (int i = 0; i < len; i++) {
        char ch = buffer[i];

        if (ch == ',' || ch == '\n') {
            char temp = ch;
            buffer[i] = '\0';
            data[row * cols + col] = atof(&buffer[start]);
            buffer[i] = temp;
            start = i + 1;
            col++;
        }

        if (ch == '\n') {
            row++;
            col = 0;
        }
    }

    set->rows = rows;
    set->cols = cols;
    set->data = data;

    free(buffer);
}

void training_set_deinit(TrainingSet *set) {
    free(set->data);
    set->data = NULL;
}
