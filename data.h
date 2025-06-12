#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_COLS = 4207
#define DATA_SIZE = 784

typedef struct data {
    int num_samples;
    double** data;
    int* targets;
} data;

void clean_data(data *d) {
    for (int i = 0; i < d->num_samples; i++) {
	free(d->data[i]);
    }
    free(d->data);
    free(d->targets);
    free(d);
}

data* initialize_data(char* file_name, int num_rows) {
    data* d = (data*)malloc(sizeof(data));
    d->num_samples = num_rows;
    d->data = (double**)calloc(num_rows, sizeof(double*));
    for (int i = 0; i < num_rows; i++) {
        d->data[i] = (double*)calloc(784, sizeof(double));
    }
    d->targets = (int*)calloc(num_rows, sizeof(int));
    FILE *file = fopen(file_name, "r");
    // handle errors
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    char line[4207];
    int line_num = 0;
    while (fgets(line, 4207, file) != NULL && line_num < num_rows) {
        // Parse Line
        char *token = strtok(line, ",");
        d->targets[line_num] = atoi(token);
        token = strtok(NULL, ",");
        int token_num = 0;
        while (token != NULL) {
            d->data[line_num][token_num] = atof(token);
            token = strtok(NULL, ",");
            token_num++;
        }
        line_num++;
    }
    //free(line);
    fclose(file);
    return d;
}
