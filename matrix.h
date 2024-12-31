#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to multiply two matrices A[][] and B[][] returning matrix C[][]
void multiply(int row1, int col1, double** A, int row2, int col2, double** B, double** C) {
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            double sum = 0;
            for (int k = 0; k < col1; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Function to transpose matrix A[][] returning matrix C
void transpose(int row1, int col1, double** A, double** C) {
    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row1; j++) {
            C[i][j] = A[j][i];
        }
    }
}

// function to add two matrices together
void add(int row1, int col1, double** A, double** B, double** C) {
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col1; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// Debugging function for printing out arrays
void toString(int row1, int col1, double** A) {
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col1; j++) {
            printf("%f, ", A[i][j]);
        }
        printf("\n");
    }
}

// function for initializing a matrix or vector
void initialize_random(int rows, int cols, double*** matrix) {
	*matrix = (double**)malloc(rows * sizeof(double*));
	for (int i = 0; i < rows; i++) {
		(*matrix)[i] = (double*)malloc(cols * sizeof(double));
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			(*matrix)[i][j] = ((double)rand() / RAND_MAX); // random value in [0.0, 1.0]
		}
	}
}

// function for initializing a matrix to 0
void initialize_to_zero(int rows, int cols, double*** matrix) {
	*matrix = (double**)calloc(rows, sizeof(double*));
	for (int i = 0; i < rows; i++) {
		(*matrix)[i] = (double*)calloc(cols, sizeof(double));
	}
}

// Struct for a single mlp
typedef struct mlp {
	// number of input features
	int n;
	// number of output neurons
	int m;
	// m x n matrix
	// must remember to initialize_random this matrix
	double** W;
	// m x 1 vector
	// must remember to initialize_random this vector
	double** b;
	// n x 1 vector
	// either the input of features or the output of last layer
	double** x;
} mlp;

// function for forwarding through the mlp
// W is an m x n matrix
// x is an n x 1 vector
// b is an m x 1 vector
// m is the number of output neurons
// n is the number of input features
void forward(int n, int m, double** W, double** x, double** b, double** output) {
	// output is a m x 1 vector where m is the number of output neurons
	initialize_to_zero(m, 1, &output);
	// multiplied is a m x 1 vector where m is the number of output neurons
	// multiplied is Wx
	double** multiplied;
	initialize_to_zero(m, 1, &multiplied);
	multiply(m, n, W, n, 1, x, multiplied);
	add(m, 1, multiplied, b, output);
	for (int i = 0; i < m; i++) {
		free(multiplied[i]);
	}
	free(multiplied);
}

// function for initializing an mlp layer
mlp* initialize_mlp(int n, int m, double** x) {
	mlp* layer = malloc(sizeof(mlp));
	layer->n = n;
	layer->m = m;

	//set W
	initialize_random(m, n, &layer->W);
	// set b
	initialize_random(m, 1, &layer->b);

	layer->x = x;
	return layer;
}

void clean_mlp(int n, int m, mlp* layer) {
	for (int i = 0; i < m; i++) {
		free(layer->W[i]);
		free(layer->b[i]);
	}
	for (int i = 0; i < n; i++) {
		free(layer->x[i]);
	}
	free(layer->W);
	free(layer->b);
	free(layer->x);
	free(layer);
}
