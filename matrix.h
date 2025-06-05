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

// Function to multiply a vector and matrix v[a] and M[a][b] returning vector r[b]
void multiply_vector(int row, int col, double* v, double** m, double* r) {
    for (int i = 0; i < col; i++) {
        double sum = 0;
        for (int j = 0 ; j < row; j++) {
            sum += v[j] * m[j][i];
        }
        r[i] = sum;
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

// function to add vector and matrix together
void add(int row1, int col1, double* A, double* B, double* C) {
    for (int i = 0; i < row1; i++) {
        C[i] = A[i] + B[i];        
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

// function for deep copying matrices
void deep_copy(int col, double* mat1, double* mat2) {
    for (int i = 0; i < col; i++) {
        mat2[i] = mat1[i];
    }
}

// function to add two matrices together
void subtract(int row1, int col1, double** A, double** B, double** C) {
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col1; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }      
    }
}

// function to add two matrices together
void subtract_vector(int col1, double* A, double* B, double* C) {
    for (int i = 0; i < col1; i++) {
        C[i] = A[i] - B[i];        
    }
}

void multiply_by_number(int row, int col, double number, double** mat) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            mat[i][j] *= number;
        }
    }
}

void multiply_by_number_vector(int col, double number, double* mat) {
    for (int i = 0; i < col; i++) {
        mat[i] *= number;
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

// function for initializing a matrix or vector
void initialize_random_vector(int cols, double** matrix) {
    *matrix = (double*)malloc(cols * sizeof(double*));
    for (int i = 0; i < cols; i++) {
        (*matrix)[i] = ((double)rand() / RAND_MAX); // random value in [0.0, 1.0]
    }
}

// function for initializing a matrix to 0
void initialize_to_zero_vector(int cols, double** matrix) {
    *matrix = (double*)calloc(cols, sizeof(double*));
}


