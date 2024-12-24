#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

// Function to multiply two matrices A[][] and B[][] returning matrix C[][]
void multiply(int row1, int col1, double A[][col1], int row2, int col2, double B[][col2], double C[][col2]) {
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
void transpose(int row1, int col1, double A[][col1], double C[][row1]) {
    for (int i = 0; i < col1; i++) {
        for (int j = 0; j < row1; j++) {
            C[i][j] = A[j][i];
        }
    }
}

// function to add two matrices together
void add(int row1, int col1, double A[][col1], double B[][col1], double C[][col1]) {
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col1; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// Debugging function for printing out arrays
void toString(int row1, int col1, int A[][col1]) {
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col1; j++) {
            printf("%d", A[i][j]);
        }
        printf("\n");
    }
}
