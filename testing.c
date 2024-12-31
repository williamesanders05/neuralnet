#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <time.h>

int main() {
	srand(time(NULL));
	int rows = 2;
	int cols = 2;
	double** arr1;
	initialize_to_zero(rows, cols, &arr1);
	arr1[0][0] = 1.0;
	arr1[1][1] = 1.0;
	double** arr2;
	initialize_to_zero(rows, cols, &arr2);
	arr2[0][1] = 1.0;
	arr2[1][0] = 1.0;
	double** arr3;
	double** arr4;
	initialize_random(rows, cols, &arr4);
	initialize_to_zero(rows, cols, &arr3);
	multiply(rows, cols, arr1, rows, cols, arr2, arr3);
	toString(rows, cols, arr3);
	toString(rows, cols, arr4);
	double** x;
	initialize_to_zero(1, 1, &x);
	mlp *layer = initialize_mlp(2, 1, x);
	printf("first value of w: %f \n", layer->W[0][0]);
	printf("first value of b: %f \n", layer->b[0][0]);
	clean_mlp(2, 1, layer);
	for (int i = 0; i < rows; i++) {
		free(arr1[i]);
		free(arr2[i]);
		free(arr3[i]);
		free(arr4[i]);
	}
	free(arr1);
	free(arr2);
	free(arr3);
	free(arr4);
	return 0;
}
