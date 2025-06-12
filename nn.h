#include "matrix.h"
#include <math.h>
#include <stdbool.h>
// Struct for a single mlp
typedef struct mlp {
  // number of input features
  int n;
  // number of output neurons
  int m;
  // m x n matrix
  // must remember to
  // initialize_random this matrix
  double **W;
  // 1 x m row vector
  // must remember to initialize_random this vector
  double *b;
  // 1 x n row vector
  // either the input of features or the output of last layer
  double *x;
  // 1 x m row vector
  // input for activation function
  double *z;
  // 1 x m vector input for next layer or output
  double *a;
  // error for the layer is 1 x m row vector
  double *delta;
} mlp;

// Struct for multiple mlp layers aka neural network
typedef struct nn {
  double learning_rate;
  int num_layers;
  mlp **layers;
  double* output;
} nn;

void RELU(double *input, double *output, int m) {
  // assume output is already initialized
  for (int i = 0; i < m; i++) {
    output[i] = (input[i] > 0) ? input[i] : 0.0;
  }
}

void RELU_derivative(double *input, double *output, int n) {
  // Assume that output is already initialized
  for (int i = 0; i < n; i++) {
    output[i] *= (input[i] > 0) ? 1.0 : 0.0;
  }
}

void softmax(double *input, double *output, int size) {
  double sum = 0.0;
  double max = -INFINITY;

  // Find the max value
  for (int i = 0; i < size; i++) {
    if (input[i] > max) {
      max = input[i];
    }
  }

  // Compute Softmax
  for (int i = 0; i < size; i++) {
    output[i] = exp(input[i] - max);
    sum += output[i];
  }

  // Normalize
  for (int i = 0; i < size; i++) {
    output[i] /= sum;
  }
}

void compute_output_gradient(double *output, double *target, double *delta,
                             int size) {
  // for softmax this is the derivative
  for (int i = 0; i < size; i++) {
    delta[i] = output[i] - target[i];
  }
}

// function for forwarding through the mlp
// W is an m x n matrix
// x is an 1 x n vector
// b is an 1 x m vector
// m is the number of output neurons
// n is the number of input features
void forward_layer(mlp *layer, double *output) {
  // assume output is initialized
  // for if it is not initialized: initialize_to_zero(1, m, &output);
  // multiplied is a 1 x m row vector where m is the number of output neurons
  // multiplied is xW^T
  double *multiplied;
  double **transposed;
  initialize_to_zero(layer->n, layer->m, &transposed);
  transpose(layer->n, layer->m, layer->W, transposed);
  initialize_to_zero_vector(layer->m, &multiplied);
  multiply_vector(layer->n, layer->m, layer->x, transposed, multiplied);
  add(layer->m, 1, multiplied, layer->b, output);
  for (int i = 0; i < layer->n; i++) {
    free(transposed[i]);
  }
  free(multiplied);
  free(transposed);
}

// function for forwarding through a neural network
// returns the output after softmaxing
void forward(nn *network, double *output) {
  int num_layers = network->num_layers;
  // loop through all layers
  for (int i = 0; i < num_layers; i++) {
    forward_layer(network->layers[i], network->layers[i]->z);
    // apply activation function if it is not last layer
    if (i != num_layers - 1) {
      RELU(network->layers[i]->z, network->layers[i]->a, network->layers[i]->m);
      // Deep copy em
      deep_copy(network->layers[i]->m, network->layers[i]->a,
                network->layers[i + 1]->x);
    } else {
      // apply softmax
      softmax(network->layers[i]->z, output, network->layers[i]->m);
    }
  }
  deep_copy(network->layers[num_layers - 1]->m, network->layers[num_layers - 1]->a, network->output);
}

double calculate_loss(double *output, int label) {
  // assume that softmax has already been applied
  double epsilon = 1e-10;
  double loss = -log(output[label] + epsilon);
  return loss;
}

// output is the final output of the neural network
// target is a one-hot encoded correct label
void backpropagate(nn *network, double *output, double *target) {
  int num_layers = network->num_layers;
  mlp **layers = network->layers;
  // compute output gradient
  compute_output_gradient(output, target, layers[num_layers - 1]->delta,
                          layers[num_layers - 1]->m);
  for (int i = num_layers - 2; i >= 0; i--) {
    // compute grad_W and grad_b
    double *next_delta = layers[i + 1]->delta;
    mlp *layer = layers[i];
    mlp *next_layer = layers[i + 1];
    // calculate delta_W
    double **delta_W;
    initialize_to_zero(layer->m, layer->n, &delta_W);
    for (int j = 0; j < layer->m; j++) {
      for (int k = 0; k < layer->n; k++) {
        delta_W[j][k] = next_delta[j] * layer->x[k];
      }
    }
    // calculate delta_b
    double *delta_b;
    initialize_to_zero_vector(layer->m, &delta_b);
    for (int j = 0; j < layer->m; j++) {
      delta_b[j] = next_delta[j];
    }

    // chain rule for this layers delta
    for (int j = 0; j < layer->m; j++) {
      for (int k = 0; k < next_layer->m; k++) {
        layer->delta[j] += next_layer->W[k][j] * next_layer->delta[k];
      }
      RELU_derivative(layer->delta, layer->z, layer->m);
    }

    // Update weights and biases
    // multiply delta by learning rate first
    multiply_by_number_vector(layer->m, network->learning_rate, delta_b);
    multiply_by_number(layer->m, layer->n, network->learning_rate, delta_W);
    // subtract delta from the weights and biases
    subtract(layer->m, layer->n, layer->W, delta_W, layer->W);
    subtract_vector(layer->m, layer->b, delta_b, layer->b);

    // clean up delta_W and delta_b
    for (int i = 0; i < layer->m; i++) {
      free(delta_W[i]);
    }
    free(delta_W);
    free(delta_b);
  }
}

// function for initializing an mlp layer
mlp *initialize_mlp(int n, int m, double *x) {
  mlp *layer = (mlp *)malloc(sizeof(mlp));
  layer->n = n;
  layer->m = m;

  // set W
  initialize_random(m, n, &layer->W);
  // set b
  initialize_random_vector(m, &layer->b);
  // set z
  initialize_to_zero_vector(m, &layer->z);
  // set a
  initialize_to_zero_vector(m, &layer->a);

  initialize_to_zero_vector(n, &layer->x);
  return layer;
}

void clean_mlp(mlp *layer) {
  for (int i = 0; i < layer->m; i++) {
    free(layer->W[i]);
  }
  free(layer->W);
  free(layer->b);
  free(layer->x);
  free(layer->z);
  free(layer->a);
  free(layer->delta);
  free(layer);
}

// size of layer_inputs is num_layers + 1
nn *initialize_nn(int num_layers, double learning_rate, int *layer_inputs) {
  nn *network = (nn *)malloc(sizeof(nn));
  network->num_layers = num_layers;
  network->learning_rate = learning_rate;
  mlp **layers = (mlp **)malloc(sizeof(mlp *) * num_layers);
  // loop through layer inputs and last one is an output
  for (int i = 0; i < num_layers; i++) {
    layers[i] = initialize_mlp(layer_inputs[i], layer_inputs[i + 1],
     (double *)NULL);
  }
  network->output = (double*)calloc(layer_inputs[num_layers], sizeof(double));
  return network;
}

void clean_nn(nn *network) {
  for (int i = 0; i < network->num_layers; i++) {
    clean_mlp(network->layers[i]);
  }
  free(network->layers);
  free(network);
}

void start_sample(nn *network, double *values) {
  mlp* layer = network->layers[0];
  deep_copy(layer->n, values, layer->x);
}