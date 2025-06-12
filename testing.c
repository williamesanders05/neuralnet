#include "data.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double* create_one_hot(int label) {
  double *result = (double*) calloc(10, sizeof(double));
  for (int i = 0; i < 10; i++) {
    result[i] = (double)(i == label);
  }
  return result;
}

void accumulate_gradients(nn* network, double* target) {
  int num_layers = network->num_layers;
  mlp** layers = network->layers;

  compute_output_gradient(network->output, target, layers[num_layers - 1]->delta, layers[num_layers - 1]->m);
  for (int i = num_layers - 2; i >= 0; i--) {
    mlp *layer = layers[i];
    mlp *next_layer = layers[i + 1];

    for (int j = 0; j < layer->m; j++) {
      layer->delta[j] = 0.0;
    }

    for (int j = 0; j < layer->m; j++) {
      for (int k = 0; k < layer->n; k++) {
	layer->delta[j] += next_layer->W[k][j] * next_layer->delta[k];
      } 
    }

    RELU_derivative(layer->z,layer->delta,layer->m);
  }
}

void update_weights_batch(nn *network, int batch_size) {
  int num_layers = network->num_layers;
  mlp **layers = network->layers;

  for (int i = 0; i < num_layers; i++) {
    mlp *layer = layers[i];

    // update W weights
    for (int j = 0; j < layer->m; j++) {
      for (int k = 0; k < layer->n; k++) {
	double gradient = (layer->x[k] * layer->delta[j]) / batch_size;
	layer->W[j][k] -= network->learning_rate * gradient;
      }
    }

    //update b weights
    for (int j = 0; j < layer->m; j++) {
      double gradient = layer->delta[j] / batch_size;
      layer->b[j] -= network->learning_rate * gradient;
    }
  }
}

void zero_gradients(nn* network) {
  for (int i = 0; i < network->num_layers; i++) {
    mlp *layer = network->layers[i];
    for (int j = 0; j < layer->m; j++) {
      layer->delta[j] = 0.0;
    }
  }
}

int main() {
  // right here I would need to read csv 2
  char *file_name = "mnist_train.csv";
  int num_samples = 30000;
  data* d = initialize_data(file_name, num_samples);
  int *layer_inputs = calloc(4, sizeof(int));
  layer_inputs[0] = 784;
  layer_inputs[1] = 128;
  layer_inputs[2] = 64;
  layer_inputs[3] = 10;
  nn* network = initialize_nn(3, 0.001, layer_inputs);
  double *outputs = (double*)calloc(10, sizeof(double));
  
  int num_batches = 300;
  int batch_size = num_samples / num_batches;
  int max_epochs = 100;

  for (int epoch = 0; epoch < max_epochs; epoch++) {
    double total_epoch_loss = 0.0;

    for (int batch = 0; batch < num_batches; batch++) {
      double total_batch_loss = 0.0;

      zero_gradients(network);

      for (int i = 0; i < batch_size; i++) {
	int sample_index = batch * batch_size + i;
	int sample_target = d->targets[sample_index];

	start_sample(network, d->data[sample_index]);
	forward(network, outputs);

	double sample_loss = calculate_loss(network->output, sample_target);
	total_batch_loss += sample_loss;

	double* one_hot = create_one_hot(sample_target);

	accumulate_gradients(network,one_hot);

	free(one_hot);
      }

      update_weights_batch(network, batch_size);

      double average_batch_loss = total_batch_loss / batch_size;
      total_epoch_loss += total_batch_loss;

      if (batch % 50 == 0) {
	printf("Epoch %d, Batch %d, Avg Loss: %.4f\n", epoch, batch, average_batch_loss);
      }

      double average_epoch_loss = total_epoch_loss / num_samples;
      printf("Epoch %d completed. Avg Loss: %.4f\n", epoch, average_epoch_loss);

      if (average_epoch_loss <= .2) {
	printf("Target loss achieved. Stopping training");
	break;
      }
    }
  }

  free(outputs);
  free(layer_inputs);
  clean_nn(network);
  clean_data(d);

  return 0;
}
