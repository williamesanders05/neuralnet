#include "data.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  // right here I would need to read csv 2
  char *file_name = "mnist_train.csv";
  data* d = initialize_data(file_name, 100);
  if (d == 0) {
    perror("Sum wrong");
    return 0;
  }
  for (int i = 0; i < 100; i++) {
    printf("%d : %d \n", i + 1, d->targets[i]);
  }
}
