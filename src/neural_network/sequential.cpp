#include "sequential.hpp"

#include <iostream>

using std::cout;
using std::endl;

sequential::sequential(vector<base_layer*> _layers, base_layer* _loss) {
  layers = _layers;
  loss = _loss;
}

mat sequential::forward(mat x) {
  mat h = x;

  for (auto l : layers) {
    h = l->forward(h);
  }

  return h;
}

mat sequential::forward(mat x, mat t) {
  mat h = loss->forward(forward(x), t);

  return h;
}

mat sequential::backward() {
  mat h = loss->backward();

  for (int i = layers.size() - 1; i >= 0; i--) h = layers[i]->backward(h);

  return h;
}

void sequential::update_param(float lr) {
  for (auto l : layers) {
    l->update_param(lr);
  }
}

void sequential::fit(vector<mat> x, vector<mat> y, int epochs, float lr, int batch_size) {
  for (int e = 0; e < epochs; e++) {
    mat sum_loss = 0;
    int b_i = 0;

    for (int j = 0; j < x.size(); j++) {
      // int i = rand() % 4;
      int i = j;

      // mat X = row(x, i);
      // mat Y = row(atleast_2d(y), i);

      mat X = x.at(j);
      mat Y = y.at(j);

      mat loss = forward(X, Y);
      sum_loss += loss;
      b_i++;

      backward();
      update_param(lr);

      if (e % ((int)(1 + epochs * (1.0 / 5))) == 0)
        cout << "loss:" << loss << "\n";
    }

    if (e % ((int)(1 + epochs * (1.0 / 5))) == 0)
      cout << "Epoch " << e << "/" << epochs
           << "\n"
           // << "average loss: " << (sum_loss / float(b_i)) << "\n"
           << "sum_loss: " << sum_loss << "\n\n";
  }
}

mat sequential::predict(mat x) {
  // Index of max output neuron
  // Can't be used if output_size is 1
  // return argmax(forward(x));

  mat y = forward(x);

  return ones_like(y) * (y > 0.5);
}