#pragma once

#include "base_layer.hpp"

class full_layer : public base_layer {
 public:
  full_layer(int in_size, int out_size, string layer_name = "full");

  int input_size;
  int output_size;

  mat weight;
  mat bias;

  mat weight_grad;
  mat bias_grad;

  mat x_store;

  void show();

  mat forward(mat x);
  mat backward(mat y_grad);
  void update_param(float lr);
};