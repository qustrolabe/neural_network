#pragma once

#include "base.hpp"

namespace layer {

class dense : public base {
 public:
  dense(int in_size, int out_size, string layer_name = "full");

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

}  // namespace layer