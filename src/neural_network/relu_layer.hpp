#pragma once

#include "base_layer.hpp"
#include "mat.hpp"

class relu_layer : public base_layer {
 public:
  relu_layer(string _name = "Relu");

  mat y;

  mat forward(mat x);
  mat backward(mat y_grad);
};