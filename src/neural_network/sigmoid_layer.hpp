#pragma once

#include "base_layer.hpp"
#include "mat.hpp"

class sigmoid_layer : public base_layer {
 public:
  sigmoid_layer(string _name = "Sigmoid");

  mat y;

  mat forward(mat x);
  mat backward(mat y_grad);
};