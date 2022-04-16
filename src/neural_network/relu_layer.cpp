#include "relu_layer.hpp"

#include <iostream>
using std::cout;

relu_layer::relu_layer(string _name) { name = _name; }

mat relu_layer::forward(mat x) {
  y = x * (x > 0);
  return y;
}

mat relu_layer::backward(mat y_grad) {
  return y_grad * (y > 0);
}