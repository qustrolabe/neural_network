#include "sigmoid_layer.hpp"

#include <iostream>
using std::cout;

sigmoid_layer::sigmoid_layer(string _name) { name = _name; }


mat sigmoid(mat x) {
    return mat(1 / (1 + exp(-x)));
}

mat sigmoid_layer::forward(mat x) {
  y = sigmoid(x);
  return y;
}

mat sigmoid_layer::backward(mat y_grad) { 
    return y_grad*y*(1-y);
}