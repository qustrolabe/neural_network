#include "sigmoid.hpp"

#include <iostream>
using std::cout;

namespace layer {

mat sig_func(mat x) { return mat(1 / (1 + exp(-x))); }

sigmoid::sigmoid(string _name) { name = _name; }

mat sigmoid::forward(mat x) {
  y = sig_func(x);
  return y;
}

mat sigmoid::backward(mat y_grad) { return y_grad * y * (1 - y); }

}  // namespace layer