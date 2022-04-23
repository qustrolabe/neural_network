#include "softmax_loss.hpp"

#include <iostream>
using std::cout;

namespace layer {

softmax_loss::softmax_loss(string _name) { name = _name; }

mat softmax_loss::forward_softmax(mat x) {
  x = exp(x);
  return x / sum(x);
}

mat softmax_loss::forward(mat x, mat t) {
  t_store = t;

  // mat z = forward_softmax(x);
  mat z = (x);
  x_store = z;

  // CrossEntropy
  return -sum(log(z) * t);
}

mat softmax_loss::backward() { return (x_store - t_store); }

}  // namespace layer