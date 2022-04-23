#include "relu.hpp"

#include <iostream>
using std::cout;

namespace layer {

relu::relu(string _name) { name = _name; }

mat relu::forward(mat x) {
  // Leaky relu implementation
  // mat y1 = ((x > 0) * x);
  // mat y2 = ((x <= 0) * x * 0.01);
  // y = y1 + y2;

  y = x * (x > 0);

  return y;
}

mat relu::backward(mat y_grad) {
  return y_grad * (y > 0);

  // mat y1 =
  // mat y2 =

  // return y1+y2;
}

}  // namespace layer