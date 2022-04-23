#pragma once

#include "base.hpp"
#include "mat.hpp"

namespace layer {

class sigmoid : public base {
 public:
  sigmoid(string _name = "Sigmoid");

  mat y;

  mat forward(mat x);
  mat backward(mat y_grad);
};

}  // namespace layer