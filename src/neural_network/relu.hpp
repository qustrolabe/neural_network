#pragma once

#include "base.hpp"
#include "mat.hpp"

namespace layer {

class relu : public base {
 public:
  relu(string _name = "Relu");

  mat y;

  mat forward(mat x);
  mat backward(mat y_grad);
};

}