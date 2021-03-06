#pragma once

#include <string>

#include "base.hpp"
#include "mat.hpp"

namespace layer {

class softmax_loss : public base {
 public:
  softmax_loss(string _name = "Softmax");

  mat x_store;
  mat t_store;

  mat forward(mat x, mat t);
  mat forward_softmax(mat x);
  mat backward();
};

}  // namespace layer