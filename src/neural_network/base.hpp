#pragma once

#include <string>
using std::string;

#include "mat.hpp"

namespace layer {

class base {
 public:
  base() {}

  string name = "";

  virtual mat forward(mat x) { return 0; }
  virtual mat forward(mat x, mat t) { return 0; }
  virtual mat backward(mat y_grad) { return 0; }
  virtual mat backward() { return 0; }
  virtual void update_param(float lr) {}
};

}  // namespace layer