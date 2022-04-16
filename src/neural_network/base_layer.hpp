#pragma once

#include <string>
using std::string;

#include "mat.hpp"

class base_layer {
 public:
  base_layer() {}

  string name = "";

  virtual mat forward(mat x) { return 0; }
  virtual mat forward(mat x, mat t) { return 0; }
  virtual mat backward(mat y_grad) { return 0; }
  virtual mat backward() { return 0; }
  virtual void update_param(float lr) {}
};