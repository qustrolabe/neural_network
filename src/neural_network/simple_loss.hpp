#pragma once

#include <string>

#include "base.hpp"
#include "mat.hpp"

namespace layer {

class simple_loss : public base {
 public:
  simple_loss(string _name = "Simple");

  mat x_store;
  mat t_store;

  mat forward(const mat x, const mat t);
  mat backward();
};

}  // namespace layer