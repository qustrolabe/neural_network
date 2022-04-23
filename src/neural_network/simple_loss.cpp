#include "simple_loss.hpp"

namespace layer {

simple_loss::simple_loss(string _name) { name = _name; }

mat simple_loss::forward(const mat x, const mat t) {
  x_store = x;
  t_store = t;

  return x_store - t_store;
}

mat simple_loss::backward() { return (x_store - t_store); }

}  // namespace layer