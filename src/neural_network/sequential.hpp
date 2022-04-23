#pragma once

#include <vector>

#include "base.hpp"
#include "mat.hpp"

using std::vector;

namespace layer {

class sequential : public base {
 public:
  sequential(vector<base*> _layers, base* _loss);

  vector<base*> layers;
  base* loss;
  // mat error;

  mat forward(mat x);
  mat forward(mat x, mat t);
  mat backward();
  void update_param(float lr);
  void fit(vector<mat> x, vector<mat> y, int epochs = 10, float lr = 0.2,
           int batch_size = 4);
  mat predict(mat x);
};

}  // namespace layer