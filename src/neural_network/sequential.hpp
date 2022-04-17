#pragma once

#include <vector>

#include "base_layer.hpp"
#include "mat.hpp"

using std::vector;

class sequential : public base_layer {
 public:
  sequential(vector<base_layer*> _layers, base_layer* _loss);

  vector<base_layer*> layers;
  base_layer* loss;
  // mat error;

  mat forward(mat x);
  mat forward(mat x, mat t);
  mat backward();
  void update_param(float lr);
  void fit(vector<mat> x, vector<mat> y, int epochs = 10, float lr = 0.2, int batch_size = 4);
  mat predict(mat x);
};