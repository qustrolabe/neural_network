#include <iostream>
using std::cout;

#include <sstream>

#include "dense.hpp"

namespace layer {

dense::dense(int in_size, int out_size, string layer_name) {
  input_size = in_size;
  output_size = out_size;

  // Add (in_size, out_size) at the end of layer name
  std::stringstream ss;
  ss << "(" << input_size << "," << output_size << ")";

  name = layer_name + ss.str();

  // float a = sqrt(2.0 / float(input_size + output_size));

  weight = random::randn<float>({input_size, output_size});

  bias = zeros<float>({output_size});

  weight_grad = zeros<float>({input_size, output_size});
  bias_grad = zeros<float>({output_size});
}

void dense::show() {
  cout << "[LAYER_INFO] name: " << name << "\n"

       << "Weight:\n"
       << weight << "\n"
       << "weight_grad:\n"
       << weight_grad << "\n"

       << "bias:\n"
       << bias << "\n"
       << "bias_grad:\n"
       << bias_grad
       << "\n"

       // << "x_store:\n" << x_store << "\n"
       // << "\n"

       << "";
}

mat dense::forward(mat x) {
  x_store = x;

  mat y = linalg::dot(x, weight) + bias;

  // cout << "shape:\n" << adapt(y.shape()) << "\n";
  assert(adapt(y.shape()) == xarray<int>({output_size}));

  return y;
}

mat dense::backward(mat y_grad) {
  assert(adapt(y_grad.shape()) == xarray<int>({output_size}));

  weight_grad = linalg::dot(transpose(atleast_2d(x_store)), atleast_2d(y_grad));
  assert(adapt(weight_grad.shape()) == xarray<int>({input_size, output_size}));

  mat x_grad = linalg::dot(y_grad, transpose(weight));
  assert(adapt(x_grad.shape()) == xarray<int>({input_size}));

  bias_grad = y_grad;

  return x_grad;
}

void dense::update_param(float lr) {
  weight = (weight - lr * weight_grad);
  bias = (bias - lr * bias_grad);
}

}  // namespace layer