#include <iostream>
using std::cout;

#include <sstream>

#include "full_layer.hpp"

// full_layer::full_layer(int in_size, int out_size, string layer_name) {
//   input_size = in_size;
//   output_size = out_size;

//   // Add (in_size, out_size) at the end of layer name
//   std::stringstream ss;
//   ss << "(" << input_size << "," << output_size << ")";

//   name = layer_name + ss.str();

//   float a = sqrt(2.0 / float(input_size + output_size));

//   weight = random::randn<float>({input_size, output_size}) * a;

//   // weight = (weight * 0) + 0.3;

//   bias = zeros<float>({output_size});

//   weight_grad = zeros<float>({input_size, output_size});
//   bias_grad = zeros<float>({output_size});
// }

void full_layer::show() {
  cout << "[LAYER_INFO] name: " << name << "\n"

       << "Weight:\n"
       << weight << "\n"
       << "weight_grad:\n"
       << weight_grad << "\n"

       << "bias:\n" << bias << "\n"
       << "bias_grad:\n" << bias_grad
       << "\n"

       // << "x_store:\n" << x_store << "\n"
       // << "\n"

       << "";
}

// mat full_layer::forward(mat x) {
//   x_store = x;
//   mat y = linalg::dot(x, weight);
//   y = y + bias;
//   return y;
// }

// mat full_layer::backward(mat y_grad) {
//   weight_grad = linalg::dot(transpose(atleast_2d(x_store)),
//   atleast_2d(y_grad)); bias_grad = y_grad;

//   mat x_grad = linalg::dot(y_grad, transpose(weight));
//   return x_grad;
// }

// void full_layer::update_param(float lr) {
//   weight = (weight - lr * weight_grad);
//   bias = (bias - lr * bias_grad);
// }








full_layer::full_layer(int in_size, int out_size, string layer_name) {
  input_size = in_size;
  output_size = out_size;

  name = layer_name;

  weight = random::randn<float>({output_size, input_size});
  bias = zeros<float>({output_size, 1});

  weight_grad = zeros<float>({output_size, input_size});
  bias_grad = zeros<float>({output_size, 1});
}

mat full_layer::forward(mat x) {
  assert(adapt(x.shape()) == xarray<int>({input_size, 1}));

  x_store = x;

  mat y = linalg::dot(weight, x) + bias;

  assert(adapt(y.shape()) == xarray<int>({output_size, 1}));

  return y;
}

mat full_layer::backward(mat y_grad) {
  assert(adapt(y_grad.shape()) == xarray<int>({output_size, 1}));

  mat x_grad = linalg::dot(transpose(weight), y_grad);

  assert(adapt(x_grad.shape()) == xarray<int>({input_size, 1}));

  weight_grad = linalg::dot(y_grad, transpose(x_store));

  assert(adapt(weight_grad.shape()) == xarray<int>({output_size, input_size}));

  bias_grad = y_grad;

  assert(adapt(bias_grad.shape()) == xarray<int>({output_size, 1}));

  return x_grad;
}

void full_layer::update_param(float lr) {
  weight = (weight - lr * weight_grad);
  bias = (bias - lr * bias_grad);
}