#pragma once

#include <string>

#include "mat.hpp"
#include "base_layer.hpp"

class softmax_loss : public base_layer {
 public:
    softmax_loss(string _name = "Softmax");

    mat x_store;
    mat t_store;

    mat forward(mat x, mat t);
    mat forward_softmax(mat x);
    mat backward();
};