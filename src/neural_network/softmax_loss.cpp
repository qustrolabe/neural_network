#include "softmax_loss.hpp"

#include <iostream>
using std::cout;

softmax_loss::softmax_loss(string _name) {
    name = _name;
}

// mat softmax_loss::forward_softmax(mat x) {
//     x = exp(x);
//     return x / sum(x);
// }

// mat softmax_loss::forward(mat x, mat t) {
//     t_store = t;

//     mat z = forward_softmax(x);
//     x_store = z;

//     // CrossEntropy
//     return -sum(log(z) * t) / float(x.shape()[0]);
// }

// mat softmax_loss::backward() {
//     return (x_store - t_store) / float(x_store.shape()[0]);
// }

mat softmax_loss::forward_softmax(mat x) {
    return x;
}

mat softmax_loss::forward(mat x, mat t) {
    t_store = t;
    x_store = x;

    // cout << "FORWARD: "<<( x - t) << "\n";
    // return x - t;

    mat loss = (x - t);
    // cout << "FORWARD: "<< loss << "\n";
    return loss;
}

mat softmax_loss::backward() {
    // return (x_store - t_store);
    return (x_store - t_store);
}