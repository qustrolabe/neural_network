#include <iostream>
// #include <utility>

#include "neural_network.hpp"

using std::cout;
using std::endl;

///////////////
//
// fix backpropagation
// fix SOFTMAX and a lot of others
//
// W*X
//
///////////////

int main(int argc, char const* argv[]) {

  #if 0

    auto s1 = sigmoid_layer("sig1");
    cout << "f{0.1,0.9}" << s1.forward({0.1,0.9}) << "\n";
    cout << "b{0.5,0.5}" << s1.backward({0.5,0.5}) << "\n";

    return 0;
  #endif

  xt::random::seed(time(NULL));

  cout << "XTENSOR 2D:\n" << atleast_2d(mat{1,2,3,4}) << "\n\n" << transpose(atleast_2d(mat{1,2,3,4})) << "\n\n";

  auto l1 = new full_layer(2, 3, "full");
  auto l2 = new full_layer(3, 4, "full");
  auto l3 = new full_layer(4, 5, "full");
  auto l4 = new full_layer(5, 4, "full");
  auto l5 = new full_layer(4, 1, "full");


  #if 0
    auto a1 = new relu_layer("relu");
    auto a2 = new relu_layer("relu");
  #else
    auto a1 = new sigmoid_layer("sig");
    auto a2 = new sigmoid_layer("sig");
    auto a3 = new sigmoid_layer("sig");
    auto a4 = new sigmoid_layer("sig");
  #endif

  auto loss = new softmax_loss("softmax_loss");

  sequential model = sequential({l1, a1, l2, a2, l3, a3, l4, a4, l5}, loss);

  l1->show();
  l2->show();

  cout << "{1,0} : " << model.forward( transpose( atleast_2d( mat{1, 0} ) ) ) << "\n";
  cout << "{0,1} : " << model.forward( transpose( atleast_2d( mat{0, 1} ) ) ) << "\n";
  cout << "{1,1} : " << model.forward( transpose( atleast_2d( mat{1, 1} ) ) ) << "\n";
  cout << "{0,0} : " << model.forward( transpose( atleast_2d( mat{0, 0} ) ) ) << "\n";

  cout << "\n"
       << "====================== TRAINING ======================\n";

  int epochs = 10000;
  float lr = 0.1;

  mat inputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0},
  };

  // mat outputs = {
  //   1,
  //   1,
  //   0,
  //   0
  // };


  mat outputs = {
    0,
    1,
    1,
    0,
  };


  model.fit(inputs,outputs, epochs, lr);

  cout << "\n"
       << "====================== AFTER FIT ======================\n";

  l1->show();
  l2->show();

  cout << "{1,0} : " << model.forward( transpose( atleast_2d( mat{1, 0} ) ) ) << "\n";
  cout << "{0,1} : " << model.forward( transpose( atleast_2d( mat{0, 1} ) ) ) << "\n";
  cout << "{1,1} : " << model.forward( transpose( atleast_2d( mat{1, 1} ) ) ) << "\n";
  cout << "{0,0} : " << model.forward( transpose( atleast_2d( mat{0, 0} ) ) ) << "\n";

  return 0;
}