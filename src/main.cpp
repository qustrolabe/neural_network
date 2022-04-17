#include <iostream>

#include "neural_network.hpp"

using std::cout;
using std::endl;

int main(int argc, char const* argv[]) {
  xt::random::seed(time(NULL));

  auto l1 = new full_layer(2, 3, "full");
  auto l2 = new full_layer(3, 1, "full");


  #if 0
    auto a1 = new relu_layer("relu");
    auto a2 = new relu_layer("relu");
  #else
    auto a1 = new sigmoid_layer("sig");
    auto a2 = new sigmoid_layer("sig");
  #endif

  auto loss = new softmax_loss("softmax_loss");

  sequential model = sequential({l1, a1, l2}, loss);

  // l1->show();
  // l2->show();

  cout << "{1,0} : " << model.forward( {1, 0} ) << "\n";
  cout << "{0,1} : " << model.forward( {0, 1} ) << "\n";
  cout << "{1,1} : " << model.forward( {1, 1} ) << "\n";
  cout << "{0,0} : " << model.forward( {0, 0} ) << "\n";

  cout << "\n"
       << "====================== TRAINING ======================\n";

  int epochs = 10000;
  float lr = 0.1;

  vector<mat> inputs = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1},
  };

  vector<mat> outputs = {
    {0},
    {1},
    {1},
    {0},
  };


  model.fit(inputs,outputs, epochs, lr);

  cout << "\n"
       << "====================== AFTER FIT ======================\n";

  // l1->show();
  // l2->show();

  cout << "{1,0} : " << model.predict( {1, 0} ) << "\n";
  cout << "{0,1} : " << model.predict( {0, 1} ) << "\n";
  cout << "{1,1} : " << model.predict( {1, 1} ) << "\n";
  cout << "{0,0} : " << model.predict( {0, 0} ) << "\n";

  return 0;
}