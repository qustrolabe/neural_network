#include <iostream>

#include "neural_network.hpp"

using std::cout;
using std::endl;

// #################
// #   TODO:
// #  - visualize data
// #  - write tests
// #  - MNIST dataset
// #  - Softmax loss
// #  - Binary loss
// #  - Other activation functions
// #  - Rename full to Dense functions
// #  - Better file\class heirarchy
// #  - Class for complete network
// #  - Batches
// #  - 2D input?
// #  - File json export/import of network
// #  - std::move semantic, containers
// #  - smart pointers
// #################

int main(int argc, char const* argv[]) {
  xt::random::seed(time(NULL));

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

  auto l1 = new layer::dense(2, 3, "full");
  auto l2 = new layer::dense(3, 1, "full");

#if 0
  auto a1 = new layer::relu("relu");
  auto a2 = new layer::relu("relu");
#else
  auto a1 = new layer::sigmoid("sig");
  auto a2 = new layer::sigmoid("sig");
#endif

  auto loss = new layer::simple_loss("softmax_loss");

  auto model = layer::sequential({l1, a1, l2}, loss);

  // l1->show();
  // l2->show();

  cout << "{1,0} : " << model.forward({1, 0}) << "\n";
  cout << "{0,1} : " << model.forward({0, 1}) << "\n";
  cout << "{1,1} : " << model.forward({1, 1}) << "\n";
  cout << "{0,0} : " << model.forward({0, 0}) << "\n";

  cout << "\n"
       << "====================== TRAINING ======================\n";

  model.fit(inputs, outputs, epochs, lr);

  cout << "\n"
       << "====================== AFTER FIT ======================\n";

  // l1->show();
  // l2->show();

  cout << "{1,0} : " << model.predict({1, 0}) << "\n";
  cout << "{0,1} : " << model.predict({0, 1}) << "\n";
  cout << "{1,1} : " << model.predict({1, 1}) << "\n";
  cout << "{0,0} : " << model.predict({0, 0}) << "\n";

  return 0;
}
