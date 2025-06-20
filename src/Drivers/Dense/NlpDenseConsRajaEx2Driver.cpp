#include "NlpDenseConsRajaEx2.hpp"
#include <vector>
#include <iostream>

int main(int argc, char** argv)
{
  int n = (argc>1) ? std::atoi(argv[1]) : 16;   // quick size tweak

  DenseConsRajaEx2 ex(n, "DEFAULT", /*unconstrained=*/true);

  std::vector<double> x(n, 1.2);   // test point
  double f;
  ex.eval_f(n, x.data(), true, f);

  std::vector<double> g(n);
  ex.eval_grad_f(n, x.data(), true, g.data());

  std::cout << "Objective at test point = " << f << "\nGradient head: ";
  for(int i=0;i<std::min(5,n);++i) std::cout << g[i] << " ";
  std::cout << "...\n";
  return 0;
}
