#ifndef HIOP_EXAMPLE_DENSE_RAJA_EX2
#define HIOP_EXAMPLE_DENSE_RAJA_EX2

#include "NlpDenseConsEx2.hpp"   // original header
#include <string>

class DenseConsRajaEx2 : public DenseConsEx2
{
public:
  DenseConsRajaEx2(int n,
                   const std::string& mem_space = "DEFAULT",
                   bool unconstrained           = false);
  ~DenseConsRajaEx2() override;

  // only these two callbacks are RAJA-ised for now
  bool eval_f (hiop::size_type n, const double* x, bool new_x, double& obj) override;
  bool eval_grad_f(hiop::size_type n, const double* x, bool new_x, double* grad) override;

private:
  std::string mem_space_;
  double*     buf_tmp_;   // example scratch buffer
};

#endif
