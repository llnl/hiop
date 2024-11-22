#include "NlpSparseEx1.hpp"
#include "hiopInterfacePrimalDecomp.hpp"

/** This class is the basecase problem for Ex9.
 *  To work for the master problem, it has a boolean include_rec_ to determine
 *  whether a recourse approximation is in the objective.
 *  There is no corresponding .cpp file.
 */
using namespace hiop;
class PriDecBasecaseProbleEx2 : public SparseEx1
{
public:
  PriDecBasecaseProbleEx2(int n)
      : SparseEx1(n, 1.0),
        rec_evaluator_(nullptr)
  {}

  virtual ~PriDecBasecaseProbleEx2() {}

  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
  {
    if(!SparseEx1::eval_f(n, x, new_x, obj_value)) {
      return false;
    }
    if(include_rec_) {  // same as include_r
      assert(rec_evaluator_->get_rgrad() != NULL);
      rec_evaluator_->eval_f(n, x, new_x, obj_value);
    }
    // add regularization to the objective based on rec_evaluator_
    return true;
  }

  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
  {
    if(!SparseEx1::eval_grad_f(n, x, new_x, gradf)) {
      return false;
    }
    // add regularization gradient
    if(include_rec_) {
      assert(rec_evaluator_->get_rgrad() != NULL);
      rec_evaluator_->eval_grad(n, x, new_x, gradf);
    }
    return true;
  }

  bool eval_Hess_Lagr(const size_type& n,
                      const size_type& m,
                      const double* x,
                      bool new_x,
                      const double& obj_factor,
                      const double* lambda,
                      bool new_lambda,
                      const int& nnzHSS,
                      int* iHSS,
                      int* jHSS,
                      double* MHSS)
  {
    if(!SparseEx1::eval_Hess_Lagr(n, m, x, new_x, obj_factor, lambda, new_lambda, nnzHSS, iHSS, jHSS, MHSS)) {
      return false;
    }
    // Add diagonal to the Hessian
    // The indices are already added through the parent

    if(MHSS != nullptr) {
      // use rec_evaluator_ to add diagonal entries in the Hessian
      assert(nnzHSS == n);
      if(include_rec_) {
        for(int i = 0; i < n; i++) {
          MHSS[i] += obj_factor * (rec_evaluator_->get_rhess()->local_data_const()[i]);
        }
      }
    }
    return true;
  }

  bool set_quadratic_terms(const int& n, hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator)
  {
    rec_evaluator_ = evaluator;
    return true;
  }

  void set_include(const bool include) { include_rec_ = include; };

  bool quad_is_defined()  // check if quadratic approximation is defined
  {
    if(rec_evaluator_ != NULL) {
      return true;
    } else {
      return false;
    }
  }

  void get_rec_obj(const size_type& n, const double* x, double& obj_value)
  {
    bool temp = rec_evaluator_->eval_f(n, x, false, obj_value);
    assert(temp);
  }

protected:
  bool include_rec_ = false;
  hiopInterfacePriDecProblem::RecourseApproxEvaluator* rec_evaluator_;  // this should be const
};
