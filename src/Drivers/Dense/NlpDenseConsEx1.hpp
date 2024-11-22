#ifndef HIOP_EXAMPLE_DENSE_EX1
#define HIOP_EXAMPLE_DENSE_EX1

#include "hiopVector.hpp"
#include "LinAlgFactory.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopInterface.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cassert>

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

#include <iostream>

/* Example 1: a simple infinite-dimensional QP in the optimiz. function variable x:[0,1]->R
 *  min   sum <c,x>+1/2*<x,x>
 *  s.t.
 *        integral(x:[0,1]) = 0.5
 *        0.1 <= x(t) <= 1.0, for all t in [0,1].
 *
 * Here c(t) = 1-t*10, for 0<=t<=1/10,
 *              0,     for 1/10<=t<=1.
 * The inner products are L2.
 *
 * We generate "distorted" meshes for [0,1] having the ratio of the smalest element and
 * the largest element given by r. The mesh is such that the consecutive elements
 * increase by h
 *           [t_0,t_1], [t_1, t_2], [t_2, t_3], ..., [t_{n-1}, t_n]  (t_0=0, t_n=1)
 * length:       m1        m1+h        m1+2h            m1+(n-1)h
 * For a given mesh size "n" and a desired distortion ratio "r" we compute h and m1 as
 *  h = 2*(1-r)/ ((1+r)*n*n(n-1)) and m1=2*r / ((1+r)*n)
 *
 * One can verify that the distortion rate, given by m1/(m1+(n-1)h), is r.
 */

class DiscretizedFunction;

using size_type = hiop::size_type;
using index_type = hiop::index_type;

/* our (admitedly weird) 1D distorted meshing */
class Ex1Meshing1D
{
public:
  Ex1Meshing1D(double a, double b, size_type glob_n, double r = 1.0, MPI_Comm comm = MPI_COMM_WORLD);
  virtual ~Ex1Meshing1D();
  virtual bool matches(Ex1Meshing1D* other) { return this == other; }
  virtual size_type size() const { return _mass->get_size(); }
  virtual size_type local_size() const { return col_partition[my_rank + 1] - col_partition[my_rank]; }

  /* the following methods are mostly for educational purposes and may not be optimized */
  // converts the local indexes to global indexes
  index_type getGlobalIndex(index_type i_local) const;
  // given a global index, returns the local index
  index_type getLocalIndex(index_type i_global) const;
  // for a function c(t), for given global index in the discretization
  //  returns the corresponding continuous argument 't', which is in this
  //  case the middle of the discretization interval.
  double getFunctionArgument(index_type i_global) const;

  virtual bool get_vecdistrib_info(size_type global_n, index_type* cols);
  index_type* get_col_partition() const { return col_partition; }
  MPI_Comm get_comm() const { return comm; }

  virtual void applyM(DiscretizedFunction& f);

protected:
  hiop::hiopVector* _mass;  // the length or the mass of the elements
  double _a, _b;            // end points
  double _r;                // distortion ratio

  MPI_Comm comm;
  int my_rank, comm_size;
  index_type* col_partition;

  friend class DiscretizedFunction;

private:
  Ex1Meshing1D(const Ex1Meshing1D& other) { assert(false); }
  Ex1Meshing1D() { assert(false); }
};

class DiscretizedFunction : public hiop::hiopVectorPar
{
public:
  DiscretizedFunction(Ex1Meshing1D* meshing);

  double dotProductWith(const hiopVector& v) const override;
  virtual double integral() const;
  double twonorm() const override;

  /* the following methods are mostly for educational purposes and may not be optimized */
  // converts the local indexes to global indexes
  index_type getGlobalIndex(index_type i_local) const;
  // for a function c(t), for given global index in the discretization
  //  returns the corresponding continuous argument 't', which is in this
  //  case the middle of the discretization interval.
  double getFunctionArgument(index_type i_global) const;
  // set the function value for a given global index
  void setFunctionValue(index_type i_global, const double& value);

protected:
  Ex1Meshing1D* _mesh;
};

class DenseConsEx1 : public hiop::hiopInterfaceDenseConstraints
{
public:
  DenseConsEx1(int n_mesh_elem = 100, double mesh_ratio = 1.0)
      : n_vars(n_mesh_elem),
        comm(MPI_COMM_WORLD),
        solver_(nullptr)
  {
    // create the members
    _mesh = new Ex1Meshing1D(0.0, 1.0, n_vars, mesh_ratio, comm);
    c = new DiscretizedFunction(_mesh);
    x = new DiscretizedFunction(_mesh);
    //_aux=new DiscretizedFunction(_mesh); // used as a auxiliary variable
    n_local = _mesh->local_size();

    set_c();
  }
  virtual ~DenseConsEx1()
  {
    delete c;
    delete x;
    // delete _aux;
    delete _mesh;
  }
  bool get_prob_sizes(size_type& n, size_type& m)
  {
    n = n_vars;
    m = 1;
    return true;
  }

  bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
  {
    for(int i_local = 0; i_local < n_local; i_local++) {
      xlow[i_local] = 0.1;
      xupp[i_local] = 1.0;
      type[i_local] = hiopNonlinear;
    }
    return true;
  }
  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m == 1);

    clow[0] = 0.5;
    cupp[0] = 0.5;
    type[0] = hiopInterfaceBase::hiopLinear;
    return true;
  }
  bool eval_f(const size_type& n, const double* x_in, bool new_x, double& obj_value)
  {
    x->copyFrom(x_in);
    obj_value = c->dotProductWith(*x);
    double xnrm = x->twonorm();
    // printf("c'x=%g   xnrm_sq=%g\n", obj_value, xnrm*xnrm);
    obj_value += 0.5 * xnrm * xnrm;

    return true;
  }
  bool eval_grad_f(const size_type& n, const double* x_in, bool new_x, double* gradf)
  {
    // gradf = m.*(x + c)
    // use x as auxiliary variable
    x->copyFrom(x_in);
    x->axpy(1.0, *c);
    _mesh->applyM(*x);
    x->copyTo(gradf);

    // x->copyFrom(x_in);
    // x->print(stdout);
    return true;
  }
  /** Sum(x[i])<=10 and sum(x[i])>= 1  (we pretend are different)
   */
  bool eval_cons(const size_type& n,
                 const size_type& m,
                 const size_type& num_cons,
                 const index_type* idx_cons,
                 const double* x_in,
                 bool new_x,
                 double* cons)
  {
    assert(n == n_vars);
    if(0 == num_cons) return true;  // this may happen when Hiop asks for inequalities, which we don't have in this example

    assert(num_cons == 1);
    x->copyFrom(x_in);
    cons[0] = x->integral();
    return true;
  }

  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const size_type& num_cons,
                     const index_type* idx_cons,
                     const double* x_in,
                     bool new_x,
                     double* Jac)
  {
    assert(n == n_vars);
    if(0 == num_cons) return true;  // this may happen when Hiop asks for inequalities, which we don't have in this example
    assert(1 == num_cons);
    // use x as auxiliary
    x->setToConstant(1.);
    _mesh->applyM(*x);
    x->copyTo(Jac);
    return true;
  }

  bool get_vecdistrib_info(size_type global_n, index_type* cols)
  {
    if(global_n == n_vars)
      return _mesh->get_vecdistrib_info(global_n, cols);
    else
      assert(false && "You shouldn't need distrib info for this size.");
    return true;
  }

  bool get_starting_point(const size_type& global_n, double* x0)
  {
    assert(global_n == n_vars);
    for(int i_local = 0; i_local < n_local; i_local++) {
      x0[i_local] = 0.5;
    }
    return true;
  }

  inline void set_solver(hiop::hiopAlgFilterIPM* alg_obj) { solver_ = alg_obj; }

  bool iterate_callback(int iter,
                        double obj_value,
                        double logbar_obj_value,
                        int n,
                        const double* x,
                        const double* z_L,
                        const double* z_U,
                        int m_ineq,
                        const double* s,
                        int m,
                        const double* g,
                        const double* lambda,
                        double inf_pr,
                        double inf_du,
                        double onenorm_pr,
                        double mu,
                        double alpha_du,
                        double alpha_pr,
                        int ls_trials);

private:
  int n_vars;
  MPI_Comm comm;
  Ex1Meshing1D* _mesh;

  int n_local;

  DiscretizedFunction* c;
  DiscretizedFunction* x;  // proxy for taking hiop's variable in and working with it as a function

  /// Pointer to the solver, to be used to checkpoint
  hiop::hiopAlgFilterIPM* solver_;

private:
  // populates the linear term c
  void set_c();
};

#endif
