#include "NlpDenseConsEx1.hpp"

#include <cmath>
#include <cstdio>
#include <cassert>

#ifdef HIOP_USE_AXOM
#include <axom/sidre/core/DataStore.hpp>
#include <axom/sidre/core/Group.hpp>
#include <axom/sidre/core/View.hpp>
#include <axom/sidre/spio/IOManager.hpp>
using namespace axom;
#endif

using namespace hiop;

Ex1Meshing1D::Ex1Meshing1D(double a, double b, size_type glob_n, double r, MPI_Comm comm_)
{
  _a = a;
  _b = b;
  _r = r;
  comm = comm_;
  comm_size = 1;
  my_rank = 0;
#ifdef HIOP_USE_MPI
  int ierr = MPI_Comm_size(comm, &comm_size);
  assert(MPI_SUCCESS == ierr);
  ierr = MPI_Comm_rank(comm, &my_rank);
  assert(MPI_SUCCESS == ierr);
#endif
  // set up vector distribution for primal variables - easier to store it as a member in this simple example
  col_partition = new index_type[comm_size + 1];
  size_type quotient = glob_n / comm_size, remainder = glob_n - comm_size * quotient;

  int i = 0;
  col_partition[i] = 0;
  i++;
  while(i <= remainder) {
    col_partition[i] = col_partition[i - 1] + quotient + 1;
    i++;
  }
  while(i <= comm_size) {
    col_partition[i] = col_partition[i - 1] + quotient;
    i++;
  }

  _mass = LinearAlgebraFactory::create_vector("DEFAULT", glob_n, col_partition, comm);

  // if(my_rank==0) printf("reminder=%d quotient=%d\n", remainder, quotient);
  // printf("left=%d right=%d\n", col_partition[my_rank], col_partition[my_rank+1]);

  // compute the mass
  double m1 = 2 * _r / ((1 + _r) * glob_n);
  double h = 2 * (1 - _r) / (1 + _r) / (glob_n - 1) / glob_n;

  size_type glob_n_start = col_partition[my_rank], glob_n_end = col_partition[my_rank + 1] - 1;

  double* mass = _mass->local_data();  // local slice
  double rescale = _b - _a;
  for(size_type k = glob_n_start; k <= glob_n_end; k++) {
    mass[k - glob_n_start] = (m1 + (k - glob_n_start) * h) * rescale;
    // printf(" proc %d k=%d  mass[k]=%g\n", my_rank, k, mass[k-glob_n_start]);
  }

  //_mass->print(stdout, NULL);
  // fflush(stdout);
}
Ex1Meshing1D::~Ex1Meshing1D()
{
  delete[] col_partition;
  delete _mass;
}

bool Ex1Meshing1D::get_vecdistrib_info(size_type global_n, index_type* cols)
{
  for(int i = 0; i <= comm_size; i++) cols[i] = col_partition[i];
  return true;
}
void Ex1Meshing1D::applyM(DiscretizedFunction& f) { f.componentMult(*this->_mass); }

// converts the local indexes to global indexes
index_type Ex1Meshing1D::getGlobalIndex(index_type i_local) const
{
  assert(0 <= i_local);
  assert(i_local < col_partition[my_rank + 1] - col_partition[my_rank]);

  return i_local + col_partition[my_rank];
}

index_type Ex1Meshing1D::getLocalIndex(index_type i_global) const
{
  assert(i_global >= col_partition[my_rank]);
  assert(i_global < col_partition[my_rank + 1]);
  return i_global - col_partition[my_rank];
}

// for a function c(t), for given global index in the discretization
//  returns the corresponding continuous argument 't', which is in this
//  case the middle of the discretization interval.
double Ex1Meshing1D::getFunctionArgument(index_type i_global) const
{
  assert(i_global >= col_partition[my_rank]);
  assert(i_global < col_partition[my_rank + 1]);

  const index_type& k = i_global;

  size_type glob_n = size();
  double m1 = 2 * _r / ((1 + _r) * glob_n);
  double h = 2 * (1 - _r) / (1 + _r) / (glob_n - 1) / glob_n;

  // t is the middle of [k*m1 + k(k-1)/2*h, (k+1)m1+ (k+1)k/2*h]
  double t = 0.5 * ((2 * k + 1) * m1 + k * k * h);
  return t;
}

/* DiscretizedFunction implementation */

DiscretizedFunction::DiscretizedFunction(Ex1Meshing1D* meshing)
    : hiopVectorPar(meshing->size(), meshing->get_col_partition(), meshing->get_comm())
{
  _mesh = meshing;
}

// u'*v = u'*M*v, where u is 'this'
double DiscretizedFunction::dotProductWith(const hiopVector& v_) const
{
  auto discretizedFunction(dynamic_cast<const DiscretizedFunction*>(&v_));
  if(discretizedFunction) {
    assert(discretizedFunction->_mesh->matches(this->_mesh));
    double* M = _mesh->_mass->local_data();
    double* u = this->data_;
    double* v = discretizedFunction->data_;

    double dot = 0.;
    for(int i = 0; i < get_local_size(); i++) dot += u[i] * M[i] * v[i];

#ifdef HIOP_USE_MPI
    double dotprodG;
    int ierr = MPI_Allreduce(&dot, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm_);
    assert(MPI_SUCCESS == ierr);
    dot = dotprodG;
#endif
    return dot;
  } else {
    return hiopVectorPar::dotProductWith(v_);
  }
}

// computes integral of 'this', that is sum (this[elem]*m[elem])
double DiscretizedFunction::integral() const
{
  // the base dotProductWith method would do it
  return hiopVectorPar::dotProductWith(*_mesh->_mass);
}

// norm(u) as sum(M[elem]*u[elem]^2)
double DiscretizedFunction::twonorm() const
{
  double* M = _mesh->_mass->local_data();
  double* u = this->data_;

  double nrm_square = 0.;
  for(int i = 0; i < get_local_size(); i++) nrm_square += u[i] * u[i] * M[i];

#ifdef HIOP_USE_MPI
  double nrm_squareG;
  int ierr = MPI_Allreduce(&nrm_square, &nrm_squareG, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  nrm_square = nrm_squareG;
#endif
  return sqrt(nrm_square);
}

// converts the local indexes to global indexes
index_type DiscretizedFunction::getGlobalIndex(index_type i_local) const { return _mesh->getGlobalIndex(i_local); }

// for a function c(t), for given global index in the discretization
//  returns the corresponding continuous argument 't', which is in this
//  case the middle of the discretization interval.
double DiscretizedFunction::getFunctionArgument(index_type i_global) const { return _mesh->getFunctionArgument(i_global); }

// set the function value for a given global index
void DiscretizedFunction::setFunctionValue(index_type i_global, const double& value)
{
  index_type i_local = _mesh->getLocalIndex(i_global);
  this->data_[i_local] = value;
}

/* DenseConsEx1 class implementation */

bool DenseConsEx1::iterate_callback(int iter,
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
                                    int ls_trials)
{
#ifdef HIOP_USE_AXOM
  // save state to sidre::Group every 5 iterations if a solver/algorithm object was provided
  if(iter > 0 && (iter % 5 == 0) && nullptr != solver_) {
    //
    // Example of how to save HiOp state to axom::sidre::Group
    //

    // We first manufacture a Group. User code supposedly already has one.
    sidre::DataStore ds;
    sidre::Group* group = ds.getRoot()->createGroup("HiOp quasi-Newton alg state");

    // the actual saving of state to group
    try {
      solver_->save_state_to_sidre_group(*group);
    } catch(std::runtime_error& e) {
      // user chooses action when an error occured in saving the state...
      // we choose to stop HiOp
      return false;
    }

    // User code can further inspect the Group or add addtl info to DataStore, with the end goal
    // of saving it to file before HiOp starts next iteration. Here we just save it.
    sidre::IOManager writer(comm);
    int n_files;
    MPI_Comm_size(comm, &n_files);
    writer.write(ds.getRoot(), n_files, "hiop_state_ex1", sidre::Group::getDefaultIOProtocol());
  }
#endif
  return true;
}

/*set c to
 *    c(t) = 1-10*t, for 0<=t<=1/10,
 *           0,      for 1/10<=t<=1.
 */
void DenseConsEx1::set_c()
{
  for(int i_local = 0; i_local < n_local; i_local++) {
    // this will be based on 'my_rank', thus, different ranks get the appropriate global indexes
    size_type n_global = c->getGlobalIndex(i_local);
    double t = c->getFunctionArgument(n_global);
    // if(t<=0.1) c->setFunctionValue(n_global, 1-10.*t);
    double cval;
    if(t <= 0.1)
      cval = -1. + 10. * t;
    else
      cval = 0.;

    c->setFunctionValue(n_global, cval);
    // printf("index %d  t=%g value %g\n", n_global, t, cval);
  }
}
