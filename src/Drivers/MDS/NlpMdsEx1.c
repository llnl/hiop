#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "hiopInterface.h"
#include <math.h>

typedef struct settings {
  hiop_size_type n; hiop_size_type m; hiop_size_type ns; hiop_size_type nd;
  hiop_size_type nx_sparse; hiop_size_type nx_dense;
  hiop_size_type nnz_sparse_Jaceq; hiop_size_type nnz_sparse_Jacineq;
  hiop_size_type nnz_sparse_Hess_Lagr_SS; hiop_size_type nnz_sparse_Hess_Lagr_SD;
  double* xlow; double* xupp; double* clow; double* cupp;
  double* Q; double* Md; double* buf_y;
} settings;

// y := alpha*A*x + beta*y
void timesVec(const double* Q,
              double beta,
              double*y,
              double alpha,
              const double* x,
              hiop_size_type ns,
              hiop_size_type nd) {
  int i=0;
  int j=0;
  for(i=0; i<nd; i=i+1) {
    double tmp = 0.0;
    for(j=0; j<ns; j=j+1) {
      tmp = tmp + alpha*Q[i*ns + j]*x[j];
    }
    y[i] = tmp + beta*y[i];
  }
}


int get_starting_point(const hiop_size_type n, double* x0, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  int i = 0;
  for(i=0; i<user_data->n; i=i+1) x0[i]=1.;
  return 0;
}

int get_prob_sizes(hiop_size_type* n_, hiop_size_type* m_, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  *n_ = user_data->n;
  *m_ = user_data->m;
  return 0;
} 

int get_vars_info(hiop_size_type n, double *xlow_, double* xupp_, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  hiop_size_type i = 0;
  for(i=0; i<user_data->n; i=i+1) xlow_[i] = user_data->xlow[i];
  for(i=0; i<user_data->n; i=i+1) xupp_[i] = user_data->xupp[i];
  return 0;
}

int get_cons_info(hiop_size_type m, double *clow_, double* cupp_, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  hiop_size_type i = 0;
  for(i=0; i<user_data->m; i=i+1) clow_[i] = user_data->clow[i];
  for(i=0; i<user_data->m; i=i+1) cupp_[i] = user_data->cupp[i];
  return 0;
}

int eval_f(hiop_size_type n, double* x, int new_x, double* obj, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  int i = 0;
  *obj = 0.;//x[0]*(x[0]-1.);
  //sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
  for(i=0; i<user_data->ns; i=i+1) *obj += x[i]*(x[i]-1.);
  *obj = *obj * 0.5;

  double term2=0.;
  const double* y = x+2*user_data->ns;
  timesVec(user_data->Q, 0.0, user_data->buf_y, 1., y, user_data->nd, user_data->nd);
  for(i=0; i<user_data->nd; i=i+1) term2 = term2 + user_data->buf_y[i] * y[i];
  *obj = *obj + 0.5*term2;
  
  const double* s=x+user_data->ns;
  double term3=0.;//s[0]*s[0];
  for(i=0; i<user_data->ns; i=i+1) term3 += s[i]*s[i];
  *obj = *obj + 0.5*term3;
  return 0;
}

int eval_grad_f(hiop_size_type n, double* x, int new_x, double* gradf, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  int i = 0;
  //x_i - 0.5 
  for(hiop_size_type  i=0; i<n; ++i) gradf[i]=0.0;
  for(i=0; i<user_data->ns; i=i+1) gradf[i] = x[i]-0.5;

  //Qd*y
  const double* y = x+2*user_data->ns;
  double* gradf_y = gradf+2*user_data->ns;
  timesVec(user_data->Q, 0.0, gradf_y, 1., y, user_data->nd, user_data->nd);

  //s
  const double* s=x+user_data->ns;
  double* gradf_s = gradf+user_data->ns;
  for(i=0; i<user_data->ns; i=i+1) gradf_s[i] = s[i];
  return 0;
}

int eval_cons(hiop_size_type n, hiop_size_type m,
    double* x, int new_x, 
    double* cons, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  assert(3+user_data->ns == m);
  const double* s = x+user_data->ns;
  const double* y = x+2*user_data->ns;

  for(int con_idx=0; con_idx<user_data->m; ++con_idx) {
    if(con_idx<user_data->ns) {
      //equalities
      cons[con_idx] = x[con_idx]+s[con_idx];
    } else {
      //inequalties
      assert(con_idx<user_data->ns+3);
      if(con_idx==user_data->ns) {
        cons[con_idx] = x[0];
        for(int i=0; i<user_data->ns; i=i+1) cons[con_idx] += s[i];
        for(int i=0; i<user_data->nd; i=i+1) cons[con_idx] += y[i];

      } else if(con_idx==user_data->ns+1) {
        cons[con_idx] = x[1];
        for(int i=0; i<user_data->nd; i=i+1) cons[con_idx] += y[i];
      } else if(con_idx==user_data->ns+2) {
        cons[con_idx] = x[2];
        for(int i=0; i<user_data->nd; i=i+1) cons[con_idx] += y[i];
      } else { assert(0); }
    }
  }

    // apply Md to y and add the result to equality part of 'cons'

    //we know that equalities are the first ns constraints so this should work
    timesVec(user_data->Md, 1.0, cons, 1.0, y, user_data->nd, user_data->ns );
    return 0;

}
int get_sparse_dense_blocks_info(hiop_size_type* nx_sparse, hiop_size_type* nx_dense,
    hiop_size_type* nnz_sparse_Jaceq, hiop_size_type* nnz_sparse_Jacineq,
    hiop_size_type* nnz_sparse_Hess_Lagr_SS, 
    hiop_size_type* nnz_sparse_Hess_Lagr_SD, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  *nx_sparse = user_data->nx_sparse;
  *nx_dense = user_data->nx_dense;
  *nnz_sparse_Jaceq = user_data->nnz_sparse_Jaceq;
  *nnz_sparse_Jacineq = user_data->nnz_sparse_Jacineq;
  *nnz_sparse_Hess_Lagr_SS = user_data->nnz_sparse_Hess_Lagr_SS;
  *nnz_sparse_Hess_Lagr_SD = user_data->nnz_sparse_Hess_Lagr_SD;
  return 0;
}

int eval_Jac_cons(hiop_size_type n, hiop_size_type m,
    double* x, int new_x,
    hiop_size_type nsparse, hiop_size_type ndense, 
    hiop_size_type nnzJacS, hiop_index_type* iJacS, hiop_index_type* jJacS, double* MJacS, 
    double* JacD, void* user_data_) {
  settings* user_data = (settings*) user_data_;
  assert(m==user_data->ns+3);

  if(iJacS!=NULL && jJacS!=NULL) {
    int nnzit=0;
    for(int con_idx=0; con_idx<user_data->ns; ++con_idx) {
      //sparse Jacobian eq w.r.t. x and s
      //x
      iJacS[nnzit] = con_idx;
      jJacS[nnzit] = con_idx;
      nnzit=nnzit+1;
      
      //s
      iJacS[nnzit] = con_idx;
      jJacS[nnzit] = con_idx+user_data->ns;
      nnzit=nnzit+1;
    }
    if(user_data->ns>0) {
      for(int con_idx=user_data->ns; con_idx<user_data->m; ++con_idx) {
        //sparse Jacobian ineq w.r.t x and s
        if(con_idx==user_data->ns) {
          //w.r.t x_1
          iJacS[nnzit] = con_idx;
          jJacS[nnzit] = 0;
          nnzit=nnzit+1;
          //w.r.t s
          for(int i=0; i<user_data->ns; i=i+1) {
            iJacS[nnzit] = con_idx;
            jJacS[nnzit] = user_data->ns+i;
            nnzit=nnzit+1;
          }
        } else {
          if(con_idx-user_data->ns==1 || con_idx-user_data->ns==2) {
            //w.r.t x_2 or x_3
            iJacS[nnzit] = con_idx;
            jJacS[nnzit] = con_idx-user_data->ns;
            nnzit=nnzit+1;
          } else { assert(0); }
        }
      }
    }
    assert(nnzit==nnzJacS);
  }
  //values for sparse Jacobian if requested by the solver
  if(MJacS!=NULL) {
    int nnzit=0;
    for(int con_idx=0; con_idx<user_data->ns; ++con_idx) {
      //sparse Jacobian EQ w.r.t. x and s
      //x
      MJacS[nnzit] = 1.;
      nnzit=nnzit+1;
      
      //s
      MJacS[nnzit] = 1.;
      nnzit=nnzit+1;
    }
    if(user_data->ns>0) {
      for(int con_idx=user_data->ns; con_idx<user_data->m; ++con_idx) {
        //sparse Jacobian INEQ w.r.t x and s
        if(con_idx-user_data->ns==0) {
          //w.r.t x_1
          MJacS[nnzit] = 1.;
          nnzit=nnzit+1;
          //w.r.t s
          for(int i=0; i<user_data->ns; i=i+1) {
            MJacS[nnzit] = 1.;
            nnzit=nnzit+1;
          }
        } else {
          if(con_idx-user_data->ns==1 || con_idx-user_data->ns==2) {
            //w.r.t x_2 or x_3
            MJacS[nnzit] = 1.;
            nnzit=nnzit+1;
          } else { assert(0); }
        }
      }
    }
    assert(nnzit==nnzJacS);
  }
  
  //dense Jacobian w.r.t y
  if(JacD!=NULL) {
    //just copy the dense Jacobian corresponding to equalities
    memcpy(JacD, user_data->Md, user_data->ns*user_data->nd*sizeof(double));
    
    assert(user_data->ns+3 == m);
    //do an in place fill-in for the ineq Jacobian corresponding to e^T
    for(int i=0; i<3*user_data->nd; ++i) JacD[i+user_data->nd*user_data->ns] = 1.;
  }
  return 0;
}

int eval_Hess_Lagr(hiop_size_type n, hiop_size_type m,
                   double* x, int new_x, double obj_factor,
                   double* lambda, int new_lambda,
                   hiop_size_type nsparse, hiop_size_type ndense, 
                   hiop_size_type nnzHSS, hiop_index_type* iHSS, hiop_index_type* jHSS, double* MHSS, 
                   double* HDD,
                   hiop_size_type nnzHSD, hiop_index_type* iHSD, hiop_index_type* jHSD, double* MHSD,
                   void* user_data_) {
  settings* user_data = (settings*) user_data_;
  //Note: lambda is not used since all the constraints are linear and, therefore, do 
    //not contribute to the Hessian of the Lagrangian

  assert(nnzHSS==2*user_data->ns);
  assert(nnzHSD==0);
  assert(iHSD==NULL); assert(jHSD==NULL); assert(MHSD==NULL);

  if(iHSS!=NULL && jHSS!=NULL) {
    for(int i=0; i<2*user_data->ns; i=i+1) iHSS[i] = jHSS[i] = i;     
  }

  if(MHSS!=NULL) {
    for(int i=0; i<2*user_data->ns; i=i+1) MHSS[i] = obj_factor;
  }

  if(HDD!=NULL) {
    const hiop_size_type nx_dense_squared = user_data->nd*user_data->nd;
    memcpy(HDD, user_data->Q, nx_dense_squared*sizeof(double));
    for(int i=0; i<nx_dense_squared; i=i+1) HDD[i] = obj_factor*user_data->Q[i];
  }
  return 0;
}


int main(int argc, char **argv) {
  int rank=0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size;
  int ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size); assert(MPI_SUCCESS==ierr);
  //int ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank); assert(MPI_SUCCESS==ierr);
  if(comm_size != 1) {
    printf("[error] driver detected more than one rank but the driver should be run "
           "in serial only; will exit\n");
    MPI_Finalize();
    return 1;
  }
#endif

  hiop_size_type ns = 400;
  hiop_size_type nd = 100;
  int i;

  // println("ns: $ns, nd: $nd")

  double* Q = (double*)malloc(nd*nd*sizeof(double));
  for(i=0; i<nd*nd; i=i+1) Q[i] = 1e-8;
  for(i=0; i<nd; i=i+1) Q[i + i*nd] += 2.0;
  for(i=1; i<nd-1; i=i+1) {
    Q[i+1 + i*nd] += 1.0;
    Q[i + (i+1)*nd] += 1.0;
  }

  double* Md = (double*)malloc(ns*nd*sizeof(double));
  for(i=0; i<ns*nd; i=i+1) Md[i] = -1.0;
  double* buf_y = (double*)malloc(nd*sizeof(double));
  for(i=0; i<nd; i=i+1) buf_y[i] = 0.0;

  hiop_size_type n = 2*ns + nd;
  hiop_size_type m = ns+3;
  hiop_size_type nx_sparse = 2*ns;
  hiop_size_type nx_dense = nd;
  hiop_size_type nnz_sparse_Jaceq = 2*ns;
  hiop_size_type nnz_sparse_Jacineq = 3+ns;
  hiop_size_type nnz_sparse_Hess_Lagr_SS = 2*ns;
  hiop_size_type nnz_sparse_Hess_Lagr_SD = 0;

  double* xlow = (double*)malloc(n*sizeof(double));
  double* xupp = (double*)malloc(n*sizeof(double));

  double* clow = (double*)malloc(m*sizeof(double));
  double* cupp = (double*)malloc(m*sizeof(double));

  for(i=0; i<ns; i=i+1) xlow[i] = -1e+20;
  for(i=ns; i<2*ns; i=i+1) xlow[i] = 0.;
  xlow[2*ns] = -4.;
  for(i=2*ns+1; i<n; ++i) xlow[i] = -1e+20;

  for(i=0; i<ns; ++i) xupp[i] = 3.;
  for(i=ns; i<2*ns; ++i) xupp[i] = +1e+20;
  xupp[2*ns] = 4.;
  for(i=2*ns+1; i<n; ++i) xupp[i] = +1e+20;

  for(i=0; i<m; i=i+1) clow[i] = 0.0;
  for(i=0; i<m; i=i+1) cupp[i] = 0.0;
  clow[m-3] = -2.;    cupp[m-3] = 2.;
  clow[m-2] = -1e+20; cupp[m-2] = 2.;
  clow[m-1] = -2.;    cupp[m-1] = 1e+20;


  settings user_data = {n, m, ns, nd, nx_sparse, nx_dense, nnz_sparse_Jaceq, nnz_sparse_Jacineq,
                        nnz_sparse_Hess_Lagr_SS, nnz_sparse_Hess_Lagr_SD,
                        xlow, xupp, clow, cupp,
                        Q,  Md, buf_y};
                        
  cHiopMDSProblem problem;
  problem.user_data = &user_data;
  problem.get_starting_point = get_starting_point;
  problem.get_prob_sizes = get_prob_sizes;  
  problem.get_vars_info = get_vars_info;
  problem.get_cons_info = get_cons_info;
  problem.eval_f = eval_f;
  problem.eval_grad_f = eval_grad_f;
  problem.eval_cons = eval_cons;
  problem.get_sparse_dense_blocks_info = get_sparse_dense_blocks_info;
  problem.eval_Jac_cons = eval_Jac_cons;
  problem.eval_Hess_Lagr = eval_Hess_Lagr;
  problem.solution = (double*)malloc(n * sizeof(double));
  for(int i=0; i<n; i++) problem.solution[i] = 0.0;
  
  hiop_mds_create_problem(&problem);
  hiop_mds_solve_problem(&problem);
  if(fabs(problem.obj_value-(-4.999509728895e+01))>1e-6) {
    printf("objective mismatch for MDS Ex1 C interface problem with 400 sparse variables and 100 "
      "dense variables did. BTW, obj=%18.12e was returned by HiOp.\n", problem.obj_value);
      return -1;
  }
  hiop_mds_destroy_problem(&problem);
  free(problem.solution);
  free(xlow); free(xupp);
  free(clow); free(cupp);
  free(Q); free(Md); free(buf_y);
#ifdef HIOP_USE_MAGMA
  magma_finalize();
#endif
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
