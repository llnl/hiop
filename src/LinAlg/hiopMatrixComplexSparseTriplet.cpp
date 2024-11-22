#include "hiopMatrixComplexSparseTriplet.hpp"

#include "hiop_blasdefs.hpp"

#include "hiopMatrixComplexDense.hpp"

#include <iostream>
#include <sstream>
#include <iostream>  // std::cout, std::fixed
#include <iomanip>   // std::setprecision

namespace hiop
{
hiopMatrixComplexSparseTriplet::hiopMatrixComplexSparseTriplet(int rows, int cols, int nnz)
{
  stM = new hiopMatrixSparseTripletStorage<int, std::complex<double> >(rows, cols, nnz);
}
hiopMatrixComplexSparseTriplet::~hiopMatrixComplexSparseTriplet() { delete stM; }

hiopMatrix* hiopMatrixComplexSparseTriplet::alloc_clone() const
{
  return new hiopMatrixComplexSparseTriplet(stM->m(), stM->n(), stM->numberOfNonzeros());
}

hiopMatrix* hiopMatrixComplexSparseTriplet::new_copy() const
{
  assert(false);
  return NULL;
}

void hiopMatrixComplexSparseTriplet::setToZero()
{
  auto* values = stM->M();
  for(int i = 0; i < stM->numberOfNonzeros(); i++) values[i] = 0.;
}
void hiopMatrixComplexSparseTriplet::setToConstant(double c)
{
  auto* values = stM->M();
  for(int i = 0; i < stM->numberOfNonzeros(); i++) values[i] = c;
}
void hiopMatrixComplexSparseTriplet::setToConstant(std::complex<double> c)
{
  auto* values = stM->M();
  for(int i = 0; i < stM->numberOfNonzeros(); i++) values[i] = c;
}

double hiopMatrixComplexSparseTriplet::max_abs_value()
{
  char norm = 'M';
  int one = 1, nnz = stM->numberOfNonzeros();
  hiop::dcomplex* M = reinterpret_cast<dcomplex*>(stM->M());

  double maxv = ZLANGE(&norm, &one, &nnz, M, &one, NULL);
  return maxv;
}

void hiopMatrixComplexSparseTriplet::timesVec(double beta,
                                              std::complex<double>* y,
                                              double alpha,
                                              const std::complex<double>* x) const
{
  int nrows = stM->m();
  // y= beta*y
  if(beta != 0.) {
    for(int i = 0; i < nrows; i++) {
      y[i] *= beta;
    }
  } else {
    for(int i = 0; i < nrows; i++) {
      y[i] = 0.;
    }
  }

  auto* values = stM->M();
  int* iRow = stM->i_row();
  int* jCol = stM->j_col();
  int nnz = stM->numberOfNonzeros();
  int ncols = stM->n();
  // y += alpha*this*x
  for(int i = 0; i < nnz; i++) {
    assert(iRow[i] < nrows);
    assert(jCol[i] < ncols);
    y[iRow[i]] += alpha * x[jCol[i]] * values[i];
  }
}

/* W = beta*W + alpha*this^T*X
 *
 * Only supports W and X of the type 'hiopMatrixComplexDense'
 */
void hiopMatrixComplexSparseTriplet::transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
{
  assert(m() == X.m());
  assert(n() == W.m());
  assert(W.n() == X.n());

  hiopMatrixComplexDense* Wd = dynamic_cast<hiopMatrixComplexDense*>(&W);
  if(Wd == NULL) {
    std::cerr << "hiopMatrixComplexSparseTriplet::transTimesMat received an unsuported type (1)\n";
    return;
  }

  const hiopMatrixComplexDense* Xd = dynamic_cast<const hiopMatrixComplexDense*>(&X);
  if(Xd == NULL) {
    std::cerr << "hiopMatrixComplexSparseTriplet::transTimesMat received an unsuported type (2)\n";
    return;
  }

  auto* W_M = Wd->get_M();
  const auto* X_M = Xd->local_data();  // same as get_M but with has const qualifier

  if(beta == 0.) {
    Wd->setToZero();
  } else {
    int N = W.m() * W.n();
    dcomplex zalpha;
    zalpha.re = beta;
    zalpha.im = 0.;
    int one = 1;
    ZSCAL(&N, &zalpha, reinterpret_cast<dcomplex*>(*W_M), &one);
  }

  int* this_irow = storage()->i_row();
  int* this_jcol = storage()->j_col();
  std::complex<double>* this_M = storage()->M();
  int nnz = numberOfNonzeros();

  for(int it = 0; it < nnz; it++) {
    const std::complex<double> aux = alpha * this_M[it];
    for(int j = 0; j < X.n(); j++) {
      W_M[this_jcol[it]][j] += aux * X_M[this_irow[it]][j];
    }
  }
}

hiopMatrixComplexSparseTriplet* hiopMatrixComplexSparseTriplet::new_slice(const int* row_idxs,
                                                                          int nrows,
                                                                          const int* col_idxs,
                                                                          int ncols) const
{
  int* src_i = this->storage()->i_row();
  int* src_j = this->storage()->j_col();
  //
  // count nnz first
  //
  int dest_nnz = 0, src_itnz = 0, src_nnz = this->stM->numberOfNonzeros();
  for(int ki = 0; ki < nrows; ki++) {
    const int& row = row_idxs[ki];
    assert(row < m());
#ifndef NDEBUG
    if(ki > 0) {
      assert(row_idxs[ki] > row_idxs[ki - 1] && "slice row indexes need to be increasingly ordered");
    }
#endif

    while(src_itnz < src_nnz && src_i[src_itnz] < row) src_itnz++;

    for(int kj = 0; kj < ncols; kj++) {
      const int& col = col_idxs[kj];
      assert(col < n());

#ifndef NDEBUG
      if(kj > 0) {
        assert(col_idxs[kj] > col_idxs[kj - 1] && "slice column indexes need to be increasingly ordered");
      }
#endif
      while(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] < col) {
        src_itnz++;
      }

      if(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] == col) {
        // std::complex<double>* src_M  = this->storage()->M();
        // printf("[%d,%d] -> %g+%g*i  (1)\n", ki, kj, src_M[src_itnz].real(), src_M[src_itnz].imag());
        dest_nnz++;
        src_itnz++;
      }
    }
  }
  assert(src_itnz <= src_nnz);
  assert(src_itnz >= dest_nnz);

  const int dest_nnz2 = dest_nnz;
  hiopMatrixComplexSparseTriplet* newMat = new hiopMatrixComplexSparseTriplet(nrows, ncols, dest_nnz2);
  //
  // populate the new slice matrix
  //
  // first pass -> populate with elements on the upper triangle of 'this'
  int* dest_i = newMat->storage()->i_row();
  int* dest_j = newMat->storage()->j_col();
  std::complex<double>* dest_M = newMat->storage()->M();
  std::complex<double>* src_M = this->storage()->M();

  dest_nnz = 0;
  src_itnz = 0;
  for(int ki = 0; ki < nrows; ki++) {
    const int& row = row_idxs[ki];

    while(src_itnz < src_nnz && src_i[src_itnz] < row) src_itnz++;

    for(int kj = 0; kj < ncols; kj++) {
      const int& col = col_idxs[kj];

      while(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] < col) {
        src_itnz++;
      }
      if(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] == col) {
        dest_i[dest_nnz] = ki;
        dest_j[dest_nnz] = kj;
        dest_M[dest_nnz] = src_M[src_itnz];

        // printf("[%d,%d] -> %g+%g*i  (2)\n", ki, kj, src_M[src_itnz].real(), src_M[src_itnz].imag());

        dest_nnz++;
        src_itnz++;
        assert(dest_nnz <= dest_nnz2);
      }
    }
  }
  assert(src_itnz <= src_nnz);
  assert(dest_nnz == dest_nnz2);

  newMat->storage()->sort_indexes();
  return newMat;
}

// builds submatrix nrows x ncols with rows and cols specified by row_idxs and cols_idx
// assumes the 'this' is symmetric
hiopMatrixComplexSparseTriplet* hiopMatrixComplexSparseTriplet::new_sliceFromSym(const int* row_idxs,
                                                                                 int nrows,
                                                                                 const int* col_idxs,
                                                                                 int ncols) const
{
  int* src_i = this->storage()->i_row();
  int* src_j = this->storage()->j_col();

  // count nnz first
  int dest_nnz = 0, src_itnz = 0, src_nnz = this->stM->numberOfNonzeros();
  for(int ki = 0; ki < nrows; ki++) {
    const int& row = row_idxs[ki];
    assert(row < m());
#ifndef NDEBUG
    if(ki > 0) {
      assert(row_idxs[ki] > row_idxs[ki - 1] && "slice row indexes need to be increasingly ordered");
    }
#endif

    while(src_itnz < src_nnz && src_i[src_itnz] < row) src_itnz++;

    for(int kj = 0; kj < ncols; kj++) {
      const int& col = col_idxs[kj];
      assert(col < n());

#ifndef NDEBUG
      if(kj > 0) {
        assert(col_idxs[kj] > col_idxs[kj - 1] && "slice column indexes need to be increasingly ordered");
      }
#endif
      // we won't have entry (row, col) with row>col since 'this' is upper triangular. Avoid the extra
      // checks below and continue the for loop
      if(col < row) continue;

      while(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] < col) {
        src_itnz++;
      }

      if(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] == col) {
        assert(row <= col);
        dest_nnz++;
        src_itnz++;
      }
    }
  }
  assert(src_itnz <= src_nnz);
  assert(src_itnz >= dest_nnz);

  // one more iteration over, like above looking in the lower triangular this time
  // this can be done by changing the order of the for(ki) and for(kj) loops
  src_itnz = 0;  // reinitialize
  for(int kj = 0; kj < ncols; kj++) {
    const int& row = col_idxs[kj];
    assert(row < m());

    while(src_itnz < src_nnz && src_i[src_itnz] < row) src_itnz++;

    for(int ki = 0; ki < nrows; ki++) {
      const int& col = row_idxs[ki];
      assert(col < n());

      // we won't have entry (row, col) with row>col since 'this' is upper triangular
      // also entries for which row==col were already counted in the above loop
      // so avoid the extra checks below and continue the for loop
      if(col <= row) continue;

      while(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] < col) {
        src_itnz++;
      }

      if(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] == col) {
        // only lower triangle elements; the others were added in the previous loop
        assert(row < col);
        dest_nnz++;
        // step one elem further
        src_itnz++;
      }
    }
  }
  assert(src_itnz <= src_nnz);

  const int dest_nnz2 = dest_nnz;
  hiopMatrixComplexSparseTriplet* newMat = new hiopMatrixComplexSparseTriplet(nrows, ncols, dest_nnz2);

  // populate the new slice matrix

  // first pass -> populate with elements on the upper triangle of 'this'
  int* dest_i = newMat->storage()->i_row();
  int* dest_j = newMat->storage()->j_col();
  std::complex<double>* dest_M = newMat->storage()->M();
  std::complex<double>* src_M = this->storage()->M();

  dest_nnz = 0;
  src_itnz = 0;
  for(int ki = 0; ki < nrows; ki++) {
    const int& row = row_idxs[ki];

    while(src_itnz < src_nnz && src_i[src_itnz] < row) src_itnz++;

    for(int kj = ki; kj < ncols; kj++) {
      const int& col = col_idxs[kj];

      if(col < row) continue;

      while(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] < col) {
        src_itnz++;
      }
      if(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] == col) {
        assert(row <= col);
        dest_i[dest_nnz] = ki;
        dest_j[dest_nnz] = kj;
        dest_M[dest_nnz] = src_M[src_itnz];

        dest_nnz++;
        src_itnz++;
        assert(dest_nnz <= dest_nnz2);
      }
    }
  }
  assert(src_itnz <= src_nnz);

  // second pass -> populate with elements on the upper triangle of 'this'
  src_itnz = 0;
  for(int kj = 0; kj < ncols; kj++) {
    const int& row = col_idxs[kj];

    while(src_itnz < src_nnz && src_i[src_itnz] < row) src_itnz++;

    for(int ki = 0; ki < nrows; ki++) {
      const int& col = row_idxs[ki];

      if(col <= row) continue;

      while(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] < col) {
        src_itnz++;
      }

      if(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] == col) {
        assert(row < col);
        dest_i[dest_nnz] = ki;
        dest_j[dest_nnz] = kj;
        dest_M[dest_nnz] = src_M[src_itnz];

        dest_nnz++;
        src_itnz++;
        assert(dest_nnz <= dest_nnz2);
      }
    }
  }
  assert(src_itnz <= src_nnz);

  newMat->storage()->sort_indexes();

  return newMat;
}

// extract a symmetric matrix (only upper triangle is stored)
hiopMatrixComplexSparseTriplet* hiopMatrixComplexSparseTriplet::new_sliceFromSymToSym(const int* row_col_idxs,
                                                                                      int ndim) const
{
  int* src_i = this->storage()->i_row();
  int* src_j = this->storage()->j_col();

  int dest_nnz = 0, src_itnz = 0, src_nnz = this->stM->numberOfNonzeros();
  for(int ki = 0; ki < ndim; ki++) {
    const int& row = row_col_idxs[ki];
    assert(row < m());
#ifndef NDEBUG
    if(ki > 0) {
      assert(row_col_idxs[ki] > row_col_idxs[ki - 1] && "slice indexes need to be increasingly ordered");
    }
#endif

    while(src_itnz < src_nnz && src_i[src_itnz] < row) src_itnz++;

    for(int kj = ki; kj < ndim; kj++) {
      const int& col = row_col_idxs[kj];
      assert(col < n());

      while(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] < col) {
        src_itnz++;
      }

      if(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] == col) {
        dest_nnz++;
        src_itnz++;
      }
    }
  }
  assert(src_itnz <= src_nnz);
  assert(src_itnz >= dest_nnz);

  hiopMatrixComplexSparseTriplet* newMat = new hiopMatrixComplexSparseTriplet(ndim, ndim, dest_nnz);

  int* dest_i = newMat->storage()->i_row();
  int* dest_j = newMat->storage()->j_col();
  std::complex<double>* dest_M = newMat->storage()->M();
  std::complex<double>* src_M = this->storage()->M();

  dest_nnz = 0;
  src_itnz = 0;
  for(int ki = 0; ki < ndim; ki++) {
    const int& row = row_col_idxs[ki];

    while(src_itnz < src_nnz && src_i[src_itnz] < row) src_itnz++;

    for(int kj = ki; kj < ndim; kj++) {
      const int& col = row_col_idxs[kj];

      while(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] < col) {
        src_itnz++;
      }
      if(src_itnz < src_nnz && src_i[src_itnz] == row && src_j[src_itnz] == col) {
        dest_i[dest_nnz] = ki;
        dest_j[dest_nnz] = kj;
        assert(dest_i[dest_nnz] <= dest_j[dest_nnz]);
        dest_M[dest_nnz] = src_M[src_itnz];
        dest_nnz++;
        src_itnz++;
      }
    }
  }
  assert(src_itnz <= src_nnz);
  assert(src_itnz >= dest_nnz);

  return newMat;
}

void hiopMatrixComplexSparseTriplet::print(FILE* file,
                                           const char* msg /*=NULL*/,
                                           int maxRows /*=-1*/,
                                           int maxCols /*=-1*/,
                                           int rank /*=-1*/) const
{
  int myrank_ = 0, numranks = 1;  // this is a local object => always print

  int max_elems = maxRows >= 0 ? maxRows : stM->numberOfNonzeros();
  max_elems = std::min(max_elems, stM->numberOfNonzeros());

  if(file == NULL) file = stdout;

  if(myrank_ == rank || rank == -1) {
    std::stringstream ss;
    if(NULL == msg) {
      if(numranks > 1) {
        fprintf(file,
                "matrix of size %d %d and nonzeros %d, printing %d elems (on rank=%d)\n",
                m(),
                n(),
                numberOfNonzeros(),
                max_elems,
                myrank_);
        ss << "matrix of size " << m() << " " << n() << " and nonzeros " << numberOfNonzeros() << ", printing " << max_elems
           << " elems (on rank=" << myrank_ << ")" << std::endl;
      } else {
        // fprintf(file, "matrix of size %d %d and nonzeros %d, printing %d elems\n",
        //	  m(), n(), numberOfNonzeros(), max_elems);
        ss << "matrix of size " << m() << " " << n() << " and nonzeros " << numberOfNonzeros() << ", printing " << max_elems
           << " elems" << std::endl;
      }
    } else {
      ss << msg << " ";
      // fprintf(file, "%s ", msg);
    }

#ifdef AAAAA
    // output matlab indices and input format
    fprintf(file, "iRow=[");
    for(int it = 0; it < max_elems; it++) fprintf(file, "%d; ", stM->irow[it] + 1);
    fprintf(file, "];\n");

    fprintf(file, "jCol=[");
    for(int it = 0; it < max_elems; it++) fprintf(file, "%d; ", stM->jcol[it] + 1);
    fprintf(file, "];\n");

    fprintf(file, "v=[");
    for(int it = 0; it < max_elems; it++)
      // fprintf(file, "%22.16e+%22.16ei; ", stM->values[it].real(), stM->values[it].imag());
      fprintf(file, "%.6g+%.6gi; ", stM->values[it].real(), stM->values[it].imag());
    fprintf(file, "];\n");

#else
    for(int it = 0; it < max_elems; it++) {
      // fprintf(file,
      //         "[%3d,%3d] = %.6g+%.6gi\n",
      //         stM->irow[it]+1,
      //         stM->jcol[it]+1,
      //         stM->values[it].real(),
      //         stM->values[it].imag());
      ss << "[" << stM->irow[it] + 1 << "," << stM->jcol[it] + 1 << " = " << std::setprecision(6) << stM->values[it].real()
         << "+" << stM->values[it].imag() << ";" << std::endl;
    }
    fprintf(file, "%s", ss.str().c_str());
#endif
  }
}

}  // namespace hiop
