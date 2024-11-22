#ifndef MATRIX_SSTRIP_COMPLEX
#define MATRIX_SSTRIP_COMPLEX

#include <vector>
#include <complex>

#include <numeric>
#include <algorithm>

#include <cstring>
#include <cassert>

namespace hiop
{

// container for sparse matrices in triplet format; implements minimal functionality for matrix ops
template<class Tidx, class Tval>
class hiopMatrixSparseTripletStorage
{
public:
  hiopMatrixSparseTripletStorage()
      : nrows(0),
        ncols(0),
        nnz(0),
        irow(NULL),
        jcol(NULL),
        values(NULL)
  {}
  hiopMatrixSparseTripletStorage(Tidx num_rows, Tidx num_cols, Tidx num_nz)
      : nrows(num_rows),
        ncols(num_cols),
        nnz(num_nz)
  {
    irow = new Tidx[nnz];
    jcol = new Tidx[nnz];
    values = new Tval[nnz];
  }

  virtual ~hiopMatrixSparseTripletStorage()
  {
    if(values) delete[] values;
    if(jcol) delete[] jcol;
    if(irow) delete[] irow;
  }

  void copyFrom(const Tidx* irow_, const Tidx* jcol_, const Tval* values_)
  {
    memcpy(irow, irow_, nnz * sizeof(Tidx));
    memcpy(jcol, jcol_, nnz * sizeof(Tidx));
    memcpy(values, values_, nnz * sizeof(Tval));
  }

  // sorts the (i,j) in increasing order of 'i' and for equal 'i's in increasing order of 'j'
  // Complexity: n*log(n)
  //
  //  Warning: irow, jcol, and values pointers will changes inside this method. Corresponding
  //  accessor methods i(), j(), M() should be called again to get the correct pointers
  void sort_indexes()
  {
    std::vector<Tidx> vIdx(nnz);
    std::iota(vIdx.begin(), vIdx.end(), 0);
    sort(vIdx.begin(), vIdx.end(), [&](const int& i1, const int& i2) {
      if(irow[i1] < irow[i2]) return true;
      if(irow[i1] > irow[i2]) return false;
      return jcol[i1] < jcol[i2];
    });

    // permute irow, jcol, and M using additional storage

    // irow and jcol can use the same  buffer
    {
      Tidx* buffer = new Tidx[nnz];
      for(int itnz = 0; itnz < nnz; itnz++) buffer[itnz] = irow[vIdx[itnz]];

      // avoid copy back
      Tidx* buffer2 = irow;
      irow = buffer;
      buffer = buffer2;

      for(int itnz = 0; itnz < nnz; itnz++) buffer[itnz] = jcol[vIdx[itnz]];

      delete[] jcol;
      jcol = buffer;
    }

    // M
    {
      Tval* buffer = new Tval[nnz];

      for(int itnz = 0; itnz < nnz; itnz++) buffer[itnz] = values[vIdx[itnz]];

      delete[] values;
      values = buffer;
    }
  }

  // add elements with identical (i,j) and update nnz, irow, jcol, and values array accordingly
  // Precondition: (irow,jcol) are assumed to be sorted (see sort_indexes())
  void sum_up_duplicates()
  {
    if(nnz <= 0) return;
    Tidx itleft = 0, itright = 1;
    Tidx currI = irow[0], currJ = jcol[0];
    Tval val1 = values[0];

    while(itright <= nnz) {
      values[itleft] = val1;

      while(itright < nnz && irow[itright] == currI && jcol[itright] == currJ) {
        values[itleft] += values[itright];
        itright++;
      }
      irow[itleft] = currI;
      jcol[itleft] = currJ;

      if(itright < nnz) {
        currI = irow[itright];
        currJ = jcol[itright];
        val1 = values[itright];
      }
      itright++;
      itleft++;
      assert(itleft <= nnz);
      assert(itleft < itright);
      // printf("-- itright=%d\n", itright);
    }

    nnz = itleft;
  }

  inline Tidx m() const { return nrows; }
  inline Tidx n() const { return ncols; }
  inline Tidx numberOfNonzeros() const { return nnz; }
  inline Tidx* i_row() const { return irow; }
  inline Tidx* j_col() const { return jcol; }
  inline Tval* M() const { return values; }

protected:
  friend class hiopMatrixComplexSparseTriplet;

  Tidx nrows, ncols, nnz;

  Tidx *irow, *jcol;
  Tval* values;
};
}  // namespace hiop
#endif
