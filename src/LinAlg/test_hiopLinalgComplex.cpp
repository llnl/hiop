#include "hiopMatrixComplexSparseTriplet.hpp"
#include "hiopMatrixComplexDense.hpp"

#include <iostream>

using namespace hiop;

int main()
{
  bool all_tests_ok = true;
  {  // TEST sparse complex matrix
    // this is a rectangular matrix to test with
    //  [ 1+i      0      0      1-i        0     ]
    //  [  0       0      i       0         1     ]
    //  [ 2-i      0      0    1.5-0.5i  0.5-0.5i ]
    //
    int m = 3, n = 5;
    int Mrow[] = {0, 0, 1, 1, 2, 2, 2};
    int Mcol[] = {0, 3, 2, 4, 0, 3, 4};
    std::complex<double> Mval[] = {{1, 1}, {1, -1}, {0, 1}, {1, 0}, {2, -1}, {1.5, -0.5}, {0.5, -0.5}};

    int nnz = sizeof(Mrow) / sizeof(Mrow[0]);
    assert(nnz == sizeof Mcol / sizeof Mcol[0]);
    assert(nnz == sizeof Mval / sizeof Mval[0]);

    hiopMatrixComplexSparseTriplet mat(m, n, nnz);
    mat.copyFrom(Mrow, Mcol, Mval);

    // test1
    double abs_nrm = mat.max_abs_value();
    double diff = std::fabs(abs_nrm - 2.23606797749979);
    if(diff > 1e-12) {
      printf("error: max_abs_value did not return the correct value. Difference: %6.3e\n", diff);
      all_tests_ok = false;
    }
    // mat.print();

    mat.storage()->sort_indexes();
    // mat.print();

    mat.storage()->sum_up_duplicates();
    double abs_nrm2 = mat.max_abs_value();
    diff = std::fabs(abs_nrm2 - abs_nrm);
    if(diff > 1e-15) {
      printf("error: postprocessing check failed\n");
      all_tests_ok = false;
    }

    // slicing -> row and cols idxs need to be sorted
    std::vector<int> rows = {1}, cols = {1, 2};
    auto* subMat = mat.new_slice(rows.data(), rows.size(), cols.data(), cols.size());
    // subMat->print();
    if(subMat->numberOfNonzeros() != 1) {
      printf("error: new_slice did not return the correct nnz.\n");
      all_tests_ok = false;
    }
    delete subMat;

    rows = {1};
    cols = {1, 2};
    subMat = mat.new_slice(rows.data(), rows.size(), cols.data(), cols.size());
    abs_nrm = subMat->max_abs_value();
    diff = std::fabs(abs_nrm - 1.0);
    if(diff > 1e-12) {
      printf("error: check of 'new_slice' failed. Difference: %6.3e [should be %20.16e]\n", diff, abs_nrm);
      all_tests_ok = false;
    }
    delete subMat;
  }

  {  // TEST dense complex matrix
    hiopMatrixComplexDense mat(3, 4);
    std::complex<double>** M = mat.get_M();
    for(int i = 0; i < mat.m(); i++)
      for(int j = 0; j < mat.n(); j++) M[i][j] = std::complex<double>(i, j);

    // hiopMatrixComplexDense mat2(3,4);
    // std::complex<double>** M2 = mat2.get_M();
    // for(int i=0; i<mat2.m(); i++)
    //  for(int j=0; j<mat2.n(); j++)
    //	M2[i][j] = std::complex<double>(i,-2.*j);

    // mat.print();
    // mat2.print();
    // mat2.addMatrix(std::complex<double>(1.,0), mat);

    // mat2.print();

    // test1
    double abs_nrm = mat.max_abs_value();
    double diff = std::fabs(abs_nrm - 3.605551275463989);
    if(diff > 1e-12) {
      printf("error: max_abs_value did not return the correct value. Difference: %6.3e [should be %20.16e]\n",
             diff,
             abs_nrm);
      return -1;
    }
  }

  // test for sparse complex symmetric
  {
    int n = 5;
    int Mrow[] = {0, 0, 0, 1, 2, 2, 3, 3, 4};
    int Mcol[] = {0, 2, 3, 2, 2, 3, 3, 4, 4};
    std::complex<double> Mval[] = {{1, 1}, {1, -1}, {0.001, 3}, {1, 2}, {2, 2}, {2, 3.333}, {3, 3}, {3, 4}, {4, 4}};
    int nnz = sizeof(Mrow) / sizeof(Mrow[0]);
    assert(nnz == sizeof Mcol / sizeof Mcol[0]);
    assert(nnz == sizeof Mval / sizeof Mval[0]);

    hiopMatrixComplexSparseTriplet mat(n, n, nnz);
    mat.copyFrom(Mrow, Mcol, Mval);

    // mat.print();

    std::vector<int> idxs = {1, 2, 4};
    hiopMatrixComplexSparseTriplet* submat_sym = mat.new_sliceFromSymToSym(idxs.data(), idxs.size());
    // submat_sym->print();
    // test2
    double abs_nrm = submat_sym->max_abs_value();
    double diff = std::fabs(abs_nrm - 5.6568542494923806);
    if(diff > 1e-12) {
      printf("error: check of '.new_sliceFromSymToSym' failed. Difference: %6.3e [should be %20.16e]\n", diff, abs_nrm);
      all_tests_ok = false;
    }
    delete submat_sym;

    std::vector<int> idxs_row = {0, 2, 3};
    std::vector<int> idxs_col = {1, 2, 4};
    auto* submat_gen = mat.new_sliceFromSym(idxs_row.data(), idxs_row.size(), idxs_col.data(), idxs_col.size());
    // submat_gen->print();
    // test2
    abs_nrm = submat_gen->max_abs_value();
    diff = std::fabs(abs_nrm - 5.0);
    if(diff > 1e-12) {
      printf("error: check of 'new_sliceFromSym' failed. Difference: %6.3e [should be %20.16e]\n", diff, abs_nrm);
      all_tests_ok = false;
    }
    delete submat_gen;
  }

  if(all_tests_ok) printf("All checks passed\n");
  return 0;
}
