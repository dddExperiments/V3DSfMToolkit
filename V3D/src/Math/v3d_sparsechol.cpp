#include "Math/v3d_sparsechol.h"
#include "Math/v3d_linear.h"

#include <iostream>

#if defined(V3DLIB_ENABLE_SUITESPARSE)
# include "colamd.h"
extern "C"
{
#include "ldl.h"
}

using namespace std;

namespace V3D
{

   SparseCholesky::SparseCholesky()
      : _isGood(false)
   {
   }

   SparseCholesky::SparseCholesky(CCS_Matrix<double> const& AtA, bool verbose)
      : _isGood(false)
   {
      this->initialize(AtA, verbose);
   } // end SparseCholesky::SparseCholesky()

   void
   SparseCholesky::initialize(CCS_Matrix<double> const& AtA, bool verbose)
   {
      int const nColumns = AtA.num_cols();
      int * colStarts = const_cast<int *>(AtA.getColumnStarts());
      int * rowIdxs   = const_cast<int *>(AtA.getRowIndices());

      int const nnzAtA = AtA.getNonzeroCount();

      _permAtA.resize(nColumns+1);
      _invPermAtA.resize(nColumns+1);

      if (nnzAtA > 0)
      {
         int stats[COLAMD_STATS];
         symamd(nColumns, rowIdxs, colStarts, &_permAtA[0], (double *) NULL, stats, &calloc, &free);
         if (verbose) symamd_report(stats);
      }
      else
      {
         for (int k = 0; k < _permAtA.size(); ++k) _permAtA[k] = k;
         for (int k = 0; k < _permAtA.size(); ++k) _invPermAtA[k] = k;
      } // end if

      vector<int> workFlags(nColumns);
      _AtA_Lp.resize(nColumns+1);
      _AtA_Parent.resize(nColumns);
      _AtA_Lnz.resize(nColumns);

      ldl_symbolic(nColumns, colStarts, rowIdxs,
                   &_AtA_Lp[0], &_AtA_Parent[0], &_AtA_Lnz[0], &workFlags[0], &_permAtA[0], &_invPermAtA[0]);

      if (verbose) cout << "SparseCholesky::SparseCholesky(): Nonzeros in LDL decomposition: "
                        << _AtA_Lp[nColumns] << endl;
   } // end SparseCholesky::initialize()


   bool
   SparseCholesky::setAtA(CCS_Matrix<double> const& AtA)
   {
      int const nColumns = AtA.num_cols();
      int * colStarts = const_cast<int *>(AtA.getColumnStarts());
      int * rowIdxs   = const_cast<int *>(AtA.getRowIndices());
      double * values = const_cast<double *>(AtA.getValues());

      int const lnz = _AtA_Lp[nColumns];
      //showSparseMatrixInfo(AtA);

      vector<double> Y(nColumns);
      vector<int> workPattern(nColumns), workFlag(nColumns);

      _D.resize(nColumns);
      _Li.resize(lnz);
      _Lx.resize(lnz);

      int const d = ldl_numeric(nColumns, colStarts, rowIdxs, values,
                                &_AtA_Lp[0], &_AtA_Parent[0], &_AtA_Lnz[0],
                                &_Li[0], &_Lx[0], &_D[0],
                                &Y[0], &workPattern[0], &workFlag[0],
                                &_permAtA[0], &_invPermAtA[0]);

      _isGood = (d == nColumns);
      return _isGood;
   } // end SparseCholesky::setAtA()

   bool
   SparseCholesky::solve(VectorBase<double> const& rhs_, VectorBase<double>& X)
   {
      if (!_isGood)
      {
         cerr << "SparseCholesky::solve(): Previous LDL decomposition (setAtA) failed." << endl;
         return false;
      }

      int const nColumns = rhs_.size();

      X.newsize(nColumns);

      VectorBase<double> rhs(nColumns);
      copyVector(rhs_, rhs);

      VectorBase<double> rhsP(nColumns);

      ldl_perm(nColumns, &rhsP[0], &rhs[0], &_permAtA[0]);
      ldl_lsolve(nColumns, &rhsP[0], &_AtA_Lp[0], &_Li[0], &_Lx[0]);
      ldl_dsolve(nColumns, &rhsP[0], &_D[0]);
      ldl_ltsolve(nColumns, &rhsP[0], &_AtA_Lp[0], &_Li[0], &_Lx[0]);
      ldl_permt(nColumns, &X[0], &rhsP[0], &_permAtA[0]);

      return true;
   } // end SparseCholesky::solve()

} // end namespace V3D

#endif // defined(V3DLIB_ENABLE_SUITESPARSE)
