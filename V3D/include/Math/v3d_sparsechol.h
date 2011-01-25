// -*- C++ -*-

#ifndef V3D_SPARSE_CHOL_H
#define V3D_SPARSE_CHOL_H

#include "Math/v3d_linearbase.h"

namespace V3D
{

# if defined(V3DLIB_ENABLE_SUITESPARSE)

   struct SparseCholesky
   {
         SparseCholesky();
         SparseCholesky(CCS_Matrix<double> const& AtA, bool verbose = false);

         void initialize(CCS_Matrix<double> const& AtA, bool verbose = false);

         bool setAtA(CCS_Matrix<double> const& AtA);
         bool solve(VectorBase<double> const& rhs, VectorBase<double>& X);

         std::vector<int> const& getPermutation() const { return _permAtA; }
         std::vector<int> const& getInvPermutation() const { return _invPermAtA; }

      protected:
         std::vector<int> _permAtA, _invPermAtA;
         std::vector<int> _AtA_Lp, _AtA_Parent, _AtA_Lnz;

         std::vector<int>    _Li;
         std::vector<double> _Lx, _D;

         bool _isGood;
   }; // end struct SparseCholesky

# endif

} // end namespace V3D


#endif
