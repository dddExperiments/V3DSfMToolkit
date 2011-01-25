// -*- C++ -*-

#ifndef V3D_SPARSE_EIG_H
#define V3D_SPARSE_EIG_H

#include "Math/v3d_linearbase.h"

namespace V3D
{

#if defined(V3DLIB_ENABLE_ARPACK)

   enum
   {
      V3D_ARPACK_LARGEST_EIGENVALUES = 0,
      V3D_ARPACK_SMALLEST_EIGENVALUES = 1,
      V3D_ARPACK_LARGEST_MAGNITUDE_EIGENVALUES = 2,
      V3D_ARPACK_SMALLEST_MAGNITUDE_EIGENVALUES = 3,
   };

   struct SparseSymmetricEigConfig
   {
         SparseSymmetricEigConfig()
            : tolerance(0.0), maxArnoldiIterations(300), nColumnsV(-1)
         { }

         double tolerance;
         int    maxArnoldiIterations;
         int    nColumnsV;
   }; // end struct SparseSymmetricEigConfig

   bool computeSparseSymmetricEig(CCS_Matrix<double> const& A, int mode, int nWanted,
                                  VectorBase<double>& lambda, MatrixBase<double>& U,
                                  SparseSymmetricEigConfig cfg = SparseSymmetricEigConfig());

   // Compute right singular vectors and values through eig(AtA).
   bool computeSparseSVD(CCS_Matrix<double> const& A, int mode, int nWanted,
                         VectorBase<double>& sigma, MatrixBase<double>& V,
                         SparseSymmetricEigConfig cfg = SparseSymmetricEigConfig());

   inline double
   matrixNorm_L2(CCS_Matrix<double> const& A)
   {
      using namespace V3D;

      VectorBase<double> sv(1);
      MatrixBase<double> V(1, A.num_rows());

      bool status = computeSparseSVD(A, V3D_ARPACK_LARGEST_MAGNITUDE_EIGENVALUES, 1, sv, V);
      return sv[0];
   } // end matrixNorm_L2()

#endif

} // end namespace V3D

#endif
