// -*- C++ -*-

#ifndef V3D_SVD_UTILITIES_H
#define V3D_SVD_UTILITIES_H

#include "Math/v3d_linear.h"
#include "Math/v3d_linear_tnt.h"

// Some applications of the SVD for minimization.

namespace V3D
{

   //! Minimize ||Ax|| s.t. ||x||=1. From H&Z, 2nd ed., p593 (Alg. A5.4).
   template <typename Mat, typename Vec>
   inline void
   minimize_norm_Ax_st_norm_x_eq_1(Mat const& A, Vec& x)
   {
      assert(x.size() == A.num_cols());

      typedef typename Vec::value_type Num;

      SVD<double> svd(A);

      Matrix<double> const& V = svd.getV();

      int const lastCol = V.num_cols() - 1;
      for (size_t i = 0; i < x.size(); ++i) x[i] = V[i][lastCol];
   } // end minimize_norm_Ax_st_norm_x_eq_1()

   //! \brief Minimize ||Ax|| s.t. ||x||=1 and x = Gy, where G has rank r.
   //! From H&Z, p566 (Alg. A3.7).
//    template <typename Mat1, typename Mat2, typename Vec>
//    inline void
//    minimize_norm_Ax_st_norm_x_eq_1_and_x_eq_Gy(Mat1 const& A, Mat2 const& G, unsigned r, Vec& x)
//    {
//       using namespace std;

//       assert(x.size() == A.num_cols());

//       typedef typename Vec::value_type Num;

//       DynMatrixContainer<Num> U(G.num_rows(), G.num_cols());
//       DynMatrixContainer<Num> Vt(G.num_cols(), G.num_cols());
//       DynVectorContainer<Num> S(G.num_cols());
//       copyMatrix(G, U);
//       SVDecomposition(U, S, Vt);
//       DynMatrixContainer<Num> U1(G.num_rows(), r);

//       // we search the r non-zero (largest) entries of S
//       unsigned dstCol = 0;
//       for (unsigned i = 0; i < r; ++i)
//       {
//          unsigned maxPos = maximumVectorElement(S);
//          copySubMatrix(U, 0, maxPos, U.num_rows(), 1, U1, 0, dstCol);
//          ++dstCol;
//          S[maxPos] = -1; // Singular values are always > 0.
//       } // end for (i)
//       DynMatrixContainer<Num> AU1(A.num_rows(), U1.num_cols());
//       multiplyMatrices(A, U1, AU1);
//       DynVectorContainer<Num> x1(AU1.num_cols());
//       minimize_norm_Ax_st_norm_x_eq_1(AU1, x1);
//       multiplyMatrixVector(U1, x1, x);
//    } // end minimize_norm_Ax_st_norm_x_eq_1_and_x_eq_Gy()

   //! \brief Minimize ||Ax|| s.t. ||x||=1 and Cx = 0, where C has rank r.
   //! From H&Z, 2nd ed., p594 (Alg. A5.5).
   template <typename MatA, typename MatC, typename Vec>
   inline void
   minimize_norm_Ax_st_norm_x_eq_1_and_Cx_eq_0(MatA const& A, MatC const& C, unsigned r, Vec& x)
   {
      using namespace std;

      assert(x.size() == A.num_cols());

      typedef typename Vec::value_type Num;

      Matrix<Num> C0(std::max(C.num_rows(), C.num_cols()), C.num_cols(), 0.0);
      copyMatrixSlice(C, 0, 0, C.num_rows(), C.num_cols(), C0, 0, 0);

      SVD<Num> svdC(C0);
      Matrix<Num> const& V_ = svdC.getV();

      // The orthogonal complement matrix
      Matrix<Num> C_(C.num_cols(), C.num_cols()-r);
      for (unsigned j = 0; j < C_.num_cols(); ++j)
         for (unsigned i = 0; i < C_.num_rows(); ++i)
            C_[i][j] = V_[i][j+r];

      Matrix<Num> AC_(A.num_rows(), C_.num_cols());
      multiply_A_B(A, C_, AC_);

      Vector<Num> x1(AC_.num_cols());
      minimize_norm_Ax_st_norm_x_eq_1(AC_, x1);
      multiply_A_v(C_, x1, x);
   } // end minimize_norm_Ax_st_norm_x_eq_1_and_Cx_eq_0()

} // end namespace V3D

#endif
