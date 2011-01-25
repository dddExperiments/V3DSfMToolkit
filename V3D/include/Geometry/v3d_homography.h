// -*- C++ -*-
#ifndef V3D_HOMOGRAPHY_H
#define V3D_HOMOGRAPHY_H

#include "Math/v3d_linear.h"
#include "Math/v3d_mathutilities.h"
#include "Geometry/v3d_mviewutilities.h"

namespace V3D
{

   Matrix3x3d computeHomographyLinearUnnormalized(std::vector<PointCorrespondence> const& corrs);
   Matrix3x3d computeHomographyLinear(std::vector<PointCorrespondence> const& corrs);
   Matrix3x3d computeHomographyLinear(double const f0, Vector2f const& pp0, std::vector<PointCorrespondence> const& corrs);

   bool computeRobustHomographyMLE(std::vector<PointCorrespondence> const& corrs,
                                   double inlierThreshold, int nRounds,
                                   Matrix3x3d& bestH, std::vector<int>& inliers);

//**********************************************************************

   inline Matrix3x3d
   getPanoramicHomography(double const f, Matrix3x3d const& R)
   {
      // H = K*R*inv(K)
      Matrix3x3d H;
      copyMatrix(R, H);
      H[0][2] *= f; H[1][2] *= f;
      H[2][0] /= f; H[2][1] /= f;
      return H;
   }

   inline Matrix3x3d
   getPanoramicHomography(double const f, Vector2d const& pp, Matrix3x3d const& R)
   {
      // H = K*R*inv(K)
      Matrix3x3d K; makeIdentityMatrix(K);
      K[0][0] = K[1][1] = f;
      K[0][2] = pp[0];
      K[1][2] = pp[1];

      Matrix3x3d invK; makeIdentityMatrix(invK);
      invK[0][0] = invK[1][1] = 1.0 / f;
      invK[0][2] = -pp[0] / f;
      invK[1][2] = -pp[1] / f;

      return K*R*invK;
   }

   // Compute the homography H = K*H*inv(K) with unknown, but common focal length from 2 point correspondences
   // Note: the image measurements must be centered w.r.t the principal point.
   bool computePanoramicHomography(Vector2f const& p1, Vector2f const& p2, // left points
                                   Vector2f const& q1, Vector2f const& q2, // right points
                                   std::vector<double>& fs, std::vector<Matrix3x3d>& Rs);

   bool computeRobustPanoramicHomographyMLE(std::vector<PointCorrespondence> const& corrs,
                                            double inlierThreshold, int nRounds,
                                            double& best_f, Matrix3x3d& bestR, std::vector<int>& inliers);

   bool refinePanoramicHomography(std::vector<Vector2d> const& left_, std::vector<Vector2d> const& right_,
                                  double& f, Vector2d& pp, Matrix3x3d& R, double inlierThreshold, int what = 0);

   inline bool
   refinePanoramicHomography(std::vector<PointCorrespondence> const& corrs,
                             double& f, Vector2d& pp, Matrix3x3d& R,
                             double inlierThreshold, int what = 0)
   {
      using namespace std;
      using namespace V3D;

      int const N = corrs.size();
      vector<Vector2d> left(N), right(N);
      for (int i = 0; i < N; ++i)
      {
         copyVector(corrs[i].left.pos, left[i]);
         copyVector(corrs[i].right.pos, right[i]);
      }
      return refinePanoramicHomography(left, right, f, pp, R, inlierThreshold, what);
   } // end refinePanoramicHomography()

   template <typename Mat3x3>
   inline double
   getFocalLengthFromPanoramicHomography(Mat3x3 const& H_)
   {
      assert(H_.num_rows() == 3);
      assert(H_.num_cols() == 3);

      Matrix<double> H(3, 3);
      copyMatrix(H_, H);

      // Scale H such that the second singular value is 1.
      SVD<double> svd(H);
      scaleMatrixIP(1.0/svd.getSingularValues()[1], H);

      // Use the image of the dual absolute conic
      double h00 = H[0][0], h01 = H[0][1], h02 = H[0][2];
      double h10 = H[1][0], h11 = H[1][1], h12 = H[1][2];
      double h20 = H[2][0], h21 = H[2][1], h22 = H[2][2];

      vector<double> c(9), rhs(9);
      rhs[0] = sqr(h02); rhs[1] = h02*h12;  rhs[2] = h02*h22;
      rhs[3] = h02*h12;  rhs[4] = sqr(h12); rhs[5] = h12*h22;
      rhs[6] = h02*h22;  rhs[7] = h12*h22;  rhs[8] = sqr(h22)-1;

      c[0] = sqr(h01)+sqr(h00)-1; c[1] = h01*h11+h00*h10;     c[2] = h01*h21+h00*h20;
      c[3] = h01*h11+h00*h10;     c[4] = sqr(h11)+sqr(h10)-1; c[5] = h11*h21+h10*h20;
      c[6] = h01*h21+h00*h20;     c[7] = h11*h21+h10*h20;     c[8] = sqr(h21)+sqr(h20);

      double f = 0.0;
      for (int i = 0; i < 9; ++i) f -= rhs[i] / c[i];
      f /= 9.0;
      f = sqrt(f);
      return f;
   } // end getFocalLengthFromPanoramicHomography()

   bool refineHomography(std::vector<Vector2d> const& left_, std::vector<Vector2d> const& right_,
                         double inlierThreshold, Matrix3x3d& H);

   inline bool
   refineHomography(std::vector<PointCorrespondence> const& corrs,
                    double inlierThreshold, Matrix3x3d& H)
   {
      using namespace std;
      using namespace V3D;

      int const N = corrs.size();
      vector<Vector2d> left(N), right(N);
      for (int i = 0; i < N; ++i)
      {
         copyVector(corrs[i].left.pos, left[i]);
         copyVector(corrs[i].right.pos, right[i]);
      }
      return refineHomography(left, right, inlierThreshold, H);
   } // end refineHomography()

   Matrix3x3d getHomographyCompatibleWithFundamental(Matrix3x3d const& F, std::vector<PointCorrespondence> const& corrs);

   bool computeRobustCompatibleHomographyMLE(Matrix3x3d const&F,
                                             std::vector<PointCorrespondence> const& corrs,
                                             double inlierThreshold, int nRounds,
                                             Matrix3x3d& bestH, std::vector<int>& inliers);

} // end namespace V3D

#endif
