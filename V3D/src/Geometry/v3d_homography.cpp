#include "v3d_homography.h"

#include "Math/v3d_linear_tnt.h"
#include "Math/v3d_linear_lu.h"
#include "Math/v3d_optimization.h"
#include "Math/v3d_svdutilities.h"

#ifdef WIN32 //HA
#include <float.h>
#define isfinite(x) _finite(x)
#endif

using namespace V3D;
using namespace std;

namespace
{

   inline int
   getShuffleCount(int N, int nRounds, int minSampleSize)
   {
      return std::max(1, (int)((float)nRounds*(float)minSampleSize/(float)N + 0.5));
   }

//**********************************************************************

   struct PanoramicHomographyMinimizer : public V3D::SimpleLevenbergOptimizer
   {
         enum { OPTIMIZE_R = 0, OPTIMIZE_R_F = 1, OPTIMIZE_R_F_PP = 2 };

         PanoramicHomographyMinimizer(Matrix3x3d const R, double f, Vector2d const& pp,
                                      vector<Vector2d> const& left, vector<Vector2d> const& right,
                                      double inlierThreshold, int optimizerLevel)
            : V3D::SimpleLevenbergOptimizer(2*left.size(), 6), // R, f, ppx, ppy
              _left(left), _right(right),
              _inlierThreshold(inlierThreshold), _optimizerLevel(optimizerLevel)
         {
            fillVector(0.0, observation);
            _currentR   = R;
            _current_f  = f;
            _current_pp = pp;
            _currentH = getPanoramicHomography(f, pp, R);
         }

         virtual void evalFunction(Vector<double>& res)
         {
            Vector2d Hp;

            for (size_t k = 0; k < _left.size(); ++k)
            {
               Vector2d const& p = _left[k];
               Vector2d const& q = _right[k];

               multiply_A_v_projective(_currentH, p, Hp);

               res[2*k+0] = Hp[0] - q[0];
               res[2*k+1] = Hp[1] - q[1];
            } // end for (k)
         } // end evalFunction()

         virtual void fillWeights(Vector<double>& w)
         {
            vector<double> es(_left.size());

            Vector2d Hp;

            for (unsigned int k = 0; k < _left.size(); ++k)
            {
               Vector2d const& p = _left[k];
               Vector2d const& q = _right[k];

               multiply_A_v_projective(_currentH, p, Hp);

               double const e = norm_L2(q - Hp);
               w[2*k+0] = w[2*k+1] = (e < _inlierThreshold) ? 1.0 : sqrt(_inlierThreshold / e);
               es[k] = e;
            } // end for (k)
         }

         virtual void fillJacobian(Matrix<double>& J)
         {
            Matrix3x3d const& R = _currentR;

            Matrix<double> innerDer(9, 6, 0.0); // dE/dOmega and dE/df
            makeZeroMatrix(innerDer);

#if 0
            double const f      = _current_f;
            double const f2     = f*f;

            // dH/dOmega_x
            innerDer[0][0] = 0; innerDer[1][0] = R[0][2];   innerDer[2][0] = -f*R[0][1];
            innerDer[3][0] = 0; innerDer[4][0] = R[1][2];   innerDer[5][0] = -f*R[1][1];
            innerDer[6][0] = 0; innerDer[7][0] = R[2][2]/f; innerDer[8][0] =   -R[2][1];

            // dH/dOmega_y
            innerDer[0][1] = -R[0][2];   innerDer[1][1] = 0; innerDer[2][1] = f*R[0][0];
            innerDer[3][1] = -R[1][2];   innerDer[4][1] = 0; innerDer[5][1] = f*R[1][0];
            innerDer[6][1] = -R[2][2]/f; innerDer[7][1] = 0; innerDer[8][1] =   R[2][0];

            // dH/dOmega_z
            innerDer[0][2] = R[0][1];   innerDer[1][2] = -R[0][0];   innerDer[2][2] = 0;
            innerDer[3][2] = R[1][1];   innerDer[4][2] = -R[1][0];   innerDer[5][2] = 0;
            innerDer[6][2] = R[2][1]/f; innerDer[7][2] = -R[2][0]/f; innerDer[8][2] = 0;

            // dH/df
            if (_optimizerLevel >= OPTIMIZE_R_F)
            {
               innerDer[0][3] = 0;           innerDer[1][3] = 0;           innerDer[2][3] = R[0][2];
               innerDer[3][3] = 0;           innerDer[4][3] = 0;           innerDer[5][3] = R[1][2];
               innerDer[6][3] = -R[2][0]/f2; innerDer[7][3] = -R[2][1]/f2; innerDer[8][3] = 0;
            }
#else
            double const f   = _current_f;
            double const ppx = _current_pp[0];
            double const ppy = _current_pp[1];

            double const rf = 1.0 / f;
            double const rppx = ppx*rf;
            double const rppy = ppy*rf;

            double const r00 = R[0][0], r01 = R[0][1], r02 = R[0][2];
            double const r10 = R[1][0], r11 = R[1][1], r12 = R[1][2];
            double const r20 = R[2][0], r21 = R[2][1], r22 = R[2][2];

            double const rf2 = rf*rf;

            // dH/dOmega_x
            innerDer[0][0] = 0; innerDer[1][0] = ppx*r22*rf+f*r02*rf; innerDer[2][0] = ppx*(r22*rppy-r21)+f*(r02*rppy-r01);
            innerDer[3][0] = 0; innerDer[4][0] = ppy*r22*rf+f*r12*rf; innerDer[5][0] = ppy*(r22*rppy-r21)+f*(r12*rppy-r11);
            innerDer[6][0] = 0; innerDer[7][0] = r22*rf;              innerDer[8][0] = r22*rppy-r21;

            // dH/dOmega_y
            innerDer[0][1] = -ppx*r22*rf-f*r02*rf; innerDer[1][1] = 0; innerDer[2][1] = ppx*(r20-r22*rppx)+f*(r00-r02*rppx);
            innerDer[3][1] = -ppy*r22*rf-f*r12*rf; innerDer[4][1] = 0; innerDer[5][1] = ppy*(r20-r22*rppx)+f*(r10-r12*rppx);
            innerDer[6][1] = -r22*rf;              innerDer[7][1] = 0; innerDer[8][1] = r20-r22*rppx;

            // dH/dOmega_z
            innerDer[0][2] = ppx*r21*rf+f*r01*rf; innerDer[1][2] = -ppx*r20*rf-f*r00*rf; innerDer[2][2] = ppx*(r21*rppx-r20*rppy)+f*(r01*rppx-r00*rppy);
            innerDer[3][2] = ppy*r21*rf+f*r11*rf; innerDer[4][2] = -ppy*r20*rf-f*r10*rf; innerDer[5][2] = ppy*(r21*rppx-r20*rppy)+f*(r11*rppx-r10*rppy);
            innerDer[6][2] = r21*rf;              innerDer[7][2] = -r20*rf;              innerDer[8][2] = r21*rppx-r20*rppy;

            if (_optimizerLevel >= OPTIMIZE_R_F)
            {
               // dH/df
               innerDer[0][3] = -ppx*r20*rf2; innerDer[1][3] = -ppx*r21*rf2; innerDer[2][3] = ppx*(ppy*r21*rf2+ppx*r20*rf2)+r02+f*(ppy*r01*rf2+ppx*r00*rf2)-ppy*r01/f-ppx*r00/f;
               innerDer[3][3] = -ppy*r20*rf2; innerDer[4][3] = -ppy*r21*rf2; innerDer[5][3] = ppy*(ppy*r21*rf2+ppx*r20*rf2)+r12+f*(ppy*r11*rf2+ppx*r10*rf2)-ppy*r11/f-ppx*r10/f;
               innerDer[6][3] = -r20*rf2;     innerDer[7][3] = -r21*rf2;     innerDer[8][3] = ppy*r21*rf2+ppx*r20*rf2;

               if (_optimizerLevel >= OPTIMIZE_R_F_PP)
               {
                  // dH/dppx
                  innerDer[0][4] = r20*rf; innerDer[1][4] = r21*rf; innerDer[2][4] = r22-ppy*r21/f-2*ppx*r20/f-r00;
                  innerDer[3][4] = 0;      innerDer[4][4] = 0;      innerDer[5][4] = -ppy*r20/f-r10;
                  innerDer[6][4] = 0;      innerDer[7][4] = 0;      innerDer[8][4] = -r20*rf;

                  // dH/dppy
                  innerDer[0][5] = 0;      innerDer[1][5] = 0;      innerDer[2][5] = -ppx*r21/f-r01;
                  innerDer[3][5] = r20*rf; innerDer[4][5] = r21*rf; innerDer[5][5] = r22-2*ppy*r21/f-ppx*r20/f-r11;
                  innerDer[6][5] = 0;      innerDer[7][5] = 0;      innerDer[8][5] = -r21*rf;
               } // end if
            } // end if
#endif

            Matrix<double> outerDer(2, 9); // dq/dH
            Matrix<double> derivative(2, 6);

            Vector3d Hp;
            for (size_t k = 0; k < _left.size(); ++k)
            {
               Vector2d const& p = _left[k];
               Vector2d const& q = _right[k];

               multiply_A_v_affine(_currentH, p, Hp);

               double const z = 1.0 / Hp[2];
               double const z2 = z*z;

               outerDer[0][0] = p[0]*z;         outerDer[0][1] = p[1]*z;         outerDer[0][2] = z;
               outerDer[0][3] = 0;              outerDer[0][4] = 0;              outerDer[0][5] = 0;
               outerDer[0][6] = -p[0]*Hp[0]*z2; outerDer[0][7] = -p[1]*Hp[0]*z2; outerDer[0][8] = -Hp[0]*z2;

               outerDer[1][0] = 0;              outerDer[1][1] = 0;              outerDer[1][2] = 0;
               outerDer[1][3] = p[0]*z;         outerDer[1][4] = p[1]*z;         outerDer[1][5] = z;
               outerDer[1][6] = -p[0]*Hp[1]*z2; outerDer[1][7] = -p[1]*Hp[1]*z2; outerDer[1][8] = -Hp[1]*z2;

               multiply_A_B(outerDer, innerDer, derivative);
               //scaleMatrixIP(-1.0, derivative);

               for (int c = 0; c < 6; ++c) J[2*k+0][c] = derivative[0][c];
               for (int c = 0; c < 6; ++c) J[2*k+1][c] = derivative[1][c];
            } // end for (k)
         } // end fillJacobian()

         virtual double getParameterLength() const
         {
            // R is a rotation matrix, i.e. with Frobenious norm 3.
            return 3.0 + fabs(_current_f);
         }

         virtual void updateCurrentParameters(Vector<double> const& delta)
         {
            Vector3d omega(delta[0], delta[1], delta[2]);

            Matrix3x3d const oldR(_currentR);

            Matrix3x3d dR;
            createRotationMatrixRodrigues(omega, dR);
            _currentR = oldR * dR;

            if (_optimizerLevel >= OPTIMIZE_R_F) _current_f  += delta[3];
            if (_optimizerLevel >= OPTIMIZE_R_F_PP) { _current_pp[0] += delta[4]; _current_pp[1] += delta[5]; }

            _currentH = getPanoramicHomography(_current_f, _current_pp, _currentR);
         }

         virtual void saveCurrentParameters()
         {
            _savedR   = _currentR;
            _saved_f  = _current_f;
            _saved_pp = _current_pp;
         }

         virtual void restoreCurrentParameters()
         {
            _currentR   = _savedR;
            _current_f  = _saved_f;
            _current_pp = _saved_pp;
            _currentH   = getPanoramicHomography(_current_f, _current_pp, _currentR);
         }

         vector<Vector2d> const& _left, _right;

         Matrix3x3d _currentR, _savedR;
         double     _current_f, _saved_f;
         Vector2d   _current_pp, _saved_pp;
         Matrix3x3d _currentH;

         double _inlierThreshold;
         int    _optimizerLevel;
   }; // end struct PanoramicHomographyMinimizer

   struct HomographyMinimizer : public V3D::SimpleLevenbergOptimizer
   {
         HomographyMinimizer(Matrix3x3d const H, vector<Vector2d> const& left, vector<Vector2d> const& right,
                                      double inlierThreshold)
            : V3D::SimpleLevenbergOptimizer(2*left.size(), 9),
              _left(left), _right(right),
              _inlierThreshold(inlierThreshold)
         {
            fillVector(0.0, observation);
            
            _currentH = H;
            this->normalizeCurrentH();
         }

         virtual void evalFunction(Vector<double>& res)
         {
            Vector2d Hp;

            for (size_t k = 0; k < _left.size(); ++k)
            {
               Vector2d const& p = _left[k];
               Vector2d const& q = _right[k];

               multiply_A_v_projective(_currentH, p, Hp);

               res[2*k+0] = Hp[0] - q[0];
               res[2*k+1] = Hp[1] - q[1];
            } // end for (k)
         } // end evalFunction()

         virtual void fillWeights(Vector<double>& w)
         {
            vector<double> es(_left.size());

            Vector2d Hp;

            for (unsigned int k = 0; k < _left.size(); ++k)
            {
               Vector2d const& p = _left[k];
               Vector2d const& q = _right[k];

               multiply_A_v_projective(_currentH, p, Hp);

               double const e = norm_L2(q - Hp);
               w[2*k+0] = w[2*k+1] = (e < _inlierThreshold) ? 1.0 : sqrt(_inlierThreshold / e);
               es[k] = e;
            } // end for (k)
         }

         virtual void fillJacobian(Matrix<double>& J)
         {
            Matrix3x3d const& H = _currentH;

            Matrix<double> derivative(2, 9); // dq/dH

            Vector3d Hp;
            for (size_t k = 0; k < _left.size(); ++k)
            {
               Vector2d const& p = _left[k];
               Vector2d const& q = _right[k];

               multiply_A_v_affine(_currentH, p, Hp);

               double const z = 1.0 / Hp[2];
               double const z2 = z*z;

               derivative[0][0] = p[0]*z;         derivative[0][1] = p[1]*z;         derivative[0][2] = z;
               derivative[0][3] = 0;              derivative[0][4] = 0;              derivative[0][5] = 0;
               derivative[0][6] = -p[0]*Hp[0]*z2; derivative[0][7] = -p[1]*Hp[0]*z2; derivative[0][8] = -Hp[0]*z2;

               derivative[1][0] = 0;              derivative[1][1] = 0;              derivative[1][2] = 0;
               derivative[1][3] = p[0]*z;         derivative[1][4] = p[1]*z;         derivative[1][5] = z;
               derivative[1][6] = -p[0]*Hp[1]*z2; derivative[1][7] = -p[1]*Hp[1]*z2; derivative[1][8] = -Hp[1]*z2;

               //scaleMatrixIP(-1.0, derivative);

               for (int c = 0; c < 9; ++c) J[2*k+0][c] = derivative[0][c];
               for (int c = 0; c < 9; ++c) J[2*k+1][c] = derivative[1][c];
            } // end for (k)
         } // end fillJacobian()

         virtual double getParameterLength() const
         {
            // We keep H normalized, so it is 1.
            return 1.0;
         }

         virtual void updateCurrentParameters(Vector<double> const& delta)
         {
            _currentH[0][0] += delta[0]; _currentH[0][1] += delta[1]; _currentH[0][2] += delta[2];
            _currentH[1][0] += delta[3]; _currentH[1][1] += delta[4]; _currentH[1][2] += delta[5];
            _currentH[2][0] += delta[6]; _currentH[2][1] += delta[7]; _currentH[2][2] += delta[8];
            this->normalizeCurrentH();
         }

         virtual void saveCurrentParameters()
         {
            _savedH = _currentH;
         }

         virtual void restoreCurrentParameters()
         {
            _currentH = _savedH;
         }

         void normalizeCurrentH()
         {
            double len2 = 0.0;
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j)
                  len2 += sqr(_currentH[i][j]);
            scaleMatrixIP(1.0/sqrt(len2), _currentH);
         }

         vector<Vector2d> const& _left, _right;

         Matrix3x3d _currentH, _savedH;

         double _inlierThreshold;
   }; // end struct HomographyMinimizer

} // end namespace <>

namespace V3D
{

   Matrix3x3d
   computeHomographyLinearUnnormalized(std::vector<PointCorrespondence> const& corrs)
   {
      int const N = corrs.size();
      Matrix<double> A(3*N, 9, 0.0);

      for (int i = 0; i < N; ++i)
      {
         Vector2f const p = corrs[i].left.pos;
         Vector2f const q = corrs[i].right.pos;

         A[3*i+0][3] = -p[0];
         A[3*i+0][4] = -p[1];
         A[3*i+0][5] = -1.0;
         A[3*i+0][6] = q[1]*p[0];
         A[3*i+0][7] = q[1]*p[1];
         A[3*i+0][8] = q[1];

         A[3*i+1][0] = -p[0];
         A[3*i+1][1] = -p[1];
         A[3*i+1][2] = -1.0;
         A[3*i+1][6] = q[0]*p[0];
         A[3*i+1][7] = q[0]*p[1];
         A[3*i+1][8] = q[0];

         A[3*i+2][0] = -q[1]*p[0];
         A[3*i+2][1] = -q[1]*p[1];
         A[3*i+2][2] = -q[1];
         A[3*i+2][3] = q[0]*p[0];
         A[3*i+2][4] = q[0]*p[1];
         A[3*i+2][5] = q[0];
      } // end for (i)

      SVD<double> svd(A);
      Matrix<double> const& V = svd.getV();

      Matrix3x3d H;
      H[0][0] = V[0][8]; H[0][1] = V[1][8]; H[0][2] = V[2][8];
      H[1][0] = V[3][8]; H[1][1] = V[4][8]; H[1][2] = V[5][8];
      H[2][0] = V[6][8]; H[2][1] = V[7][8]; H[2][2] = V[8][8];
      return H;
   } // end computeHomographyLinearUnnormalized()

   Matrix3x3d
   computeHomographyLinear(std::vector<PointCorrespondence> const& corrs_)
   {
      int const N = corrs_.size();

      Vector2f centerL(0.0, 0.0);
      Vector2f centerR(0.0, 0.0);

      for (int i = 0; i < N; ++i)
      {
         addVectors(centerL, corrs_[i].left.pos, centerL);
         addVectors(centerR, corrs_[i].right.pos, centerR);
      }
      scaleVectorIP(1.0/N, centerL);
      scaleVectorIP(1.0/N, centerR);

      vector<PointCorrespondence> corrs(N);
      for (int i = 0; i < N; ++i)
      {
         corrs[i].left.pos  = corrs_[i].left.pos - centerL;
         corrs[i].right.pos = corrs_[i].right.pos - centerR;
      }

      double varL = 0.0, varR = 0.0;

      for (int i = 0; i < N; ++i)
      {
         varL += sqrNorm_L2(corrs[i].left.pos);
         varR += sqrNorm_L2(corrs[i].right.pos);
      }
      varL /= N; varR /= N;
      double const scaleL = 1.0 / sqrt(varL);
      double const scaleR = 1.0 / sqrt(varR);

      for (int i = 0; i < N; ++i)
      {
         scaleVectorIP(scaleL, corrs[i].left.pos);
         scaleVectorIP(scaleR, corrs[i].right.pos);
      }

      Matrix3x3d const Hnorm = computeHomographyLinearUnnormalized(corrs);
      //cout << "Hnorm = "; displayMatrix(Hnorm);

      Matrix3x3d Tleft, invTright;
      makeZeroMatrix(Tleft);
      makeZeroMatrix(invTright);

      Tleft[0][0] = Tleft[1][1] = scaleL;
      Tleft[0][2] = -scaleL * centerL[0];
      Tleft[1][2] = -scaleL * centerL[1];
      Tleft[2][2] = 1.0;

      invTright[0][0] = invTright[1][1] = 1.0 / scaleR;
      invTright[0][2] = centerR[0];
      invTright[1][2] = centerR[1];
      invTright[2][2] = 1.0;

      Matrix3x3d H = invTright * Hnorm * Tleft;
      //cout << "H = "; displayMatrix(H);

      // Choose sign, such that x2^T * H * x1 > 0 for all correspondences x1 <-> x2.
      int nNegative = 0;
      Vector3f p, q, Hp;
      for (int i = 0; i < N; ++i)
      {
         p[0] = corrs[i].left.pos[0];  p[1] = corrs[i].left.pos[1];  p[2] = 1.0f;
         q[0] = corrs[i].right.pos[0]; q[1] = corrs[i].right.pos[1]; q[2] = 1.0f;
         multiply_A_v(Hnorm, p, Hp);
         if (innerProduct(q, Hp) < 0) ++nNegative;
      }

      if (2*nNegative > N) scaleMatrixIP(-1.0, H);
      return H;
   } // end computeHomographyLinear()

   Matrix3x3d
   computeHomographyLinear(double const f0, Vector2f const& pp0, std::vector<PointCorrespondence> const& corrs)
   {
      float const d  = 1.0/f0;

      int const N = corrs.size();
      Matrix<double> A(3*N, 9, 0.0);

      for (int i = 0; i < N; ++i)
      {
         Vector2f const p = d*(corrs[i].left.pos - pp0);
         Vector2f const q = d*(corrs[i].right.pos - pp0);

         A[3*i+0][3] = -p[0];
         A[3*i+0][4] = -p[1];
         A[3*i+0][5] = -1.0;
         A[3*i+0][6] = q[1]*p[0];
         A[3*i+0][7] = q[1]*p[1];
         A[3*i+0][8] = q[1];

         A[3*i+1][0] = -p[0];
         A[3*i+1][1] = -p[1];
         A[3*i+1][2] = -1.0;
         A[3*i+1][6] = q[0]*p[0];
         A[3*i+1][7] = q[0]*p[1];
         A[3*i+1][8] = q[0];

         A[3*i+2][0] = -q[1]*p[0];
         A[3*i+2][1] = -q[1]*p[1];
         A[3*i+2][2] = -q[1];
         A[3*i+2][3] = q[0]*p[0];
         A[3*i+2][4] = q[0]*p[1];
         A[3*i+2][5] = q[0];
      } // end for (i)

      SVD<double> svd(A);
      Matrix<double> const& V = svd.getV();

      Matrix3x3d H;
      H[0][0] = V[0][8]; H[0][1] = V[1][8]; H[0][2] = V[2][8];
      H[1][0] = V[3][8]; H[1][1] = V[4][8]; H[1][2] = V[5][8];
      H[2][0] = V[6][8]; H[2][1] = V[7][8]; H[2][2] = V[8][8];
      return H;
   } // end computeHomographyLinear()

   bool
   computeRobustHomographyMLE(std::vector<PointCorrespondence> const& corrs,
                              double inlierThreshold, int nRounds,
                              Matrix3x3d& bestH, std::vector<int>& inliers)
   {
      inliers.clear();

      int const N = corrs.size();
      int const minSize = 4;
      if (N < minSize) return false;

      double const sqrThreshold = inlierThreshold*inlierThreshold;

      vector<int> indices(corrs.size());
      for (size_t i = 0; i < N; ++i) indices[i] = i;

      int const nShuffles = getShuffleCount(N, nRounds, minSize);
      //cout << "nShuffles = " << nShuffles << endl;
      int round = 0;

      vector<PointCorrespondence> minCorrs(minSize);

      double bestScore = 1e30;
      vector<int> curInliers;

      for (int s = 0; s < nShuffles; ++s)
      {
         //cout << "s = " << s << ", round = " << round << endl;

         if (round >= nRounds) break;

         // shuffle indices 
         random_shuffle(indices.begin(), indices.end());

         for (int j = 0; j < N-minSize; j += minSize, ++round)
         {
            if (round >= nRounds) break;

            minCorrs[0].left.pos = corrs[indices[j+0]].left.pos; minCorrs[0].right.pos = corrs[indices[j+0]].right.pos;
            minCorrs[1].left.pos = corrs[indices[j+1]].left.pos; minCorrs[1].right.pos = corrs[indices[j+1]].right.pos;
            minCorrs[2].left.pos = corrs[indices[j+2]].left.pos; minCorrs[2].right.pos = corrs[indices[j+2]].right.pos;
            minCorrs[3].left.pos = corrs[indices[j+3]].left.pos; minCorrs[3].right.pos = corrs[indices[j+3]].right.pos;

            Matrix3x3d H = computeHomographyLinear(minCorrs);

            curInliers.clear();
            double score = 0.0;

            Vector2f Hp;
            for (int i = 0; i < N; ++i)
            {
               multiply_A_v_projective(H, corrs[i].left.pos, Hp);
               double const d2 = sqrNorm_L2(Hp - corrs[i].right.pos);
               if (d2 < sqrThreshold)
               {
                  score += d2;
                  curInliers.push_back(i);
               }
               else
                  score += sqrThreshold;
            } // end for (i)

            if (score < bestScore)
            {
               bestH = H;
               bestScore = score;
               inliers = curInliers;
            } // end if
         } // end for (j)
      } // end for (s)

      //cout << "used rounds = " << round << endl;
      return true;
   } // end computeRobustHomographyMLE()

   bool
   refineHomography(std::vector<Vector2d> const& left,
                    std::vector<Vector2d> const& right,
                    double inlierThreshold, Matrix3x3d& H)
   {
      HomographyMinimizer opt(H, left, right, inlierThreshold);

      optimizerVerbosenessLevel = 0;
      opt.minimize();
      optimizerVerbosenessLevel = 0;

      H = opt._currentH;

      return true;
   } // end refinePanoramicHomography()

//**********************************************************************

   bool
   computePanoramicHomography(Vector2f const& p1, Vector2f const& p2, // left points
                              Vector2f const& q1, Vector2f const& q2, // right points
                              std::vector<double>& fs, std::vector<Matrix3x3d>& Rs)
   {
      double const a12 = innerProduct(p1, p2) + 1.0;
      double const a1 = sqrNorm_L2(p1) + 1.0;
      double const a2 = sqrNorm_L2(p2) + 1.0;

      double const b12 = innerProduct(q1, q2) + 1.0;
      double const b1 = sqrNorm_L2(q1) + 1.0;
      double const b2 = sqrNorm_L2(q2) + 1.0;

      double const c3 = b2 - 2*b12 + b1 - a2 + 2*a12 - a1;
      double const c2 = b1*b2 + 2*a12*b2 - b12*b12 - 2*a2*b12 - 2*a1*b12 + 2*a12*b1 - a1*a2 + a12*a12;
      double const c1 = 2*a12*b1*b2 + a12*a12*b2 - a2*b12*b12 - a1*b12*b12 - 2*a1*a2*b12 + a12*a12*b1;
      double const c0 = a12*a12*b1*b2 - a1*a2*b12*b12;

      double roots[3];

      int const nRoots = getRealRootsOfCubicPolynomial(c3, c2, c1, c0, roots);

      fs.clear();
      Rs.clear();

      double const eps = 0.1;

      for (int i = 0; i < nRoots; ++i)
      {
         if (isfinite(roots[i]) && roots[i] > eps)
         {
            double const f = sqrt(roots[i]);
            fs.push_back(f);
            //cout << "f = " << f << endl;

            Vector3d u1, u2, v1, v2;
            u1[0] = p1[0]; u1[1] = p1[1]; u1[2] = f; normalizeVector(u1);
            u2[0] = p2[0]; u2[1] = p2[1]; u2[2] = f; normalizeVector(u2);

            v1[0] = q1[0]; v1[1] = q1[1]; v1[2] = f; normalizeVector(v1);
            v2[0] = q2[0]; v2[1] = q2[1]; v2[2] = f; normalizeVector(v2);

            Matrix<double> C(3, 3, 0.0);
            Matrix3x3d uv;
            makeOuterProductMatrix(v1, u1, C);
            makeOuterProductMatrix(v2, u2, uv);
            addMatricesIP(uv, C);

            //cout << "C = "; displayMatrix(C);
            SVD<double> svd(C);
            //cout << "done with SVD." << endl;
            Matrix3x3d U;
            copyMatrix(svd.getU(), U);
            Matrix<double> const& V = svd.getV();
            Matrix3x3d Vt;
            makeTransposedMatrix(V, Vt);

            Matrix3x3d D; makeIdentityMatrix(D);

            double const det = matrixDeterminant3x3(U*Vt);
            if (det < 0) D[2][2] = -1;
            Matrix3x3d const R = U*D*Vt;
            Rs.push_back(R);
         } // end if
      } // end for (i)

      return fs.size() > 0;
   } // end computePanoramicHomography()

   bool
   computeRobustPanoramicHomographyMLE(std::vector<PointCorrespondence> const& corrs,
                                       double inlierThreshold, int nRounds,
                                       double& best_f, Matrix3x3d& bestR, std::vector<int>& inliers)
   {
      inliers.clear();

      int const N = corrs.size();
      int const minSize = 2;
      if (N < minSize) return false;

      double const sqrThreshold = inlierThreshold*inlierThreshold;

      vector<int> indices(corrs.size());
      for (size_t i = 0; i < N; ++i) indices[i] = i;

      int const nShuffles = getShuffleCount(N, nRounds, 2);
      //cout << "nShuffles = " << nShuffles << endl;
      int round = 0;

      Vector2f left[2], right[2];
      double bestScore = 1e30;
      vector<int> curInliers;

      for (int s = 0; s < nShuffles; ++s)
      {
         //cout << "s = " << s << ", round = " << round << endl;

         if (round >= nRounds) break;

         // shuffle indices 
         random_shuffle(indices.begin(), indices.end());

         for (int j = 0; j < N-2; j += 2, ++round)
         {
            if (round >= nRounds) break;

            left[0] = corrs[indices[j+0]].left.pos; right[0] = corrs[indices[j+0]].right.pos;
            left[1] = corrs[indices[j+1]].left.pos; right[1] = corrs[indices[j+1]].right.pos;

            vector<double> fs;
            vector<Matrix3x3d> Rs;

            computePanoramicHomography(left[0], left[1], right[0], right[1], fs, Rs);

            for (int k = 0; k < fs.size(); ++k)
            {
               curInliers.clear();
               Matrix3x3d const H = getPanoramicHomography(fs[k], Rs[k]);

               double score = 0.0;

               Vector2f Hp;
               for (int i = 0; i < N; ++i)
               {
                  multiply_A_v_projective(H, corrs[i].left.pos, Hp);
                  double const d2 = sqrNorm_L2(Hp - corrs[i].right.pos);
                  if (d2 < sqrThreshold)
                  {
                     score += d2;
                     curInliers.push_back(i);
                  }
                  else
                     score += sqrThreshold;
               } // end for (i)
               //cout << "f = " << f << ", curInliers.size() = " << curInliers.size() << ", score = " << score << endl;

               if (score < bestScore)
               {
                  best_f = fs[k];
                  bestR = Rs[k];
                  bestScore = score;
                  inliers = curInliers;
               } // end if
            } // end for (k)
         } // end for (j)
      } // end for (s)

      //cout << "used rounds = " << round << endl;
      return true;
   } // end computeRobustPanoramicHomographyMLE()

   bool
   refinePanoramicHomography(std::vector<Vector2d> const& left_,
                             std::vector<Vector2d> const& right_,
                             double& f, Vector2d& pp, Matrix3x3d& R,
                             double inlierThreshold, int what)
   {
#define REFINE_WITH_NORMALIZATION 1
#ifdef REFINE_WITH_NORMALIZATION
      // Normalize to f = 1.
      double const f0   = 1.0;
      double const rcpF = 1.0 / f;
      Vector2d const pp0(0.0, 0.0);

      int const N = left_.size();
      vector<Vector2d> left(N), right(N);
      for (int i = 0; i < N; ++i)
      {
         scaleVector(rcpF, left_[i]-pp, left[i]);
         scaleVector(rcpF, right_[i]-pp, right[i]);
      }
#else
      double const f0   = f;
      double const rcpF = 1.0;
      Vector2d& pp0     = pp;

      vector<Vector2d> const& left  = left_;
      vector<Vector2d> const& right = right_;
#endif

      int mode = PanoramicHomographyMinimizer::OPTIMIZE_R;
      switch (what)
      {
         case 1:
            mode = PanoramicHomographyMinimizer::OPTIMIZE_R_F;
            break;
         case 2:
            mode = PanoramicHomographyMinimizer::OPTIMIZE_R_F_PP;
            break;
      }

      PanoramicHomographyMinimizer opt(R, f0, pp0, left, right, inlierThreshold*rcpF, mode);

      optimizerVerbosenessLevel = 0;
      opt.minimize();
      optimizerVerbosenessLevel = 0;

      R = opt._currentR;
      f = opt._current_f / rcpF;
#ifdef REFINE_WITH_NORMALIZATION
      pp = pp + (1.0/rcpF)*opt._current_pp;
#else
      pp = opt._current_pp;
#endif

      return true;
#undef REFINE_WITH_NORMALIZATION
   } // end refinePanoramicHomography()

   Matrix3x3d
   getHomographyCompatibleWithFundamental(Matrix3x3d const& F, vector<PointCorrespondence> const& corrs)
   {
      int const N = corrs.size();
      double const rcpN = 1.0 / N;

      Vector2f meanL, meanR;
      makeZeroVector(meanL);
      makeZeroVector(meanR);

      for (int i = 0; i < N; ++i)
      {
         addVectorsIP(corrs[i].left.pos, meanL);
         addVectorsIP(corrs[i].right.pos, meanR);
      }
      scaleVectorIP(rcpN, meanL);
      scaleVectorIP(rcpN, meanR);

      double scaleL = 0, scaleR = 0;

      for (int i = 0; i < N; ++i)
      {
         scaleL += sqrDistance_L2(corrs[i].left.pos, meanL);
         scaleR += sqrDistance_L2(corrs[i].right.pos, meanR);
      }

      scaleL *= rcpN; scaleR *= rcpN;

      float const rcpScaleL = 1.0 / scaleL;
      float const rcpScaleR = 1.0 / scaleR;

      Matrix3x3d T1, invT2;
      makeIdentityMatrix(T1);
      makeIdentityMatrix(invT2);

      T1[0][0] = T1[1][1] = rcpScaleL;
      T1[0][2] = -rcpScaleL * meanL[0];
      T1[1][2] = -rcpScaleL * meanL[1];

      invT2[0][0] = invT2[1][1] = scaleR;
      invT2[0][2] = meanR[0];
      invT2[1][2] = meanR[1];

      // Fundamental between normalized correspondences
      Matrix3x3d const F0 = invT2.transposed() * F * invertedMatrix(T1);

      Matrix<double> A(3*N, 9, 0.0);

      for (int i = 0; i < N; ++i)
      {
         Vector2f const p = rcpScaleL*(corrs[i].left.pos - meanL);
         Vector2f const q = rcpScaleR*(corrs[i].right.pos - meanR);

         A[3*i+0][3] = -p[0];
         A[3*i+0][4] = -p[1];
         A[3*i+0][5] = -1.0;
         A[3*i+0][6] = q[1]*p[0];
         A[3*i+0][7] = q[1]*p[1];
         A[3*i+0][8] = q[1];

         A[3*i+1][0] = -p[0];
         A[3*i+1][1] = -p[1];
         A[3*i+1][2] = -1.0;
         A[3*i+1][6] = q[0]*p[0];
         A[3*i+1][7] = q[0]*p[1];
         A[3*i+1][8] = q[0];

         A[3*i+2][0] = -q[1]*p[0];
         A[3*i+2][1] = -q[1]*p[1];
         A[3*i+2][2] = -q[1];
         A[3*i+2][3] = q[0]*p[0];
         A[3*i+2][4] = q[0]*p[1];
         A[3*i+2][5] = q[0];
      } // end for (i)

      Matrix<double> C(9, 9, 0.0);

      C[0][0] = 2*F0[0][0]; C[0][3] = 2*F0[1][0]; C[0][6] = 2*F0[2][0];
      C[1][0] =   F0[0][1]; C[1][3] =   F0[1][1]; C[1][6] =   F0[2][1];
      C[2][0] =   F0[0][2]; C[2][3] =   F0[1][2]; C[2][6] =   F0[2][2];

      C[3][1] =   F0[0][0]; C[3][4] =   F0[1][0]; C[3][7] =   F0[2][0];
      C[4][1] = 2*F0[0][1]; C[4][4] = 2*F0[1][1]; C[4][7] = 2*F0[2][1];
      C[5][1] =   F0[0][2]; C[5][4] =   F0[1][2]; C[5][7] =   F0[2][2];

      C[6][2] =   F0[0][0]; C[6][5] =   F0[1][0]; C[6][8] =   F0[2][0];
      C[7][2] =   F0[0][1]; C[7][5] =   F0[1][1]; C[7][8] =   F0[2][1];
      C[8][2] = 2*F0[0][2]; C[8][5] = 2*F0[1][2]; C[8][8] = 2*F0[2][2];

      C[1][1] += F0[0][0]; C[1][4] += F0[1][0]; C[1][7] += F0[2][0];
      C[2][2] += F0[0][0]; C[2][5] += F0[1][0]; C[2][8] += F0[2][0];
      C[3][0] += F0[0][1]; C[3][3] += F0[1][1]; C[3][6] += F0[2][1];
      C[5][2] += F0[0][1]; C[5][5] += F0[1][1]; C[5][8] += F0[2][1];
      C[6][0] += F0[0][2]; C[6][3] += F0[1][2]; C[6][6] += F0[2][2];
      C[7][1] += F0[0][2]; C[7][4] += F0[1][2]; C[7][7] += F0[2][2];

      Vector<double> X(A.num_cols());

      // Rank C is 5 (3 d.o.f. for a compatible H + scale)
      minimize_norm_Ax_st_norm_x_eq_1_and_Cx_eq_0(A, C, 5, X);

      Matrix3x3d H;
      H[0][0] = X[0]; H[0][1] = X[1]; H[0][2] = X[2];
      H[1][0] = X[3]; H[1][1] = X[4]; H[1][2] = X[5];
      H[2][0] = X[6]; H[2][1] = X[7]; H[2][2] = X[8];

      H = invT2 * H * T1;

      // Choose sign, such that x2^t * H * x1 > 0 for all correspondences x1 <-> x2.
      int nNegative = 0;
      Vector3d p, q;
      p[2] = 1.0; q[2] = 1.0;
      for (int i = 0; i < N; ++i)
      {
         PointCorrespondence const& corr = corrs[i];
         p[0] = corr.left.pos[0];
         p[1] = corr.left.pos[1];
         q[0] = corr.right.pos[0];
         q[1] = corr.right.pos[1];
         if (innerProduct(q, H * p) < 0) ++nNegative;

//          Vector2f Hp;
//          multiply_A_v_projective(H, corrs[i].left.pos, Hp);
//          double const d2 = sqrNorm_L2(Hp - corrs[i].right.pos);
//          cout << d2 << " ";
      } // end for (i)
      //cout << endl;

      //cout << "H^T F + F^T H = "; displayMatrix(transposedMatrix(F)*H + transposedMatrix(H)*F);

      if (2*nNegative > N) scaleMatrixIP(-1.0, H);
      return H;
   } // end getHomographyCompatibleWithFundamental()

   bool
   computeRobustCompatibleHomographyMLE(Matrix3x3d const&F,
                                        std::vector<PointCorrespondence> const& corrs,
                                        double inlierThreshold, int nRounds,
                                        Matrix3x3d& bestH, std::vector<int>& inliers)
   {
      inliers.clear();

      int const N = corrs.size();
      int const minSize = 3;
      if (N < minSize) return false;

      double const sqrThreshold = inlierThreshold*inlierThreshold;

      vector<int> indices(corrs.size());
      for (size_t i = 0; i < N; ++i) indices[i] = i;

      int const nShuffles = getShuffleCount(N, nRounds, minSize);
      //cout << "nShuffles = " << nShuffles << endl;
      int round = 0;

      vector<PointCorrespondence> minCorrs(minSize);

      double bestScore = 1e30;
      vector<int> curInliers;

      for (int s = 0; s < nShuffles; ++s)
      {
         //cout << "s = " << s << ", round = " << round << endl;

         if (round >= nRounds) break;

         // shuffle indices 
         random_shuffle(indices.begin(), indices.end());

         for (int j = 0; j < N-minSize; j += minSize, ++round)
         {
            if (round >= nRounds) break;

            minCorrs[0].left.pos = corrs[indices[j+0]].left.pos; minCorrs[0].right.pos = corrs[indices[j+0]].right.pos;
            minCorrs[1].left.pos = corrs[indices[j+1]].left.pos; minCorrs[1].right.pos = corrs[indices[j+1]].right.pos;
            minCorrs[2].left.pos = corrs[indices[j+2]].left.pos; minCorrs[2].right.pos = corrs[indices[j+2]].right.pos;

            Matrix3x3d const H = getHomographyCompatibleWithFundamental(F, minCorrs);
            curInliers.clear();
            double score = 0.0;

            //cout << "H = "; displayMatrix(H);

            Vector2f Hp;
            for (int i = 0; i < N; ++i)
            {
               multiply_A_v_projective(H, corrs[i].left.pos, Hp);
               double const d2 = sqrNorm_L2(Hp - corrs[i].right.pos);

               //cout << d2 << " ";

               if (d2 < sqrThreshold)
               {
                  score += d2;
                  curInliers.push_back(i);
               }
               else
                  score += sqrThreshold;
            } // end for (i)
            //cout << endl;

            if (score < bestScore)
            {
               bestH = H;
               bestScore = score;
               inliers = curInliers;
            } // end if
         } // end for (j)
      } // end for (s)

      //cout << "used rounds = " << round << endl;
      return true;
   } // end computeRobustCompatibleHomographyMLE()

} // end namespace V3D
