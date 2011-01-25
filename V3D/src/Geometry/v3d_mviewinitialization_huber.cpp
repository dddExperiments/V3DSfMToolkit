#include "Math/v3d_optimization.h"
#include "Math/v3d_sparseeig.h"
#include "Geometry/v3d_mviewinitialization.h"

using namespace std;
using namespace V3D;

namespace
{

   template <typename T>
   inline T
   clamp(T x, T a, T b)
   {
      return std::max(a, std::min(b, x));
   }

   inline double
   huber(double x, double y)
   {
#if 0
      if (fabs(x) > y) return fabs(x);
      return x*x/(2*y) + y/2;
#else
      if (fabs(x) > y) return fabs(x) - y/2;
      return x*x/(2*y);
#endif
   }

   inline float
   cubicRoot(float x)
   {
#if 0
      return cbrtf(x);
#else
      float const sgn = (x >= 0.0f) ? 1.0f : -1.0f;

      // assumes sizeof(unsigned int) == 4 (32 bits)
      float const R = fabs(x);
      if (R < 1e-10) return 0.0f;

      // We need to declare some variables volatile to prevent to optimizer to optimize too much
      volatile float y = R;
      unsigned int const C = 709921077;
      volatile unsigned int * p = (volatile unsigned int *)(&y);
      *p = (*p)/3 + C;
      x = y;

#if 0
      // Newton iterations
      x = (R / (x*x) + 2*x) / 3.0f;
      x = (R / (x*x) + 2*x) / 3.0f;
      x = (R / (x*x) + 2*x) / 3.0f;
#elif 0
      // Halley's method, cubic convergence
      float x3;
      x3 = x*x*x; x = x - (x3-R)*x/(2*x3+R);
      x3 = x*x*x; x = x - (x3-R)*x/(2*x3+R);
      x3 = x*x*x; x = x - (x3-R)*x/(2*x3+R);
#else
      // Quartic convergence
      float x3, s;
      x3 = x*x*x; s = (x3 - R)/R;
      x = x - x*((14.0f/81.0f*s - 2.0f/9.0f)*s + 1.0f/3.0f)*s;
      x3 = x*x*x; s = (x3 - R)/R;
      x = x - x*((14.0f/81.0f*s - 2.0f/9.0f)*s + 1.0f/3.0f)*s;
#endif

      return sgn * x;
#endif
   }

   double
   computeTranslationRegistrationLipschitzConstant(float const sigma,
                                                   std::vector<Matrix3x3d> const& rotations,
                                                   std::vector<TriangulatedPoint>& sparseModel)
   {
      int const N = rotations.size();
      int const M = sparseModel.size();

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      // We have 4*K + K dual variables and 3*N+3*M primal ones

      vector<pair<int, int> > nz;
      vector<double> values;
      nz.reserve(5*K);
      values.reserve(5*K);

      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];
         Matrix3x3d const& R = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         int const Ti0 = 3*i + 0;
         int const Ti1 = 3*i + 1;
         int const Ti2 = 3*i + 2;

         int const Xj0 = 3*N + 3*j + 0;
         int const Xj1 = 3*N + 3*j + 1;
         int const Xj2 = 3*N + 3*j + 2;

         int const Yp0 = 5*k + 0;
         int const Yp1 = 5*k + 1;
         int const Ym0 = 5*k + 2;
         int const Ym1 = 5*k + 3;
         int const Zk  = 5*k + 4;

         nz.push_back(make_pair(Yp0, Xj0)); values.push_back(-R[0][0] + (u1 - sigma)*R[2][0]);
         nz.push_back(make_pair(Yp0, Xj1)); values.push_back(-R[0][1] + (u1 - sigma)*R[2][1]);
         nz.push_back(make_pair(Yp0, Xj2)); values.push_back(-R[0][2] + (u1 - sigma)*R[2][2]);
         nz.push_back(make_pair(Yp0, Ti0)); values.push_back(-1.0);
         nz.push_back(make_pair(Yp0, Ti2)); values.push_back(u1 - sigma);

         nz.push_back(make_pair(Ym0, Xj0)); values.push_back(-R[0][0] - (u1 + sigma)*R[2][0]);
         nz.push_back(make_pair(Ym0, Xj1)); values.push_back(-R[0][1] - (u1 + sigma)*R[2][1]);
         nz.push_back(make_pair(Ym0, Xj2)); values.push_back(-R[0][2] - (u1 + sigma)*R[2][2]);
         nz.push_back(make_pair(Ym0, Ti0)); values.push_back(-1.0);
         nz.push_back(make_pair(Ym0, Ti2)); values.push_back(-u1 - sigma);

         nz.push_back(make_pair(Yp1, Xj0)); values.push_back(-R[1][0] + (u2 - sigma)*R[2][0]);
         nz.push_back(make_pair(Yp1, Xj1)); values.push_back(-R[1][1] + (u2 - sigma)*R[2][1]);
         nz.push_back(make_pair(Yp1, Xj2)); values.push_back(-R[1][2] + (u2 - sigma)*R[2][2]);
         nz.push_back(make_pair(Yp1, Ti1)); values.push_back(-1.0);
         nz.push_back(make_pair(Yp1, Ti2)); values.push_back(u2 - sigma);

         nz.push_back(make_pair(Ym1, Xj0)); values.push_back(-R[1][0] - (u2 + sigma)*R[2][0]);
         nz.push_back(make_pair(Ym1, Xj1)); values.push_back(-R[1][1] - (u2 + sigma)*R[2][1]);
         nz.push_back(make_pair(Ym1, Xj2)); values.push_back(-R[1][2] - (u2 + sigma)*R[2][2]);
         nz.push_back(make_pair(Ym1, Ti1)); values.push_back(-1.0);
         nz.push_back(make_pair(Ym1, Ti2)); values.push_back(-u2 - sigma);

         nz.push_back(make_pair(Zk, Xj0)); values.push_back(-R[1][0]);
         nz.push_back(make_pair(Zk, Xj1)); values.push_back(-R[1][1]);
         nz.push_back(make_pair(Zk, Xj2)); values.push_back(-R[1][2]);
         nz.push_back(make_pair(Zk, Ti2)); values.push_back(-1.0);
      } // end for (k)

      CCS_Matrix<double> A(5*K, 3*N+3*M, nz, values);

      Vector<double> sv(1);
      Matrix<double> V(1, 5*K);

      bool status = computeSparseSVD(A, V3D_ARPACK_LARGEST_MAGNITUDE_EIGENVALUES, 1, sv, V);
      return sv[0];
   } // end computeTranslationRegistrationLipschitzConstant()

   static int nCubicRoots = 0;

   inline void
   projectZs_Huber(float& z1, float& z2)
   {
#if 1
      // Project (z1, z2) onto the convex set z2 <= -0.5*z1^2, |z1| <= 1

      float const z1mag = fabsf(z1);

      if (z2 <= -0.5f*z1*z1 && z1mag <= 1.0f) return; // Nothing to do

      // The simple case, just clamp z1
      if (z2 <= -0.5f && z1mag > 1.0f)
      {
         z1 = std::max(-1.0f, std::min(1.0f, z1));
         return;
      }

      if (z2 <= z1 - 1.5f)
      {
         z1 = 1.0f; z2 = -0.5f;
         return;
      }
      if (z2 <= -z1 - 1.5f)
      {
         z1 = -1.0f; z2 = -0.5f;
         return;
      }

# if 1
      // This is now the projection on the parabola step rising to a 3rd order polynomial.
      // Since z2 > 0.5, we can use the simplified Cardano formulas not using complex numbers.
      // We also know to obtain one real root.
      float const p = (2.0+ 2.0*z2)/3.0;
      float const q = -z1;
      float const D2 = std::max(0.0f, q*q + p*p*p); // Should be non-negative
      float const D = sqrtf(D2);
      ++nCubicRoots;

      float const u = cubicRoot(-q + D);
      float const v = cubicRoot(-q - D);
      z1 = u+v;
      z1 = std::max(-1.0f, std::min(1.0f, z1));
      z2 = -0.5f*z1*z1;
# else
      z1 = std::max(-1.0f, std::min(1.0f, z1));
      z2 = std::min(z2, -0.5f*z1*z1);
# endif
#else
      z1 = std::max(-1.0f, std::min(1.0f, z1));
      z2 = std::min(z2, -0.5f*z1*z1);
#endif

      z2 = std::max(z2, -10.0f);

      //if (fabs(z1) > 1.0) cout << "z1 = " << z1 << endl;
   } // end projectZs_Huber()

   void
   rescaleStructure(std::vector<Matrix3x3d> const& rotations,
                    vector<int> const& correspondingView, vector<int> const& correspondingPoint,
                    std::vector<Vector3d>& Ts, std::vector<Vector3d>& Xs, std::vector<double>& Ds)
   {
      double minDist = 1e30;

      int const N = rotations.size();
      int const M = Xs.size();
      int const K = Ds.size();

      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R = rotations[i];
         Vector3d   const& T = Ts[i];
         Vector3d   const& X = Xs[j];

         double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2] + Ds[k];
         minDist = std::min(minDist, XX3);
      }
      //cout << "minDist = " << minDist << endl;

      double const stretch = 1.0 / minDist;

      for (int i = 0; i < N; ++i) scaleVectorIP(stretch, Ts[i]);
      for (int j = 0; j < M; ++j) scaleVectorIP(stretch, Xs[j]);
      for (int k = 0; k < K; ++k) Ds[k] *= stretch;
   } // end rescaleStructure()

   inline double
   computeStructureSimilarity(std::vector<Vector3d> const& Xs1, std::vector<Vector3d> const& Xs2)
   {
      int const M = Xs1.size();

      Vector3d mean1, mean2;
      makeZeroVector(mean1);
      makeZeroVector(mean2);

      for (int j = 0; j < M; ++j) addVectorsIP(Xs1[j], mean1);
      for (int j = 0; j < M; ++j) addVectorsIP(Xs2[j], mean2);

      scaleVectorIP(1.0/M, mean1);
      scaleVectorIP(1.0/M, mean2);

      double var1 = 0.0, var2 = 0.0;
      for (int j = 0; j < M; ++j) var1 += sqrDistance_L2(Xs1[j], mean1);
      for (int j = 0; j < M; ++j) var2 += sqrDistance_L2(Xs2[j], mean2);

      if (var1 < 1e-3 || var2 < 1e-3) return 1e6;

      double const scale1 = 1.0 / sqrt(var1);
      double const scale2 = 1.0 / sqrt(var2);

      double res = 0.0;
#if 0
      for (int j = 0; j < M; ++j)
         res += sqrDistance_L2(scale1 * (Xs1[j] - mean1), scale2 * (Xs2[j] - mean2));
      return res / M;
#else
      for (int j = 0; j < M; ++j)
         res = std::max(res, sqrDistance_L2(scale1 * (Xs1[j] - mean1), scale2 * (Xs2[j] - mean2)));
      return res;
#endif
   } // end computeStructureSimilarity()

   void
   selectCheiralityConstraints(vector<int> const& correspondingView, vector<int> const& correspondingPoint,
                               vector<int>& constraints)
   {
      constraints.clear();
      int const K = correspondingView.size();
      vector<int> allKs(K);
      for (int k = 0; k < K; ++k) allKs[k] = k;
      random_shuffle(allKs.begin(), allKs.end());

      set<int> handledViews, handledPoints;

      for (int l = 0; l < K; ++l)
      {
         int const k = allKs[l];
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         if (handledViews.find(i) == handledViews.end() && handledPoints.find(j) == handledPoints.end())
         {
            constraints.push_back(k);
            handledViews.insert(i);
            handledPoints.insert(j);
         }
      } // end for (l)
   } // end selectCheiralityConstraints()

   void
   enforceCheiralityConstraints(vector<Matrix3x3d> const& rotations,
                                vector<int> const& correspondingView, vector<int> const& correspondingPoint,
                                vector<int> const& constraints, vector<Vector3d>& Ts, vector<Vector3d>& Xs, vector<double>& Ds)
   {
      typedef InlineVector<double, 5> Vector5d;

#if 1
      for (int l = 0; l < constraints.size(); ++l)
      {
         int const k = constraints[l];
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R = rotations[i];
         Vector3d& T = Ts[i];
         Vector3d& X = Xs[j];
         double&   D = Ds[k];

         double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2] + D;
         if (XX3 < 1.0)
         {
            // Determine the closest point satisfying (RX + T)_3 + D = 1

            Vector5d n, X0, n_norm;
            n[0] = R[2][0]; n[1] = R[2][1]; n[2] = R[2][2]; n[3] = 1.0; n[4] = 1.0;
            X0[0] = X[0]; X0[1] = X[1]; X0[2] = X[2]; X0[3] = T[2]; X0[4] = D;

            double const len = norm_L2(n);
            scaleVector(1.0/len, n, n_norm);
            X0 = X0 - (innerProduct(X0, n_norm) - 1.0/len)*n_norm;
            if (X0[4] < 0.0)
            {
               // D clamped to 0
               n[4] = 0; X0[4] = 0;
               double const len = norm_L2(n);
               scaleVector(1.0/len, n, n_norm);
               X0 = X0 - (innerProduct(X0, n_norm) - 1.0/len)*n_norm;
            } // end if
            X[0] = X0[0]; X[1] = X0[1]; X[2] = X0[2]; T[2] = X0[3]; D = X0[4];

            double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2] + D;

            //cout << "i = " << i << " j = " << j << " XX3 = " << XX3 << ": "; displayVector(X0);
         } // end if
      } // end for (l)
      //cout << "--------------------" << endl;
#endif

      for (int k = 0; k < Ds.size(); ++k) Ds[k] = std::max(0.0, Ds[k]);
   } // end enforceCheiralityConstraints()

} // end namespace <>

namespace V3D
{

   void
   computeConsistentTranslationsConic_Huber_PD(float const sigma,
                                               std::vector<Matrix3x3d> const& rotations,
                                               std::vector<Vector3d>& translations,
                                               std::vector<TriangulatedPoint>& sparseModel,
                                               TranslationRegistrationPD_Params const& params)
   {
//       cout << "sizeof(unsigned int) = " << sizeof(unsigned int) << endl;
//       cout << "cbrt(5) = " << cubicRoot(5) << endl;
//       cout << "cbrt(10) = " << cubicRoot(10) << endl;
//       cout << "cbrt(2) = " << cubicRoot(2) << endl;
//       cout << "cbrt(-5) = " << cubicRoot(-5) << endl;

//       {
//          float z1 = -1.0, z2 = 0.0;
//          projectZs_Huber(z1, z2);
//          cout << "(z1, z2) = " << z1 << ", " << z2 << endl;
//       }

      // 0 is the hybrid scheme, 1 is Chambolle's method
#define NUMERICAL_SCHEME 1
//#define USE_FIXED_BALLOONING_TERM 1

      // sigma (X_ij^3 + D_ij) >= eta
#if !defined(USE_FIXED_BALLOONING_TERM)
      float const eta = sigma;
#else
      float const eta = 0.0;
#endif

      double const L = computeTranslationRegistrationLipschitzConstant(sigma, rotations, sparseModel);
      double const tau_p = params.timestepMultiplier * sqrt(params.timestepRatio) / L;
      double const tau_d = params.timestepMultiplier / sqrt(params.timestepRatio) / L;

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      vector<int> cheiralityConstraints;
      selectCheiralityConstraints(correspondingView, correspondingPoint, cheiralityConstraints);

#if !defined(USE_FIXED_BALLOONING_TERM)
      float const initZ = 0.0f;
#else
      float const initZ = -0.5f;
#endif
      double const rhoWeight = 1.0;

      vector<Vector3d> Xs(M);
      vector<double>   Ds(K, 1.0);
      vector<Vector4f> Zs(K, Vector4f(0.0f, initZ, 0.0f, initZ));

      for (int i = 0; i < N; ++i) makeZeroVector(translations[i]);
      for (int j = 0; j < M; ++j) makeZeroVector(Xs[j]);

#if NUMERICAL_SCHEME == 1
      vector<Vector3d> XXs(M);
      vector<Vector3d> Xs_saved(M);
      vector<double>   DDs(K, 0.0);
      vector<double>   Ds_saved(K, 1.0);
      vector<Vector3d> TTs(N);
      vector<Vector3d> Ts_saved(N);
      for (int i = 0; i < N; ++i) makeZeroVector(TTs[i]);
      for (int j = 0; j < M; ++j) makeZeroVector(XXs[j]);
#endif

      vector<Vector3d> Xs_last(Xs);

      vector<Vector3d> dTs_k(K), dXs_k(K);
      vector<double> dDs_k(K);

      nCubicRoots = 0;

      for (int iter = 0; iter < params.nMaxIterations; ++iter)
      {
         // Primal updates
         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            Vector3d   const& T = translations[i];
            Vector3d   const& X = Xs[j];
            Vector2d   const& m  = bundleStruct.measurements[k];
            float      const  u1 = m[0];
            float      const  u2 = m[1];

            float const z1x = Zs[k][0];
            float const z2x = Zs[k][1];

            float const z1y = Zs[k][2];
            float const z2y = Zs[k][3];

            Vector3d dTs_cur, dXs_cur;

            dTs_cur[0] = -z1x;
            dTs_cur[1] = -z1y;
            dTs_cur[2] = z1x*u1 + z1y*u2 + sigma*(z2x + z2y);

            dXs_cur[0] = z1x * (u1*R[2][0] - R[0][0]) + z2x * sigma*R[2][0];
            dXs_cur[1] = z1x * (u1*R[2][1] - R[0][1]) + z2x * sigma*R[2][1];
            dXs_cur[2] = z1x * (u1*R[2][2] - R[0][2]) + z2x * sigma*R[2][2];

            dXs_cur[0] += z1y * (u2*R[2][0] - R[1][0]) + z2y * sigma*R[2][0];
            dXs_cur[1] += z1y * (u2*R[2][1] - R[1][1]) + z2y * sigma*R[2][1];
            dXs_cur[2] += z1y * (u2*R[2][2] - R[1][2]) + z2y * sigma*R[2][2];

            dTs_k[k] = dTs_cur;
            dXs_k[k] = dXs_cur;

            dDs_k[k] = z1x*u1 + z1y*u2 + sigma*(z2x + z2y) + rhoWeight;
         } // end for (k)

         vector<Vector3d> dTs(N);
         vector<Vector3d> dXs(M);
         for (int i = 0; i < N; ++i) makeZeroVector(dTs[i]);
         for (int j = 0; j < M; ++j) makeZeroVector(dXs[j]);

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            addVectorsIP(dTs_k[k], dTs[i]);
            addVectorsIP(dXs_k[k], dXs[j]);
         } // end for (k)

         // Keep T[0] fixed to origin
#if NUMERICAL_SCHEME == 1
         // Chambolle
         std::copy(translations.begin(), translations.end(), Ts_saved.begin());
         std::copy(Xs.begin(), Xs.end(), Xs_saved.begin());
         std::copy(Ds.begin(), Ds.end(), Ds_saved.begin());

         for (int i = 1; i < N; ++i) addVectorsIP(-tau_p * dTs[i], translations[i]);

//          if (0) {
//             Matrix3x3d const& R = rotations[1];
//             Vector3d T = translations[1];
//             Vector3d u(-R[0][1], -R[1][1], -R[2][1]);
//             normalizeVector(u);
//             translations[1] = T - (innerProduct(u, T) - 1.0)*u;
//          }

         for (int j = 0; j < M; ++j) addVectorsIP(-tau_p * dXs[j], Xs[j]);
         for (int k = 0; k < K; ++k) Ds[k] -= tau_p * dDs_k[k];

         enforceCheiralityConstraints(rotations, correspondingView, correspondingPoint, cheiralityConstraints, translations, Xs, Ds);

         for (int i = 1; i < N; ++i) TTs[i] = 2.0*translations[i] - Ts_saved[i];
         for (int j = 0; j < M; ++j) XXs[j] = 2.0*Xs[j] - Xs_saved[j];
         for (int k = 0; k < K; ++k) DDs[k] = 2*Ds[k] - Ds_saved[k];
#else
         for (int i = 1; i < N; ++i) addVectorsIP(-tau_p * dTs[i], translations[i]);
         for (int j = 0; j < M; ++j) addVectorsIP(-tau_p * dXs[j], Xs[j]);
         for (int k = 0; k < K; ++k) Ds[k] -= tau_p * dDs_k[k];

         enforceCheiralityConstraints(rotations, correspondingView, correspondingPoint, cheiralityConstraints, translations, Xs, Ds);
#endif

         // Dual updates
         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
#if NUMERICAL_SCHEME == 1
            Vector3d   const& X = XXs[j];
            Vector3d   const& T = TTs[i];
            double     const& D = DDs[k];
#else
            Vector3d   const& X = Xs[j];
            Vector3d   const& T = translations[i];
            double     const& D = Ds[k];
#endif
            Vector2d   const& m = bundleStruct.measurements[k];
            float const u1 = m[0];
            float const u2 = m[1];

            double const XX1 = R[0][0]*X[0] + R[0][1]*X[1] + R[0][2]*X[2] + T[0];
            double const XX2 = R[1][0]*X[0] + R[1][1]*X[1] + R[1][2]*X[2] + T[1];
            double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2] + D;

#if !defined(USE_FIXED_BALLOONING_TERM)
            Zs[k][0] += tau_d * (u1*XX3 - XX1 - eta*Zs[k][0]);
            Zs[k][1] += tau_d * (sigma * XX3 - eta);

            Zs[k][2] += tau_d * (u2*XX3 - XX2 - eta*Zs[k][2]);
            Zs[k][3] += tau_d * (sigma * XX3 - eta);

            projectZs_Huber(Zs[k][0], Zs[k][1]);
            projectZs_Huber(Zs[k][2], Zs[k][3]);
#else
            Zs[k][0] += tau_d * (u1*XX3 - XX1 - sigma*Zs[k][0]);
            Zs[k][2] += tau_d * (u2*XX3 - XX2 - sigma*Zs[k][2]);

            Zs[k][0] = clamp(Zs[k][0], -1.0f, 1.0f);
            Zs[k][2] = clamp(Zs[k][2], -1.0f, 1.0f);
#endif
         } // end for (k)

         if ((iter % params.reportFrequency) == 0 || iter == params.nMaxIterations-1)
         {
            cout << "iter = " << iter << endl;

            //for (int i = 1; i < N; ++i) { cout << "dT[" << i << "] = "; displayVector(dTs[i]); }
            //for (int i = 1; i < N; ++i) { cout << "T[" << i << "] = "; displayVector(translations[i]); }

            double E = 0.0, E_data = 0.0, E_balloon = 0.0;

            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               Vector2d   const& m = bundleStruct.measurements[k];
               float const u1 = m[0];
               float const u2 = m[1];

               double const XX1 = R[0][0]*X[0] + R[0][1]*X[1] + R[0][2]*X[2] + T[0];
               double const XX2 = R[1][0]*X[0] + R[1][1]*X[1] + R[1][2]*X[2] + T[1];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2] + Ds[k];

               double const c = sigma*XX3;
               E_data += huber(XX1 - u1*XX3, c);
               E_data += huber(XX2 - u2*XX3, c);
               E += rhoWeight * Ds[k];
#if 1 || defined(USE_FIXED_BALLOONING_TERM)
               E_balloon += Zs[k][1] * sigma * XX3;
               E_balloon += Zs[k][3] * sigma * XX3;
#endif
            } // end for (k)
            cout << "E = " << E+E_data+E_balloon << ", E_data = " << E_data << ", E_balloon = " << E_balloon << endl;

            //cout << "Ds = "; displayVector(Ds);

            int nFeasible = 0;
            vector<double> X3s(K);
            double meanX3 = 0.0, meanD = 0.0;
            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];

               Matrix3x3d const& R = rotations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + translations[i][2] + Ds[k];
               if (XX3 >= 1.0) ++nFeasible;
               X3s[k] = XX3;
               meanX3 += XX3;
               meanD  += Ds[k];
            }
            cout << nFeasible << "/" << K << " measurements satisfy cheirality." << endl;
            //cout << "X3s = "; displayVector(X3s);
            cout << "avg X_3 = " << meanX3 / K << ", avg_D = " << meanD / K << endl;

            //cout << "dDs = "; displayVector(dDs_k);
            //cout << "Ds = "; displayVector(Ds);

#if !defined(USE_FIXED_BALLOONING_TERM)
            cout << "nCubicRoots = " << nCubicRoots << endl;
            nCubicRoots = 0;
#endif
            double const sim = computeStructureSimilarity(Xs, Xs_last);
            cout << "structure similarity = " << sim << endl;
            Xs_last = Xs;
            if (sim < params.similarityThreshold) break;
         } // end scope
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }
#undef NUMERICAL_SCHEME
   } // end computeConsistentTranslationsConic_Huber_PD()

   void
   computeConsistentTranslationsConic_Huber_PD_Popov(float const sigma,
                                                     std::vector<Matrix3x3d> const& rotations,
                                                     std::vector<Vector3d>& translations,
                                                     std::vector<TriangulatedPoint>& sparseModel,
                                                     TranslationRegistrationPD_Params const& params)
   {
      double const L = computeTranslationRegistrationLipschitzConstant(sigma, rotations, sparseModel);
      double const tau_p = params.timestepMultiplier * sqrt(params.timestepRatio) / L / 4;
      double const tau_d = params.timestepMultiplier / sqrt(params.timestepRatio) / L / 4;

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      vector<int> cheiralityConstraints;
      selectCheiralityConstraints(correspondingView, correspondingPoint, cheiralityConstraints);

      vector<Vector3d> Xs(M);
      vector<double>   Ds(K, 1.0);
      vector<Vector4f> Zs(K);

      for (int k = 0; k < K; ++k) makeZeroVector(Zs[k]);
      for (int i = 0; i < N; ++i) makeZeroVector(translations[i]);
      for (int j = 0; j < M; ++j) makeZeroVector(Xs[j]);

      vector<Vector3d> XXs(Xs);
      vector<double>   DDs(Ds);
      vector<Vector3d> TTs(translations);
      vector<Vector4f> ZZs(Zs);

      for (int i = 0; i < N; ++i) makeZeroVector(TTs[i]);
      for (int j = 0; j < M; ++j) makeZeroVector(XXs[j]);

      vector<Vector3d> dTs_k(K), dXs_k(K);
      vector<double> dDs_k(K);
      vector<Vector3d> dTs(N);
      vector<Vector3d> dXs(M);
      vector<Vector4d> dZs(K);
      vector<Vector3d> Xs_last(Xs);

      for (int iter = 0; iter < params.nMaxIterations; ++iter)
      {
         // Primal updates
         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            Vector3d   const& T = TTs[i];
            Vector3d   const& X = XXs[j];
            double     const& D = DDs[k];

            Vector2d   const& m  = bundleStruct.measurements[k];
            float      const  u1 = m[0];
            float      const  u2 = m[1];

            float const z1x = ZZs[k][0];
            float const z2x = ZZs[k][1];

            float const z1y = ZZs[k][2];
            float const z2y = ZZs[k][3];

            Vector3d dTs_cur, dXs_cur;

            dTs_cur[0] = -z1x;
            dTs_cur[1] = -z1y;
            dTs_cur[2] = z1x*u1 + z1y*u2 + sigma*(z2x + z2y);

            dXs_cur[0] = z1x * (u1*R[2][0] - R[0][0]) + z2x * sigma*R[2][0];
            dXs_cur[1] = z1x * (u1*R[2][1] - R[0][1]) + z2x * sigma*R[2][1];
            dXs_cur[2] = z1x * (u1*R[2][2] - R[0][2]) + z2x * sigma*R[2][2];

            dXs_cur[0] += z1y * (u2*R[2][0] - R[1][0]) + z2y * sigma*R[2][0];
            dXs_cur[1] += z1y * (u2*R[2][1] - R[1][1]) + z2y * sigma*R[2][1];
            dXs_cur[2] += z1y * (u2*R[2][2] - R[1][2]) + z2y * sigma*R[2][2];

            dTs_k[k] = dTs_cur;
            dXs_k[k] = dXs_cur;

            dDs_k[k] = z1x*u1 + z1y*u2 + sigma*(z2x + z2y) + 1.0;

            double const XX1 = R[0][0]*X[0] + R[0][1]*X[1] + R[0][2]*X[2] + T[0];
            double const XX2 = R[1][0]*X[0] + R[1][1]*X[1] + R[1][2]*X[2] + T[1];
            double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2] + D;

            dZs[k][0] = tau_d * (u1*XX3 - XX1 - sigma*Zs[k][0]);
            dZs[k][1] = tau_d * (sigma*XX3 - sigma);

            dZs[k][2] = tau_d * (u2*XX3 - XX2 - sigma*Zs[k][2]);
            dZs[k][3] = tau_d * (sigma*XX3 - sigma);
         } // end for (k)

         for (int i = 0; i < N; ++i) makeZeroVector(dTs[i]);
         for (int j = 0; j < M; ++j) makeZeroVector(dXs[j]);

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            addVectorsIP(dTs_k[k], dTs[i]);
            addVectorsIP(dXs_k[k], dXs[j]);
         } // end for (k)

         // Keep T[0] fixed to origin
         for (int i = 1; i < N; ++i) addVectorsIP(-tau_p * dTs[i], translations[i]);
         for (int j = 0; j < M; ++j) addVectorsIP(-tau_p * dXs[j], Xs[j]);
         for (int k = 0; k < K; ++k) Ds[k] = std::max(0.0, Ds[k] - tau_p * dDs_k[k]);

         //rescaleStructure(rotations, correspondingView, correspondingPoint, translations, Xs, Ds);
         enforceCheiralityConstraints(rotations, correspondingView, correspondingPoint, cheiralityConstraints, translations, Xs, Ds);

         // Dual updates
         for (int k = 0; k < K; ++k)
         {
            Zs[k][0] = Zs[k][0] + dZs[k][0];
            Zs[k][1] = Zs[k][1] + dZs[k][1];
            Zs[k][2] = Zs[k][2] + dZs[k][2];
            Zs[k][3] = Zs[k][3] + dZs[k][3];

            projectZs_Huber(Zs[k][0], Zs[k][1]);
            projectZs_Huber(Zs[k][2], Zs[k][3]);
         } // end for (k)

         // Leading point update:
         for (int i = 1; i < N; ++i) addVectors(translations[i], -tau_p * dTs[i], TTs[i]);
         for (int j = 0; j < M; ++j) addVectors(Xs[j], -tau_p * dXs[j], XXs[j]);
         for (int k = 0; k < K; ++k) DDs[k] = std::max(0.0, Ds[k] - tau_p * dDs_k[k]);

         //rescaleStructure(rotations, correspondingView, correspondingPoint, TTs, XXs, DDs);
         enforceCheiralityConstraints(rotations, correspondingView, correspondingPoint, cheiralityConstraints, TTs, XXs, DDs);

         // Dual updates
         for (int k = 0; k < K; ++k)
         {
            ZZs[k][0] = Zs[k][0] + dZs[k][0];
            ZZs[k][1] = Zs[k][1] + dZs[k][1];
            ZZs[k][2] = Zs[k][2] + dZs[k][2];
            ZZs[k][3] = Zs[k][3] + dZs[k][3];

            projectZs_Huber(ZZs[k][0], ZZs[k][1]);
            projectZs_Huber(ZZs[k][2], ZZs[k][3]);
         } // end for (k)

         if ((iter % params.reportFrequency) == 0 || iter == params.nMaxIterations-1)
         {
            cout << "iter = " << iter << endl;

            //for (int i = 1; i < N; ++i) { cout << "dT[" << i << "] = "; displayVector(dTs[i]); }
            //for (int i = 1; i < N; ++i) { cout << "T[" << i << "] = "; displayVector(translations[i]); }

            double E = 0.0;

            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               Vector2d   const& m = bundleStruct.measurements[k];
               float const u1 = m[0];
               float const u2 = m[1];

               double const XX1 = R[0][0]*X[0] + R[0][1]*X[1] + R[0][2]*X[2] + T[0];
               double const XX2 = R[1][0]*X[0] + R[1][1]*X[1] + R[1][2]*X[2] + T[1];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2] + Ds[k];

               double const c = sigma*XX3;
               E += huber(XX1 - u1*XX3, c);
               E += huber(XX2 - u2*XX3, c);
               E += Ds[k];
            } // end for (k)
            cout << "E = " << E << endl;

            //cout << "Ds = "; displayVector(Ds);

            int nFeasible = 0;
            vector<double> X3s(K);
            double meanX3 = 0.0;
            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];

               Matrix3x3d const& R = rotations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + translations[i][2] + Ds[k];
               if (XX3 >= 1.0) ++nFeasible;
               X3s[k] = XX3;
               meanX3 += XX3;
            }
            cout << nFeasible << "/" << K << " measurements satisfy cheirality." << endl;
            //cout << "X3s = "; displayVector(X3s);
            cout << "avg X_3 = " << meanX3 / K << endl;

            double const sim = computeStructureSimilarity(Xs, Xs_last);
            cout << "structure similarity = " << sim << endl;
            Xs_last = Xs;
            if (sim < params.similarityThreshold) break;
         } // end scope
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }
#undef NUMERICAL_SCHEME
   } // end computeConsistentTranslationsConic_Huber_PD()

} // end namespace V3D
