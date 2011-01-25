#include "Math/v3d_linear_lu.h"
#include "Math/v3d_optimization.h"
#include "Math/v3d_sparseeig.h"
#include "Math/v3d_sparsechol.h"
#include "Geometry/v3d_mviewinitialization.h"

using namespace std;
using namespace V3D;

float cbrtf(float xx); //HA

#if defined(V3DLIB_ENABLE_LPSOLVE)

extern "C"
{
#include "lpsolve/lp_lib.h"
}

namespace V3D
{

   void
   computeConsistentTranslationsOSE_L1(std::vector<Matrix3x3d> const& rotations,
                                       std::vector<float> const& weights,
                                       std::vector<Vector3d>& translations,
                                       std::vector<TriangulatedPoint>& sparseModel)
   {
      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      // We have 3N + 3M + K variables and 6K inequalities.
      int const nVars = 3*N + 3*M + K;

      int const transStart = 1; // Columns are 1-based in lp-solve
      int const pointStart = transStart + 3*N;
      int const distStart  = pointStart + 3*M;

#define TVAR(i, el) (transStart + 3*(i) + (el))
#define XVAR(j, el) (pointStart + 3*(j) + (el))
#define DISTVAR(k) (distStart + (k))

      lprec * lp = make_lp(0, nVars);

      set_add_rowmode(lp, TRUE);

      {
         // Set objective coefficients.
         vector<REAL> c(K);
         vector<int> colNo(K);
         for (int k = 0; k < K; ++k)
         {
            c[k]     = weights[k];
            colNo[k] = DISTVAR(k);
         }
         set_obj_fnex(lp, K, &c[0], &colNo[0]);
      }

      Vector3d u;
      Matrix3x3d uut, Pu, Qu;

      int k = 0;

      Matrix3x3d I; makeIdentityMatrix(I);

      {
         int colNo[7];
         REAL row[7];
         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            Vector2d const& m = bundleStruct.measurements[k];

            u[0] = m[0];
            u[1] = m[1];
            u[2] = 1.0;
            normalizeVector(u);
            makeOuterProductMatrix(u, u, uut);

            Pu = (I - uut) * R;

            // Camera center.
            colNo[0] = TVAR(i, 0); colNo[1] = TVAR(i, 1); colNo[2] = TVAR(i, 2);
            // 3D point.
            colNo[3] = XVAR(j, 0); colNo[4] = XVAR(j, 1); colNo[5] = XVAR(j, 2);

            // X component
            colNo[6] = DISTVAR(k);

            row[0] = -Pu[0][0]; row[1] = -Pu[0][1]; row[2] = -Pu[0][2];
            row[3] =  Pu[0][0]; row[4] =  Pu[0][1]; row[5] =  Pu[0][2];
            row[6] = -1;
            add_constraintex(lp, 7, row, colNo, LE, 0.0);

            colNo[0] = TVAR(i, 0); colNo[1] = TVAR(i, 1); colNo[2] = TVAR(i, 2);
            colNo[3] = XVAR(j, 0); colNo[4] = XVAR(j, 1); colNo[5] = XVAR(j, 2);
            colNo[6] = DISTVAR(k);

            row[0] = -Pu[0][0]; row[1] = -Pu[0][1]; row[2] = -Pu[0][2];
            row[3] =  Pu[0][0]; row[4] =  Pu[0][1]; row[5] =  Pu[0][2];
            row[6] = 1;
            add_constraintex(lp, 7, row, colNo, GE, 0.0);

            // Y component
            colNo[0] = TVAR(i, 0); colNo[1] = TVAR(i, 1); colNo[2] = TVAR(i, 2);
            colNo[3] = XVAR(j, 0); colNo[4] = XVAR(j, 1); colNo[5] = XVAR(j, 2);
            colNo[6] = DISTVAR(k);

            row[0] = -Pu[1][0]; row[1] = -Pu[1][1]; row[2] = -Pu[1][2];
            row[3] =  Pu[1][0]; row[4] =  Pu[1][1]; row[5] =  Pu[1][2];
            row[6] = -1;
            add_constraintex(lp, 7, row, colNo, LE, 0.0);

            colNo[0] = TVAR(i, 0); colNo[1] = TVAR(i, 1); colNo[2] = TVAR(i, 2);
            colNo[3] = XVAR(j, 0); colNo[4] = XVAR(j, 1); colNo[5] = XVAR(j, 2);
            colNo[6] = DISTVAR(k);

            row[0] = -Pu[1][0]; row[1] = -Pu[1][1]; row[2] = -Pu[1][2];
            row[3] =  Pu[1][0]; row[4] =  Pu[1][1]; row[5] =  Pu[1][2];
            row[6] = 1;
            add_constraintex(lp, 7, row, colNo, GE, 0.0);

            // Z component
            colNo[0] = TVAR(i, 0); colNo[1] = TVAR(i, 1); colNo[2] = TVAR(i, 2);
            colNo[3] = XVAR(j, 0); colNo[4] = XVAR(j, 1); colNo[5] = XVAR(j, 2);
            colNo[6] = DISTVAR(k);

            row[0] = -Pu[2][0]; row[1] = -Pu[2][1]; row[2] = -Pu[2][2];
            row[3] =  Pu[2][0]; row[4] =  Pu[2][1]; row[5] =  Pu[2][2];
            row[6] = -1;
            add_constraintex(lp, 7, row, colNo, LE, 0.0);

            colNo[0] = TVAR(i, 0); colNo[1] = TVAR(i, 1); colNo[2] = TVAR(i, 2);
            colNo[3] = XVAR(j, 0); colNo[4] = XVAR(j, 1); colNo[5] = XVAR(j, 2);
            colNo[6] = DISTVAR(k);

            row[0] = -Pu[2][0]; row[1] = -Pu[2][1]; row[2] = -Pu[2][2];
            row[3] =  Pu[2][0]; row[4] =  Pu[2][1]; row[5] =  Pu[2][2];
            row[6] = 1;
            add_constraintex(lp, 7, row, colNo, GE, 0.0);
         } // end for (l)
      } // end for (j)

      if (1) {
         // Fix the translation ambiguity.
         int colNo[1];
         REAL row[1] = { 1.0 };
         colNo[0] = TVAR(0, 0); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
         colNo[0] = TVAR(0, 1); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
         colNo[0] = TVAR(0, 2); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
      }

      // Fix the scale ambiguity.
      {
         int colNo[6];
         REAL row[6];

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            Vector2d const& m = bundleStruct.measurements[k];

            u[0] = m[0];
            u[1] = m[1];
            u[2] = 1.0;
            normalizeVector(u);

            u = R.transposed() * u;

            colNo[0] = TVAR(i, 0); colNo[1] = TVAR(i, 1); colNo[2] = TVAR(i, 2);
            colNo[3] = XVAR(j, 0); colNo[4] = XVAR(j, 1); colNo[5] = XVAR(j, 2);

            row[0] = -u[0]; row[1] = -u[1]; row[2] = -u[2];
            row[3] =  u[0]; row[4] =  u[1]; row[5] =  u[2];
            add_constraintex(lp, 6, row, colNo, GE, 1.0);
            //break;
         } // end for (l)
      } // end for (j)

      set_add_rowmode(lp, FALSE);

      for (int i = transStart; i < distStart; ++i) set_unbounded(lp, i);

      //set_timeout(lp, 1000);

      set_verbose(lp, 4);
      int res = solve(lp);
      cout << "lp return code = " << res << endl;

      REAL * Y;
      get_ptr_primal_solution(lp, &Y);
      int m = get_Nrows(lp);

//       for (int i = 0; i < N; ++i)
//       {
//          cout << "cam" << i << " = ["
//               << Y[transStart+m+3*i] << " "
//               << Y[transStart+m+3*i+1] << " "
//               << Y[transStart+m+3*i+2] << "]" << endl;
//       }

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];

         X.pos[0] = Y[m + XVAR(j, 0)];
         X.pos[1] = Y[m + XVAR(j, 1)];
         X.pos[2] = Y[m + XVAR(j, 2)];
      }

      for (int i = 0; i < N; ++i)
      {
         Vector3d center;
         center[0] = Y[m + TVAR(i, 0)];
         center[1] = Y[m + TVAR(i, 1)];
         center[2] = Y[m + TVAR(i, 2)];

         translations[i] = -(rotations[i] * center);
      }

      delete_lp(lp);

#undef TVAR
#undef XVAR
#undef DISTVAR
   } // end computeConsistentTranslationsOSE_L1()

//**********************************************************************

   void
   computeConsistentTranslationsConic_L1(float const sigma,
                                         std::vector<Matrix3x3d> const& rotations,
                                         std::vector<float> const& weights,
                                         std::vector<Vector3d>& translations,
                                         std::vector<TriangulatedPoint>& sparseModel)
   {
      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

#if 0
      int const nVars = 3*N + 3*M + 2*K + K;

      int const transStart  = 1; // Columns are 1-based in lp-solve
      int const pointStart  = transStart + 3*N;
      int const offsetStart = pointStart + 3*M;
      int const auxStart    = offsetStart + 2*K;

# define TVAR(i, el) (transStart + 3*(i) + (el))
# define XVAR(j, el) (pointStart + 3*(j) + (el))
# define OFSVAR(k, el) (offsetStart + 2*(k) + (el))
# define AUXVAR(k) (auxStart + (k))

      lprec * lp = make_lp(0, nVars);

      set_add_rowmode(lp, TRUE);

      {
         // Set objective coefficients.
         vector<REAL> c(K);
         vector<int> colNo(K);
         for (int k = 0; k < K; ++k)
         {
            c[k] = weights[k];
            colNo[k] = AUXVAR(k);
         }
         set_obj_fnex(lp, K, &c[0], &colNo[0]);
      }

      for (int k = transStart; k < auxStart; ++k) set_unbounded(lp, k);

      // Add the cheirality conditions
      {
         double row[4];
         int    colNo[4];

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            row[0] = R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[2][2]; colNo[2] = XVAR(j, 2);
            row[3] = 1.0;     colNo[3] = TVAR(i, 2);

            add_constraintex(lp, 4, row, colNo, GE, 1.0);
         } // end for (k)
      } // end scope

      // Add the "conic" constraints
      {
         double row[6];
         int    colNo[6];

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            Vector2d const& m = bundleStruct.measurements[k];
            float const u1 = m[0];
            float const u2 = m[1];

            // x-component
            row[0] = R[0][0] - (sigma + u1)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[0][1] - (sigma + u1)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[0][2] - (sigma + u1)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;           colNo[3] = TVAR(i, 0);
            row[4] = -(sigma + u1); colNo[4] = TVAR(i, 2);
            row[5] = 1.0;           colNo[5] = OFSVAR(k, 0);

            add_constraintex(lp, 6, row, colNo, LE, 0.0);

            row[0] = R[0][0] + (sigma - u1)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[0][1] + (sigma - u1)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[0][2] + (sigma - u1)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;          colNo[3] = TVAR(i, 0);
            row[4] = (sigma - u1); colNo[4] = TVAR(i, 2);
            row[5] = 1.0;          colNo[5] = OFSVAR(k, 0);

            add_constraintex(lp, 6, row, colNo, GE, 0.0);

            // y-component
            row[0] = R[1][0] - (sigma + u2)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[1][1] - (sigma + u2)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[1][2] - (sigma + u2)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;           colNo[3] = TVAR(i, 1);
            row[4] = -(sigma + u2); colNo[4] = TVAR(i, 2);
            row[5] = 1.0;           colNo[5] = OFSVAR(k, 1);

            add_constraintex(lp, 6, row, colNo, LE, 0.0);

            row[0] = R[1][0] + (sigma - u2)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[1][1] + (sigma - u2)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[1][2] + (sigma - u2)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;          colNo[3] = TVAR(i, 1);
            row[4] = (sigma - u2); colNo[4] = TVAR(i, 2);
            row[5] = 1.0;          colNo[5] = OFSVAR(k, 1);

            add_constraintex(lp, 6, row, colNo, GE, 0.0);
         } // end for (k)
      } // end scope

      if (1)
      {
         // Fix the translation ambiguity.
         int colNo[1];
         REAL row[1] = { 1.0 };
         row[0] = 1.0; colNo[0] = TVAR(0, 0); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
         row[0] = 1.0; colNo[0] = TVAR(0, 1); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
         row[0] = 1.0; colNo[0] = TVAR(0, 2); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
      }

      // Add auxVar >= |ofsVar|
      {
         int colNo[2];
         REAL row[2];

         for (int k = 0; k < K; ++k)
         {
            row[0] = 1.0;  colNo[0] = AUXVAR(k);
            row[1] = -1.0; colNo[1] = OFSVAR(k, 0);
            add_constraintex(lp, 2, row, colNo, GE, 0.0);

            row[0] = 1.0; colNo[0] = AUXVAR(k);
            row[1] = 1.0; colNo[1] = OFSVAR(k, 0);
            add_constraintex(lp, 2, row, colNo, GE, 0.0);

            row[0] = 1.0;  colNo[0] = AUXVAR(k);
            row[1] = -1.0; colNo[1] = OFSVAR(k, 1);
            add_constraintex(lp, 2, row, colNo, GE, 0.0);

            row[0] = 1.0; colNo[0] = AUXVAR(k);
            row[1] = 1.0; colNo[1] = OFSVAR(k, 1);
            add_constraintex(lp, 2, row, colNo, GE, 0.0);
         } // end for (k)
      }

      set_add_rowmode(lp, FALSE);

      //set_timeout(lp, 1000);

      set_verbose(lp, 4);
      int res = solve(lp);
      cout << "lp return code = " << res << endl;

      REAL * Y;
      get_ptr_primal_solution(lp, &Y);
      int m = get_Nrows(lp);


      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];

         X.pos[0] = Y[m + XVAR(j, 0)];
         X.pos[1] = Y[m + XVAR(j, 1)];
         X.pos[2] = Y[m + XVAR(j, 2)];
      }

      for (int i = 0; i < N; ++i)
      {
         translations[i][0] = Y[m + TVAR(i, 0)];
         translations[i][1] = Y[m + TVAR(i, 1)];
         translations[i][2] = Y[m + TVAR(i, 2)];
      }

      delete_lp(lp);

# undef TVAR
# undef XVAR
# undef OFSVAR
# undef AUXVAR
#else
      int const nVars = 3*N + 3*M + 2*K;

      int const transStart  = 1; // Columns are 1-based in lp-solve
      int const pointStart  = transStart + 3*N;
      int const offsetStart = pointStart + 3*M;

# define TVAR(i, el) (transStart + 3*(i) + (el))
# define XVAR(j, el) (pointStart + 3*(j) + (el))
# define OFSVAR(k, el) (offsetStart + 2*(k) + (el))

      lprec * lp = make_lp(0, nVars);

      set_add_rowmode(lp, TRUE);

      {
         // Set objective coefficients.
         vector<REAL> c(2*K);
         vector<int> colNo(2*K);
         for (int k = 0; k < K; ++k)
         {
            c[2*k+0]     = weights[k];
            colNo[2*k+0] = OFSVAR(k, 0);
            c[2*k+1]     = weights[k];
            colNo[2*k+1] = OFSVAR(k, 1);
         }
         set_obj_fnex(lp, 2*K, &c[0], &colNo[0]);
      }

      for (int k = transStart; k < offsetStart; ++k) set_unbounded(lp, k);

      // Add the cheirality conditions
      {
         double row[4];
         int    colNo[4];

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            row[0] = R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[2][2]; colNo[2] = XVAR(j, 2);
            row[3] = 1.0;     colNo[3] = TVAR(i, 2);

            add_constraintex(lp, 4, row, colNo, GE, 1.0);
         } // end for (k)
      } // end scope

      // Add the "conic" constraints
      {
         double row[6];
         int    colNo[6];

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            Vector2d const& m = bundleStruct.measurements[k];
            float const u1 = m[0];
            float const u2 = m[1];

            // x-component
            row[0] = R[0][0] + (sigma - u1)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[0][1] + (sigma - u1)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[0][2] + (sigma - u1)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;          colNo[3] = TVAR(i, 0);
            row[4] = (sigma - u1); colNo[4] = TVAR(i, 2);
            row[5] = 1.0;          colNo[5] = OFSVAR(k, 0);

            add_constraintex(lp, 6, row, colNo, GE, 0.0);

            row[0] = R[0][0] - (sigma + u1)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[0][1] - (sigma + u1)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[0][2] - (sigma + u1)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;           colNo[3] = TVAR(i, 0);
            row[4] = -(sigma + u1); colNo[4] = TVAR(i, 2);
            row[5] = -1.0;          colNo[5] = OFSVAR(k, 0);

            add_constraintex(lp, 6, row, colNo, LE, 0.0);

            // y-component
            row[0] = R[1][0] + (sigma - u2)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[1][1] + (sigma - u2)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[1][2] + (sigma - u2)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;          colNo[3] = TVAR(i, 1);
            row[4] = (sigma - u2); colNo[4] = TVAR(i, 2);
            row[5] = 1.0;          colNo[5] = OFSVAR(k, 1);

            add_constraintex(lp, 6, row, colNo, GE, 0.0);

            row[0] = R[1][0] - (sigma + u2)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[1][1] - (sigma + u2)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[1][2] - (sigma + u2)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;           colNo[3] = TVAR(i, 1);
            row[4] = -(sigma + u2); colNo[4] = TVAR(i, 2);
            row[5] = -1.0;          colNo[5] = OFSVAR(k, 1);

            add_constraintex(lp, 6, row, colNo, LE, 0.0);
         } // end for (k)
      } // end scope

      if (1)
      {
         // Fix the translation ambiguity.
         int colNo[1];
         REAL row[1] = { 1.0 };
         row[0] = 1.0; colNo[0] = TVAR(0, 0); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
         row[0] = 1.0; colNo[0] = TVAR(0, 1); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
         row[0] = 1.0; colNo[0] = TVAR(0, 2); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
      }

      set_add_rowmode(lp, FALSE);

      //set_timeout(lp, 1000);

      set_verbose(lp, 4);
      int res = solve(lp);
      cout << "lp return code = " << res << endl;

      REAL * Y;
      get_ptr_primal_solution(lp, &Y);
      int m = get_Nrows(lp);


      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];

         X.pos[0] = Y[m + XVAR(j, 0)];
         X.pos[1] = Y[m + XVAR(j, 1)];
         X.pos[2] = Y[m + XVAR(j, 2)];
      }

      for (int i = 0; i < N; ++i)
      {
         translations[i][0] = Y[m + TVAR(i, 0)];
         translations[i][1] = Y[m + TVAR(i, 1)];
         translations[i][2] = Y[m + TVAR(i, 2)];
      }

      delete_lp(lp);

# undef TVAR
# undef XVAR
# undef OFSVAR
#endif
   } // end computeConsistentTranslationsConic_L1()

   void
   computeConsistentTranslationsConic_L1_reduced(float const sigma,
                                                 std::vector<Matrix3x3d> const& rotations,
                                                 std::vector<Vector3d>& translations,
                                                 std::vector<TriangulatedPoint>& sparseModel)
   {
      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      int const nVars = 3*N + 3*M + M;

      int const transStart = 1; // Columns are 1-based in lp-solve
      int const pointStart = transStart + 3*N;
      int const residStart = pointStart + 3*M;

#define TVAR(i, el) (transStart + 3*(i) + (el))
#define XVAR(j, el) (pointStart + 3*(j) + (el))
#define RESVAR(j) (residStart + (j))

      lprec * lp = make_lp(0, nVars);

      set_add_rowmode(lp, TRUE);

      {
         // Set objective coefficients.
         vector<REAL> c(M, 1.0);
         vector<int> colNo(M);
         for (int j = 0; j < M; ++j) colNo[j] = RESVAR(j);

         set_obj_fnex(lp, M, &c[0], &colNo[0]);
      }

      for (int k = transStart; k < residStart; ++k) set_unbounded(lp, k);

      // Add the cheirality conditions
      {
         double row[4];
         int    colNo[4];

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            row[0] = R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[2][2]; colNo[2] = XVAR(j, 2);
            row[3] = 1.0;     colNo[3] = TVAR(i, 2);

            add_constraintex(lp, 4, row, colNo, GE, 1.0);
            break;
         } // end for (k)
      } // end scope

      // Add the "conic" constraints
      {
         double row[6];
         int    colNo[6];

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            Vector2d const& m = bundleStruct.measurements[k];
            float const u1 = m[0];
            float const u2 = m[1];

            // x-component
            row[0] = R[0][0] - (sigma + u1)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[0][1] - (sigma + u1)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[0][2] - (sigma + u1)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;           colNo[3] = TVAR(i, 0);
            row[4] = -(sigma + u1); colNo[4] = TVAR(i, 2);
            row[5] = -1.0;          colNo[5] = RESVAR(j);

            add_constraintex(lp, 6, row, colNo, LE, 0.0);

            row[0] = R[0][0] + (sigma - u1)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[0][1] + (sigma - u1)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[0][2] + (sigma - u1)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;          colNo[3] = TVAR(i, 0);
            row[4] = (sigma - u1); colNo[4] = TVAR(i, 2);
            row[5] = 1.0;          colNo[5] = RESVAR(j);

            add_constraintex(lp, 6, row, colNo, GE, 0.0);

            // y-component
            row[0] = R[1][0] - (sigma + u2)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[1][1] - (sigma + u2)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[1][2] - (sigma + u2)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;           colNo[3] = TVAR(i, 1);
            row[4] = -(sigma + u2); colNo[4] = TVAR(i, 2);
            row[5] = -1.0;          colNo[5] = RESVAR(j);

            add_constraintex(lp, 6, row, colNo, LE, 0.0);

            row[0] = R[1][0] + (sigma - u2)*R[2][0]; colNo[0] = XVAR(j, 0);
            row[1] = R[1][1] + (sigma - u2)*R[2][1]; colNo[1] = XVAR(j, 1);
            row[2] = R[1][2] + (sigma - u2)*R[2][2]; colNo[2] = XVAR(j, 2);

            row[3] = 1.0;          colNo[3] = TVAR(i, 1);
            row[4] = (sigma - u2); colNo[4] = TVAR(i, 2);
            row[5] = 1.0;          colNo[5] = RESVAR(j);

            add_constraintex(lp, 6, row, colNo, GE, 0.0);
         } // end for (k)
      } // end scope

      if (1)
      {
         // Fix the translation ambiguity.
         int colNo[1];
         REAL row[1] = { 1.0 };
         row[0] = 1.0; colNo[0] = TVAR(0, 0); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
         row[0] = 1.0; colNo[0] = TVAR(0, 1); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
         row[0] = 1.0; colNo[0] = TVAR(0, 2); add_constraintex(lp, 1, row, colNo, EQ, 0.0);
      }

      set_add_rowmode(lp, FALSE);

      //set_timeout(lp, 1000);

      set_verbose(lp, 4);
      //set_simplextype(lp, SIMPLEX_DUAL_DUAL);
      int res = solve(lp);
      cout << "lp return code = " << res << endl;

      REAL * Y;
      get_ptr_primal_solution(lp, &Y);
      int m = get_Nrows(lp);


      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];

         X.pos[0] = Y[m + XVAR(j, 0)];
         X.pos[1] = Y[m + XVAR(j, 1)];
         X.pos[2] = Y[m + XVAR(j, 2)];
      }

      for (int i = 0; i < N; ++i)
      {
         translations[i][0] = Y[m + TVAR(i, 0)];
         translations[i][1] = Y[m + TVAR(i, 1)];
         translations[i][2] = Y[m + TVAR(i, 2)];
      }

      delete_lp(lp);

#undef TVAR
#undef XVAR
#undef RESVAR
   } // end computeConsistentTranslationsConic_L1_reduced()

} // end namespace V3D

#endif

//**********************************************************************

namespace V3D
{

   void
   computeConsistentTranslationsConic_L1_New(float const sigma,
                                             std::vector<Matrix3x3d> const& rotations,
                                             std::vector<float> const& weights,
                                             std::vector<Vector3d>& translations,
                                             std::vector<TriangulatedPoint>& sparseModel,
                                             bool useIP, bool useInitialValue)
   {
      LP_SolverType solver = useIP ? LP_DSDP : LP_LPSOLVE55;
#if !defined(V3DLIB_ENABLE_DSDP)
      if (solver == LP_DSDP)
      {
         solver = LP_LPSOLVE55;
         cerr << "computeConsistentTranslationsConic_L1_New(): Switching to lp_solve, since DSDP not enabled in V3D." << endl;
      }
#endif

#if !defined(V3DLIB_ENABLE_LPSOLVE)
      if (solver == LP_LPSOLVE55)
         throwV3DErrorHere("computeConsistentTranslationsConic_L1_New(): lp_solve not enabled in V3D.");
#endif

      int const N = rotations.size();
      int const M = sparseModel.size();

      if (!useInitialValue) translations.resize(N);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      int const nVars = 3*N + 3*M + 3*K;

      int const transStart  = 0;
      int const pointStart  = transStart + 3*N;
      int const offsetStart = pointStart + 3*M;

# define TVAR(i, el) (transStart + 3*(i) + (el))
# define XVAR(j, el) (pointStart + 3*(j) + (el))
# define OFSVAR(k, el) (offsetStart + 3*(k) + (el))

      vector<pair<int, int> > nz;
      vector<double> values;

      // Set objective coefficients.
      vector<double> obj(nVars, 0.0);
      for (int k = 0; k < K; ++k)
      {
         obj[OFSVAR(k, 0)] = weights[k];
         obj[OFSVAR(k, 1)] = weights[k];
         obj[OFSVAR(k, 2)] = weights[k];
      }

      // Set the non-negative vars.
      vector<int> nonNegativeVariables(3*K);
      for (int k = 0; k < 3*K; ++k) nonNegativeVariables[k] = offsetStart + k;

      vector<double> b;
      vector<ConstraintType> constraintTypes;

      b.reserve(K + 4*K + 1);
      constraintTypes.reserve(K + 4*K + 1);

      int rowPos = 0;

#if 1
      // Add the cheirality conditions (R_i*X_j + T_i)_3 + Z_ij >= 1
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R = rotations[i];
         nz.push_back(make_pair(rowPos, XVAR(j, 0))); values.push_back(R[2][0]);
         nz.push_back(make_pair(rowPos, XVAR(j, 1))); values.push_back(R[2][1]);
         nz.push_back(make_pair(rowPos, XVAR(j, 2))); values.push_back(R[2][2]);
         nz.push_back(make_pair(rowPos, TVAR(i, 2))); values.push_back(1.0);
         nz.push_back(make_pair(rowPos, OFSVAR(k, 2))); values.push_back(1.0);

         constraintTypes.push_back(LP_GREATER_EQUAL); b.push_back(1.0); ++rowPos;
      } // end for (k)
#else
      {
         // Add just one cheirality condition.
         int const k = 0;
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R = rotations[i];
         nz.push_back(make_pair(rowPos, XVAR(j, 0))); values.push_back(R[2][0]);
         nz.push_back(make_pair(rowPos, XVAR(j, 1))); values.push_back(R[2][1]);
         nz.push_back(make_pair(rowPos, XVAR(j, 2))); values.push_back(R[2][2]);
         nz.push_back(make_pair(rowPos, TVAR(i, 2))); values.push_back(1.0);
         nz.push_back(make_pair(rowPos, OFSVAR(k, 2))); values.push_back(1.0);

         constraintTypes.push_back(LP_EQUAL); b.push_back(1.0); ++rowPos;
      }
#endif

      // Add the "conic" constraints
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R = rotations[i];
         Vector2d const& m = bundleStruct.measurements[k];
         float const u1 = m[0];
         float const u2 = m[1];

         // x-component
         nz.push_back(make_pair(rowPos, XVAR(j, 0))); values.push_back(R[0][0] + (sigma - u1)*R[2][0]);
         nz.push_back(make_pair(rowPos, XVAR(j, 1))); values.push_back(R[0][1] + (sigma - u1)*R[2][1]);
         nz.push_back(make_pair(rowPos, XVAR(j, 2))); values.push_back(R[0][2] + (sigma - u1)*R[2][2]);

         nz.push_back(make_pair(rowPos, TVAR(i, 0))); values.push_back(1.0);
         nz.push_back(make_pair(rowPos, TVAR(i, 2))); values.push_back(sigma - u1);
         nz.push_back(make_pair(rowPos, OFSVAR(k, 0))); values.push_back(1.0);

         constraintTypes.push_back(LP_GREATER_EQUAL); b.push_back(0.0); ++rowPos;

         nz.push_back(make_pair(rowPos, XVAR(j, 0))); values.push_back(R[0][0] - (sigma + u1)*R[2][0]);
         nz.push_back(make_pair(rowPos, XVAR(j, 1))); values.push_back(R[0][1] - (sigma + u1)*R[2][1]);
         nz.push_back(make_pair(rowPos, XVAR(j, 2))); values.push_back(R[0][2] - (sigma + u1)*R[2][2]);

         nz.push_back(make_pair(rowPos, TVAR(i, 0))); values.push_back(1.0);
         nz.push_back(make_pair(rowPos, TVAR(i, 2))); values.push_back(-(sigma + u1));
         nz.push_back(make_pair(rowPos, OFSVAR(k, 0))); values.push_back(-1.0);

         constraintTypes.push_back(LP_LESS_EQUAL); b.push_back(0.0); ++rowPos;

         // y-component
         nz.push_back(make_pair(rowPos, XVAR(j, 0))); values.push_back(R[1][0] + (sigma - u2)*R[2][0]);
         nz.push_back(make_pair(rowPos, XVAR(j, 1))); values.push_back(R[1][1] + (sigma - u2)*R[2][1]);
         nz.push_back(make_pair(rowPos, XVAR(j, 2))); values.push_back(R[1][2] + (sigma - u2)*R[2][2]);

         nz.push_back(make_pair(rowPos, TVAR(i, 1))); values.push_back(1.0);
         nz.push_back(make_pair(rowPos, TVAR(i, 2))); values.push_back(sigma - u2);
         nz.push_back(make_pair(rowPos, OFSVAR(k, 1))); values.push_back(1.0);

         constraintTypes.push_back(LP_GREATER_EQUAL); b.push_back(0.0); ++rowPos;

         nz.push_back(make_pair(rowPos, XVAR(j, 0))); values.push_back(R[1][0] - (sigma + u2)*R[2][0]);
         nz.push_back(make_pair(rowPos, XVAR(j, 1))); values.push_back(R[1][1] - (sigma + u2)*R[2][1]);
         nz.push_back(make_pair(rowPos, XVAR(j, 2))); values.push_back(R[1][2] - (sigma + u2)*R[2][2]);

         nz.push_back(make_pair(rowPos, TVAR(i, 1))); values.push_back(1.0);
         nz.push_back(make_pair(rowPos, TVAR(i, 2))); values.push_back(-(sigma + u2));
         nz.push_back(make_pair(rowPos, OFSVAR(k, 1))); values.push_back(-1.0);

         constraintTypes.push_back(LP_LESS_EQUAL); b.push_back(0.0); ++rowPos;
      } // end for (k)

      // Fix the translation ambiguity.
      nz.push_back(make_pair(rowPos, TVAR(0, 0))); values.push_back(1.0);
      constraintTypes.push_back(LP_EQUAL); b.push_back(0.0); ++rowPos;
      nz.push_back(make_pair(rowPos, TVAR(0, 1))); values.push_back(1.0);
      constraintTypes.push_back(LP_EQUAL); b.push_back(0.0); ++rowPos;
      nz.push_back(make_pair(rowPos, TVAR(0, 2))); values.push_back(1.0);
      constraintTypes.push_back(LP_EQUAL); b.push_back(0.0); ++rowPos;

      cout << "rowPos = " << rowPos << ", b.size() = " << b.size() << endl;

      CCS_Matrix<double> A(b.size(), nVars, nz, values);

      vector<pair<int, double> > const emptyBounds;

      vector<double> Y(nVars);

      if (useInitialValue)
      {
         for (int i = 0; i < N; ++i)
         {
            Y[TVAR(i, 0)] = translations[i][0];
            Y[TVAR(i, 1)] = translations[i][1];
            Y[TVAR(i, 2)] = translations[i][2];
         }

         for (int j = 0; j < M; ++j)
         {
            TriangulatedPoint& X = sparseModel[j];
            Y[XVAR(j, 0)] = X.pos[0];
            Y[XVAR(j, 1)] = X.pos[1];
            Y[XVAR(j, 2)] = X.pos[2];
         }

         double const eps = 1e-3; // Add eps to be in the interior

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R = rotations[i];
            Vector2d const& m = bundleStruct.measurements[k];
            float const u1 = m[0];
            float const u2 = m[1];

            Vector3d const XX = R*sparseModel[j].pos + translations[i];

            Y[OFSVAR(k, 0)] = std::max(0.0, fabs(u1*XX[2] - XX[0]) - sigma*XX[2]) + eps;
            Y[OFSVAR(k, 1)] = std::max(0.0, fabs(u2*XX[2] - XX[1]) - sigma*XX[2]) + eps;
            Y[OFSVAR(k, 2)] = std::max(0.0, 1.0 - XX[2]) + eps;
         } // end for (k)
      } // end if (useInitialValue)

      LP_Configuration conf;
      conf.solverType = solver;
      conf.verbose    = true;
      conf.useInitialValue = useInitialValue;
         
      bool status = solveLinearProgram(obj, A, constraintTypes, b, emptyBounds, emptyBounds, nonNegativeVariables, Y, conf);

      cout << "LP solver status is " << status << endl;

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];

         X.pos[0] = Y[XVAR(j, 0)];
         X.pos[1] = Y[XVAR(j, 1)];
         X.pos[2] = Y[XVAR(j, 2)];
      }

      for (int i = 0; i < N; ++i)
      {
         translations[i][0] = Y[TVAR(i, 0)];
         translations[i][1] = Y[TVAR(i, 1)];
         translations[i][2] = Y[TVAR(i, 2)];
      }

# undef TVAR
# undef XVAR
# undef OFSVAR
   } // end computeConsistentTranslationsConic_L1_New()

} // end namespace V3D

//**********************************************************************

namespace
{

   template <typename T>
   inline T
   clamp(T x, T a, T b)
   {
      return std::max(a, std::min(b, x));
   }

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

} // end namespace <>

//**********************************************************************

namespace
{

   inline void
   projectZs(float& z1p, float& z1m, float& z2)
   {
#if 1
      float const z1mag = z1p + z1m;

      if (z1mag <= 1.0f && z2 < -z1mag) return;

      float const z1 = z1p - z1m;

      // Case 1: z2 >= |z1|, closest point is (0, 0)
      if (z2 >= z1mag)
      {
         z1p = z1m = z2 = 0.0f;
      }
      // Case 2: |z1| > 1, z2 < -|z1|
      else if (z1mag > 1.0f && z2 < -z1mag)
      {
         // Only need to clamp z1
         if (z1 > 0.0)
            z1p = 1.0, z1m = 0.0;
         else
            z1p = 0.0, z1m = 1.0;
      }
      else if (z1 >= 0.0f)
      {
         z1m = 0.0f;
         if (z1 > 1.0f && z2 < z1 - 2.0f)
         {
            // (z1, z2) = (1, -1)
            z1p = 1.0f; z2 = -1.0f;
         }
         else
         {
            // Closest point on line segment (0,0) to (1,-1)
            z1p = 0.5f * (z1 - z2);
            z2  = -z1p;
         }
      }
      else
      {
         // z1 < 0
         z1p = 0.0f;
         if (z1 < -1.0f && z2 < -z1 - 2.0f)
         {
            // (z1, z2) = (-1, -1)
            z1m = 1.0f; z2 = -1.0f;
         }
         else
         {
            // Closest point on line segment (0,0) to (-1,-1)
            z2 = 0.5f * (z1 + z2);
            z1m = -z2;
         }
      } // end if
#else
      z1p = clamp(z1p, 0.0f, 1.0f);
      z1m = clamp(z1m, 0.0f, 1.0f);
      z2 = std::min(z2, -(z1p+z1m));
#endif
//       if (z1p < 0.0 || z1p > 1.0 || z1m < 0.0 || z1m > 1.0 || z2 > -(z1p+z1m))
//       {
//          cout << "z = (" << z1p << ", " << z1m << ", " << z2 << ")" << endl;
//       }
   } // end projectZs()

   inline void
   projectZs(float& z1, float& z2)
   {
#if 1
      float const z1mag = fabs(z1);

      if (z1mag <= 1.0f && z2 < -z1mag) return;

      // Case 1: z2 >= |z1|, closest point is (0, 0)
      if (z2 >= z1mag)
      {
         z1 = z2 = 0.0f;
      }
      // Case 2: |z1| > 1, z2 < -|z1|
      else if (z1mag > 1.0f && z2 < -z1mag)
      {
         // Only need to clamp z1
         z1 = clamp(z1, -1.0f, 1.0f);
      }
      else if (z1 >= 0.0f)
      {
         if (z1 > 1.0f && z2 < z1 - 2.0f)
         {
            // (z1, z2) = (1, -1)
            z1 = 1.0f; z2 = -1.0f;
         }
         else
         {
            // Closest point on line segment (0,0) to (1,-1)
            z1 = 0.5f * (z1 - z2);
            z2  = -z1;
         }
      }
      else
      {
         // z1 < 0
         if (z1 < -1.0f && z2 < -z1 - 2.0f)
         {
            // (z1, z2) = (-1, -1)
            z1 = -1.0f; z2 = -1.0f;
         }
         else
         {
            // Closest point on line segment (0,0) to (-1,-1)
            z2 = 0.5f * (z1 + z2);
            z1 = z2;
         }
      } // end if
#else
      z1 = clamp(z1, -1.0f, 1.0f);
      z2 = std::min(z2, -fabs(z1));
#endif
   } // end projectZs()

} // end namepace <>

//**********************************************************************

namespace V3D
{

   bool
   computeConsistentCameraCenters_LP(std::vector<Vector3d> const& c_ji, std::vector<Vector3d> const& c_jk,
                                     std::vector<Vector3i> const& ijks,
                                     std::vector<Vector3d>& centers, bool verbose)
   {
#if defined(V3DLIB_ENABLE_LPSOLVE)
      typedef InlineVector<double, 6> Vector6d;

      int const N = centers.size();
      int const K = ijks.size();

      int const nVars = 3*N + K + 2*K;

      int const centersStart = 0;
      int const scalesStart = centersStart + 3*N;
      int const distStart   = scalesStart + K;

# define CENTERVAR(i, el) (centersStart + 3*(i) + (el))
# define SCALEVAR(k)      (scalesStart + (k))
# define DISTVAR(k, el)   (distStart + 2*(k) + (el))

      vector<double> costs(nVars, 0);
      for (int k = distStart; k < nVars; ++k) costs[k] = 1.0;

      // Set the non-negative vars.
      vector<int> nonNegativeVariables; nonNegativeVariables.reserve(2*K);
      for (int k = 0; k < K; ++k) { nonNegativeVariables.push_back(DISTVAR(k, 0)); nonNegativeVariables.push_back(DISTVAR(k, 1)); }

      vector<double> b;
      vector<ConstraintType> constraintTypes;

      vector<pair<int, int> > nz;
      vector<double> values;

      int const nRows = K*3*4;
      b.reserve(nRows); constraintTypes.reserve(nRows);

      int const nnz = K*3*4*4;
      nz.reserve(nnz); values.reserve(nnz);

      int rowPos = 0;

      // Add constraints |c_ij * s_ijk - c_j + c_i| \le d_ijk
      for (int k = 0; k < K; ++k)
      {
         int const i0 = ijks[k][0];
         int const i1 = ijks[k][1];
         int const i2 = ijks[k][2];

         Vector6d c;
         copyVectorSlice(c_ji[k], 0, 3, c, 0);
         copyVectorSlice(c_jk[k], 0, 3, c, 3);
         normalizeVector(c);

         for (int el = 0; el < 3; ++el)
         {
            nz.push_back(make_pair(rowPos, SCALEVAR(k)));       values.push_back(c[el]);
            nz.push_back(make_pair(rowPos, CENTERVAR(i1, el))); values.push_back(1.0);
            nz.push_back(make_pair(rowPos, CENTERVAR(i0, el))); values.push_back(-1.0);
            nz.push_back(make_pair(rowPos, DISTVAR(k, 0)));     values.push_back(1.0);
            constraintTypes.push_back(LP_GREATER_EQUAL); b.push_back(0.0); ++rowPos;

            nz.push_back(make_pair(rowPos, SCALEVAR(k)));       values.push_back(c[el]);
            nz.push_back(make_pair(rowPos, CENTERVAR(i1, el))); values.push_back(1.0);
            nz.push_back(make_pair(rowPos, CENTERVAR(i0, el))); values.push_back(-1.0);
            nz.push_back(make_pair(rowPos, DISTVAR(k, 0)));     values.push_back(-1.0);
            constraintTypes.push_back(LP_LESS_EQUAL); b.push_back(0.0); ++rowPos;

            nz.push_back(make_pair(rowPos, SCALEVAR(k)));       values.push_back(c[el+3]);
            nz.push_back(make_pair(rowPos, CENTERVAR(i1, el))); values.push_back(1.0);
            nz.push_back(make_pair(rowPos, CENTERVAR(i2, el))); values.push_back(-1.0);
            nz.push_back(make_pair(rowPos, DISTVAR(k, 1)));     values.push_back(1.0);
            constraintTypes.push_back(LP_GREATER_EQUAL); b.push_back(0.0); ++rowPos;

            nz.push_back(make_pair(rowPos, SCALEVAR(k)));       values.push_back(c[el+3]);
            nz.push_back(make_pair(rowPos, CENTERVAR(i1, el))); values.push_back(1.0);
            nz.push_back(make_pair(rowPos, CENTERVAR(i2, el))); values.push_back(-1.0);
            nz.push_back(make_pair(rowPos, DISTVAR(k, 1)));     values.push_back(-1.0);
            constraintTypes.push_back(LP_LESS_EQUAL); b.push_back(0.0); ++rowPos;
         } // end for (el)
      } // end for (k)

      vector<pair<int, double> > lowerBounds, upperBounds;
      for (int k = scalesStart; k < distStart; ++k) lowerBounds.push_back(make_pair(k, 1.0));

      CCS_Matrix<double> A(b.size(), nVars, nz, values);

      LP_Configuration conf;
      conf.verbose = verbose;
      conf.solverType = LP_LPSOLVE55;

      vector<double> Y(nVars);
      bool status = solveLinearProgram(costs, A, constraintTypes, b, lowerBounds, upperBounds, nonNegativeVariables, Y, conf);

      for (int i = 0; i < N; ++i)
      {
         centers[i][0] = Y[CENTERVAR(i, 0)];
         centers[i][1] = Y[CENTERVAR(i, 1)];
         centers[i][2] = Y[CENTERVAR(i, 2)];
      }

      return status;

# undef CENTERVAR
# undef SCALEVAR
# undef DISVAR
#else
      cerr << "computeConsistentTranslations_LP(): support for LP_SOLVE not compiled into V3D." << endl;
      return false;
#endif
   } // end computeConsistentTranslations_LP()

} // end namespace V3D

//**********************************************************************

namespace
{

   inline void projectP(Vector3d& p)
   {
      double const denom = std::max(1.0, norm_L2(p));
      scaleVectorIP(1.0 / denom, p);
   }

} // end namespace <>

namespace V3D
{

   bool
   computeConsistentCameraCenters_L2_BOS(std::vector<Vector3d> const& c_ji, std::vector<Vector3d> const& c_jk,
                                         std::vector<Vector3i> const& ijks, std::vector<Vector3d>& centers,
                                         MultiViewInitializationParams_BOS const& params)
   {
      int const K = ijks.size();
      int const N = centers.size();

      double L = params.L;
      if (L <= 0.0)
      {
#if defined(V3DLIB_ENABLE_ARPACK)
         // We have 6*K dual variables and 3*N + K primal ones
         vector<pair<int, int> > nz;
         vector<double> values;
         nz.reserve(3*6*K);
         values.reserve(3*6*K);

         int row = 0;
         for (int k = 0; k < K; ++k)
         {
            int const i0 = ijks[k][0];
            int const i1 = ijks[k][1];
            int const i2 = ijks[k][2];

            int const Sijk = k;
            int const Ci0  = K + 3*i0 + 0;
            int const Ci1  = K + 3*i0 + 1;
            int const Ci2  = K + 3*i0 + 2;
            int const Cj0  = K + 3*i1 + 0;
            int const Cj1  = K + 3*i1 + 1;
            int const Cj2  = K + 3*i1 + 2;
            int const Ck0  = K + 3*i2 + 0;
            int const Ck1  = K + 3*i2 + 1;
            int const Ck2  = K + 3*i2 + 2;

            nz.push_back(make_pair(row, Sijk)); values.push_back(-c_ji[k][0]);
            nz.push_back(make_pair(row, Ci0));  values.push_back(1.0);
            nz.push_back(make_pair(row, Cj0));  values.push_back(-1.0);
            ++row;

            nz.push_back(make_pair(row, Sijk)); values.push_back(-c_ji[k][1]);
            nz.push_back(make_pair(row, Ci1));  values.push_back(1.0);
            nz.push_back(make_pair(row, Cj1));  values.push_back(-1.0);
            ++row;

            nz.push_back(make_pair(row, Sijk)); values.push_back(-c_ji[k][2]);
            nz.push_back(make_pair(row, Ci2));  values.push_back(1.0);
            nz.push_back(make_pair(row, Cj2));  values.push_back(-1.0);
            ++row;

            nz.push_back(make_pair(row, Sijk)); values.push_back(-c_jk[k][0]);
            nz.push_back(make_pair(row, Ck0));  values.push_back(1.0);
            nz.push_back(make_pair(row, Cj0));  values.push_back(-1.0);
            ++row;

            nz.push_back(make_pair(row, Sijk)); values.push_back(-c_jk[k][1]);
            nz.push_back(make_pair(row, Ck1));  values.push_back(1.0);
            nz.push_back(make_pair(row, Cj1));  values.push_back(-1.0);
            ++row;

            nz.push_back(make_pair(row, Sijk)); values.push_back(-c_jk[k][2]);
            nz.push_back(make_pair(row, Ck2));  values.push_back(1.0);
            nz.push_back(make_pair(row, Cj2));  values.push_back(-1.0);
            ++row;
         } // end for (k)

         CCS_Matrix<double> A(6*K, 3*N + K, nz, values);
         L = sqr(matrixNorm_L2(A));
         if (params.verbose) cout << "|A|^2 = " << L << endl;
#else
         cerr << "computeConsistentCameraCenters_L2_BOS(): ARPACK not available for |A|_2 computation, assuming default value for L" << endl;
         L = 1000.0;
#endif
      } // end if

      double const delta = 0.95 / L / params.alpha;

      vector<Vector3d> centersB(N);

      std::fill(centers.begin(), centers.end(), makeVector3(0.0, 0.0, 0.0));
      std::fill(centersB.begin(), centersB.end(), makeVector3(0.0, 0.0, 0.0));

      vector<double> scales(K, 0.0);
      vector<double> scalesB(K, 0.0);

      vector<Vector3d> p1(K, makeVector3(0.0, 0.0, 0.0));
      vector<Vector3d> p2(K, makeVector3(0.0, 0.0, 0.0));

      vector<Vector3d> Atp_centers(N);
      vector<double>   Atp_scales(K);

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         // Update centers and scales

         // compute A^T * p
         std::fill(Atp_centers.begin(), Atp_centers.end(), makeVector3(0.0, 0.0, 0.0));
         std::fill(Atp_scales.begin(), Atp_scales.end(), 0.0);
         for (int k = 0; k < K; ++k)
         {
            int const i0 = ijks[k][0];
            int const i1 = ijks[k][1];
            int const i2 = ijks[k][2];
            addVectorsIP( p1[k], Atp_centers[i0]);
            addVectorsIP(-p1[k], Atp_centers[i1]);
            addVectorsIP( p2[k], Atp_centers[i2]);
            addVectorsIP(-p2[k], Atp_centers[i1]);

            Atp_scales[k] -= innerProduct(c_ji[k], p1[k]);
            Atp_scales[k] -= innerProduct(c_jk[k], p2[k]);
         } // end for (k)

         // Real update
         for (int i = 0; i < N; ++i)
         {
            Vector3d center_new = centers[i] - params.alpha * Atp_centers[i];
            centersB[i] = 2.0 * center_new - centers[i];
            centers[i]  = center_new;
         }
         for (int k = 0; k < K; ++k)
         {
            double scale_new = scales[k] - params.alpha * Atp_scales[k];
            scale_new  = std::max(1.0, scale_new);
            scalesB[k] = 2.0 * scale_new - scales[k];
            scales[k]  = scale_new;
         }

         // Update p1 and p2
         for (int k = 0; k < K; ++k)
         {
            int const i0 = ijks[k][0];
            int const i1 = ijks[k][1];
            int const i2 = ijks[k][2];

            Vector3d u1 = centersB[i0] - centersB[i1] - scalesB[k]*c_ji[k];
            Vector3d u2 = centersB[i2] - centersB[i1] - scalesB[k]*c_jk[k];

            p1[k] = p1[k] + delta*u1;
            p2[k] = p2[k] + delta*u2;

            projectP(p1[k]);
            projectP(p2[k]);
         } // end for (k)

         if (params.verbose && ((iter + 1) % params.reportFrequency) == 0)
         {
            double E = 0.0;
            for (int k = 0; k < K; ++k)
            {
               int const i0 = ijks[k][0];
               int const i1 = ijks[k][1];
               int const i2 = ijks[k][2];

               Vector3d u1 = centers[i0] - centers[i1] - scales[k]*c_ji[k];
               Vector3d u2 = centers[i2] - centers[i1] - scales[k]*c_jk[k];

               E += norm_L2(u1) + norm_L2(u2);
            } // end for (k)
            cout << "iter = " << iter << ", E = " << E << endl;
         } // end if
      } // end for (iter)
      return true;
   } // end computeConsistentCameraCenters_L2_BOS()

} // end namespace V3D

//**********************************************************************

namespace
{

   inline Vector3d
   proxNormL2(double const gamma, Vector3d const& xx)
   {
      double const l = norm_L2(xx);
      if (l > gamma)
         return (1.0 - gamma/l)*xx;

      return Vector3d(0, 0, 0);
   } // end proxNormL2()

} // end namespace <>

namespace V3D
{

   bool
   computeConsistentCameraCenters_L2_SDMM(int const nSubModels, std::vector<Vector3d> const& c_ij,
                                          std::vector<Vector2i> const& ijs, std::vector<int> const& submodelIndices,
                                          std::vector<double> const& weights, std::vector<Vector3d>& centers,
                                          MultiViewInitializationParams_BOS const& params)
   {
      double const gamma = params.alpha;

      int const K = ijs.size();
      int const N = centers.size();

      vector<pair<int, int> > nz;
      vector<double> values;
      nz.reserve(3*3*K + nSubModels);
      values.reserve(3*3*K + nSubModels);

      int const nVars = nSubModels + 3*N;

      int row = 0;
      for (int k = 0; k < K; ++k)
      {
         int const i = ijs[k][0];
         int const j = ijs[k][1];

         int const Sijk = submodelIndices[k];
         int const Ci0  = nSubModels + 3*i + 0;
         int const Ci1  = nSubModels + 3*i + 1;
         int const Ci2  = nSubModels + 3*i + 2;
         int const Cj0  = nSubModels + 3*j + 0;
         int const Cj1  = nSubModels + 3*j + 1;
         int const Cj2  = nSubModels + 3*j + 2;

         nz.push_back(make_pair(row, Sijk)); values.push_back(c_ij[k][0]);
         nz.push_back(make_pair(row, Ci0));  values.push_back(1.0);
         nz.push_back(make_pair(row, Cj0));  values.push_back(-1.0);
         ++row;

         nz.push_back(make_pair(row, Sijk)); values.push_back(c_ij[k][1]);
         nz.push_back(make_pair(row, Ci1));  values.push_back(1.0);
         nz.push_back(make_pair(row, Cj1));  values.push_back(-1.0);
         ++row;

         nz.push_back(make_pair(row, Sijk)); values.push_back(c_ij[k][2]);
         nz.push_back(make_pair(row, Ci2));  values.push_back(1.0);
         nz.push_back(make_pair(row, Cj2));  values.push_back(-1.0);
         ++row;
      } // end for (k)

      for (int t = 0; t < nSubModels; ++t)
      {
         nz.push_back(make_pair(row, t)); values.push_back(1.0);
         ++row;
      }

      // Add 3 rows sum(C) = 0 to make A of full rank (and to remove the gauge freedom)
#if 0
      for (int i = 0; i < N; ++i)
      {
         nz.push_back(make_pair(row, nSubModels + 3*i + 0)); values.push_back(1.0);
      }
      ++row;
      for (int i = 0; i < N; ++i)
      {
         nz.push_back(make_pair(row, nSubModels + 3*i + 1)); values.push_back(1.0);
      }
      ++row;
      for (int i = 0; i < N; ++i)
      {
         nz.push_back(make_pair(row, nSubModels + 3*i + 2)); values.push_back(1.0);
      }
      ++row;
#else
      // This code path is much better for cholesky...
      nz.push_back(make_pair(row, nSubModels + 0)); values.push_back(1.0);
      ++row;
      nz.push_back(make_pair(row, nSubModels + 1)); values.push_back(1.0);
      ++row;
      nz.push_back(make_pair(row, nSubModels + 2)); values.push_back(1.0);
      ++row;
#endif

      CCS_Matrix<double> A(3*K + nSubModels + 3, nVars, nz, values);
      CCS_Matrix<double> AtA(nVars, nVars, vector<pair<int, int> >());
      multiply_At_A_SparseSparse(A, AtA);

      SparseCholesky chol(AtA);
      chol.setAtA(AtA);

      Vector<double> X(nVars);

      vector<Vector3d> Y(K, Vector3d(0, 0, 0)); // These correspond to |s_t * c_ij + c_i - c_j|_2
      vector<Vector3d> Z(K, Vector3d(0, 0, 0));

      vector<double> Ys(nSubModels, 1); // These correspond to s_t >= 1
      vector<double> Zs(nSubModels, 0);

      vector<Vector3d> YZ(3*K);
      Vector<double> rhs(nVars);

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         for (int k = 0; k < K; ++k) YZ[k] = Y[k] - Z[k];

         // compute A^T * YZ = rhs
         makeZeroVector(rhs);
         for (int k = 0; k < K; ++k)
         {
            int const i = ijs[k][0];
            int const j = ijs[k][1];
            int const t = submodelIndices[k];

            rhs[nSubModels + 3*i + 0] += YZ[k][0];
            rhs[nSubModels + 3*i + 1] += YZ[k][1];
            rhs[nSubModels + 3*i + 2] += YZ[k][2];

            rhs[nSubModels + 3*j + 0] -= YZ[k][0];
            rhs[nSubModels + 3*j + 1] -= YZ[k][1];
            rhs[nSubModels + 3*j + 2] -= YZ[k][2];

            rhs[t] += innerProduct(c_ij[k], YZ[k]);
         } // end for (k)

         for (int t = 0; t < nSubModels; ++t) rhs[t] += (Ys[t] - Zs[t]);

         chol.solve(rhs, X);

         for (int i = 0; i < N; ++i)
         {
            centers[i][0] = X[nSubModels + 3*i + 0];
            centers[i][1] = X[nSubModels + 3*i + 1];
            centers[i][2] = X[nSubModels + 3*i + 2];
         }

         for (int k = 0; k < K; ++k)
         {
            int const i = ijs[k][0];
            int const j = ijs[k][1];
            int const t = submodelIndices[k];

            double const s = X[t];
            Vector3d const& Ci = centers[i];
            Vector3d const& Cj = centers[j];

            Vector3d const XX = s * c_ij[k] + Ci - Cj;
            Vector3d const YY = proxNormL2(gamma * weights[k], XX + Z[k]);
            Y[k] = YY;
            Z[k] = Z[k] + XX - YY;
         } // end for (k)

         for (int t = 0; t < nSubModels; ++t)
         {
            // prox for f = i(s >= 1), i.e. clamp to [1, infty]
            double const s = X[t];
            Ys[t] = std::max(1.0, s + Zs[t]);
            Zs[t] += s - Ys[t];
         } // end for (t)

         if (params.verbose && ((iter + 1) % params.reportFrequency) == 0)
         {
            double E = 0.0;
            for (int k = 0; k < K; ++k)
            {
               int const i = ijs[k][0];
               int const j = ijs[k][1];
               int const t = submodelIndices[k];
               Vector3d u = centers[i] - centers[j] + X[t]*c_ij[k];
               E += weights[k] * norm_L2(u);
            } // end for (k)
            cout << "iter = " << iter << ", E = " << E << endl;

//             for (int t = 0; t < nSubModels; ++t)
//                cout << X[t] << " ";
//             cout << endl;
         } // end if
      } // end for (iter)
      return true;
   } // end computeConsistentCameraCenters_L2_SDMM()

} // end namespace V3D

//**********************************************************************

namespace
{

   template <typename T>
   inline void
   projectZs(T& z1, T& z2)
   {
#if 1
      T const z1mag = fabs(z1);

      if (z1mag <= T(1) && z2 <= -z1mag) return;

      // Case 1: z2 >= |z1|, closest point is (0, 0)
      if (z2 >= z1mag)
      {
         z1 = z2 = 0;
      }
      // Case 2: |z1| > 1, z2 < -|z1|
      else if (z1mag > T(1) && z2 < -z1mag)
      {
         // Only need to clamp z1
         z1 = clamp(z1, T(-1), T(1));
      }
      else if (z1 >= 0.0f)
      {
         if (z1 > 1 && z2 < z1 - 2)
         {
            // (z1, z2) = (1, -1)
            z1 = 1; z2 = -1;
         }
         else
         {
            // Closest point on line segment (0,0) to (1,-1)
            z1 = 0.5 * (z1 - z2);
            z2  = -z1;
         }
      }
      else
      {
         // z1 < 0
         if (z1 < T(-1) && z2 < -z1 - 2)
         {
            // (z1, z2) = (-1, -1)
            z1 = -1; z2 = -1;
         }
         else
         {
            // Closest point on line segment (0,0) to (-1,-1)
            z2 = 0.5 * (z1 + z2);
            z1 = z2;
         }
      } // end if
#else
      z1 = clamp(z1, T(-1), T(1));
      z2 = std::min(z2, -fabs(z1));
#endif
   } // end projectZs()

   inline void
   prox_g1(double gamma, double& x, double& y)
   {
#if 0
      double const x0 = x;
      double const y0 = y;

      if (y0 >= 0)
      {
         //if (fabs(x0) <= y0) return; // Nothing to do

         if (x0 > y0)
         {
            double const step = std::min(gamma, (x0-y0)/2);
            x = x0 - step;
            y = y0 + step;
         }
         else if (x0 < -y0)
         {
            double const step = std::min(gamma, (-x0-y0)/2);
            x = x0 + step;
            y = y0 + step;
         } // end if
      }
      else
      {
         y = 0;
         if (x0 > gamma)
         {
            x = x0 - gamma;
         }
         else if (-x0 > gamma)
         {
            x = x0 + gamma;
         }
      } // end if (y0 >= 0)
#else
      // Use Moreau's decomposition
      double z1 = x / gamma, z2 = y / gamma;
      projectZs(z1, z2);
      x = x - gamma * z1;
      y = y - gamma * z2;
#endif
   } // end prox_g1()

   inline void
   prox_g2(double const gamma, double& x1, double& x2, double& y)
   {
      // Use Moreau's decomposition to compute prox_g2
      double z1 = x1/gamma;
      double z2 = x2/gamma;
      double v = y/gamma;

      double const normZ = sqrt(z1*z1 + z2*z2);

      // Project (z1,z2,v) onto the set |z| <= 1, v <= -|z|
      if (normZ > 1 || v > -normZ)
      {
         if (v > normZ)
         {
            v = z1 = z2 = 0.0;
         }
         else if (v < -1)
         {
            z1 /= std::max(1.0, normZ);
            z2 /= std::max(1.0, normZ);
         }
         else if (v > normZ - 2)
         {
            double const e = v + normZ;
            v = v - e/2;
            double const newNormZ = normZ - e/2;
            z1 *= newNormZ / normZ;
            z2 *= newNormZ / normZ;
         }
         else
         {
            v = -1;
            z1 /= std::max(1.0, normZ);
            z2 /= std::max(1.0, normZ);
         }
      } // end if

      x1 -= gamma * z1;
      x2 -= gamma * z2;
      y  -= gamma * v;
   } // end prox_g2()

   template <typename T>
   inline void
   projectZs_Huber(T& z1, T& z2)
   {
#if 1
      // Project (z1, z2) onto the convex set z2 <= -0.5*z1^2, |z1| <= 1

      float const z1mag = fabsf(z1);

      if (z2 <= -0.5f*z1*z1 && z1mag <= 1.0f) return; // Nothing to do

      // The simple case, just clamp z1
      if (z2 <= -0.5f && z1mag > 1.0f)
      {
         z1 = std::max(T(-1), std::min(T(1), z1));
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
      float const p = (2.0 + 2.0*z2)/3.0;
      float const q = -z1;
      float const D2 = std::max(0.0f, q*q + p*p*p); // Should be non-negative
      float const D = sqrtf(D2);

      float const u = cbrtf(-q + D);
      float const v = cbrtf(-q - D);
      z1 = u+v;
      z1 = std::max(T(-1), std::min(T(1), z1));
      z2 = -0.5f*z1*z1;
# else
      z1 = std::max(T(-1), std::min(T(1), z1));
      z2 = std::min(z2, -0.5f*z1*z1);
# endif
#else
      z1 = std::max(T(-1), std::min(T(1), z1));
      z2 = std::min(z2, -0.5f*z1*z1);
#endif

      z2 = std::max(z2, T(-10));

      //if (fabs(z1) > 1.0) cout << "z1 = " << z1 << endl;
   } // end projectZs_Huber()

   inline void
   prox_H(double const gamma, double& x, double& y)
   {
      // Use Moreau's decomposition to compute prox_g2
      float z = x/gamma;
      float v = y/gamma;

      projectZs_Huber(z, v);

      x -= gamma * z;
      y -= gamma * v;
   } // end prox_H()

   inline double
   computeStructureDistance(std::vector<Vector3d> const& Xs1, std::vector<Vector3d> const& Xs2)
   {
#if 1
      return computeStructureSimilarity(Xs1, Xs2);
#else
      int const M = Xs1.size();

      double res = 0.0;
# if 0
      for (int j = 0; j < M; ++j)
         res = std::max(res, sqrDistance_L2(Xs1[j], Xs2[j]));
      return sqrt(res);
# else
      for (int j = 0; j < M; ++j)
         res += sqrDistance_L2(Xs1[j], Xs2[j]);
      return sqrt(res / M);
# endif
#endif
   } // end computeStructureDistance()

   typedef InlineVector<double, 5> Vector5d;
   typedef InlineVector<double, 6> Vector6d;

   template <typename Elem>
   inline InlineVector<Elem, 5>
   makeVector5(Elem a, Elem b, Elem c, Elem d, Elem e)
   {
      InlineVector<Elem, 5> res;
      res[0] = a; res[1] = b; res[2] = c; res[3] = d; res[4] = e;
      return res;
   }

   template <typename Elem>
   inline InlineVector<Elem, 6>
   makeVector6(Elem a, Elem b, Elem c, Elem d, Elem e, Elem f)
   {
      InlineVector<Elem, 6> res;
      res[0] = a; res[1] = b; res[2] = c; res[3] = d; res[4] = e; res[5] = f;
      return res;
   }

   template <typename T> inline T sqr(T x) { return x*x; }

   inline Vector3d
   transformIntoMeasurementCone(int const k, float const sigma, BundlePointStructure const& bundleStruct,
                                std::vector<Matrix3x3d> const& rotations, vector<Vector3d> const& Ts, vector<Vector3d> const& Xs)
   {
      Vector3d res;

      int const i = bundleStruct.correspondingView[k];
      int const j = bundleStruct.correspondingPoint[k];
      Matrix3x3d const& R = rotations[i];
      Vector3d   const& T = Ts[i];
      Vector3d   const& X = Xs[j];
      Vector2d   const& m = bundleStruct.measurements[k];
      float const u1 = m[0];
      float const u2 = m[1];

      Vector3d const XX = R*X + T;

      res[0] = u1*XX[2] - XX[0];
      res[1] = u2*XX[2] - XX[1];
      res[2] = XX[2];

      return res;
   } // end transformIntoMeasurementCone()

   // Compute A^T (w1*Y1 + w2*Y2)
   inline void
   compute_At_Y_Aniso(float const sigma, BundlePointStructure const& bundleStruct,
                      std::vector<Matrix3x3d> const& rotations,
                      double const w1, vector<Vector5d> const& Y1, double const w2, vector<Vector5d> const& Y2,
                      Vector<double>& AtY)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;
      int const K = correspondingView.size();
      int const N = rotations.size();

      makeZeroVector(AtY);
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         float const x1 = w1*Y1[k][0] + w2*Y2[k][0];
         float const y1 = w1*Y1[k][1] + w2*Y2[k][1];
         float const x2 = w1*Y1[k][2] + w2*Y2[k][2];
         float const y2 = w1*Y1[k][3] + w2*Y2[k][3];
         float const z  = w1*Y1[k][4] + w2*Y2[k][4];

         if (i > 0)
         {
            AtY[TVAR(i, 0)] += -x1;
            AtY[TVAR(i, 1)] += -x2;
            AtY[TVAR(i, 2)] += x1*u1 + x2*u2 + sigma*(y1 + y2);
            AtY[TVAR(i, 2)] += z;
         }

         AtY[XVAR(j, 0)] += x1 * (u1*R[2][0] - R[0][0]) + sigma * y1 * R[2][0];
         AtY[XVAR(j, 1)] += x1 * (u1*R[2][1] - R[0][1]) + sigma * y1 * R[2][1];
         AtY[XVAR(j, 2)] += x1 * (u1*R[2][2] - R[0][2]) + sigma * y1 * R[2][2];

         AtY[XVAR(j, 0)] += x2 * (u2*R[2][0] - R[1][0]) + sigma * y2 * R[2][0];
         AtY[XVAR(j, 1)] += x2 * (u2*R[2][1] - R[1][1]) + sigma * y2 * R[2][1];
         AtY[XVAR(j, 2)] += x2 * (u2*R[2][2] - R[1][2]) + sigma * y2 * R[2][2];

         AtY[XVAR(j, 0)] += z * R[2][0];
         AtY[XVAR(j, 1)] += z * R[2][1];
         AtY[XVAR(j, 2)] += z * R[2][2];
      } // end for (k)
#undef TVAR
#undef XVAR
   } // end compute_At_Y_Aniso()

} // end namespace <>

# if defined(V3DLIB_ENABLE_SUITESPARSE)

#include "Math/v3d_sparsechol.h"

namespace V3D
{

   void
   computeConsistentTranslationsConic_Aniso_SDMM(float const sigma,
                                                 std::vector<Matrix3x3d> const& rotations,
                                                 std::vector<Vector3d>& translations,
                                                 std::vector<TriangulatedPoint>& sparseModel,
                                                 MultiViewInitializationParams_BOS const& params,
                                                 bool const strictCheirality)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);
      makeZeroVector(translations[0]);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      int const nVars = 3*(N-1) + 3*M; // First T is 0

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      vector<pair<int, int> > nzL;
      vector<double>          valsL;
      nzL.reserve(22*K);
      valsL.reserve(22*K);

      int row = 0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         if (i > 0)
         {
            nzL.push_back(make_pair(row, TVAR(i, 0))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u1);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 1))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u2);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(1.0);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            ++row;
         }
         else
         {
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            ++row;
         } // end if
      } // end for (k)

      CCS_Matrix<double> L(row, nVars, nzL, valsL);
      CCS_Matrix<double> Q;

      multiply_At_A_SparseSparse(L, Q);

      SparseCholesky chol(Q, params.verbose);
      bool status = chol.setAtA(Q);
      cout << "Cholesky status = " << (status ? "good" : "bad") << endl;

      vector<Vector3d> Xs(M);
      vector<Vector5d> Ys(K, makeVector5(0.0, double(sigma), 0.0, double(sigma), 0.0));
      vector<Vector5d> Zs(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));

      vector<Vector3d> savedXs(M);
      double E_saved = 1e30, E_cur = 0;
      double E_primal_best = 1e40, E_dual_best = -1e40;

      Vector<double> rhs(nVars), X_new(nVars), mu(nVars);;

      double gamma = params.alpha;

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, Ys, -1.0, Zs, rhs);

         //cout << "rhs = "; displayVector(rhs);

         chol.solve(rhs, X_new);
         //cout << "X_new = "; displayVector(X_new);

         // Vector<double> QX(nVars);
         // makeZeroVector(QX);
         //cout << "Q.num_rows() = " << Q.num_rows() << " Q.num_cols() = " << Q.num_cols() << endl;
         //cout << "X_new.size() = " << X_new.size() << " QX.size() = " << QX.size() << endl;
         // multiply_At_v_Sparse(Q, X_new, QX);
         //cout << "|Q*X|^2 = " << sqrNorm_L2(QX) << endl;

         for (int i = 1; i < N; ++i) copyVectorSlice(X_new, TVAR(i, 0), 3, translations[i], 0);
         for (int j = 0; j < M; ++j) copyVectorSlice(X_new, XVAR(j, 0), 3, Xs[j], 0);

         if (iter == 0) savedXs = Xs;

         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

            double const x1 = XX[0];
            double const y1 = sigma*XX[2];
            double const x2 = XX[1];
            double const y2 = sigma*XX[2];

            double xx1 = x1 + Zs[k][0];
            double yy1 = y1 + Zs[k][1];
            double xx2 = x2 + Zs[k][2];
            double yy2 = y2 + Zs[k][3];

            prox_g1(gamma, xx1, yy1);
            prox_g1(gamma, xx2, yy2);

            Ys[k][0] = xx1; Ys[k][1] = yy1;
            Ys[k][2] = xx2; Ys[k][3] = yy2;

            Zs[k][0] += x1 - xx1; Zs[k][1] += y1 - yy1;
            Zs[k][2] += x2 - xx2; Zs[k][3] += y2 - yy2;

            double const w  = XX[2];
            double       ww = w + Zs[k][4];

            if (strictCheirality)
               ww = std::max(ww, 1.0);
            else
            {
               // Prox for gamma*max{0, -ww+1}
               if (ww < 1.0 - gamma)
                  ww += gamma;
               else
                  ww = std::max(ww, 1.0);
            }

            Ys[k][4] = ww;
            Zs[k][4] += w - ww;
         } // end for (k)

         double const eps = 1e-5;

         if (((iter+1) % params.checkFrequency) == 0)
         {
            E_cur = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               E_cur += std::max(0.0, fabs(XX[0]) - c);
               E_cur += std::max(0.0, fabs(XX[1]) - c);

               if (strictCheirality)
               {
                  if (XX[2] < 1.0 - eps) E_cur += 1e30;
               }
               else
               {
                  E_cur += std::max(0.0, -XX[2] + 1.0);
               }
            } // end for (k)
         } // end if

         if (params.verbose && (((iter+1) % params.reportFrequency) == 0 || iter == params.nIterations-1 || iter == 9))
         {
            cout << "iter = " << iter << endl;

            int    nFeasible = 0;

            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

               //if (XX3 >= 0.9999) ++nFeasible;
               if (XX3 > 1.0 - eps) ++nFeasible;
            } // end for (k)

            double E_primal = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               E_primal += std::max(0.0, fabs(XX[0]) - c);
               E_primal += std::max(0.0, fabs(XX[1]) - c);

               if (strictCheirality)
               {
                  if (XX[2] < 1.0 - eps) E_primal += 1e30;
               }
               else
               {
                  E_primal += std::max(0.0, -(XX[2]-1.0));
               }
            } // end for (k)

            cout << "E_cur = " << E_cur << ", E_saved = " << E_saved << endl;
            cout << "nFeasible = " << nFeasible << "/" << K << endl;
#if 0
            cout << "Current structure distance = " << computeStructureDistance(savedXs, Xs) << endl;
#else
            cout << "Relative energy change = " << fabs(E_cur - E_saved) / std::min(E_cur, E_saved) << endl;
#endif

            double E_dual = 0;
            vector<Vector3d> mu_Ts(N);
            vector<Vector3d> mu_Xs(M);
            {
               compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0/gamma, Zs, 0.0, Zs, rhs);
               chol.solve(rhs, mu);

               for (int i = 1; i < N; ++i) copyVectorSlice(mu, TVAR(i, 0), 3, mu_Ts[i], 0);
               for (int j = 0; j < M; ++j) copyVectorSlice(mu, XVAR(j, 0), 3, mu_Xs[j], 0);
               makeZeroVector(mu_Ts[0]);
            }

            vector<Vector5d> PPs(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, mu_Ts, mu_Xs);

               double const x1 = XX[0];
               double const y1 = sigma*XX[2];
               double const x2 = XX[1];
               double const y2 = sigma*XX[2];
               double const z  = XX[2];

               // Note: the dual vars are really z/gamma (see the A-L formulation)
               PPs[k][0] = Zs[k][0] / gamma - x1;
               PPs[k][1] = Zs[k][1] / gamma - y1;
               PPs[k][2] = Zs[k][2] / gamma - x2;
               PPs[k][3] = Zs[k][3] / gamma - y2;
               PPs[k][4] = Zs[k][4] / gamma - z;
            } // end for (k)

            compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, PPs, 0.0, PPs, rhs);
            cout << "norm(At*lambda) = " << norm_L2(rhs) << endl;

            for (int k = 0; k < K; ++k)
            {
               double const x1 = PPs[k][0], y1 = PPs[k][1];
               double const x2 = PPs[k][2], y2 = PPs[k][3];
               double const z  = PPs[k][4];

               double accum = 0;
               //accum += std::min(0.0, z);
               accum += z;
               if (strictCheirality)
               {
                  if (z > eps) accum += 1e30;
               }
               else
               {
                  if (z > eps || z < -1.0-eps) accum += 1e30;
               }

               if (fabs(x1) > 1.0 + eps) accum += 1e30;
               if (fabs(x1) > -y1 + eps) accum += 1e30;
               if (fabs(x2) > 1.0 + eps) accum += 1e30;
               if (fabs(x2) > -y2 + eps) accum += 1e30;

               E_dual -= accum;
            } // end for (k)

            // compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, Zs, 0.0, Zs, rhs);
            // cout << "||Q||_1 = " << norm_L1(rhs) << endl;

            // double const B = 10;
            // E_dual -= B * norm_L1(rhs) / gamma;

            E_primal_best = std::min(E_primal_best, E_primal);
            E_dual_best   = std::max(E_dual_best, E_dual);

            //cout << "E_primal = "  << E_primal << ", E_dual = " << E_dual << ", duality gap = " << (E_primal - E_dual)/E_primal * 100 << "%" << endl;
            cout << "E_primal = "  << E_primal_best << ", E_dual = " << E_dual_best << ", duality gap = " << (E_primal_best - E_dual_best)/E_primal_best * 100 << "%" << endl;

            // cout << "Ts = [ ";
            // for (int i = 0; i < N; ++i) cout << translations[i][0] << " " << translations[i][1] << " " << translations[i][2] << " / ";
            // cout << "]" << endl;

            cout << "=== " << (iter+1) << " " << E_primal_best << " " << E_dual_best << endl;

//             cout << "Vs = [ ";
//             for (int k = 0; k < K; ++k) cout << Vs[k] << " ";
//             cout << "]" << endl;

            //gamma = std::max(gamma / 1.1, 1e-3);
            //gamma = std::min(1.0, 1.1*gamma);
            //cout << "new gamma = " << gamma << endl;
         } // end if

         if (((iter+1) % params.checkFrequency) == 0)
         {
#if 0
            if (computeStructureDistance(savedXs, Xs) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            savedXs = Xs;
#else
            if (E_cur < 1e30 && fabs(E_cur - E_saved) / std::min(E_cur, E_saved) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            E_saved = E_cur;
#endif
         }
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }

#undef TVAR
#undef XVAR
   } // end computeConsistentTranslationsConic_Aniso_SDMM()

   void
   computeConsistentTranslationsConic_Iso_SDMM(float const sigma,
                                               std::vector<Matrix3x3d> const& rotations,
                                               std::vector<Vector3d>& translations,
                                               std::vector<TriangulatedPoint>& sparseModel,
                                               MultiViewInitializationParams_BOS const& params)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);
      makeZeroVector(translations[0]);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      int const nVars = 3*(N-1) + 3*M; // First T is 0

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      vector<pair<int, int> > nzL;
      vector<double>          valsL;
      nzL.reserve(22*K);
      valsL.reserve(22*K);

      int row = 0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         if (i > 0)
         {
            nzL.push_back(make_pair(row, TVAR(i, 0))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u1);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 1))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u2);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(1.0);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            ++row;
         }
         else
         {
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            ++row;
         } // end if
      } // end for (k)

      CCS_Matrix<double> L(row, nVars, nzL, valsL);
      CCS_Matrix<double> Q;

      multiply_At_A_SparseSparse(L, Q);

      SparseCholesky chol(Q, params.verbose);
      bool status = chol.setAtA(Q);
      cout << "Cholesky status = " << (status ? "good" : "bad") << endl;

      vector<Vector3d> Xs(M);
      vector<Vector3d> Ys(K, makeVector3(0.0, 0.0, double(sigma)));
      vector<Vector3d> Zs(K, makeVector3(0.0, 0.0, 0.0));

      vector<Vector3d> savedXs(M);
      double E_saved = 1e30, E_cur = 0;

      vector<double> Ws(K, 0.0), Vs(K, 0.0);

      Vector<double> rhs(nVars), X_new(nVars);

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         makeZeroVector(rhs);
         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            Matrix3x3d const& R  = rotations[i];
            Vector3d   const& T  = translations[i];
            Vector3d   const& X  = Xs[j];
            Vector2d   const& m  = bundleStruct.measurements[k];
            float      const  u1 = m[0];
            float      const  u2 = m[1];

            float const x1 = Ys[k][0] - Zs[k][0], x2 = Ys[k][1] - Zs[k][1], y = Ys[k][2] - Zs[k][2];

            float const z = Ws[k] - Vs[k];

            if (i > 0)
            {
               rhs[TVAR(i, 0)] += -x1;
               rhs[TVAR(i, 1)] += -x2;
               rhs[TVAR(i, 2)] += x1*u1 + x2*u2 + sigma*y;

               rhs[TVAR(i, 2)] += z;
            }

            rhs[XVAR(j, 0)] += x1 * (u1*R[2][0] - R[0][0]);
            rhs[XVAR(j, 1)] += x1 * (u1*R[2][1] - R[0][1]);
            rhs[XVAR(j, 2)] += x1 * (u1*R[2][2] - R[0][2]);

            rhs[XVAR(j, 0)] += x2 * (u2*R[2][0] - R[1][0]);
            rhs[XVAR(j, 1)] += x2 * (u2*R[2][1] - R[1][1]);
            rhs[XVAR(j, 2)] += x2 * (u2*R[2][2] - R[1][2]);

            rhs[XVAR(j, 0)] += sigma * y * R[2][0];
            rhs[XVAR(j, 1)] += sigma * y * R[2][1];
            rhs[XVAR(j, 2)] += sigma * y * R[2][2];

            rhs[XVAR(j, 0)] += z * R[2][0];
            rhs[XVAR(j, 1)] += z * R[2][1];
            rhs[XVAR(j, 2)] += z * R[2][2];
         } // end for (k)

         chol.solve(rhs, X_new);

         Vector<double> QX(nVars);
         makeZeroVector(QX);
         multiply_At_v_Sparse(Q, X_new, QX);
         //cout << "|Q*X|^2 = " << sqrNorm_L2(QX) << endl;

         for (int i = 1; i < N; ++i) copyVectorSlice(X_new, TVAR(i, 0), 3, translations[i], 0);
         for (int j = 0; j < M; ++j) copyVectorSlice(X_new, XVAR(j, 0), 3, Xs[j], 0);

         if (iter == 0) savedXs = Xs;

         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

            double const x1 = XX[0];
            double const x2 = XX[1];
            double const y  = sigma*XX[2];

            double xx1 = x1 + Zs[k][0];
            double xx2 = x2 + Zs[k][1];
            double yy  = y  + Zs[k][2];

            prox_g2(params.alpha, xx1, xx2, yy);

            Ys[k][0] = xx1; Ys[k][1] = xx2; Ys[k][2] = yy;

            Zs[k][0] += x1 - xx1; Zs[k][1] += x2 - xx2; Zs[k][2] += y - yy;

            double const w  = XX[2];
            double const ww = std::max(w + Vs[k], 1.0);
            Ws[k] = ww;
            Vs[k] += w - ww;
         } // end for (k)

         if (((iter+1) % params.checkFrequency) == 0 || iter == 9)
         {
            E_cur = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
#if 0
               E_cur += std::max(0.0, fabs(XX[0]) - c);
               E_cur += std::max(0.0, fabs(XX[1]) - c);
#else
               double e1 = XX[0];
               double e2 = XX[1];
               E_cur += std::max(0.0, sqrt(e1*e1 + e2*e2) - c);
#endif
            } // end for (k)
         }

         if (params.verbose && (((iter+1) % params.reportFrequency) == 0 || iter == params.nIterations-1 || iter == 9))
         {
            cout << "iter = " << iter << endl;

            int    nFeasible = 0;

            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

               if (XX3 >= 0.99) ++nFeasible;
            } // end for (k)
            cout << "E_cur = " << E_cur << ", E_saved = " << E_saved << endl;
            cout << "nFeasible = " << nFeasible << "/" << K << endl;
#if 0
            cout << "Current structure distance = " << computeStructureDistance(savedXs, Xs) << endl;
#else
            cout << "Relative energy change = " << fabs(E_cur - E_saved) / std::min(E_cur, E_saved) << endl;
#endif

            cout << "Ts = [ ";
            for (int i = 0; i < N; ++i) cout << translations[i][0] << " " << translations[i][1] << " " << translations[i][2] << " / ";
            cout << "]" << endl;

            cout << "=== " << (iter+1) << " " << E_cur << endl;
         } // end if

         if (((iter+1) % params.checkFrequency) == 0)
         {
#if 0
            if (computeStructureDistance(savedXs, Xs) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            savedXs = Xs;
#else
            if (fabs(E_cur - E_saved) / std::min(E_cur, E_saved) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            E_saved = E_cur;
#endif
         }
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }

#undef TVAR
#undef XVAR
   } // end computeConsistentTranslationsConic_Iso_SDMM()

   void
   computeConsistentTranslationsConic_Huber_SDMM(float const sigma,
                                                 std::vector<Matrix3x3d> const& rotations,
                                                 std::vector<Vector3d>& translations,
                                                 std::vector<TriangulatedPoint>& sparseModel,
                                                 MultiViewInitializationParams_BOS const& params)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);
      makeZeroVector(translations[0]);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      int const nVars = 3*(N-1) + 3*M; // First T is 0

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      vector<pair<int, int> > nzL;
      vector<double>          valsL;
      nzL.reserve(22*K);
      valsL.reserve(22*K);

      int row = 0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         if (i > 0)
         {
            nzL.push_back(make_pair(row, TVAR(i, 0))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u1);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 1))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u2);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(1.0);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            ++row;
         }
         else
         {
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            ++row;
         } // end if
      } // end for (k)

      CCS_Matrix<double> L(row, nVars, nzL, valsL);
      CCS_Matrix<double> Q;

      multiply_At_A_SparseSparse(L, Q);

      SparseCholesky chol(Q, params.verbose);
      bool status = chol.setAtA(Q);
      cout << "Cholesky status = " << (status ? "good" : "bad") << endl;

      vector<Vector3d> Xs(M);
      vector<Vector5d> Ys(K, makeVector5(0.0, double(sigma), 0.0, double(sigma), 1.0));
      vector<Vector5d> Zs(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));

      vector<Vector3d> savedXs(M);
      double E_saved = 1e30, E_cur = 0;

      Vector<double> rhs(nVars), X_new(nVars);

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, Ys, -1.0, Zs, rhs);

         chol.solve(rhs, X_new);

         Vector<double> QX(nVars);
         makeZeroVector(QX);
         multiply_At_v_Sparse(Q, X_new, QX);
         //cout << "|Q*X|^2 = " << sqrNorm_L2(QX) << endl;

         for (int i = 1; i < N; ++i) copyVectorSlice(X_new, TVAR(i, 0), 3, translations[i], 0);
         for (int j = 0; j < M; ++j) copyVectorSlice(X_new, XVAR(j, 0), 3, Xs[j], 0);

         if (iter == 0) savedXs = Xs;

         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

            double const x1 = XX[0];
            double const y1 = sigma*XX[2];
            double const x2 = XX[1];
            double const y2 = sigma*XX[2];

            double xx1 = x1 + Zs[k][0];
            double yy1 = y1 + Zs[k][1];
            double xx2 = x2 + Zs[k][2];
            double yy2 = y2 + Zs[k][3];

            prox_H(params.alpha, xx1, yy1);
            prox_H(params.alpha, xx2, yy2);

            Ys[k][0] = xx1; Ys[k][1] = yy1;
            Ys[k][2] = xx2; Ys[k][3] = yy2;

            Zs[k][0] += x1 - xx1; Zs[k][1] += y1 - yy1;
            Zs[k][2] += x2 - xx2; Zs[k][3] += y2 - yy2;

            double const w  = XX[2];
            double const ww = std::max(w + Zs[k][4], 1.0);
            Ys[k][4] = ww;
            Zs[k][4] += w - ww;
         } // end for (k)

         if (((iter+1) % params.checkFrequency) == 0 || iter == 9)
         {
            E_cur = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               double e1 = fabs(XX[0]);
               double e2 = fabs(XX[1]);

               E_cur += (e1 > c) ? e1 : (e1*e1/2/c + c/2);
               E_cur += (e2 > c) ? e2 : (e2*e2/2/c + c/2);
            } // end for (k)
         }

         if (params.verbose && (((iter+1) % params.reportFrequency) == 0 || iter == params.nIterations-1 || iter == 9))
         {
            cout << "iter = " << iter << endl;

            double E = 0.0;
            int    nFeasible = 0;

            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

               if (XX3 >= 0.99) ++nFeasible;
            } // end for (k)

            cout << "E_cur = " << E_cur << ", E_saved = " << E_saved << endl;
            cout << "nFeasible = " << nFeasible << "/" << K << endl;
#if 0
            cout << "Current structure distance = " << computeStructureDistance(savedXs, Xs) << endl;
#else
            cout << "Relative energy change = " << fabs(E_cur - E_saved) / std::min(E_cur, E_saved) << endl;
#endif
            cout << "Ts = [ ";
            for (int i = 0; i < N; ++i) cout << translations[i][0] << " " << translations[i][1] << " " << translations[i][2] << " / ";
            cout << "]" << endl;

            cout << "=== " << (iter+1) << " " << E_cur << endl;
         } // end if

         if (((iter+1) % params.checkFrequency) == 0)
         {
#if 0
            if (computeStructureDistance(savedXs, Xs) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            savedXs = Xs;
#else
            if (fabs(E_cur - E_saved) / std::min(E_cur, E_saved) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            E_saved = E_cur;
#endif
         }
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }

#undef TVAR
#undef XVAR
   } // end computeConsistentTranslationsConic_Huber_SDMM()

} // end namespace V3D

#endif // defined (V3DLIB_ENABLE_SUITESPARSE)

//======================================================================

namespace
{

} // end namespace <>

namespace V3D
{

   void
   computeConsistentTranslationsConic_Aniso_BOS(float const sigma,
                                                std::vector<Matrix3x3d> const& rotations,
                                                std::vector<Vector3d>& translations,
                                                std::vector<TriangulatedPoint>& sparseModel,
                                                MultiViewInitializationParams_BOS const& params)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);
      makeZeroVector(translations[0]);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      int const nVars = 3*(N-1) + 3*M; // First T is 0

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      vector<pair<int, int> > nzL;
      vector<double>          valsL;
      nzL.reserve(22*K);
      valsL.reserve(22*K);

      int row = 0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         if (i > 0)
         {
            nzL.push_back(make_pair(row, TVAR(i, 0))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u1);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 1))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u2);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(1.0);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            ++row;
         }
         else
         {
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            ++row;
         } // end if
      } // end for (k)

      CCS_Matrix<double> L(row, nVars, nzL, valsL);
      CCS_Matrix<double> Q;

      multiply_At_A_SparseSparse(L, Q);

      double const normQ_L2 = sparseMatrixNorm_L2_Bound(Q);
      cout << "normQ_L2 = " << normQ_L2 << endl;

#if 1
      double const alpha = params.alpha;
      double const delta = 0.25 / alpha / normQ_L2;
#else
      double const theta = 0.9;
      double const alpha = sqrt(theta / normQ_L2);
      double const delta = sqrt(theta / normQ_L2);
#endif

      double const gamma = alpha;
      double const ralpha = 1.0f / alpha;

      vector<Vector3d> Xs(M);
      vector<Vector5d> Zs(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));
      vector<Vector5d> mu(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));
      vector<Vector5d> P(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));

      vector<Vector3d> savedXs(M);
      double E_saved = 1e30, E_cur = 0;

      Vector<double> LtP(nVars);

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

            double const x1 = XX[0];
            double const y1 = sigma*XX[2];
            double const x2 = XX[1];
            double const y2 = sigma*XX[2];

            P[k][0] = mu[k][0] + alpha*(x1 - Zs[k][0]);
            P[k][1] = mu[k][1] + alpha*(y1 - Zs[k][1]);
            P[k][2] = mu[k][2] + alpha*(x2 - Zs[k][2]);
            P[k][3] = mu[k][3] + alpha*(y2 - Zs[k][3]);
            P[k][4] = mu[k][4] + alpha*(XX[2] - Zs[k][4]);
         } // end for (k)

         // X = X_old - delta * L^t P
         compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, P, 0.0, P, LtP);

         for (int i = 1; i < N; ++i)
         {
            translations[i][0] -= delta * LtP[TVAR(i, 0)];
            translations[i][1] -= delta * LtP[TVAR(i, 1)];
            translations[i][2] -= delta * LtP[TVAR(i, 2)];
         }

         for (int j = 0; j < M; ++j)
         {
            Xs[j][0] -= delta * LtP[XVAR(j, 0)];
            Xs[j][1] -= delta * LtP[XVAR(j, 1)];
            Xs[j][2] -= delta * LtP[XVAR(j, 2)];
         }

         if (iter == 0) savedXs = Xs;

         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

            double const x1 = XX[0];
            double const y1 = sigma*XX[2];
            double const x2 = XX[1];
            double const y2 = sigma*XX[2];

            double xx1 = x1 + ralpha*mu[k][0];
            double yy1 = y1 + ralpha*mu[k][1];
            double xx2 = x2 + ralpha*mu[k][2];
            double yy2 = y2 + ralpha*mu[k][3];

            prox_g1(ralpha, xx1, yy1);
            prox_g1(ralpha, xx2, yy2);

            Zs[k][0] = xx1; Zs[k][1] = yy1;
            Zs[k][2] = xx2; Zs[k][3] = yy2;

            double const ww = std::max(XX[2] + ralpha*mu[k][4], 1.0);
            Zs[k][4] = ww;

            mu[k][0] += gamma*(x1 - Zs[k][0]);
            mu[k][1] += gamma*(y1 - Zs[k][1]);
            mu[k][2] += gamma*(x2 - Zs[k][2]);
            mu[k][3] += gamma*(y2 - Zs[k][3]);
            mu[k][4] += gamma*(XX[2] - Zs[k][4]);
         } // end for (k)

         double const eps = 1e-3;

         if (((iter+1) % params.checkFrequency) == 0)
         {
            E_cur = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               E_cur += std::max(0.0, fabs(XX[0]) - c);
               E_cur += std::max(0.0, fabs(XX[1]) - c);

               if (XX[2] < 1.0 - eps) E_cur += 1e30;
            } // end for (k)
         } // end if

         if (params.verbose && (((iter+1) % params.reportFrequency) == 0 || iter == params.nIterations-1 || iter == 9))
         {
            cout << "iter = " << iter << endl;

            int    nFeasible = 0;

            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

               //if (XX3 >= 0.9999) ++nFeasible;
               if (XX3 > 1.0 - eps) ++nFeasible;
            } // end for (k)

            double E_primal = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               E_primal += std::max(0.0, fabs(XX[0]) - c);
               E_primal += std::max(0.0, fabs(XX[1]) - c);

               //if (XX3 < 1.0 - eps) E_primal += 1e30;
            } // end for (k)

            cout << "E_cur = " << E_cur << ", E_saved = " << E_saved << endl;
            cout << "nFeasible = " << nFeasible << "/" << K << endl;
#if 0
            cout << "Current structure distance = " << computeStructureDistance(savedXs, Xs) << endl;
#else
            cout << "Relative energy change = " << fabs(E_cur - E_saved) / std::min(E_cur, E_saved) << endl;
#endif

            double E_dual = 0;
            cout << "E_primal = "  << E_primal << ", E_dual = " << E_dual << ", duality gap = " << (E_primal - E_dual)/E_primal * 100 << "%" << endl;

            cout << "Ts = [ ";
            for (int i = 0; i < N; ++i) cout << translations[i][0] << " " << translations[i][1] << " " << translations[i][2] << " / ";
            cout << "]" << endl;

            cout << "=== " << (iter+1) << " " << E_primal << " " << E_dual << endl;

//             cout << "Vs = [ ";
//             for (int k = 0; k < K; ++k) cout << Vs[k] << " ";
//             cout << "]" << endl;

            //gamma = std::max(gamma / 1.1, 1e-3);
            //gamma = std::min(1.0, 1.1*gamma);
            //cout << "new gamma = " << gamma << endl;
         } // end if

         if (((iter+1) % params.checkFrequency) == 0)
         {
#if 0
            if (computeStructureDistance(savedXs, Xs) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            savedXs = Xs;
#else
            if (E_cur < 1e30 && fabs(E_cur - E_saved) / std::min(E_cur, E_saved) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            E_saved = E_cur;
#endif
         }
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }

#undef TVAR
#undef XVAR
   } // end computeConsistentTranslationsConic_Aniso_BOS()

   void
   computeConsistentTranslationsConic_Aniso_LS_Free(float const sigma,
                                                    std::vector<Matrix3x3d> const& rotations,
                                                    std::vector<Vector3d>& translations,
                                                    std::vector<TriangulatedPoint>& sparseModel,
                                                    MultiViewInitializationParams_BOS const& params)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);
      makeZeroVector(translations[0]);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      int const nVars = 3*(N-1) + 3*M; // First T is 0

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      double const theta = 2.0;
      double const alpha = params.alpha;
      double const ralpha = 1.0 / alpha;

      vector<Vector3d> Xs(M);
      vector<Vector5d> Ys(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));
      vector<Vector5d> Ws(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));
      vector<Vector3d> UTs(N, makeVector3(0.0, 0.0, 0.0));
      vector<Vector3d> UXs(M);
      vector<Vector5d> Vs(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));

      vector<Vector3d> savedXs(M);
      double E_saved = 1e30, E_cur = 0;

      Vector<double> LtY(nVars);
      Vector<double> LtV(nVars);

      double gamma;

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         // Ws = arg min_W J(W) + alpha/2 |W - (A*X + Y)|^2
         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

            double const x1 = XX[0];
            double const y1 = sigma*XX[2];
            double const x2 = XX[1];
            double const y2 = sigma*XX[2];

            double xx1 = x1 + Ys[k][0];
            double yy1 = y1 + Ys[k][1];
            double xx2 = x2 + Ys[k][2];
            double yy2 = y2 + Ys[k][3];

            prox_g1(ralpha, xx1, yy1);
            prox_g1(ralpha, xx2, yy2);

            Ws[k][0] = xx1; Ws[k][1] = yy1;
            Ws[k][2] = xx2; Ws[k][3] = yy2;

            double const ww = std::max(XX[2] + Ys[k][4], 1.0);
            Ws[k][4] = ww;
         } // end for (k)

         // Compute u = x - A^T y
         // Simulatenously compute numer = |x-u|^2 + |y-v|^2 = |A^T y|^2 + |w - Ax|^2
         // and denom = numer + |A(x - u)|^2 + |A^T (y - v)|^2

         compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, Ys, 0.0, Ys, LtY);

         double numer = sqrNorm_L2(LtY);

         for (int i = 1; i < N; ++i)
         {
            UTs[i][0] = translations[i][0] - LtY[TVAR(i, 0)];
            UTs[i][1] = translations[i][1] - LtY[TVAR(i, 1)];
            UTs[i][2] = translations[i][2] - LtY[TVAR(i, 2)];
         }

         for (int j = 0; j < M; ++j)
         {
            UXs[j][0] = Xs[j][0] - LtY[XVAR(j, 0)];
            UXs[j][1] = Xs[j][1] - LtY[XVAR(j, 1)];
            UXs[j][2] = Xs[j][2] - LtY[XVAR(j, 2)];
         }

         // Compute v = Ax + y - w
         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

            double const x1 = XX[0];
            double const y1 = sigma*XX[2];
            double const x2 = XX[1];
            double const y2 = sigma*XX[2];
            double const z  = XX[2];

            Vs[k][0] = x1 + Ys[k][0] - Ws[k][0];
            Vs[k][1] = y1 + Ys[k][1] - Ws[k][1];
            Vs[k][2] = x2 + Ys[k][2] - Ws[k][2];
            Vs[k][3] = y2 + Ys[k][3] - Ws[k][3];
            Vs[k][4] = z  + Ys[k][4] - Ws[k][4];

            numer += sqr(Ws[k][0] - x1);
            numer += sqr(Ws[k][1] - y1);
            numer += sqr(Ws[k][2] - x2);
            numer += sqr(Ws[k][3] - y2);
            numer += sqr(Ws[k][4] - z);
         } // end for (k)

         double denom = numer;
         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];
            Matrix3x3d const& R = rotations[i];
            Vector3d   const  T = translations[i] - UTs[i];
            Vector3d   const  X = Xs[j] - UXs[j];
            Vector2d   const& m = bundleStruct.measurements[k];
            float const u1 = m[0];
            float const u2 = m[1];

            double const XX1 = R[0][0]*X[0] + R[0][1]*X[1] + R[0][2]*X[2] + T[0];
            double const XX2 = R[1][0]*X[0] + R[1][1]*X[1] + R[1][2]*X[2] + T[1];
            double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

            double const x1 = u1*XX3 - XX1;
            double const y1 = sigma*XX3;
            double const x2 = u2*XX3 - XX2;
            double const y2 = sigma*XX3;
            double const z  = XX3;

            denom += sqr(x1) + sqr(y1);
            denom += sqr(x2) + sqr(y2);
            denom += sqr(z);
         } // end for (k)

         compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, Vs, 0.0, Vs, LtV);
         for (int i = 0; i < nVars; ++i)
            denom += sqr(LtY[i] - LtV[i]);

         gamma = theta * numer / denom;
         //cout << "gamma = " << gamma << endl;

         // x_new = x - gamma*(x - u - A^T y + A^T v) = x - gamma A^T v
         for (int i = 1; i < N; ++i)
         {
            translations[i][0] -= gamma * LtV[TVAR(i, 0)];
            translations[i][1] -= gamma * LtV[TVAR(i, 1)];
            translations[i][2] -= gamma * LtV[TVAR(i, 2)];
         }
         for (int j = 0; j < M; ++j)
         {
            Xs[j][0] -= gamma * LtV[XVAR(j, 0)];
            Xs[j][1] -= gamma * LtV[XVAR(j, 1)];
            Xs[j][2] -= gamma * LtV[XVAR(j, 2)];
         }

         // y_new = y - gamma*(Ax - Au + y - v) = y - gamma*(w - Au)
         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, UTs, UXs);

            double const x1 = XX[0];
            double const y1 = sigma*XX[2];
            double const x2 = XX[1];
            double const y2 = sigma*XX[2];
            double const z  = XX[2];

            Ys[k][0] -= gamma * (Ws[k][0] - x1);
            Ys[k][1] -= gamma * (Ws[k][1] - y1);
            Ys[k][2] -= gamma * (Ws[k][2] - x2);
            Ys[k][3] -= gamma * (Ws[k][3] - y2);
            Ys[k][4] -= gamma * (Ws[k][4] - z);
         }

         if (iter == 0) savedXs = Xs;

         double const eps = 1e-6;

         if (((iter+1) % params.checkFrequency) == 0)
         {
            E_cur = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               E_cur += std::max(0.0, fabs(XX[0]) - c);
               E_cur += std::max(0.0, fabs(XX[1]) - c);

               if (XX[2] < 1.0 - eps) E_cur += 1e30;
            } // end for (k)
         } // end if

         if (params.verbose && (((iter+1) % params.reportFrequency) == 0 || iter == params.nIterations-1 || iter == 9))
         {
            cout << "iter = " << iter << endl;
            cout << "gamma = " << gamma << endl;

            int    nFeasible = 0;

            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

               //if (XX3 >= 0.9999) ++nFeasible;
               if (XX3 > 1.0 - eps) ++nFeasible;
            } // end for (k)

            double E_primal = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               E_primal += std::max(0.0, fabs(XX[0]) - c);
               E_primal += std::max(0.0, fabs(XX[1]) - c);

               //if (XX3 < 1.0 - eps) E_primal += 1e30;
            } // end for (k)

            cout << "E_cur = " << E_cur << ", E_saved = " << E_saved << endl;
            cout << "nFeasible = " << nFeasible << "/" << K << endl;
#if 0
            cout << "Current structure distance = " << computeStructureDistance(savedXs, Xs) << endl;
#else
            cout << "Relative energy change = " << fabs(E_cur - E_saved) / std::min(E_cur, E_saved) << endl;
#endif

            double E_dual = 0;
//             for (int k = 0; k < K; ++k)
//             {
//                // Note: the dual vars are really z/gamma (see the A-L formulation)
//                double const x1 = Zs[k][0] / gamma, y1 = Zs[k][1] / gamma;
//                double const x2 = Zs[k][2] / gamma, y2 = Zs[k][3] / gamma;
//                double const z = Vs[k] / gamma;

//                double accum = 0;
//                accum += std::min(0.0, z);
//                if (z > eps) accum += 1e30;

// #if 0
//                if (fabs(x1) > std::min(1.0, -y1) + eps) accum += 1e30;
//                if (fabs(x2) > std::min(1.0, -y2) + eps) accum += 1e30;
// #else
//                if (fabs(x1) > 1.0 + eps) accum += 1e30;
//                if (fabs(x1) > -y1 + eps) accum += 1e30;
//                if (fabs(x2) > 1.0 + eps) accum += 1e30;
//                if (fabs(x2) > -y2 + eps) accum += 1e30;
// #endif

//                E_dual -= accum;
//             } // end for (k)

//             makeZeroVector(LtP);
//             for (int k = 0; k < K; ++k)
//             {
//                int const i = correspondingView[k];
//                int const j = correspondingPoint[k];

//                Matrix3x3d const& R  = rotations[i];
//                Vector3d   const& T  = translations[i];
//                Vector3d   const& X  = Xs[j];
//                Vector2d   const& m  = bundleStruct.measurements[k];
//                float      const  u1 = m[0];
//                float      const  u2 = m[1];

//                float const x1 = Zs[k][0], y1 = Zs[k][1];
//                float const x2 = Zs[k][2], y2 = Zs[k][3];
//                float const z = Vs[k];

//                if (i > 0)
//                {
//                   rhs[TVAR(i, 0)] += -x1;
//                   rhs[TVAR(i, 1)] += -x2;
//                   rhs[TVAR(i, 2)] += x1*u1 + x2*u2 + sigma*(y1 + y2);
//                   rhs[TVAR(i, 2)] += z;
//                }

//                rhs[XVAR(j, 0)] += x1 * (u1*R[2][0] - R[0][0]) + sigma * y1 * R[2][0];
//                rhs[XVAR(j, 1)] += x1 * (u1*R[2][1] - R[0][1]) + sigma * y1 * R[2][1];
//                rhs[XVAR(j, 2)] += x1 * (u1*R[2][2] - R[0][2]) + sigma * y1 * R[2][2];

//                rhs[XVAR(j, 0)] += x2 * (u2*R[2][0] - R[1][0]) + sigma * y2 * R[2][0];
//                rhs[XVAR(j, 1)] += x2 * (u2*R[2][1] - R[1][1]) + sigma * y2 * R[2][1];
//                rhs[XVAR(j, 2)] += x2 * (u2*R[2][2] - R[1][2]) + sigma * y2 * R[2][2];

//                rhs[XVAR(j, 0)] += z * R[2][0];
//                rhs[XVAR(j, 1)] += z * R[2][1];
//                rhs[XVAR(j, 2)] += z * R[2][2];
//             } // end for (k)
//             cout << "||Q||_1 = " << norm_L1(rhs) << endl;

//             double const B = 10;
//             E_dual -= B * norm_L1(rhs) / gamma;

            cout << "E_primal = "  << E_primal << ", E_dual = " << E_dual << ", duality gap = " << (E_primal - E_dual)/E_primal * 100 << "%" << endl;

            cout << "Ts = [ ";
            for (int i = 0; i < N; ++i) cout << translations[i][0] << " " << translations[i][1] << " " << translations[i][2] << " / ";
            cout << "]" << endl;

            cout << "=== " << (iter+1) << " " << E_primal << " " << E_dual << endl;

//             cout << "Vs = [ ";
//             for (int k = 0; k < K; ++k) cout << Vs[k] << " ";
//             cout << "]" << endl;

            //gamma = std::max(gamma / 1.1, 1e-3);
            //gamma = std::min(1.0, 1.1*gamma);
            //cout << "new gamma = " << gamma << endl;
         } // end if

         if (((iter+1) % params.checkFrequency) == 0)
         {
#if 0
            if (computeStructureDistance(savedXs, Xs) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            savedXs = Xs;
#else
            if (E_cur < 1e30 && fabs(E_cur - E_saved) / std::min(E_cur, E_saved) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            E_saved = E_cur;
#endif
         }
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }

#undef TVAR
#undef XVAR
   } // end computeConsistentTranslationsConic_Aniso_LS_Free()

} // end namespace V3D

//**********************************************************************

namespace
{

   inline void
   reflect_g1(double gamma, double& x, double& y)
   {
      double const x0 = x;
      double const y0 = y;
      prox_g1(gamma, x, y);
      x = 2*x - x0;
      y = 2*y - y0;
   }

   inline void
   reflect_h(double& z)
   {
      double const z1 = std::max(1.0, z);
      z = 2*z1 - z;
   }

   typedef InlineVector<double, 6> Vector6d;

   typedef InlineMatrix<double, 5, 3> Matrix5x3d;
   typedef InlineMatrix<double, 5, 5> Matrix5x5d;
   typedef InlineMatrix<double, 5, 6> Matrix5x6d;
   typedef InlineMatrix<double, 6, 6> Matrix6x6d;

   // prox_f(xx,yy) = arg min (|x-xx|^2 + |y-yy|^2 s.t. Lx = y), i.e. x = inv(I + L^T L)(xx + L^T yy)
   // L = [R I], since Lx = RX+T, and L^T L = [I R^T; R I] and inv(I + L^T L) = 1/3 * [2I  -R^T; -R  2I].
   inline void
   prox_Lk(int const i, double const sigma, Matrix3x3d const& R, Vector2d const& m,
           Vector3d& X, Vector3d& T, Vector5d& Y)
   {
      float const u1 = m[0];
      float const u2 = m[1];

      if (i > 0)
      {
         Matrix5x6d L;

         L[0][0] = u1*R[2][0] - R[0][0];
         L[0][1] = u1*R[2][1] - R[0][1];
         L[0][2] = u1*R[2][2] - R[0][2];
         L[0][3] = -1;
         L[0][4] = 0;
         L[0][5] = u1;

         L[1][0] = sigma*R[2][0];
         L[1][1] = sigma*R[2][1];
         L[1][2] = sigma*R[2][2];
         L[1][3] = 0;
         L[1][4] = 0;
         L[1][5] = sigma;

         L[2][0] = u2*R[2][0] - R[1][0];
         L[2][1] = u2*R[2][1] - R[1][1];
         L[2][2] = u2*R[2][2] - R[1][2];
         L[2][3] = 0;
         L[2][4] = -1;
         L[2][5] = u2;

         L[3][0] = sigma*R[2][0];
         L[3][1] = sigma*R[2][1];
         L[3][2] = sigma*R[2][2];
         L[3][3] = 0;
         L[3][4] = 0;
         L[3][5] = sigma;

         L[4][0] = R[2][0];
         L[4][1] = R[2][1];
         L[4][2] = R[2][2];
         L[4][3] = 0;
         L[4][4] = 0;
         L[4][5] = 1.0;

         Matrix<double> LtL(6, 6);
         multiply_At_A(L, LtL);
         for (int k = 0; k < LtL.num_rows(); ++k) LtL[k][k] += 1.0;

         if (0 && i == 1)
         {
            cout << "L1 = "; displayMatrix(L);
            cout << "A1 = "; displayMatrix(LtL);
         }

         Vector<double> rhs(6);
         multiply_At_v(L, Y, rhs);
         //cout << __FILE__ << ":" << __LINE__ << endl;
         rhs[0] += X[0]; rhs[1] += X[1]; rhs[2] += X[2];
         rhs[3] += T[0]; rhs[4] += T[1]; rhs[5] += T[2];
         //addVectorsIP(X, rhs);

         LU<double> lu(LtL);
         rhs = lu.solve(rhs); // reuse rhs for solution

         multiply_A_v(L, rhs, Y);
         copyVectorSlice(rhs, 0, 3, X, 0);
         copyVectorSlice(rhs, 3, 3, T, 0);
      }
      else
      {
         // T_0 is always 0.
         Matrix5x3d L;

         L[0][0] = u1*R[2][0] - R[0][0];
         L[0][1] = u1*R[2][1] - R[0][1];
         L[0][2] = u1*R[2][2] - R[0][2];

         L[1][0] = sigma*R[2][0];
         L[1][1] = sigma*R[2][1];
         L[1][2] = sigma*R[2][2];

         L[2][0] = u2*R[2][0] - R[1][0];
         L[2][1] = u2*R[2][1] - R[1][1];
         L[2][2] = u2*R[2][2] - R[1][2];

         L[3][0] = sigma*R[2][0];
         L[3][1] = sigma*R[2][1];
         L[3][2] = sigma*R[2][2];

         L[4][0] = R[2][0];
         L[4][1] = R[2][1];
         L[4][2] = R[2][2];

         Matrix<double> LtL(3, 3);
         multiply_At_A(L, LtL);
         for (int k = 0; k < LtL.num_rows(); ++k) LtL[k][k] += 1.0;

         Vector<double> rhs(3);
         multiply_At_v(L, Y, rhs);
         //cout << __FILE__ << ":" << __LINE__ << endl;
         rhs[0] += X[0]; rhs[1] += X[1]; rhs[2] += X[2];
         //addVectorsIP(X, rhs);

         LU<double> lu(LtL);
         rhs = lu.solve(rhs); // reuse rhs for solution

         multiply_A_v(L, rhs, Y);
         copyVectorSlice(rhs, 0, 3, X, 0);
         makeZeroVector(T);
      } // end if (i > 0)
   } // end prox_Lk()

   inline void
   reflect_Lk(int const i, double const sigma, Matrix3x3d const& R, Vector2d const& m,
              Vector3d& X, Vector3d& T, Vector5d& Y)
   {
      Vector3d const X0 = X;
      Vector3d const T0 = T;
      Vector5d const Y0 = Y;
      prox_Lk(i, sigma, R, m, X, T, Y);
      X = 2.0*X - X0;
      T = 2.0*T - T0;
      Y = 2.0*Y - Y0;
   }

   inline void
   fillCachedMatrices(int const i, double const sigma, Matrix3x3d const& R, Vector2d const& m,
                      Matrix5x6d& Ldst, Matrix6x6d& invLtLdst)
   {
      float const u1 = m[0];
      float const u2 = m[1];

      if (i > 0)
      {
         Matrix5x6d L;

         L[0][0] = u1*R[2][0] - R[0][0];
         L[0][1] = u1*R[2][1] - R[0][1];
         L[0][2] = u1*R[2][2] - R[0][2];
         L[0][3] = -1;
         L[0][4] = 0;
         L[0][5] = u1;

         L[1][0] = sigma*R[2][0];
         L[1][1] = sigma*R[2][1];
         L[1][2] = sigma*R[2][2];
         L[1][3] = 0;
         L[1][4] = 0;
         L[1][5] = sigma;

         L[2][0] = u2*R[2][0] - R[1][0];
         L[2][1] = u2*R[2][1] - R[1][1];
         L[2][2] = u2*R[2][2] - R[1][2];
         L[2][3] = 0;
         L[2][4] = -1;
         L[2][5] = u2;

         L[3][0] = sigma*R[2][0];
         L[3][1] = sigma*R[2][1];
         L[3][2] = sigma*R[2][2];
         L[3][3] = 0;
         L[3][4] = 0;
         L[3][5] = sigma;

         L[4][0] = R[2][0];
         L[4][1] = R[2][1];
         L[4][2] = R[2][2];
         L[4][3] = 0;
         L[4][4] = 0;
         L[4][5] = 1.0;

         Matrix<double> LtL(6, 6);
         multiply_At_A(L, LtL);
         for (int k = 0; k < LtL.num_rows(); ++k) LtL[k][k] += 1.0;

         Matrix<double> invLtL(6, 6);
         invertMatrix(LtL, invLtL);

         copyMatrix(L, Ldst);
         copyMatrix(invLtL, invLtLdst);
      }
      else
      {
         // T_0 is always 0.
         Matrix5x3d L;

         L[0][0] = u1*R[2][0] - R[0][0];
         L[0][1] = u1*R[2][1] - R[0][1];
         L[0][2] = u1*R[2][2] - R[0][2];

         L[1][0] = sigma*R[2][0];
         L[1][1] = sigma*R[2][1];
         L[1][2] = sigma*R[2][2];

         L[2][0] = u2*R[2][0] - R[1][0];
         L[2][1] = u2*R[2][1] - R[1][1];
         L[2][2] = u2*R[2][2] - R[1][2];

         L[3][0] = sigma*R[2][0];
         L[3][1] = sigma*R[2][1];
         L[3][2] = sigma*R[2][2];

         L[4][0] = R[2][0];
         L[4][1] = R[2][1];
         L[4][2] = R[2][2];

         Matrix<double> LtL(3, 3);
         multiply_At_A(L, LtL);
         for (int k = 0; k < LtL.num_rows(); ++k) LtL[k][k] += 1.0;

         Matrix<double> invLtL(3, 3);
         invertMatrix(LtL, invLtL);

         copyMatrixSlice(L, 0, 0, 5, 3, Ldst, 0, 0);
         copyMatrixSlice(invLtL, 0, 0, 3, 3, invLtLdst, 0, 0);
      } // end if (i > 0)
   } // fillCachedMatrices()

   inline void
   prox_Lk(int const i, Matrix5x6d const& L, Matrix6x6d const& invLtL, Vector3d& X, Vector3d& T, Vector5d& Y)
   {
      if (i > 0)
      {
         Vector6d rhs;
         multiply_At_v(L, Y, rhs);
         rhs[0] += X[0]; rhs[1] += X[1]; rhs[2] += X[2];
         rhs[3] += T[0]; rhs[4] += T[1]; rhs[5] += T[2];
         //addVectorsIP(X, rhs);

         rhs = invLtL * rhs;

         multiply_A_v(L, rhs, Y);
         copyVectorSlice(rhs, 0, 3, X, 0);
         copyVectorSlice(rhs, 3, 3, T, 0);
      }
      else
      {
         // T_0 is always 0.
         Matrix5x3d L1;
         copyMatrixSlice(L, 0, 0, 5, 3, L1, 0, 0);
         Matrix3x3d invLtL1;
         copyMatrixSlice(invLtL, 0, 0, 3, 3, invLtL1, 0, 0);

         Vector3d rhs;
         multiply_At_v(L1, Y, rhs);
         rhs[0] += X[0]; rhs[1] += X[1]; rhs[2] += X[2];

         rhs = invLtL1 * rhs; // reuse rhs for solution

         multiply_A_v(L1, rhs, Y);
         copyVectorSlice(rhs, 0, 3, X, 0);
         makeZeroVector(T);
      } // end if (i > 0)
   } // end prox_Lk()

   inline void
   reflect_Lk(int const i, Matrix5x6d const& L, Matrix6x6d const& invLtL, Vector3d& X, Vector3d& T, Vector5d& Y)
   {
      Vector3d const X0 = X;
      Vector3d const T0 = T;
      Vector5d const Y0 = Y;
      prox_Lk(i, L, invLtL, X, T, Y);
      X = 2.0*X - X0;
      T = 2.0*T - T0;
      Y = 2.0*Y - Y0;
   }

} // end namespace <>

namespace V3D
{

   void
   computeConsistentTranslationsConic_Aniso_ADMM(float const sigma,
                                                 std::vector<Matrix3x3d> const& rotations,
                                                 std::vector<Vector3d>& translations,
                                                 std::vector<TriangulatedPoint>& sparseModel,
                                                 MultiViewInitializationParams_BOS const& params)
   {
      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);
      makeZeroVector(translations[0]);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      int const nVars = 3*(N-1) + 3*M; // First T is 0

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      double const gamma = params.alpha;
      double const lambda = 0.5;
      double const lambda_ = 1.0 - lambda;

      vector<Vector3d>& Ts = translations;
      vector<Vector3d>  Xs(M);

      vector<Vector3d> TTs(K);
      vector<Vector3d> XXs(K);
      vector<Vector5d> YYs(K);

      vector<Vector3d> TT1s(K);
      vector<Vector3d> XX1s(K);
      vector<Vector5d> YY1s(K);

      vector<double> denomTs(N, 0.0);
      vector<double> denomXs(M, 0.0);

      vector<Matrix5x6d> Ls(K);
      vector<Matrix6x6d> invLtLs(K);
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         Matrix3x3d const& R = rotations[i];
         Vector2d   const& m = bundleStruct.measurements[k];

         fillCachedMatrices(i, sigma, R, m, Ls[k], invLtLs[k]);
      } // end for (i)

      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         denomTs[i] += 1.0;
         denomXs[j] += 1.0;
      }

      // cout << "denomTs = "; displayVector(denomTs);
      // cout << "denomXs = "; displayVector(denomXs);

      for (int i = 0; i < N; ++i) denomTs[i] = 1.0 / denomTs[i];
      for (int j = 0; j < M; ++j) denomXs[j] = 1.0 / denomXs[j];

      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

         TTs[k] = translations[i];
         XXs[k] = Xs[j];

         YYs[k][0] = XX[0];
         YYs[k][1] = sigma*XX[2];
         YYs[k][2] = XX[1];
         YYs[k][3] = sigma*XX[2];
         YYs[k][4] = XX[2];
      } // end for (k)

      vector<Vector3d> savedXs(M);
      double E_saved = 1e30, E_cur = 0;

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
#if 0
         // First reflect
         for (int k = 0; k < K; ++k)
         {
            YY1s[k] = YYs[k];
            reflect_g1(gamma, YY1s[k][0], YY1s[k][1]);
            reflect_g1(gamma, YY1s[k][2], YY1s[k][3]);
            reflect_h(YY1s[k][4]);
         }

         for (int i = 0; i < N; ++i) makeZeroVector(Ts[i]);
         for (int j = 0; j < M; ++j) makeZeroVector(Xs[j]);

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            if (i > 0) addVectorsIP(TTs[k], Ts[i]);
            addVectorsIP(XXs[k], Xs[j]);
         } // end for (k)

         for (int i = 0; i < N; ++i) scaleVectorIP(denomTs[i], Ts[i]);
         for (int j = 0; j < M; ++j) scaleVectorIP(denomXs[j], Xs[j]);

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            if (i > 0) TT1s[k] = 2.0*Ts[i] - TTs[k];
            XX1s[k] = 2.0*Xs[j] - XXs[k];
         } // end for (k)

         // Second reflect
         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            Matrix3x3d const& R = rotations[i];
            Vector2d   const& m = bundleStruct.measurements[k];

            //reflect_Lk(i, sigma, R, m, XX1s[k], TT1s[k], YY1s[k]);
            reflect_Lk(i, Ls[k], invLtLs[k], XX1s[k], TT1s[k], YY1s[k]);
         } // end for (i)
#else
         // In this code path change the order of reflections
         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            Matrix3x3d const& R = rotations[i];
            Vector2d   const& m = bundleStruct.measurements[k];

            YY1s[k] = YYs[k];
            TT1s[k] = TTs[k];
            YY1s[k] = YYs[k];

            //reflect_Lk(i, sigma, R, m, XX1s[k], TT1s[k], YY1s[k]);
            reflect_Lk(i, Ls[k], invLtLs[k], XX1s[k], TT1s[k], YY1s[k]);
         } // end for (i)

         for (int k = 0; k < K; ++k)
         {
            reflect_g1(gamma, YY1s[k][0], YY1s[k][1]);
            reflect_g1(gamma, YY1s[k][2], YY1s[k][3]);
            reflect_h(YY1s[k][4]);
         }

         for (int i = 0; i < N; ++i) makeZeroVector(Ts[i]);
         for (int j = 0; j < M; ++j) makeZeroVector(Xs[j]);

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            if (i > 0) addVectorsIP(TT1s[k], Ts[i]);
            addVectorsIP(XX1s[k], Xs[j]);
         } // end for (k)

         for (int i = 0; i < N; ++i) scaleVectorIP(denomTs[i], Ts[i]);
         for (int j = 0; j < M; ++j) scaleVectorIP(denomXs[j], Xs[j]);

         for (int k = 0; k < K; ++k)
         {
            int const i = correspondingView[k];
            int const j = correspondingPoint[k];

            if (i > 0) TT1s[k] = 2.0*Ts[i] - TT1s[k];
            XX1s[k] = 2.0*Xs[j] - XX1s[k];
         } // end for (k)
#endif

         // Final averaging
         for (int k = 0; k < K; ++k)
         {
            TTs[k] = lambda_*TTs[k] + lambda*TT1s[k];
            XXs[k] = lambda_*XXs[k] + lambda*XX1s[k];
            YYs[k] = lambda_*YYs[k] + lambda*YY1s[k];
         }

         if (iter == 0) savedXs = Xs;

         double const eps = 1e-6;

         if (((iter+1) % params.checkFrequency) == 0)
         {
            E_cur = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               E_cur += std::max(0.0, fabs(XX[0]) - c);
               E_cur += std::max(0.0, fabs(XX[1]) - c);

               if (XX[2] < 1.0 - eps) E_cur += 1e30;
            } // end for (k)
         } // end if

         if (params.verbose && (((iter+1) % params.reportFrequency) == 0 || iter == params.nIterations-1 || iter == 9))
         {
            cout << "iter = " << iter << endl;
            cout << "gamma = " << gamma << endl;

            int    nFeasible = 0;

            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

               //if (XX3 >= 0.9999) ++nFeasible;
               if (XX3 > 1.0 - eps) ++nFeasible;
            } // end for (k)

            double E_primal = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               E_primal += std::max(0.0, fabs(XX[0]) - c);
               E_primal += std::max(0.0, fabs(XX[1]) - c);

               //if (XX3 < 1.0 - eps) E_primal += 1e30;
            } // end for (k)

            cout << "E_cur = " << E_cur << ", E_saved = " << E_saved << endl;
            cout << "nFeasible = " << nFeasible << "/" << K << endl;
#if 0
            cout << "Current structure distance = " << computeStructureDistance(savedXs, Xs) << endl;
#else
            cout << "Relative energy change = " << fabs(E_cur - E_saved) / std::min(E_cur, E_saved) << endl;
#endif

            double E_dual = 0;

            cout << "E_primal = "  << E_primal << ", E_dual = " << E_dual << ", duality gap = " << (E_primal - E_dual)/E_primal * 100 << "%" << endl;

            cout << "Ts = [ ";
            for (int i = 0; i < N; ++i) cout << translations[i][0] << " " << translations[i][1] << " " << translations[i][2] << " / ";
            cout << "]" << endl;

            // cout << "Xs = [ ";
            // for (int i = 0; i < M; ++i) cout << Xs[i][0] << " " << Xs[i][1] << " " << Xs[i][2] << " / ";
            // cout << "]" << endl;

            cout << "=== " << (iter+1) << " " << E_primal << " " << E_dual << endl;

//             cout << "Vs = [ ";
//             for (int k = 0; k < K; ++k) cout << Vs[k] << " ";
//             cout << "]" << endl;

            //gamma = std::max(gamma / 1.1, 1e-3);
            //gamma = std::min(1.0, 1.1*gamma);
            //cout << "new gamma = " << gamma << endl;
         } // end if

         if (((iter+1) % params.checkFrequency) == 0)
         {
#if 0
            if (computeStructureDistance(savedXs, Xs) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            savedXs = Xs;
#else
            if (E_cur < 1e30 && fabs(E_cur - E_saved) / std::min(E_cur, E_saved) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            E_saved = E_cur;
#endif
         }
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }
   } // end computeConsistentTranslationsConic_Aniso_ADMM()

} // end namespace V3D

#if defined(V3DLIB_ENABLE_SUITESPARSE)

namespace
{

   // h1(XX3, rho) = rho s.t. rho >= 0 and XX3+rho >= 1.
   inline void
   prox_h1(double const gamma, double& XX3, double& rho)
   {
      rho -= gamma;

      if (rho >= 0.0 && XX3 + rho >= 1.0) return;

      if (XX3 >= 1.0)
      {
         rho = std::max(0.0, rho);
         return;
      }

      if (XX3 < 1.0 + rho)
      {
         double const delta = (1.0 - XX3 - rho) / 2;
         XX3 += delta;
         rho += delta;
         return;
      }

      XX3 = 1.0;
      rho = 0.0;
   } // end prox_h1()

   inline Vector3d
   transformIntoMeasurementCone(int const k, float const sigma, BundlePointStructure const& bundleStruct,
                                std::vector<Matrix3x3d> const& rotations, vector<Vector3d> const& Ts,
                                vector<Vector3d> const& Xs, vector<double> const& rhos)
   {
      Vector3d res;

      int const i = bundleStruct.correspondingView[k];
      int const j = bundleStruct.correspondingPoint[k];
      Matrix3x3d const& R = rotations[i];
      Vector3d   const& T = Ts[i];
      Vector3d   const& X = Xs[j];
      Vector2d   const& m = bundleStruct.measurements[k];
      float const u1 = m[0];
      float const u2 = m[1];

      Vector3d XX = R*X + T;

      res[0] = u1*XX[2] - XX[0];
      res[1] = u2*XX[2] - XX[1];
      res[2] = XX[2] + rhos[k];

      return res;
   } // end transformIntoMeasurementCone()

   // Compute A^T (w1*Y1 + w2*Y2)
   inline void
   compute_At_Y_Aniso(float const sigma, BundlePointStructure const& bundleStruct,
                      std::vector<Matrix3x3d> const& rotations,
                      double const w1, vector<Vector6d> const& Y1, double const w2, vector<Vector6d> const& Y2,
                      Vector<double>& AtY)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))
#define RVAR(k) (3*(N-1) + 3*M + (k))

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;
      int const K = correspondingView.size();
      int const N = rotations.size();
      int const M = bundleStruct.points3d.size();

      makeZeroVector(AtY);
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         float const x1 = w1*Y1[k][0] + w2*Y2[k][0];
         float const y1 = w1*Y1[k][1] + w2*Y2[k][1];
         float const x2 = w1*Y1[k][2] + w2*Y2[k][2];
         float const y2 = w1*Y1[k][3] + w2*Y2[k][3];
         float const z  = w1*Y1[k][4] + w2*Y2[k][4];
         float const w =  w1*Y1[k][5] + w2*Y2[k][5];

         if (i > 0)
         {
            AtY[TVAR(i, 0)] += -x1;
            AtY[TVAR(i, 1)] += -x2;
            AtY[TVAR(i, 2)] += x1*u1 + x2*u2 + sigma*(y1 + y2);
            AtY[TVAR(i, 2)] += z;
         }

         AtY[XVAR(j, 0)] += x1 * (u1*R[2][0] - R[0][0]) + sigma * y1 * R[2][0];
         AtY[XVAR(j, 1)] += x1 * (u1*R[2][1] - R[0][1]) + sigma * y1 * R[2][1];
         AtY[XVAR(j, 2)] += x1 * (u1*R[2][2] - R[0][2]) + sigma * y1 * R[2][2];

         AtY[XVAR(j, 0)] += x2 * (u2*R[2][0] - R[1][0]) + sigma * y2 * R[2][0];
         AtY[XVAR(j, 1)] += x2 * (u2*R[2][1] - R[1][1]) + sigma * y2 * R[2][1];
         AtY[XVAR(j, 2)] += x2 * (u2*R[2][2] - R[1][2]) + sigma * y2 * R[2][2];

         AtY[XVAR(j, 0)] += z * R[2][0];
         AtY[XVAR(j, 1)] += z * R[2][1];
         AtY[XVAR(j, 2)] += z * R[2][2];

         AtY[RVAR(k)] += sigma*(y1 + y2) + z + w;
      } // end for (k)
#undef TVAR
#undef XVAR
#undef RVAR
   } // end compute_At_Y_Aniso()

   inline void
   compute_At_Y_Iso(float const sigma, BundlePointStructure const& bundleStruct,
                    std::vector<Matrix3x3d> const& rotations,
                    double const w1, vector<Vector5d> const& Y1, double const w2, vector<Vector5d> const& Y2,
                    Vector<double>& AtY)
   {
      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;
      int const K = correspondingView.size();
      int const N = rotations.size();
      int const M = bundleStruct.points3d.size();

      makeZeroVector(AtY);

#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))
#define RVAR(k) (3*(N-1) + 3*M + (k))
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         float const x1 = w1*Y1[k][0] + w2*Y2[k][0];
         float const x2 = w1*Y1[k][1] + w2*Y2[k][1];
         float const y  = w1*Y1[k][2] + w2*Y2[k][2];
         float const z  = w1*Y1[k][3] + w2*Y2[k][3];
         float const w  = w1*Y1[k][4] + w2*Y2[k][4];

         if (i > 0)
         {
            AtY[TVAR(i, 0)] += -x1;
            AtY[TVAR(i, 1)] += -x2;
            AtY[TVAR(i, 2)] += x1*u1 + x2*u2 + sigma*y;
            AtY[TVAR(i, 2)] += z;
         }

         AtY[XVAR(j, 0)] += x1 * (u1*R[2][0] - R[0][0]);
         AtY[XVAR(j, 1)] += x1 * (u1*R[2][1] - R[0][1]);
         AtY[XVAR(j, 2)] += x1 * (u1*R[2][2] - R[0][2]);

         AtY[XVAR(j, 0)] += x2 * (u2*R[2][0] - R[1][0]);
         AtY[XVAR(j, 1)] += x2 * (u2*R[2][1] - R[1][1]);
         AtY[XVAR(j, 2)] += x2 * (u2*R[2][2] - R[1][2]);

         AtY[XVAR(j, 0)] += sigma * y * R[2][0];
         AtY[XVAR(j, 1)] += sigma * y * R[2][1];
         AtY[XVAR(j, 2)] += sigma * y * R[2][2];

         AtY[XVAR(j, 0)] += z * R[2][0];
         AtY[XVAR(j, 1)] += z * R[2][1];
         AtY[XVAR(j, 2)] += z * R[2][2];

         AtY[RVAR(k)] += sigma*y + z + w;
      } // end for (k)
#undef TVAR
#undef XVAR
#undef RVAR
   } // end compute_At_Y_Iso()

} // end namespace <>

namespace V3D
{

   void
   computeConsistentTranslationsRelaxedConic_Aniso_SDMM(float const sigma,
                                                        std::vector<Matrix3x3d> const& rotations,
                                                        std::vector<Vector3d>& translations,
                                                        std::vector<TriangulatedPoint>& sparseModel,
                                                        MultiViewInitializationParams_BOS const& params)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))
#define RVAR(k) (3*(N-1) + 3*M + (k))

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);
      makeZeroVector(translations[0]);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      int const nVars = 3*(N-1) + 3*M + K; // First T is 0

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      vector<pair<int, int> > nzL;
      vector<double>          valsL;
      nzL.reserve(22*K);
      valsL.reserve(22*K);

      int row = 0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         if (i > 0)
         {
            nzL.push_back(make_pair(row, TVAR(i, 0))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u1);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 1))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u2);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(1.0);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;

            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;
         }
         else
         {
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;

            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;
         } // end if
      } // end for (k)

      CCS_Matrix<double> L(row, nVars, nzL, valsL);
      CCS_Matrix<double> Q;

      multiply_At_A_SparseSparse(L, Q);

      SparseCholesky chol(Q, params.verbose);
      bool status = chol.setAtA(Q);
      cout << "Cholesky status = " << (status ? "good" : "bad") << endl;

      vector<Vector3d> Xs(M);
      vector<double>   rhos(K, 0.0);
      vector<Vector6d> Ys(K, makeVector6(0.0, double(sigma), 0.0, double(sigma), 0.0, 0.0));
      vector<Vector6d> Zs(K, makeVector6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

      vector<Vector3d> savedXs(M);
      double E_saved = 1e30, E_cur = 0;
      double E_primal_best = 1e40, E_dual_best = -1e40;

      Vector<double> rhs(nVars), X_new(nVars), mu(nVars);;

      double gamma = params.alpha;

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, Ys, -1.0, Zs, rhs);

         chol.solve(rhs, X_new);

         for (int i = 1; i < N; ++i) copyVectorSlice(X_new, TVAR(i, 0), 3, translations[i], 0);
         for (int j = 0; j < M; ++j) copyVectorSlice(X_new, XVAR(j, 0), 3, Xs[j], 0);
         for (int k = 0; k < K; ++k) rhos[k] = X_new[RVAR(k)];

         if (iter == 0) savedXs = Xs;

         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs, rhos);
            double const r  = rhos[k];

            double const x1 = XX[0];
            double const y1 = sigma*XX[2];
            double const x2 = XX[1];
            double const y2 = sigma*XX[2];

            double xx1 = x1 + Zs[k][0];
            double yy1 = y1 + Zs[k][1];
            double xx2 = x2 + Zs[k][2];
            double yy2 = y2 + Zs[k][3];

            prox_g1(gamma, xx1, yy1);
            prox_g1(gamma, xx2, yy2);

            Ys[k][0] = xx1; Ys[k][1] = yy1;
            Ys[k][2] = xx2; Ys[k][3] = yy2;

            Zs[k][0] += x1 - xx1; Zs[k][1] += y1 - yy1;
            Zs[k][2] += x2 - xx2; Zs[k][3] += y2 - yy2;

            double const z = XX[2];

            double zz = z + Zs[k][4];
            double rr = r + Zs[k][5];

            //prox_h1(gamma, zz, rr);
            zz = std::max(1.0, zz);
            rr -= gamma;
            rr = std::max(0.0, rr);

            Ys[k][4] = zz;
            Ys[k][5] = rr;

            Zs[k][4] += z - zz;
            Zs[k][5] += r - rr;
         } // end for (k)

         double const eps = 1e-4;

         if (((iter+1) % params.checkFrequency) == 0)
         {
            E_cur = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs, rhos);

               double const c = sigma*XX[2];
               E_cur += std::max(0.0, fabs(XX[0]) - c);
               E_cur += std::max(0.0, fabs(XX[1]) - c);

               double rho = rhos[k], XX3 = XX[2] - rho;
               if (XX3 < 1.0) rho = std::max(rho, 1.0 - XX3);
               E_cur += rho;
            } // end for (k)
         } // end if

         if (params.verbose && (((iter+1) % params.reportFrequency) == 0 || iter == params.nIterations-1 || iter == 9))
         {
            cout << "iter = " << iter << endl;

            int    nFeasible = 0;
            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

               if (XX3 > 1.0 - eps) ++nFeasible;
            } // end for (k)

            double E_primal = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs, rhos);

               double const c = sigma*XX[2];
               E_primal += std::max(0.0, fabs(XX[0]) - c);
               E_primal += std::max(0.0, fabs(XX[1]) - c);

               double rho = rhos[k], XX3 = XX[2] - rho;
               if (XX3 < 1.0) rho = std::max(rho, 1.0 - XX3);
               E_primal += rho;
            } // end for (k)

            cout << "E_cur = " << E_cur << ", E_saved = " << E_saved << endl;
            cout << "nFeasible = " << nFeasible << "/" << K << endl;
#if 0
            cout << "Current structure distance = " << computeStructureDistance(savedXs, Xs) << endl;
#else
            cout << "Relative energy change = " << fabs(E_cur - E_saved) / std::min(E_cur, E_saved) << endl;
#endif

            double E_dual = 0;
#if 0
            vector<Vector3d> mu_Ts(N);
            vector<Vector3d> mu_Xs(M);
            vector<double>   mu_rhos(K);
            {
               compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0/gamma, Zs, 0.0, Zs, rhs);
               chol.solve(rhs, mu);

               for (int i = 1; i < N; ++i) copyVectorSlice(mu, TVAR(i, 0), 3, mu_Ts[i], 0);
               for (int j = 0; j < M; ++j) copyVectorSlice(mu, XVAR(j, 0), 3, mu_Xs[j], 0);
               makeZeroVector(mu_Ts[0]);
               for (int k = 0; k < K; ++k) mu_rhos[k] = mu[RVAR(k)];
            }

            vector<Vector6d> PPs(K, makeVector6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, mu_Ts, mu_Xs, mu_rhos);

               double const x1 = XX[0];
               double const y1 = sigma*XX[2];
               double const x2 = XX[1];
               double const y2 = sigma*XX[2];
               double const z  = XX[2];
               double const r  = mu_rhos[k];

               // Note: the dual vars are really z/gamma (see the A-L formulation)
               PPs[k][0] = Zs[k][0] / gamma - x1;
               PPs[k][1] = Zs[k][1] / gamma - y1;
               PPs[k][2] = Zs[k][2] / gamma - x2;
               PPs[k][3] = Zs[k][3] / gamma - y2;
               PPs[k][4] = Zs[k][4] / gamma - z;
               PPs[k][5] = Zs[k][5] / gamma - r;
            } // end for (k)

            compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, PPs, 0.0, PPs, rhs);
            cout << "norm(At*lambda) = " << norm_L2(rhs) << endl;

            for (int k = 0; k < K; ++k)
            {
               double const x1 = PPs[k][0], y1 = PPs[k][1];
               double const x2 = PPs[k][2], y2 = PPs[k][3];
               double const z  = PPs[k][4], r  = PPs[k][5];

               double accum = 0;
               //accum += std::min(0.0, z);
               accum += z;

               if (z > eps) accum += 1e30;
               if (r > 1+eps) accum += 1e30;

               if (fabs(x1) > 1.0 + eps) accum += 1e30;
               if (fabs(x1) > -y1 + eps) accum += 1e30;
               if (fabs(x2) > 1.0 + eps) accum += 1e30;
               if (fabs(x2) > -y2 + eps) accum += 1e30;

               E_dual -= accum;
            } // end for (k)
#else
            vector<Vector6d> PPs(K, makeVector6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
            for (int k = 0; k < K; ++k)
            {
               // Note: the dual vars are really z/gamma (see the A-L formulation)
               double x1 = Zs[k][0] / gamma;
               double y1 = Zs[k][1] / gamma;
               double x2 = Zs[k][2] / gamma;
               double y2 = Zs[k][3] / gamma;
               double z  = Zs[k][4] / gamma;
               double r  = Zs[k][5] / gamma;

               // x1 = clamp(x1, -1.0, 1.0);
               // y1 = std::max(y1, -fabs(x1));
               // x2 = clamp(x2, -1.0, 1.0);
               // y2 = std::max(y2, -fabs(x2));
               projectZs(x1, y1);
               projectZs(x2, y2);
               z = std::min(0.0, z);
               r = std::min(1.0, r);

               PPs[k][0] = x1; PPs[k][1] = y1;
               PPs[k][2] = x2; PPs[k][3] = y2;
               PPs[k][4] = z;  PPs[k][5] = r;

               double accum = 0;
               accum += z;

               E_dual -= accum;
            } // end for (k)
            compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, PPs, 0.0, PPs, rhs);
            cout << "||Q||_1 = " << norm_L1(rhs) << endl;

            //double const B = 10;
# if 0
            double B = 0.0;
            for (int i = 0; i < N; ++i) B = std::max(B, norm_Linf(translations[i]));
            for (int j = 0; j < M; ++j) B = std::max(B, norm_Linf(Xs[j]));
            cout << "B = " << B << endl;
            E_dual -= B * norm_L1(rhs);
# else
            for (int i = 1; i < N; ++i)
            {
               E_dual -= fabs(translations[i][0] * rhs[TVAR(i, 0)]);
               E_dual -= fabs(translations[i][1] * rhs[TVAR(i, 1)]);
               E_dual -= fabs(translations[i][2] * rhs[TVAR(i, 2)]);
            }
            for (int j = 0; j < M; ++j)
            {
               E_dual -= fabs(Xs[j][0] * rhs[XVAR(j, 0)]);
               E_dual -= fabs(Xs[j][1] * rhs[XVAR(j, 1)]);
               E_dual -= fabs(Xs[j][2] * rhs[XVAR(j, 2)]);
            }
# endif
#endif

            E_primal_best = std::min(E_primal_best, E_primal);
            E_dual_best   = std::max(E_dual_best, E_dual);

            //cout << "E_primal = "  << E_primal << ", E_dual = " << E_dual << ", duality gap = " << (E_primal - E_dual)/E_primal * 100 << "%" << endl;
            cout << "E_primal = "  << E_primal_best << ", E_dual = " << E_dual_best << ", duality gap = " << (E_primal_best - E_dual_best)/E_primal_best * 100 << "%" << endl;

            // cout << "Ts = [ ";
            // for (int i = 0; i < N; ++i) cout << translations[i][0] << " " << translations[i][1] << " " << translations[i][2] << " / ";
            // cout << "]" << endl;

            cout << "=== " << (iter+1) << " " << E_primal_best << " " << E_dual_best << endl;
         } // end if

         if (((iter+1) % params.checkFrequency) == 0)
         {
#if 0
            if (computeStructureDistance(savedXs, Xs) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            savedXs = Xs;
#else
            if (E_cur < 1e30 && fabs(E_cur - E_saved) / std::min(E_cur, E_saved) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            E_saved = E_cur;
#endif
         }
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }

#undef TVAR
#undef XVAR
#undef RVAR
   } // end computeConsistentTranslationsRelaxedConic_Aniso_SDMM()

   void
   computeConsistentTranslationsRelaxedConic_Iso_SDMM(float const sigma,
                                                      std::vector<Matrix3x3d> const& rotations,
                                                      std::vector<Vector3d>& translations,
                                                      std::vector<TriangulatedPoint>& sparseModel,
                                                      MultiViewInitializationParams_BOS const& params)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))
#define RVAR(k) (3*(N-1) + 3*M + (k))

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);
      makeZeroVector(translations[0]);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      int const nVars = 3*(N-1) + 3*M + K; // First T is 0

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      vector<pair<int, int> > nzL;
      vector<double>          valsL;
      nzL.reserve(22*K);
      valsL.reserve(22*K);

      int row = 0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         if (i > 0)
         {
            nzL.push_back(make_pair(row, TVAR(i, 0))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u1);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 1))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u2);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(1.0);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;

            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;
         }
         else
         {
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;

            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;
         } // end if
      } // end for (k)

      CCS_Matrix<double> L(row, nVars, nzL, valsL);
      CCS_Matrix<double> Q;

      multiply_At_A_SparseSparse(L, Q);

      SparseCholesky chol(Q, params.verbose);
      bool status = chol.setAtA(Q);
      cout << "Cholesky status = " << (status ? "good" : "bad") << endl;

      vector<Vector3d> Xs(M);
      vector<double>   rhos(K, 0.0);
      vector<Vector5d> Ys(K, makeVector5(0.0, 0.0, double(sigma), 0.0, 0.0));
      vector<Vector5d> Zs(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));

      vector<Vector3d> savedXs(M);
      double E_saved = 1e30, E_cur = 0;
      double E_primal_best = 1e40, E_dual_best = -1e40;

      Vector<double> rhs(nVars), X_new(nVars), mu(nVars);;

      double gamma = params.alpha;

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         compute_At_Y_Iso(sigma, bundleStruct, rotations, 1.0, Ys, -1.0, Zs, rhs);

         chol.solve(rhs, X_new);

         for (int i = 1; i < N; ++i) copyVectorSlice(X_new, TVAR(i, 0), 3, translations[i], 0);
         for (int j = 0; j < M; ++j) copyVectorSlice(X_new, XVAR(j, 0), 3, Xs[j], 0);
         for (int k = 0; k < K; ++k) rhos[k] = X_new[RVAR(k)];

         if (iter == 0) savedXs = Xs;

         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs, rhos);
            double const r  = rhos[k];

            double const x1 = XX[0];
            double const x2 = XX[1];
            double const y  = sigma*XX[2];

            double xx1 = x1 + Zs[k][0];
            double xx2 = x2 + Zs[k][1];
            double yy  = y  + Zs[k][2];

            prox_g2(gamma, xx1, xx2, yy);

            Ys[k][0] = xx1; Ys[k][1] = xx2; Ys[k][2] = yy;

            Zs[k][0] += x1 - xx1; Zs[k][1] += x2 - xx2; Zs[k][2] += y - yy;

            double const z = XX[2];

            double zz = z + Zs[k][3];
            double rr = r + Zs[k][4];

            //prox_h1(gamma, zz, rr);
            zz = std::max(1.0, zz);
            rr -= gamma;
            rr = std::max(0.0, rr);

            Ys[k][3] = zz;
            Ys[k][4] = rr;

            Zs[k][3] += z - zz;
            Zs[k][4] += r - rr;
         } // end for (k)

         double const eps = 1e-4;

         if (((iter+1) % params.checkFrequency) == 0)
         {
            E_cur = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs, rhos);

               double const c = sigma*XX[2];
               double e1 = XX[0];
               double e2 = XX[1];
               E_cur += std::max(0.0, sqrt(e1*e1 + e2*e2) - c);

               double rho = rhos[k], XX3 = XX[2] - rho;
               if (XX3 < 1.0) rho = std::max(rho, 1.0 - XX3);
               E_cur += rho;
            } // end for (k)
         } // end if

         if (params.verbose && (((iter+1) % params.reportFrequency) == 0 || iter == params.nIterations-1 || iter == 9))
         {
            cout << "iter = " << iter << endl;

            int    nFeasible = 0;
            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

               if (XX3 > 1.0 - eps) ++nFeasible;
            } // end for (k)

            double E_primal = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs);

               double const c = sigma*XX[2];
               double e1 = XX[0];
               double e2 = XX[1];
               E_primal += std::max(0.0, sqrt(e1*e1 + e2*e2) - c);

               double rho = rhos[k], XX3 = XX[2] - rho;
               if (XX3 < 1.0) rho = std::max(rho, 1.0 - XX3);
               E_primal += rho;
            } // end for (k)

            cout << "E_cur = " << E_cur << ", E_saved = " << E_saved << endl;
            cout << "nFeasible = " << nFeasible << "/" << K << endl;
#if 0
            cout << "Current structure distance = " << computeStructureDistance(savedXs, Xs) << endl;
#else
            cout << "Relative energy change = " << fabs(E_cur - E_saved) / std::min(E_cur, E_saved) << endl;
#endif

            double E_dual = 0;
            vector<Vector3d> mu_Ts(N);
            vector<Vector3d> mu_Xs(M);
            vector<double>   mu_rhos(K);
            {
               compute_At_Y_Iso(sigma, bundleStruct, rotations, 1.0/gamma, Zs, 0.0, Zs, rhs);
               chol.solve(rhs, mu);

               for (int i = 1; i < N; ++i) copyVectorSlice(mu, TVAR(i, 0), 3, mu_Ts[i], 0);
               for (int j = 0; j < M; ++j) copyVectorSlice(mu, XVAR(j, 0), 3, mu_Xs[j], 0);
               makeZeroVector(mu_Ts[0]);
               for (int k = 0; k < K; ++k) mu_rhos[k] = mu[RVAR(k)];
            }

            vector<Vector5d> PPs(K, makeVector5(0.0, 0.0, 0.0, 0.0, 0.0));
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, mu_Ts, mu_Xs, mu_rhos);

               double const x1 = XX[0];
               double const x2 = XX[1];
               double const y  = sigma*XX[2];
               double const z  = XX[2];
               double const r  = mu_rhos[k];

               // Note: the dual vars are really z/gamma (see the A-L formulation)
               PPs[k][0] = Zs[k][0] / gamma - x1;
               PPs[k][1] = Zs[k][1] / gamma - x2;
               PPs[k][2] = Zs[k][2] / gamma - y;
               PPs[k][3] = Zs[k][3] / gamma - z;
               PPs[k][4] = Zs[k][4] / gamma - r;
            } // end for (k)

            compute_At_Y_Iso(sigma, bundleStruct, rotations, 1.0, PPs, 0.0, PPs, rhs);
            cout << "norm(At*lambda) = " << norm_L2(rhs) << endl;

            for (int k = 0; k < K; ++k)
            {
               double const x1 = PPs[k][0], x2 = PPs[k][1], y = PPs[k][2];
               double const z  = PPs[k][3], r  = PPs[k][4];

               double accum = 0;
               //accum += std::min(0.0, z);
               accum += z;

               if (z > eps) accum += 1e30;
               if (r > 1+eps) accum += 1e30;

               // |x| <= 1, y <= -|x|
               double const normX = sqrt(x1*x1 + x2*x2);

               if (normX > 1.0 + eps) accum += 1e30;
               if (normX > -y + eps)  accum += 1e30;

               E_dual -= accum;
            } // end for (k)

            E_primal_best = std::min(E_primal_best, E_primal);
            E_dual_best   = std::max(E_dual_best, E_dual);

            //cout << "E_primal = "  << E_primal << ", E_dual = " << E_dual << ", duality gap = " << (E_primal - E_dual)/E_primal * 100 << "%" << endl;
            cout << "E_primal = "  << E_primal_best << ", E_dual = " << E_dual_best << ", duality gap = " << (E_primal_best - E_dual_best)/E_primal_best * 100 << "%" << endl;

            // cout << "Ts = [ ";
            // for (int i = 0; i < N; ++i) cout << translations[i][0] << " " << translations[i][1] << " " << translations[i][2] << " / ";
            // cout << "]" << endl;

            cout << "=== " << (iter+1) << " " << E_primal_best << " " << E_dual_best << endl;
         } // end if

         if (((iter+1) % params.checkFrequency) == 0)
         {
#if 0
            if (computeStructureDistance(savedXs, Xs) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            savedXs = Xs;
#else
            if (E_cur < 1e30 && fabs(E_cur - E_saved) / std::min(E_cur, E_saved) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            E_saved = E_cur;
#endif
         }
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }

#undef TVAR
#undef XVAR
#undef RVAR
   } // end computeConsistentTranslationsRelaxedConic_Iso_SDMM()

   void
   computeConsistentTranslationsRelaxedConic_Huber_SDMM(float const sigma,
                                                        std::vector<Matrix3x3d> const& rotations,
                                                        std::vector<Vector3d>& translations,
                                                        std::vector<TriangulatedPoint>& sparseModel,
                                                        MultiViewInitializationParams_BOS const& params)
   {
#define TVAR(i, el) (3*((i)-1) + (el))
#define XVAR(j, el) (3*(N-1) + 3*(j) + (el))
#define RVAR(k) (3*(N-1) + 3*M + (k))

      int const N = rotations.size();
      int const M = sparseModel.size();

      translations.resize(N);
      makeZeroVector(translations[0]);

      BundlePointStructure bundleStruct(sparseModel, 2);
      int const K = bundleStruct.measurements.size();

      int const nVars = 3*(N-1) + 3*M + K; // First T is 0

      vector<int> const& correspondingView  = bundleStruct.correspondingView;
      vector<int> const& correspondingPoint = bundleStruct.correspondingPoint;

      vector<pair<int, int> > nzL;
      vector<double>          valsL;
      nzL.reserve(22*K);
      valsL.reserve(22*K);

      int row = 0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];

         Matrix3x3d const& R  = rotations[i];
         Vector2d   const& m  = bundleStruct.measurements[k];
         float      const  u1 = m[0];
         float      const  u2 = m[1];

         if (i > 0)
         {
            nzL.push_back(make_pair(row, TVAR(i, 0))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u1);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 1))); valsL.push_back(-1.0);
            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(u2);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(sigma);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, TVAR(i, 2))); valsL.push_back(1.0);
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;

            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;
         }
         else
         {
            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u1*R[2][0] - R[0][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u1*R[2][1] - R[0][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u1*R[2][2] - R[0][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(u2*R[2][0] - R[1][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(u2*R[2][1] - R[1][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(u2*R[2][2] - R[1][2]);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(sigma*R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(sigma*R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(sigma*R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(sigma*1.0);
            ++row;

            nzL.push_back(make_pair(row, XVAR(j, 0))); valsL.push_back(R[2][0]);
            nzL.push_back(make_pair(row, XVAR(j, 1))); valsL.push_back(R[2][1]);
            nzL.push_back(make_pair(row, XVAR(j, 2))); valsL.push_back(R[2][2]);
            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;

            nzL.push_back(make_pair(row, RVAR(k)));    valsL.push_back(1.0);
            ++row;
         } // end if
      } // end for (k)

      CCS_Matrix<double> L(row, nVars, nzL, valsL);
      CCS_Matrix<double> Q;

      multiply_At_A_SparseSparse(L, Q);

      SparseCholesky chol(Q, params.verbose);
      bool status = chol.setAtA(Q);
      cout << "Cholesky status = " << (status ? "good" : "bad") << endl;

      vector<Vector3d> Xs(M);
      vector<double>   rhos(K, 0.0);
      vector<Vector6d> Ys(K, makeVector6(0.0, double(sigma), 0.0, double(sigma), 0.0, 0.0));
      vector<Vector6d> Zs(K, makeVector6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

      vector<Vector3d> savedXs(M);
      double E_saved = 1e30, E_cur = 0;
      double E_primal_best = 1e40, E_dual_best = -1e40;

      Vector<double> rhs(nVars), X_new(nVars), mu(nVars);;

      double gamma = params.alpha;

      for (int iter = 0; iter < params.nIterations; ++iter)
      {
         compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, Ys, -1.0, Zs, rhs);

         chol.solve(rhs, X_new);

         for (int i = 1; i < N; ++i) copyVectorSlice(X_new, TVAR(i, 0), 3, translations[i], 0);
         for (int j = 0; j < M; ++j) copyVectorSlice(X_new, XVAR(j, 0), 3, Xs[j], 0);
         for (int k = 0; k < K; ++k) rhos[k] = X_new[RVAR(k)];

         if (iter == 0) savedXs = Xs;

         for (int k = 0; k < K; ++k)
         {
            Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs, rhos);
            double const r  = rhos[k];

            double const x1 = XX[0];
            double const y1 = sigma*XX[2];
            double const x2 = XX[1];
            double const y2 = sigma*XX[2];

            double xx1 = x1 + Zs[k][0];
            double yy1 = y1 + Zs[k][1];
            double xx2 = x2 + Zs[k][2];
            double yy2 = y2 + Zs[k][3];

            prox_H(params.alpha, xx1, yy1);
            prox_H(params.alpha, xx2, yy2);

            Ys[k][0] = xx1; Ys[k][1] = yy1;
            Ys[k][2] = xx2; Ys[k][3] = yy2;

            Zs[k][0] += x1 - xx1; Zs[k][1] += y1 - yy1;
            Zs[k][2] += x2 - xx2; Zs[k][3] += y2 - yy2;

            double const z = XX[2];

            double zz = z + Zs[k][4];
            double rr = r + Zs[k][5];

            //prox_h1(gamma, zz, rr);
            zz = std::max(1.0, zz);
            rr -= gamma;
            rr = std::max(0.0, rr);

            Ys[k][4] = zz;
            Ys[k][5] = rr;

            Zs[k][4] += z - zz;
            Zs[k][5] += r - rr;
         } // end for (k)

         double const eps = 1e-4;

         if (((iter+1) % params.checkFrequency) == 0)
         {
            E_cur = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs, rhos);

               double const c = sigma*XX[2];
               double e1 = fabs(XX[0]);
               double e2 = fabs(XX[1]);

               // E_cur += (e1 > c) ? e1 : (e1*e1/2/c + c/2);
               // E_cur += (e2 > c) ? e2 : (e2*e2/2/c + c/2);
               E_cur += (e1 > c) ? (e1-c/2) : (e1*e1/2/c);
               E_cur += (e2 > c) ? (e2-c/2) : (e2*e2/2/c);

               double rho = rhos[k], XX3 = XX[2] - rho;
               if (XX3 < 1.0) rho = std::max(rho, 1.0 - XX3);
               E_cur += rho;
            } // end for (k)
         } // end if

         if (params.verbose && (((iter+1) % params.reportFrequency) == 0 || iter == params.nIterations-1 || iter == 9))
         {
            cout << "iter = " << iter << endl;

            int    nFeasible = 0;
            for (int k = 0; k < K; ++k)
            {
               int const i = correspondingView[k];
               int const j = correspondingPoint[k];
               Matrix3x3d const& R = rotations[i];
               Vector3d   const& T = translations[i];
               Vector3d   const& X = Xs[j];
               double const XX3 = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];

               if (XX3 > 1.0 - eps) ++nFeasible;
            } // end for (k)

            double E_primal = 0;
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, translations, Xs, rhos);

               double const c = sigma*XX[2];
               double e1 = fabs(XX[0]);
               double e2 = fabs(XX[1]);

               // E_primal += (e1 > c) ? e1 : (e1*e1/2/c + c/2);
               // E_primal += (e2 > c) ? e2 : (e2*e2/2/c + c/2);
               E_primal += (e1 > c) ? (e1-c/2) : (e1*e1/2/c);
               E_primal += (e2 > c) ? (e2-c/2) : (e2*e2/2/c);

               double rho = rhos[k], XX3 = XX[2] - rho;
               if (XX3 < 1.0) rho = std::max(rho, 1.0 - XX3);
               E_primal += rho;
            } // end for (k)

            cout << "E_cur = " << E_cur << ", E_saved = " << E_saved << endl;
            cout << "nFeasible = " << nFeasible << "/" << K << endl;
#if 0
            cout << "Current structure distance = " << computeStructureDistance(savedXs, Xs) << endl;
#else
            cout << "Relative energy change = " << fabs(E_cur - E_saved) / std::min(E_cur, E_saved) << endl;
#endif

            double E_dual = 0;
#if 0
            vector<Vector3d> mu_Ts(N);
            vector<Vector3d> mu_Xs(M);
            vector<double>   mu_rhos(K);
            {
               compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0/gamma, Zs, 0.0, Zs, rhs);
               chol.solve(rhs, mu);

               for (int i = 1; i < N; ++i) copyVectorSlice(mu, TVAR(i, 0), 3, mu_Ts[i], 0);
               for (int j = 0; j < M; ++j) copyVectorSlice(mu, XVAR(j, 0), 3, mu_Xs[j], 0);
               makeZeroVector(mu_Ts[0]);
               for (int k = 0; k < K; ++k) mu_rhos[k] = mu[RVAR(k)];
            }

            vector<Vector6d> PPs(K, makeVector6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
            for (int k = 0; k < K; ++k)
            {
               Vector3d const XX = transformIntoMeasurementCone(k, sigma, bundleStruct, rotations, mu_Ts, mu_Xs, mu_rhos);

               double const x1 = XX[0];
               double const y1 = sigma*XX[2];
               double const x2 = XX[1];
               double const y2 = sigma*XX[2];
               double const z  = XX[2];
               double const r  = mu_rhos[k];

               // Note: the dual vars are really z/gamma (see the A-L formulation)
               PPs[k][0] = Zs[k][0] / gamma - x1;
               PPs[k][1] = Zs[k][1] / gamma - y1;
               PPs[k][2] = Zs[k][2] / gamma - x2;
               PPs[k][3] = Zs[k][3] / gamma - y2;
               PPs[k][4] = Zs[k][4] / gamma - z;
               PPs[k][5] = Zs[k][5] / gamma - r;
            } // end for (k)

            compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, PPs, 0.0, PPs, rhs);
            cout << "norm(At*lambda) = " << norm_L2(rhs) << endl;

            for (int k = 0; k < K; ++k)
            {
               double const x1 = PPs[k][0], y1 = PPs[k][1];
               double const x2 = PPs[k][2], y2 = PPs[k][3];
               double const z  = PPs[k][4], r  = PPs[k][5];

               double accum = 0;
               //accum += std::min(0.0, z);
               accum += z;

               if (z > eps) accum += 1e30;
               if (r > 1+eps) accum += 1e30;

               // y <= -0.5*x^2 (i.e. x^2 <= -2*y), |x| <= 1

               if (fabs(x1) > 1.0 + eps) accum += 1e30;
               if (x1*x1 > -2*y1 + eps)  accum += 1e30;
               if (fabs(x2) > 1.0 + eps) accum += 1e30;
               if (x2*x2 > -2*y2 + eps)  accum += 1e30;

               E_dual -= accum;
            } // end for (k)
#else
            vector<Vector6d> PPs(K, makeVector6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
            for (int k = 0; k < K; ++k)
            {
               // Note: the dual vars are really z/gamma (see the A-L formulation)
               double x1 = Zs[k][0] / gamma;
               double y1 = Zs[k][1] / gamma;
               double x2 = Zs[k][2] / gamma;
               double y2 = Zs[k][3] / gamma;
               double z  = Zs[k][4] / gamma;
               double r  = Zs[k][5] / gamma;

               projectZs_Huber(x1, y1);
               projectZs_Huber(x2, y2);
               z = std::min(0.0, z);
               r = std::min(1.0, r);

               PPs[k][0] = x1; PPs[k][1] = y1;
               PPs[k][2] = x2; PPs[k][3] = y2;
               PPs[k][4] = z;  PPs[k][5] = r;

               double accum = 0;
               accum += z;

               E_dual -= accum;
            } // end for (k)
            compute_At_Y_Aniso(sigma, bundleStruct, rotations, 1.0, PPs, 0.0, PPs, rhs);
            cout << "||Q||_1 = " << norm_L1(rhs) << endl;

            for (int i = 1; i < N; ++i)
            {
               E_dual -= fabs(translations[i][0] * rhs[TVAR(i, 0)]);
               E_dual -= fabs(translations[i][1] * rhs[TVAR(i, 1)]);
               E_dual -= fabs(translations[i][2] * rhs[TVAR(i, 2)]);
            }
            for (int j = 0; j < M; ++j)
            {
               E_dual -= fabs(Xs[j][0] * rhs[XVAR(j, 0)]);
               E_dual -= fabs(Xs[j][1] * rhs[XVAR(j, 1)]);
               E_dual -= fabs(Xs[j][2] * rhs[XVAR(j, 2)]);
            }
#endif

            E_primal_best = std::min(E_primal_best, E_primal);
            E_dual_best   = std::max(E_dual_best, E_dual);

            //cout << "E_primal = "  << E_primal << ", E_dual = " << E_dual << ", duality gap = " << (E_primal - E_dual)/E_primal * 100 << "%" << endl;
            cout << "E_primal = "  << E_primal_best << ", E_dual = " << E_dual_best << ", duality gap = " << (E_primal_best - E_dual_best)/E_primal_best * 100 << "%" << endl;

            // cout << "Ts = [ ";
            // for (int i = 0; i < N; ++i) cout << translations[i][0] << " " << translations[i][1] << " " << translations[i][2] << " / ";
            // cout << "]" << endl;

            cout << "=== " << (iter+1) << " " << E_primal_best << " " << E_dual_best << endl;
         } // end if

         if (((iter+1) % params.checkFrequency) == 0)
         {
#if 0
            if (computeStructureDistance(savedXs, Xs) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            savedXs = Xs;
#else
            if (E_cur < 1e30 && fabs(E_cur - E_saved) / std::min(E_cur, E_saved) < params.stoppingThreshold)
            {
               if (params.verbose)
                  cout << "Converged at iteration " << iter << endl;
               break;
            }
            E_saved = E_cur;
#endif
         }
      } // end for (iter)

      for (int j = 0; j < M; ++j)
      {
         TriangulatedPoint& X = sparseModel[j];
         X.pos = Xs[j];
      }

#undef TVAR
#undef XVAR
#undef RVAR
   } // end computeConsistentTranslationsRelaxedConic_Huber_SDMM()

} // end namespace V3D

#endif //defined(V3DLIB_ENABLE_SUITESPARSE)
