#include "Math/v3d_linear.h"
#include "Math/v3d_optimization.h"
#include "Math/v3d_mathutilities.h"
#include "Geometry/v3d_poseutilities.h"

#define X2_POS  0
#define XY_POS  1
#define XZ_POS  2
#define X_POS   3
#define Y2_POS  6
#define YZ_POS  9
#define Y_POS   10
#define Z2_POS  17
#define Z_POS   18
#define ONE_POS 19

namespace
{

   using namespace V3D;

   inline void
   generateConstraintEG1(double p1, double p2, double q1, double q2,
                         double x1, double x2, double x3, double y1, double y2, double y3,
                         double b1, double b2, double b3, double * dst)
   {
      double const one = 1.0;

      dst[Z2_POS] = p1*x2-p2*x1-b1*p2+b2*p1;
      dst[YZ_POS] = -p1*x3+p1*q2*x2+(one-p2*q2)*x1+(b2*p1-b1*p2)*q2-b3*p1+b1;
      dst[XZ_POS] = p2*x3+(p1*q1-one)*x2-p2*q1*x1+(b2*p1-b1*p2)*q1+b3*p2-b2;
      dst[Y2_POS] = -p1*q2*x3+q2*x1+(b1-b3*p1)*q2;
      dst[XY_POS] = (p2*q2-p1*q1)*x3-q2*x2+q1*x1+(b3*p2-b2)*q2+(b1-b3*p1)*q1;
      dst[X2_POS] = p2*q1*x3-q1*x2+(b3*p2-b2)*q1;
      dst[Z_POS]  = (+(-p2*q2-p1*q1)*y3+p2*y2+p1*y1+(p2*q2+p1*q1)*x3-q2*x2-q1*x1-b2*q2
                     -b1*q1+b2*p2+b1*p1);
      dst[Y_POS]  = (+q2*y3
                     +(-p1*q1-one)*y2+p1*q2*y1-p2*x3+(p1*q1+one)*x2-p2*q1*x1+(b1*p1+b3)*q2
                     -b1*p2*q1-b3*p2);
      dst[X_POS]  = (+q1*y3
                     +p2*q1*y2+(-p2*q2-one)*y1-p1*x3-p1*q2*x2+(p2*q2+one)*x1-b2*p1*q2
                     +(b2*p2+b3)*q1-b3*p1);
      dst[ONE_POS] = (+p1*q2*y3-p2*q1*y3+q1*y2
                      -p1*y2-q2*y1+p2*y1-p1*q2*x3+p2*q1*x3-q1*x2+p1*x2+q2*x1-p2*x1);
   } // end generateConstraintEG1()

   inline void
   generateConstraintEG0(double p1, double p2, double q1, double q2,
                         double x1, double x2, double x3, double y1, double y2, double y3,
                         double * dst)
   {
      generateConstraintEG1(p1, p2, q1, q2, x1, x2, x3, y1, y2, y3, 0, 0, 0, dst);
   } // end generateConstraintEG0()

   inline void
   generateConstraintEG1_MRP(double p1, double p2, double q1, double q2,
                             double x1, double x2, double x3, double y1, double y2, double y3,
                             double b1, double b2, double b3, double * dst)
   {
      double const four = 4.0;
      double const f16 = 16.0;

      dst[Z2_POS] = f16*p1*x2-f16*p2*x1-f16*b1*p2+f16*b2*p1;
      dst[YZ_POS] = (-f16*p1*x3+f16*p1*q2*x2+(f16-f16*p2*q2)*x1+(f16*b2*p1-f16*b1*p2)*q2-f16*b3*p1
                     +f16*b1);
      dst[XZ_POS] = (f16*p2*x3+(f16*p1*q1-f16)*x2-f16*p2*q1*x1+(f16*b2*p1-f16*b1*p2)*q1+f16*b3*p2
                     -f16*b2);
      dst[Y2_POS] = -f16*p1*q2*x3+f16*q2*x1+(f16*b1-f16*b3*p1)*q2;
      dst[XY_POS] = ((f16*p2*q2-f16*p1*q1)*x3-f16*q2*x2+f16*q1*x1+(f16*b3*p2-f16*b2)*q2
                     +(f16*b1-f16*b3*p1)*q1);
      dst[X2_POS] = f16*p2*q1*x3-f16*q1*x2+(f16*b3*p2-f16*b2)*q1;
      dst[Z_POS]  = (+(-four*p2*q2-four*p1*q1)*y3+four*p2*y2+four*p1*y1+(four*p2*q2+four*p1*q1)*x3-four*q2*x2
                     -four*q1*x1-four*b2*q2-four*b1*q1+four*b2*p2+four*b1*p1);
      dst[Y_POS]  = (+four*q2*y3+(-four*p1*q1-four)*y2+four*p1*q2*y1-four*p2*x3+(four*p1*q1+four)*x2
                     -four*p2*q1*x1+(four*b1*p1+four*b3)*q2-four*b1*p2*q1-four*b3*p2);
      dst[X_POS]  = (+four*q1*y3+four*p2*q1*y2+(-four*p2*q2-four)*y1-four*p1*x3-four*p1*q2*x2
                     +(four*p2*q2+four)*x1-four*b2*p1*q2+(four*b2*p2+four*b3)*q1-four*b3*p1);
      dst[ONE_POS] = (+p1*q2*y3-p2*q1*y3+q1*y2-p1*y2-q2*y1+p2*y1
                      -p1*q2*x3+p2*q1*x3-q1*x2+p1*x2+q2*x1-p2*x1);
   } // end generateConstraintEG1_MRP()

   inline void
   generateConstraintEG0_MRP(double p1, double p2, double q1, double q2,
                             double x1, double x2, double x3, double y1, double y2, double y3,
                             double * dst)
   {
      generateConstraintEG1_MRP(p1, p2, q1, q2, x1, x2, x3, y1, y2, y3, 0, 0, 0, dst);
   } // end generateConstraintEG0_MRP()

   inline void
   generateConstraintEG1_MRP2(double p1, double p2, double q1, double q2,
                              double x1, double x2, double x3, double y1, double y2, double y3,
                              double b1, double b2, double b3, double * dst)
   {
      double const F4 = 4;
      double const F8 = 8;
      double const F16 = 16;

      dst[Z2_POS] = ((F8*p2*q1-F8*p1*q2)*y3+F8*p1*y2-F8*p2*y1+(F8*p1*q2-F8*p2*q1)*x3+F8*q1*x2
                     -F8*q2*x1-F8*b1*q2+F8*b2*q1-F8*b1*p2+F8*b2*p1);
      dst[YZ_POS] = (-F8*q1*y3+F8*p2*q1*y2+(F8-F8*p2*q2)*y1-F8*p1*x3+F8*p1*q2*x2+(F8-F8*p2*q2)*x1
                     +(F8*b2*p1-F16*b1*p2)*q2+(F8*b2*p2-F8*b3)*q1-F8*b3*p1+F16*b1);
      dst[XZ_POS] = (F8*q2*y3+(F8*p1*q1-F8)*y2-F8*p1*q2*y1+F8*p2*x3+(F8*p1*q1-F8)*x2-F8*p2*q1*x1
                     +(F8*b3-F8*b1*p1)*q2+(F16*b2*p1-F8*b1*p2)*q1+F8*b3*p2-F16*b2);
      dst[Y2_POS] = (-F8*p1*q2*y3+(F8*p1-F8*q1)*y2+F8*q2*y1-F8*p2*q1*x3+(F8*q1-F8*p1)*x2+F8*p2*x1
                     +(F8*b1-F8*b3*p1)*q2-F8*b3*p2*q1+F8*b1*p2);
      dst[XY_POS] = ((F8*p2*q2-F8*p1*q1)*y3-F8*p2*y2+F8*p1*y1+(F8*p2*q2-F8*p1*q1)*x3-F8*q2*x2
                     +F8*q1*x1+(F16*b3*p2-F8*b2)*q2+(F8*b1-F16*b3*p1)*q1
                     -F8*b2*p2+F8*b1*p1);
      dst[X2_POS] = (F8*p2*q1*y3-F8*q1*y2+(F8*q2-F8*p2)*y1+F8*p1*q2*x3-F8*p1*x2+(F8*p2-F8*q2)*x1
                     +F8*b3*p1*q2+(F8*b3*p2-F8*b2)*q1-F8*b2*p1);
      dst[Z_POS]  = (+(F4*p2*q2+F4*p1*q1)*y3-F4*p2*y2-F4*p1*y1+(-F4*p2*q2-F4*p1*q1)*x3+F4*q2*x2
                     +F4*q1*x1+F4*b2*q2+F4*b1*q1-F4*b2*p2-F4*b1*p1);
      dst[Y_POS]  = (-F4*q2*y3+(F4*p1*q1+F4)*y2-F4*p1*q2*y1+F4*p2*x3+(-F4*p1*q1-F4)*x2
                     +F4*p2*q1*x1+(-F4*b1*p1-F4*b3)*q2+F4*b1*p2*q1+F4*b3*p2);
      dst[X_POS]  = (-F4*q1*y3-F4*p2*q1*y2+(F4*p2*q2+F4)*y1+F4*p1*x3+F4*p1*q2*x2
                     +(-F4*p2*q2-F4)*x1+F4*b2*p1*q2+(-F4*b2*p2-F4*b3)*q1+F4*b3*p1);
      dst[ONE_POS] = (+p1*q2*y3-p2*q1*y3+q1*y2-p1*y2-q2*y1+p2*y1
                      -p1*q2*x3+p2*q1*x3-q1*x2+p1*x2+q2*x1-p2*x1);
   } // end generateConstraintEG1_MRP2()

   inline void
   generateConstraintEG0_MRP2(double p1, double p2, double q1, double q2,
                              double x1, double x2, double x3, double y1, double y2, double y3,
                              double * dst)
   {
      generateConstraintEG1_MRP2(p1, p2, q1, q2, x1, x2, x3, y1, y2, y3, 0, 0, 0, dst);
   } // end generateConstraintEG0_MRP2()

   struct ClassicalRodrigues_2_1Plus1PointOptimizer : public SimpleLevenbergOptimizer
   {
         ClassicalRodrigues_2_1Plus1PointOptimizer(Vector2d const& pL1, Vector2d const& pL2, Vector2d const& pR,
                                                   Vector2d const& qL1, Vector2d const& qL2, Vector2d const& qR,
                                                   Vector3d const& X, Vector3d const& Y, Vector3d const& B,
                                                   Matrix3x3d const& R0)
            : SimpleLevenbergOptimizer(3, 3),
              _pL1(homogenizeVector(pL1)), _pL2(homogenizeVector(pL2)), _pR(homogenizeVector(pR)),
              _qL1(homogenizeVector(qL1)), _qL2(homogenizeVector(qL2)), _qR(homogenizeVector(qR)),
              _X(X), _Y(Y), _B(B)
         {
            copyMatrix(R0, _R);
            makeZeroVector(currentParameters);
         }

         virtual void evalFunction(Vector<double>& res)
         {
            Matrix3x3d XX, YY;
            makeCrossProductMatrix(_X, XX);
            makeCrossProductMatrix(_Y, YY);

            res[0] = innerProduct(_qL1, YY*_R*_pL1 - _R*XX*_pL1);
            res[1] = innerProduct(_qL2, YY*_R*_pL2 - _R*XX*_pL2);

            makeCrossProductMatrix(_X+_B, XX);
            makeCrossProductMatrix(_Y+_B, YY);
            res[2] = innerProduct(_qR, YY*_R*_pR - _R*XX*_pR);
         }

         virtual void fillJacobian(Matrix<double>& J)
         {
            Matrix3x3d XX, YY;
            makeCrossProductMatrix(_X, XX);
            makeCrossProductMatrix(_Y, YY);

            Vector3d d_dw;
            d_dw = _qL1 * (crossProductMatrix(_R*XX*_pL1) - (YY * crossProductMatrix(_R*_pL1)));
            J.setRowSlice(0, 0, 3, d_dw);

            d_dw = _qL2 * (crossProductMatrix(_R*XX*_pL2) - (YY * crossProductMatrix(_R*_pL2)));
            J.setRowSlice(1, 0, 3, d_dw);

            makeCrossProductMatrix(_X+_B, XX);
            makeCrossProductMatrix(_Y+_B, YY);
            d_dw = _qR * (crossProductMatrix(_R*XX*_pR) - (YY * crossProductMatrix(_R*_pR)));
            J.setRowSlice(2, 0, 3, d_dw);
         }

         virtual void updateCurrentParameters(Vector<double> const& delta)
         {
            // Create incremental rotation using Rodriguez formula.
            Matrix3x3d dR;
            createRotationMatrixRodrigues(delta, dR);
            _R = dR * _R;
         }

         virtual void saveCurrentParameters()
         {
            _Rsaved = _R;
         }

         virtual void restoreCurrentParameters()
         {
            _R = _Rsaved;
         }

         Vector3d const  _pL1, _pL2, _pR, _qL1, _qL2, _qR;
         Vector3d const& _X, _Y, _B;

         Matrix3x3d _R, _Rsaved;
   }; // end struct ClassicalRodrigues_2_1Plus1PointOptimizer

   struct ModifiedRodrigues_2_1Plus1PointOptimizer : public SimpleLevenbergOptimizer
   {
         ModifiedRodrigues_2_1Plus1PointOptimizer(Vector2d const& pL1, Vector2d const& pL2, Vector2d const& pR,
                                                  Vector2d const& qL1, Vector2d const& qL2, Vector2d const& qR,
                                                  Vector3d const& X, Vector3d const& Y, Vector3d const& B,
                                                  Matrix3x3d const& R0)
            : SimpleLevenbergOptimizer(3, 3),
              _pL1(homogenizeVector(pL1)), _pL2(homogenizeVector(pL2)), _pR(homogenizeVector(pR)),
              _qL1(homogenizeVector(qL1)), _qL2(homogenizeVector(qL2)), _qR(homogenizeVector(qR)),
              _X(X), _Y(Y), _B(B)
         {
            copyMatrix(R0, _R);
            makeZeroVector(currentParameters);
         }

         virtual void evalFunction(Vector<double>& res)
         {
            Matrix3x3d XX, YY;
            makeCrossProductMatrix(_X, XX);
            makeCrossProductMatrix(_Y, YY);

            res[0] = innerProduct(_qL1, YY*_R*_pL1 - _R*XX*_pL1);
            res[1] = innerProduct(_qL2, YY*_R*_pL2 - _R*XX*_pL2);

            makeCrossProductMatrix(_X+_B, XX);
            makeCrossProductMatrix(_Y+_B, YY);
            res[2] = innerProduct(_qR, YY*_R*_pR - _R*XX*_pR);
         }

         virtual void fillJacobian(Matrix<double>& J)
         {
            Matrix3x3d XX, YY;
            makeCrossProductMatrix(_X, XX);
            makeCrossProductMatrix(_Y, YY);

            Vector3d d_dw;
            d_dw = -4.0 * _qL1 * (crossProductMatrix(_R*XX*_pL1) - (YY * crossProductMatrix(_R*_pL1)));
            J.setRowSlice(0, 0, 3, d_dw);

            d_dw = -4.0 * _qL2 * (crossProductMatrix(_R*XX*_pL2) - (YY * crossProductMatrix(_R*_pL2)));
            J.setRowSlice(1, 0, 3, d_dw);

            makeCrossProductMatrix(_X+_B, XX);
            makeCrossProductMatrix(_Y+_B, YY);
            d_dw = -4.0 * _qR * (crossProductMatrix(_R*XX*_pR) - (YY * crossProductMatrix(_R*_pR)));
            J.setRowSlice(2, 0, 3, d_dw);
         }

         virtual void updateCurrentParameters(Vector<double> const& delta)
         {
            // Create incremental rotation using Rodriguez formula.
            Matrix3x3d dR;
            createRotationMatrixFromModifiedRodrigues(delta, dR);
            _R = dR * _R;
         }

         virtual void saveCurrentParameters()
         {
            _Rsaved = _R;
         }

         virtual void restoreCurrentParameters()
         {
            _R = _Rsaved;
         }

         Vector3d const  _pL1, _pL2, _pR, _qL1, _qL2, _qR;
         Vector3d const& _X, _Y, _B;

         Matrix3x3d _R, _Rsaved;
   }; // end struct ModifiedRodrigues_2_1Plus1PointOptimizer

   struct Quaternion_2_1Plus1PointOptimizer : public SimpleLevenbergOptimizer
   {
         Quaternion_2_1Plus1PointOptimizer(Vector2d const& pL1, Vector2d const& pL2, Vector2d const& pR,
                                           Vector2d const& qL1, Vector2d const& qL2, Vector2d const& qR,
                                           Vector3d const& X, Vector3d const& Y, Vector3d const& B,
                                           Matrix3x3d const& R0)
            : SimpleLevenbergOptimizer(3, 4),
              _pL1(homogenizeVector(pL1)), _pL2(homogenizeVector(pL2)), _pR(homogenizeVector(pR)),
              _qL1(homogenizeVector(qL1)), _qL2(homogenizeVector(qL2)), _qR(homogenizeVector(qR)),
              _X(X), _Y(Y), _B(B)
         {
            createQuaternionFromRotationMatrix(R0, _Q);
            makeZeroVector(currentParameters);
         }

         virtual void evalFunction(Vector<double>& res)
         {
            Matrix3x3d R;
            createRotationMatrixFromQuaternion(_Q, R);

            Matrix3x3d XX, YY;
            makeCrossProductMatrix(_X, XX);
            makeCrossProductMatrix(_Y, YY);

            res[0] = innerProduct(_qL1, YY*R*_pL1 - R*XX*_pL1);
            res[1] = innerProduct(_qL2, YY*R*_pL2 - R*XX*_pL2);

            makeCrossProductMatrix(_X+_B, XX);
            makeCrossProductMatrix(_Y+_B, YY);
            res[2] = innerProduct(_qR, YY*R*_pR - R*XX*_pR);
         }

         void fillDerivativeRow(double p1, double p2, double q1, double q2,
                                double X1, double X2, double X3, double Y1, double Y2, double Y3,
                                double J[4])
         {
            double const x = _Q[0];
            double const y = _Q[1];
            double const z = _Q[2];
            double const w = _Q[3];

            J[0] = (q2*(p2*(2*y*Y3-2*w*Y1+2*y*X3+2*w*X1)+p1*(2*x*Y3-2*z*Y1+2*x*X3-2*w*X2)
                        +2*z*Y3+2*x*Y1-2*y*X2-2*x*X1)
                    +q1*(p1*(-2*y*Y3+2*z*Y2-2*y*X3+2*z*X2)
                         +p2*(2*x*Y3+2*w*Y2+2*x*X3-2*z*X1)+2*w*Y3-2*x*Y2-2*x*X2+2*y*X1)
                    +p2*(-2*y*Y2-2*x*Y1+2*z*X3+2*x*X1)+p1*(-2*x*Y2+2*y*Y1-2*w*X3-2*x*X2)
                    -2*z*Y2-2*w*Y1-2*z*X2+2*w*X1);
            J[1] = (q1*(p2*(-2*y*Y3+2*z*Y2-2*y*X3-2*w*X1)
                        +p1*(-2*x*Y3-2*w*Y2-2*x*X3+2*w*X2)-2*z*Y3-2*y*Y2+2*y*X2+2*x*X1)
                    +q2*(p1*(-2*y*Y3+2*w*Y1-2*y*X3+2*z*X2)
                         +p2*(2*x*Y3-2*z*Y1+2*x*X3-2*z*X1)+2*w*Y3+2*y*Y1-2*x*X2+2*y*X1)
                    +p1*(2*y*Y2+2*x*Y1-2*z*X3-2*y*X2)+p2*(-2*x*Y2+2*y*Y1-2*w*X3+2*y*X1)
                    -2*w*Y2+2*z*Y1+2*w*X2+2*z*X1);
            J[2] = (q1*(p2*(2*z*Y3+2*y*Y2-2*z*X3-2*x*X1)+p1*(-2*w*Y3+2*x*Y2+2*w*X3+2*x*X2)
                        -2*y*Y3+2*z*Y2+2*z*X2-2*w*X1)
                    +q2*(p1*(-2*z*Y3-2*x*Y1+2*z*X3+2*y*X2)
                         +p2*(-2*w*Y3-2*y*Y1+2*w*X3-2*y*X1)+2*x*Y3-2*z*Y1-2*w*X2-2*z*X1)
                    +p1*(2*z*Y2+2*w*Y1-2*y*X3+2*z*X2)+p2*(2*w*Y2-2*z*Y1+2*x*X3-2*z*X1)
                    -2*x*Y2+2*y*Y1-2*x*X2+2*y*X1);
            J[3] = (q1*(p1*(-2*z*Y3-2*y*Y2+2*z*X3+2*y*X2)
                        +p2*(-2*w*Y3+2*x*Y2+2*w*X3-2*y*X1)+2*x*Y3+2*w*Y2-2*w*X2-2*z*X1)
                    +q2*(p2*(-2*z*Y3-2*x*Y1+2*z*X3+2*x*X1)
                         +p1*(2*w*Y3+2*y*Y1-2*w*X3-2*x*X2)+2*y*Y3-2*w*Y1-2*z*X2+2*w*X1)
                    +p2*(2*z*Y2+2*w*Y1-2*y*X3-2*w*X1)+p1*(-2*w*Y2+2*z*Y1-2*x*X3+2*w*X2)
                    -2*y*Y2-2*x*Y1+2*y*X2+2*x*X1);

            // Project onto tangential plane of unit sphere
            double ofs = J[0]*x + J[1]*y + J[2]*z + J[3]*w;
            J[0] -= ofs*x;
            J[1] -= ofs*y;
            J[2] -= ofs*z;
            J[3] -= ofs*w;
         } // end fillDerivativeRow

         virtual void fillJacobian(Matrix<double>& J)
         {
            this->fillDerivativeRow(_pL1[0], _pL1[1], _qL1[0], _qL1[1],
                                    _X[0], _X[1], _X[2], _Y[0], _Y[1], _Y[2], J[0]);

            this->fillDerivativeRow(_pL2[0], _pL2[1], _qL2[0], _qL2[1],
                                    _X[0], _X[1], _X[2], _Y[0], _Y[1], _Y[2], J[1]);

            Vector3d const X = _X + _B;
            Vector3d const Y = _Y + _B;
            this->fillDerivativeRow(_pR[0], _pR[1], _qR[0], _qR[1],
                                    X[0], X[1], X[2], Y[0], Y[1], Y[2], J[2]);
         }

         virtual void updateCurrentParameters(Vector<double> const& delta)
         {
            addVectors(_Q, delta, _Q);
            normalizeVector(_Q);
         }

         virtual void saveCurrentParameters()
         {
            _Qsaved = _Q;
         }

         virtual void restoreCurrentParameters()
         {
            _Q = _Qsaved;
         }

         Vector3d const  _pL1, _pL2, _pR, _qL1, _qL2, _qR;
         Vector3d const& _X, _Y, _B;

         Vector4d _Q, _Qsaved;
   }; // end struct Quaternion_2_1Plus1PointOptimizer

} // end namespace

namespace V3D
{

   template <typename Num>
   bool computeRelativePose_2_1Plus1Point(Vector2d const& pL1, Vector2d const& pL2, Vector2d const& pR,
                                          Vector2d const& qL1, Vector2d const& qL2, Vector2d const& qR,
                                          Vector3d const& X, Vector3d const& Y, Vector3d const& B,
                                          std::vector<Matrix3x3d>& Rs, std::vector<Vector3d>& Ts, int method)
   {
      switch (method)
      {
         case V3D_2_1P1P_METHOD_LINEARIZED:
         {
            Rs.clear();
            Ts.clear();

            Matrix<double> F(3, 20);
            makeZeroMatrix(F);

            generateConstraintEG0(pL1[0], pL1[1], qL1[0], qL1[1], X[0], X[1], X[2], Y[0], Y[1], Y[2], F[0]);
            generateConstraintEG0(pL2[0], pL2[1], qL2[0], qL2[1], X[0], X[1], X[2], Y[0], Y[1], Y[2], F[1]);
            generateConstraintEG1(pR[0],  pR[1],  qR[0],  qR[1],  X[0], X[1], X[2], Y[0], Y[1], Y[2], B[0], B[1], B[2], F[2]);

            vector<Num> zs;
            zs.reserve(8);

            Num G[12][20];
//             for (int i = 0; i < 12; ++i)
//                for (int j = 0; j < 20; ++j) G[i][j] = 0;

            // Solve for z:
            convertToRowEchelonMatrix(F);
            convertToReducedRowEchelonMatrix(F);

            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 20; ++j)
                  G[i][j] = F[i][j];

#include "v3d_211pt_quadratic_xyz_generated.h"

            Num coeffs[9];
            for (int i = 0; i <= 8; ++i) coeffs[i] = G[11][19-i];

//             for (int i = 0; i < 9; ++i) cout << coeffs[i] << " ";
//             cout << endl;

            RootFindingParameters<Num> rootParams;
            rootParams.maxBisectionIterations = 100;
            computeRealRootsOfPolynomial(8, coeffs, zs, rootParams);

            for (size_t i = 0; i < zs.size(); ++i)
            {
               double z = zs[i];
               double z2 = z*z;
               double z3 = z2*z;
               double z4 = z2*z2;
//          double z5 = z3*z2;
//          double z6 = z3*z3;
//          double z7 = z4*z3;

               //double y = -(+G[10][12]*z7+G[10][13]*z6+G[10][14]*z5+G[10][15]*z4+G[10][16]*z3+G[10][17]*z2+G[10][18]*z+G[10][19]*1);
               //double y = -(+G[9][13]*z6+G[9][14]*z5+G[9][15]*z4+G[9][16]*z3+G[9][17]*z2+G[9][18]*z+G[9][19]*1) / (+G[9][9]*z+G[9][10]);
               double y = -(+G[7][15]*z4+G[7][16]*z3+G[7][17]*z2+G[7][18]*z+G[7][19]*1) / (+G[7][7]*z3+G[7][8]*z2+G[7][9]*z+G[7][10]);

               double y2 = y*y;
               double y3 = y2*y;

               double x = -(+G[3][4]*y3+G[3][5]*y2*z+G[3][6]*y2+G[3][8]*y*z2+G[3][9]*y*z+G[3][10]*y+G[3][16]*z3+G[3][17]*z2+G[3][18]*z+G[3][19]*1);

               //if (x*x + y*y + z*z > 1) continue; // Skip everything with too large angle

               Vector3d om = makeVector3(x, y, z);

               Matrix3x3d R;
               createRotationMatrixRodrigues(om, R);
               Rs.push_back(R);
               Ts.push_back(Y - R*X);
            } // end for (i)
            break;
         }
         case V3D_2_1P1P_METHOD_LINEARIZED_MRP:
         {
            Rs.clear();
            Ts.clear();

            Matrix<double> F(3, 20);
            makeZeroMatrix(F);

            generateConstraintEG0_MRP(pL1[0], pL1[1], qL1[0], qL1[1], X[0], X[1], X[2], Y[0], Y[1], Y[2], F[0]);
            generateConstraintEG0_MRP(pL2[0], pL2[1], qL2[0], qL2[1], X[0], X[1], X[2], Y[0], Y[1], Y[2], F[1]);
            generateConstraintEG1_MRP(pR[0],  pR[1],  qR[0],  qR[1],  X[0], X[1], X[2], Y[0], Y[1], Y[2], B[0], B[1], B[2], F[2]);

            vector<Num> zs;
            zs.reserve(8);

            Num G[12][20];
//             for (int i = 0; i < 12; ++i)
//                for (int j = 0; j < 20; ++j) G[i][j] = 0;

            // Solve for z:
            convertToRowEchelonMatrix(F);
            convertToReducedRowEchelonMatrix(F);

            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 20; ++j)
                  G[i][j] = F[i][j];

#include "v3d_211pt_mrp_linearized_xyz_generated.h"

            Num coeffs[9];
            for (int i = 0; i <= 8; ++i) coeffs[i] = G[11][19-i];

//             for (int i = 0; i < 9; ++i) cout << coeffs[i] << " ";
//             cout << endl;

            RootFindingParameters<Num> rootParams;
            rootParams.maxBisectionIterations = 200;
            computeRealRootsOfPolynomial(8, coeffs, zs, rootParams);

            for (size_t i = 0; i < zs.size(); ++i)
            {
               double z = zs[i];
               double z2 = z*z;
               double z3 = z2*z;
               double z4 = z2*z2;
//          double z5 = z3*z2;
//          double z6 = z3*z3;
//          double z7 = z4*z3;

               //double y = -(+G[10][12]*z7+G[10][13]*z6+G[10][14]*z5+G[10][15]*z4+G[10][16]*z3+G[10][17]*z2+G[10][18]*z+G[10][19]*1);
               //double y = -(+G[9][13]*z6+G[9][14]*z5+G[9][15]*z4+G[9][16]*z3+G[9][17]*z2+G[9][18]*z+G[9][19]*1) / (+G[9][9]*z+G[9][10]);
               double y = -(+G[7][15]*z4+G[7][16]*z3+G[7][17]*z2+G[7][18]*z+G[7][19]*1) / (+G[7][7]*z3+G[7][8]*z2+G[7][9]*z+G[7][10]);

               double y2 = y*y;
               double y3 = y2*y;

               double x = -(+G[3][4]*y3+G[3][5]*y2*z+G[3][6]*y2+G[3][8]*y*z2+G[3][9]*y*z+G[3][10]*y+G[3][16]*z3+G[3][17]*z2+G[3][18]*z+G[3][19]*1);

               //if (x*x + y*y + z*z > 1) continue; // Skip everything with too large angle

               // Got the signs wrong, so the reported solutions are actually the negated ones.
               Vector3d sigma = makeVector3(-x, -y, -z);

               Matrix3x3d R;
               createRotationMatrixFromModifiedRodrigues(sigma, R);
               Rs.push_back(R);
               Ts.push_back(Y - R*X);
            } // end for (i)
            break;
         }
         case V3D_2_1P1P_METHOD_QUADRATIC_MRP:
         {
            Rs.clear();
            Ts.clear();

            Matrix<double> F(3, 20);
            makeZeroMatrix(F);

            generateConstraintEG0_MRP2(pL1[0], pL1[1], qL1[0], qL1[1], X[0], X[1], X[2], Y[0], Y[1], Y[2], F[0]);
            generateConstraintEG0_MRP2(pL2[0], pL2[1], qL2[0], qL2[1], X[0], X[1], X[2], Y[0], Y[1], Y[2], F[1]);
            generateConstraintEG1_MRP2(pR[0],  pR[1],  qR[0],  qR[1],  X[0], X[1], X[2], Y[0], Y[1], Y[2], B[0], B[1], B[2], F[2]);

            vector<Num> zs;
            zs.reserve(8);

            Num G[12][20];

            // Solve for z:
            convertToRowEchelonMatrix(F);
            convertToReducedRowEchelonMatrix(F);

            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 20; ++j)
                  G[i][j] = F[i][j];

#include "v3d_211pt_mrp_xyz_generated.h"

            Num coeffs[9];
            for (int i = 0; i <= 8; ++i) coeffs[i] = G[11][19-i];

            RootFindingParameters<Num> rootParams;
            rootParams.maxBisectionIterations = 200;
            computeRealRootsOfPolynomial(8, coeffs, zs, rootParams);

            for (size_t i = 0; i < zs.size(); ++i)
            {
               double z = zs[i];
               double z2 = z*z;
               double z3 = z2*z;
               double z4 = z2*z2;

               double y = -(+G[7][15]*z4+G[7][16]*z3+G[7][17]*z2+G[7][18]*z+G[7][19]*1)
                  / (+G[7][7]*z3+G[7][8]*z2+G[7][9]*z+G[7][10]);

               double y2 = y*y;
               double y3 = y2*y;

               double x = -(+G[3][4]*y3+G[3][5]*y2*z+G[3][6]*y2+G[3][8]*y*z2+G[3][9]*y*z+G[3][10]*y+G[3][16]*z3
                            +G[3][17]*z2+G[3][18]*z+G[3][19]*1);

               //if (x*x + y*y + z*z > 1) continue; // Skip everything with too large angle

               Vector3d sigma = makeVector3(x, y, z);

               Matrix3x3d R;
               createRotationMatrixFromModifiedRodrigues(sigma, R);
               Rs.push_back(R);
               Ts.push_back(Y - R*X);
            } // end for (i)
            break;
         }
         case V3D_2_1P1P_METHOD_REFINE_NONLINEAR:
         {
            for (size_t i = 0; i < Rs.size(); ++i)
            {
               Matrix3x3d& R = Rs[i];
               ClassicalRodrigues_2_1Plus1PointOptimizer opt(pL1, pL2, pR, qL1, qL2, qR, X, Y, B, R);
               opt.minimize();
               R = opt._R;
               Ts[i] = Y - R*X;
            } // end for (i)
            break;
         }
         case V3D_2_1P1P_METHOD_REFINE_MODIFIED_RODRIGUES:
         {
            for (size_t i = 0; i < Rs.size(); ++i)
            {
               Matrix3x3d& R = Rs[i];
               ModifiedRodrigues_2_1Plus1PointOptimizer opt(pL1, pL2, pR, qL1, qL2, qR, X, Y, B, R);
               opt.minimize();
               R = opt._R;
               Ts[i] = Y - R*X;
            } // end for (i)
            break;
         }
         case V3D_2_1P1P_METHOD_REFINE_QUATERNION:
         {
            for (size_t i = 0; i < Rs.size(); ++i)
            {
               Matrix3x3d& R = Rs[i];
               Quaternion_2_1Plus1PointOptimizer opt(pL1, pL2, pR, qL1, qL2, qR, X, Y, B, R);
               opt.minimize();
               createRotationMatrixFromQuaternion(opt._Q, R);
               Ts[i] = Y - R*X;
            } // end for (i)
            break;
         }
         default:
            cerr << "computeRelativePose_2_1Plus1Point<>(): Unknown method specifier." << endl;
            return false;
      } // end switch
      return true;
   } // end computeRelativePose_2_1Plus1Point()

   template bool
   computeRelativePose_2_1Plus1Point<double>(Vector2d const& pL1, Vector2d const& pL2, Vector2d const& pR,
                                             Vector2d const& qL1, Vector2d const& qL2, Vector2d const& qR,
                                             Vector3d const& X, Vector3d const& Y, Vector3d const& B,
                                             std::vector<Matrix3x3d>& Rs, std::vector<Vector3d>& Ts, int method);

   template bool
   computeRelativePose_2_1Plus1Point<long double>(Vector2d const& pL1, Vector2d const& pL2, Vector2d const& pR,
                                                  Vector2d const& qL1, Vector2d const& qL2, Vector2d const& qR,
                                                  Vector3d const& X, Vector3d const& Y, Vector3d const& B,
                                                  std::vector<Matrix3x3d>& Rs, std::vector<Vector3d>& Ts, int method);

} // end namespace V3D
