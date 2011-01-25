#include "Base/v3d_exception.h"
#include "Math/v3d_optimization.h"
#include "Geometry/v3d_selfcalibration.h"

using namespace V3D;

namespace
{

   struct ConstantIntrinsicCalibratorLM : public SimpleLevenbergOptimizer
   {
      protected:
         static int const parameterCountFromMode[4];

      public:
         ConstantIntrinsicCalibratorLM(std::vector<Matrix3x3d > const& fundamentals,
                                       std::vector<double> const& weights,
                                       Matrix3x3d const& K0,
                                       int mode)
            : SimpleLevenbergOptimizer(fundamentals.size(), parameterCountFromMode[mode]),
              _fundamentals(fundamentals),
              _weights(weights), _mode(mode)
         {
            _K0 = K0;
            _knownAspectRatio = K0(2, 2) / K0(1, 1);

            switch (mode)
            {
               case V3D_INTRINSIC_SELF_CALIBRATION_UNCONSTRAINED:
                  currentParameters[0] = K0(1, 1);
                  currentParameters[1] = K0(1, 2);
                  currentParameters[2] = K0(1, 3);
                  currentParameters[3] = K0(2, 2);
                  currentParameters[4] = K0(2, 3);
                  break;
               case V3D_INTRINSIC_SELF_CALIBRATION_KNOWN_SKEW:
                  currentParameters[0] = K0(1, 1);
                  currentParameters[1] = K0(1, 3);
                  currentParameters[2] = K0(2, 2);
                  currentParameters[3] = K0(2, 3);
                  break;
               case V3D_INTRINSIC_SELF_CALIBRATION_KNOWN_SKEW_AND_ASPECT:
                  currentParameters[0] = K0(1, 1);
                  currentParameters[1] = K0(1, 3);
                  currentParameters[2] = K0(2, 3);
                  break;
               case V3D_INTRINSIC_SELF_CALIBRATION_ONLY_FOCAL_LENGTH:
                  currentParameters[0] = K0(1, 1);
                  break;
               default:
                  throw V3D::Exception("ConstantIntrinsicCalibratorLM::ConstantIntrinsicCalibratorLM(): Unknown mode.");
            } // end switch (mode)

            for (unsigned i = 0; i < nParameters; ++i)
               numDiffDelta[i] = std::max(0.001, 0.05 * currentParameters[i]);
         }

         virtual void evalFunction(Vector<double>& res)
         {
            assert(res.size() == _fundamentals.size());

            Matrix3x3d K, Kt, E, Vt;
            Matrix<double> E_dyn(3, 3);

            generateKfromParamVector(this->currentParameters, K);

            Kt = K.transposed();

            for (size_t i = 0; i < _fundamentals.size(); ++i)
            {
               E = Kt * _fundamentals[i] * K;

               copyMatrix(E, E_dyn);
               SVD<double> svd(E_dyn);
               Vector<double> const& S = svd.getSingularValues();
               res[i] = sqrt(_weights[i] * (S[0]-S[1]) / (S[0]+S[1]));
            } // end for (i)
         } // end evalFunction()

         void generateKfromParamVector(Vector<double> const& v, Matrix3x3d& K) const
         {
            switch (_mode)
            {
               case V3D_INTRINSIC_SELF_CALIBRATION_UNCONSTRAINED:
               {
                  K(1, 1) = v[0]; K(1, 2) = v[1]; K(1, 3) = v[2];
                  K(2, 2) = v[3]; K(2, 3) = v[4];
                  K(3, 3) = _K0(3, 3);
                  break;
               }
               case V3D_INTRINSIC_SELF_CALIBRATION_KNOWN_SKEW:
               {
                  K(1, 1) = v[0]; K(1, 2) = _K0(1, 2); K(1, 3) = v[1];
                  K(2, 2) = v[2]; K(2, 3) = v[3];
                  K(3, 3) = _K0(3, 3);
                  break;
               }
               case V3D_INTRINSIC_SELF_CALIBRATION_KNOWN_SKEW_AND_ASPECT:
               {
                  K(1, 1) = v[0]; K(1, 2) = _K0(1, 2); K(1, 3) = v[1];
                  K(2, 2) = _knownAspectRatio * v[0]; K(2, 3) = v[2];
                  K(3, 3) = _K0(3, 3);
                  break;
               }
               case V3D_INTRINSIC_SELF_CALIBRATION_ONLY_FOCAL_LENGTH:
               {
                  K(1, 1) = v[0]; K(1, 2) = _K0(1, 2); K(1, 3) = _K0(1, 3);
                  K(2, 2) = _knownAspectRatio * v[0]; K(2, 3) = _K0(2, 3);
                  K(3, 3) = _K0(3, 3);
                  break;
               }
            } // end switch (mode)
         } // end generateKfromVector()

      protected:
         std::vector<Matrix3x3d > const& _fundamentals;
         std::vector<double>   const& _weights;
         Matrix3x3d      _K0;
         double                       _knownAspectRatio;
         int                          _mode;
   }; // end struct ConstantIntrinsicCalibratorLM

   int const ConstantIntrinsicCalibratorLM::parameterCountFromMode[4] = { 5, 4, 3, 1 };

} // end namespace <>

namespace V3D_Calibration
{

   bool
   calibrateIntrinsic(std::vector<Matrix3x3d > const& fundamentals,
                      std::vector<double> const& weights,
                      Matrix3x3d& K, int mode, int const nIterations)
   {
      if (fundamentals.size() != weights.size())
      {
         throw V3D::Exception("calibrateIntrinsic(): There should be one weight for each fundamental.");
         return false;
      }

      Matrix3x3d const K0(K);

      ConstantIntrinsicCalibratorLM calibrator(fundamentals, weights, K0, mode);

      Vector<double> residuum(calibrator.nMeasurements);
      calibrator.evalFunction(residuum);
      //cout << "initial objective = " << sqrVectorLength(residuum) << endl;

      calibrator.maxIterations = nIterations;
      calibrator.minimize();

      calibrator.evalFunction(residuum);
      //cout << "final objective = " << sqrVectorLength(residuum) << endl;

      calibrator.generateKfromParamVector(calibrator.currentParameters, K);

      return true;
   } // end calibrateIntrinsic()

} // end namespace V3D_Calibration
