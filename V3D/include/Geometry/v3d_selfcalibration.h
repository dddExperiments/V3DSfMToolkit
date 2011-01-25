// -*- C++ -*-

#ifndef V3D_SELF_CALIBRATION_H
#define V3D_SELF_CALIBRATION_H

#include <Math/v3d_linear.h>

namespace V3D
{

   enum
   {
      V3D_INTRINSIC_SELF_CALIBRATION_UNCONSTRAINED = 0, //!< K = [ fx s px; 0 fy py; 0 0 1 ]
      V3D_INTRINSIC_SELF_CALIBRATION_KNOWN_SKEW = 1, //!< K = [ fx s0 px; 0 fy py; 0 0 1 ]
      V3D_INTRINSIC_SELF_CALIBRATION_KNOWN_SKEW_AND_ASPECT = 2, //!< K = [ f s0 px; 0 a0*f py; 0 0 1 ]
      V3D_INTRINSIC_SELF_CALIBRATION_ONLY_FOCAL_LENGTH = 3 //!< K = [ f 0 0; 0 f 0; 0 0 1 ]
   };

   //! Calibrate focal length and principal point from fundamental matrices.
   bool calibrateIntrinsic(std::vector<Matrix3x3d> const& fundamentals,
                           std::vector<double> const& weights,
                           Matrix3x3d & K, int mode, int const nIterations = 200);

} // end namespace V3D

#endif
