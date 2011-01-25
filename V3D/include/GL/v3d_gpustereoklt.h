// -*- C++ -*-
#ifndef V3D_GPU_STEREO_KLT_H
#define V3D_GPU_STEREO_KLT_H

# if defined(V3DLIB_GPGPU_ENABLE_CG)

#include "Math/v3d_linear.h"
#include "v3d_gpubase.h"
#include "v3d_gpupyramid.h"
#include "v3d_gpuklt.h"

namespace V3D_GPU
{

   struct KLT_StereoTracker : public KLT_TrackerBase
   {
         KLT_StereoTracker(int nIterations = 5, int nLevels = 4, int levelSkip = -1, int windowWidth = 7)
            : KLT_TrackerBase(nIterations, nLevels, levelSkip, windowWidth),
              _pointsBufferA("rgba=32f", "KLT_StereoTracker::_pointsBufferA"),
              _pointsBufferB("rgba=32f", "KLT_StereoTracker::_pointsBufferB"),
              _trackingShader(0)
         {
            _featuresBuffer0 = &_pointsBufferA;
            _featuresBuffer1 = &_pointsBufferB;
         }

         ~KLT_StereoTracker() { }

         void allocate(int width, int height, int pointsWidth, int pointsHeight);
         void deallocate();

         void setProjections(V3D::Matrix3x4d const& leftP, V3D::Matrix3x4d const& rightP);
         void providePoints(V3D::Vector4f const * points);
         void readPoints(V3D::Vector4f * points);
         void trackPoints(unsigned int pyrTexLeft0, unsigned int pyrTexRight0,
                          unsigned int pyrTexLeft1, unsigned int pyrTexRight1);

      protected:
         V3D::Matrix3x4d _leftP, _rightP;

         RTT_Buffer _pointsBufferA, _pointsBufferB;

         Cg_FragmentProgram * _trackingShader;
   }; // end struct KLT_StereoTracker

} // end namespace V3D_GPU

# endif // defined(V3DLIB_GPGPU_ENABLE_CG)

#endif
