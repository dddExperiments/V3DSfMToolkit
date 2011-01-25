// -*- C++ -*-
#ifndef V3D_CUDA_FLOW_H
#define V3D_CUDA_FLOW_H

#if defined(V3DLIB_ENABLE_CUDA)

#include <vector>

#include <Base/v3d_image.h>

namespace V3D_CUDA
{

   struct TVL1_FlowEstimationBase
   {
         TVL1_FlowEstimationBase()
            : _lambda(10.0f), _nOuterIterations(1), _nInnerIterations(50), _startLevel(0), _nLevels(3),
              _width(-1), _height(-1)
         { }

         void setLambda(float lambda)        { _lambda = lambda; }
         void setOuterIterations(int nIters) { _nOuterIterations = nIters; }
         void setInnerIterations(int nIters) { _nInnerIterations = nIters; }
         void setStartLevel(int startLevel)  { _startLevel = startLevel; }

         void allocate(int w, int h, int nLevels);
         void deallocate();

      protected:
         void warpImage(int level, float const * I0, float const * I1, float * d_u, float * d_v);

         float _lambda;
         int _nOuterIterations, _nInnerIterations;
         int _startLevel, _nLevels;
         int _width, _height;

         std::vector<int>     _widths, _heights;
         std::vector<int>     _imageSizes;
         std::vector<float *> _d_coeffs;
   }; // end struct TVL1_FlowEstimationBase

   struct TVL1_FlowEstimation_Relaxed : public TVL1_FlowEstimationBase
   {
      public:
         struct Config
         {
               Config(float tau = 0.249f, float theta = 0.1f)
                  : _tau(tau), _theta(theta)
               { }

               float _tau, _theta;
         };

         TVL1_FlowEstimation_Relaxed()
            : TVL1_FlowEstimationBase()
         { }

         void configure(Config const& cfg) { _cfg = cfg; }

         void allocate(int w, int h, int nLevels);
         void deallocate();

         void setImage(int index, unsigned char const * pixels);
         void setImage(int index, float const * pixels);
         void run();

         void getFlowField(float * uDst, float * vDst);

         void swapPyramids() { std::swap(_leftPyr, _rightPyr); }

      protected:
         void clearBuffers(int level);
         void upsampleBuffers(int dstLevel);

         Config _cfg;

         V3D::Image<float> * _leftPyr;
         V3D::Image<float> * _rightPyr;

         std::vector<int>     _bufferSizes;
         std::vector<float *> _d_u, _d_v, _d_pu, _d_pv;
   }; // end struct TVL1_FlowEstimation_Relaxed

} // end namespace V3D_CUDA

#endif // defined(V3DLIB_ENABLE_CUDA)

#endif
