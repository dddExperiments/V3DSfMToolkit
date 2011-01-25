// -*- C++ -*-
#ifndef V3D_CUDA_SEGMENTATION_H
#define V3D_CUDA_SEGMENTATION_H

#if defined(V3DLIB_ENABLE_CUDA)

namespace V3D_CUDA
{

   struct BinarySegmentation
   {
         enum
         {
            L2_METRIC = 0,
            L1_METRIC = 1,
            WEIGHTED_L2_METRIC = 2,
         };

         BinarySegmentation()
            : _metric(L2_METRIC)
         { }

         void setRegularizationMetric(int metric) { _metric = metric; }

         void allocate(int w, int h);
         void deallocate();

         void setImageData(float const * fSrc);
         void setWeightData(float const * wSrc);
         void initSegmentation();
         void getResult(float * uDst, float * p1Dst = 0, float * p2Dst = 0);

         void run(int nIterations, float alpha, float tau);

      private:
         int _w, _h;
         int _metric;

         float *_d_u, *_d_p1, *_d_p2;
         float *_d_g, *_d_c;
   }; // end struct BinarySegmentation

} // end namespace V3D_CUDA

# endif // defined(V3DLIB_ENABLE_CUDA)

#endif
