// -*- C++ -*-
#ifndef V3D_GPU_CONVEX_MRF_H
#define V3D_GPU_CONVEX_MRF_H

//#define GPU_POTTS_LABELING_USE_PACKED_P 1

# if defined(V3DLIB_GPGPU_ENABLE_CG)

#include "v3d_gpubase.h"

namespace V3D_GPU
{

   struct ConvexMRF_3Labels_Base
   {
         ConvexMRF_3Labels_Base()
            : _width(-1), _height(-1), _tau_primal(0.5f), _tau_dual(0.5f),
              _c1(1.0f), _c2(100.0f)
         { }

         void setTimesteps(float tau_primal, float tau_dual)
         {
            _tau_primal = tau_primal;
            _tau_dual = tau_dual;
         }

         void setRegularizationShape(float c1, float c2)
         {
            _c1 = c1;
            _c2 = c2;
         }

      protected:
         int _width, _height;
         float _tau_primal, _tau_dual;
         float _c1, _c2;
   }; // end struct ConvexMRF_3Labels_Base

   struct ConvexMRF_3Labels_Generic : public ConvexMRF_3Labels_Base
   {
         ConvexMRF_3Labels_Generic()
         {
            for (int i = 0; i < 2; ++i)
            {
               _uBufs[i] = new RTT_Buffer("rgb=16f", "ConvexMRF_3Labels_Generic::uBuf");
               _pqFbos[i] = new FrameBufferObject("ConvexMRF_3Labels_Generic::pqFbo");
               _rsFbos[i] = new FrameBufferObject("ConvexMRF_3Labels_Generic::rsFbo");
            }
         }

         ~ConvexMRF_3Labels_Generic()
         { }

         void allocate(int w, int h);
         void deallocate();

         void setZero();
         //void copyBuffersFrom(ConvexMRF_3Labels_Generic& src);
         void iterate(unsigned int costTexId, int nIterations);

         RTT_Buffer& getResultBuffer() { return *_uBufs[0]; }

      protected:
         RTT_Buffer * _uBufs[2];
         ImageTexture2D _p1Texs[2], _p2Texs[2], _qTexs[2];
         ImageTexture2D _r1Texs[2], _r2Texs[2], _s1Texs[2], _s2Texs[2];
         FrameBufferObject *_pqFbos[2], *_rsFbos[2];
   }; // end struct ConvexMRF_3Labels_Generic

   struct ConvexMRF_3Labels_Opt : public ConvexMRF_3Labels_Base
   {
         ConvexMRF_3Labels_Opt()
         {
            for (int i = 0; i < 2; ++i)
            {
               _uBufs[i] = new RTT_Buffer("rgb=16f", "ConvexMRF_3Labels_Opt::uBuf");
               _rsBufs[i] = new RTT_Buffer("rgba=16f", "ConvexMRF_3Labels_Opt::rsBuf");
               _pqFbos[i] = new FrameBufferObject("ConvexMRF_3Labels_Opt::pqFbo");
            }
         }

         ~ConvexMRF_3Labels_Opt()
         { }

         void allocate(int w, int h);
         void deallocate();

         void setZero();
         //void copyBuffersFrom(ConvexMRF_3Labels_Opt& src);
         void iterate(unsigned int costTexId, int nIterations);

         RTT_Buffer& getResultBuffer() { return *_uBufs[0]; }

      protected:
         RTT_Buffer * _uBufs[2];
         RTT_Buffer * _rsBufs[2];
         ImageTexture2D _p1Texs[2], _p2Texs[2], _qTexs[2];
         FrameBufferObject *_pqFbos[2];
   }; // end struct ConvexMRF_3Labels_Opt

} // end namespace V3D_GPU

# endif // defined(V3DLIB_GPGPU_ENABLE_CG)

#endif // !defined(V3D_GPU_CONVEX_MRF_H)
