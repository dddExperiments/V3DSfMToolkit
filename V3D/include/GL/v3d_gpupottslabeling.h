// -*- C++ -*-
#ifndef V3D_GPU_POTTS_LABELING_H
#define V3D_GPU_POTTS_LABELING_H

//#define GPU_POTTS_LABELING_USE_PACKED_P 1

# if defined(V3DLIB_GPGPU_ENABLE_CG)

#include "v3d_gpubase.h"

namespace V3D_GPU
{

   //! Label assignment using the Potts discontinuity model for 3 labels.
   // This corresponds to the VMV 2008 method using a stricly convex relaxation.
   struct PottsLabeling3_Relaxed
   {
         PottsLabeling3_Relaxed()
            : _width(-1), _height(-1), _theta(0.1f), _timestep(0.249f)
#if !defined(GPU_POTTS_LABELING_USE_PACKED_P)
            , _p1TexA("PottsLabeling3_Relaxed::p1TexA"),
              _p2TexA("PottsLabeling3_Relaxed::p2TexA"),
              _p1TexB("PottsLabeling3_Relaxed::p1TexB"),
              _p2TexB("PottsLabeling3_Relaxed::p2TexB")
#endif
         {
            _uBufA = new RTT_Buffer("rgb=16f", "PottsLabeling3_Relaxed::uBufA");
            _uBufB = new RTT_Buffer("rgb=16f", "PottsLabeling3_Relaxed::uBufB");
#if defined(GPU_POTTS_LABELING_USE_PACKED_P)
            _pBufA = new RTT_Buffer("rgb=32f", "PottsLabeling3_Relaxed::pBufA");
            _pBufB = new RTT_Buffer("rgb=32f", "PottsLabeling3_Relaxed::pBufB");
#else
            _pFboA = new FrameBufferObject("PottsLabeling3_Relaxed::pFboA");
            _pFboB = new FrameBufferObject("PottsLabeling3_Relaxed::pFboB");
#endif
         }

         ~PottsLabeling3_Relaxed()
         { }

         void allocate(int w, int h);
         void deallocate();

         void setTheta(float theta) { _theta = theta; }
         void setTimestep(float step) { _timestep = step; }

         void setZero();
         void copyBuffersFrom(PottsLabeling3_Relaxed& src);
         void iterate(unsigned int costTexId, int nIterations);

         RTT_Buffer& getResultBuffer() { return *_uBufA; }

      protected:
         int _width, _height;

         float _theta, _timestep;

         RTT_Buffer * _uBufA;
         RTT_Buffer * _uBufB;
#if defined(GPU_POTTS_LABELING_USE_PACKED_P)
         RTT_Buffer * _pBufA;
         RTT_Buffer * _pBufB;
#else
         ImageTexture2D _p1TexA, _p2TexA, _p1TexB, _p2TexB;
         FrameBufferObject * _pFboA;
         FrameBufferObject * _pFboB;
#endif
   }; // end struct PottsLabeling3_Relaxed

//======================================================================

   //! Label assignment using the Potts discontinuity model for 3 labels.
   // This version computes the level functions of the labels in the first place
   // (i.e. it is based on binary segmentation in the image x labels volume).
   struct PottsLabeling3_LevelFun
   {
         PottsLabeling3_LevelFun()
            : _width(-1), _height(-1), _tau_primal(0.4f), _tau_dual(0.4f)
            , _p1TexA("PottsLabeling3_LevelFun::p1TexA"),
              _p2TexA("PottsLabeling3_LevelFun::p2TexA"),
              _p1TexB("PottsLabeling3_LevelFun::p1TexB"),
              _p2TexB("PottsLabeling3_LevelFun::p2TexB")
         {
            _uBufA = new RTT_Buffer("rgb=16f", "PottsLabeling3_LevelFun::uBufA");
            _uBufB = new RTT_Buffer("rgb=16f", "PottsLabeling3_LevelFun::uBufB");
            _pFboA = new FrameBufferObject("PottsLabeling3_LevelFun::pFboA");
            _pFboB = new FrameBufferObject("PottsLabeling3_LevelFun::pFboB");
         }

         ~PottsLabeling3_LevelFun()
         { }

         void allocate(int w, int h);
         void deallocate();

         void setTimesteps(float tau_primal, float tau_dual)
         {
            _tau_primal = tau_primal;
            _tau_dual = tau_dual;
         }

         void setZero();
         void copyBuffersFrom(PottsLabeling3_LevelFun& src);
         void iterate(unsigned int costTexId, int nIterations);

         RTT_Buffer& getResultBuffer() { return *_uBufA; }

      protected:
         int _width, _height;
         float _tau_primal, _tau_dual;

         RTT_Buffer * _uBufA;
         RTT_Buffer * _uBufB;
         ImageTexture2D _p1TexA, _p2TexA, _p1TexB, _p2TexB, _qTexA, _qTexB;
         FrameBufferObject * _pFboA;
         FrameBufferObject * _pFboB;
   }; // end struct PottsLabeling3_LevelFun

   struct PottsLabeling4_LevelFun
   {
         PottsLabeling4_LevelFun()
            : _width(-1), _height(-1), _tau_primal(0.4f), _tau_dual(0.4f)
            , _p1TexA("PottsLabeling4_LevelFun::p1TexA"),
              _p2TexA("PottsLabeling4_LevelFun::p2TexA"),
              _p1TexB("PottsLabeling4_LevelFun::p1TexB"),
              _p2TexB("PottsLabeling4_LevelFun::p2TexB")
         {
            _uBufA = new RTT_Buffer("rgba=16f", "PottsLabeling4_LevelFun::uBufA");
            _uBufB = new RTT_Buffer("rgba=16f", "PottsLabeling4_LevelFun::uBufB");
            _pFboA = new FrameBufferObject("PottsLabeling4_LevelFun::pFboA");
            _pFboB = new FrameBufferObject("PottsLabeling4_LevelFun::pFboB");
         }

         ~PottsLabeling4_LevelFun()
         { }

         void allocate(int w, int h);
         void deallocate();

         void setTimesteps(float tau_primal, float tau_dual)
         {
            _tau_primal = tau_primal;
            _tau_dual = tau_dual;
         }

         void setZero();
         void copyBuffersFrom(PottsLabeling4_LevelFun& src);
         void iterate(unsigned int costTexId, int nIterations);

         RTT_Buffer& getResultBuffer() { return *_uBufA; }

      protected:
         int _width, _height;
         float _tau_primal, _tau_dual;

         RTT_Buffer * _uBufA;
         RTT_Buffer * _uBufB;
         ImageTexture2D _p1TexA, _p2TexA, _p1TexB, _p2TexB, _qTexA, _qTexB;
         FrameBufferObject * _pFboA;
         FrameBufferObject * _pFboB;
   }; // end struct PottsLabeling4_LevelFun

} // end namespace V3D_GPU

# endif

#endif
