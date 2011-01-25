#include "v3d_gpupottslabeling.h"

#if defined(V3DLIB_GPGPU_ENABLE_CG)

#include <GL/glew.h>

namespace V3D_GPU
{

   void
   PottsLabeling3_Relaxed::allocate(int w, int h)
   {
      _width = w;
      _height = h;

      _uBufA->allocate(w, h);
      _uBufB->allocate(w, h);

#if defined(GPU_POTTS_LABELING_USE_PACKED_P)
      _pBufA->allocate(w, h);
      _pBufB->allocate(w, h);
#else
      _p1TexA.allocateID();
      _p1TexA.reserve(w, h, TextureSpecification("rgb=16f"));
      _p2TexA.allocateID();
      _p2TexA.reserve(w, h, TextureSpecification("rgb=16f"));
      _p1TexB.allocateID();
      _p1TexB.reserve(w, h, TextureSpecification("rgb=16f"));
      _p2TexB.allocateID();
      _p2TexB.reserve(w, h, TextureSpecification("rgb=16f"));

      _pFboA->allocate();
      _pFboA->makeCurrent();
      _pFboA->attachTexture2D(_p1TexA, GL_COLOR_ATTACHMENT0_EXT);
      _pFboA->attachTexture2D(_p2TexA, GL_COLOR_ATTACHMENT1_EXT);

      _pFboB->allocate();
      _pFboB->makeCurrent();
      _pFboB->attachTexture2D(_p1TexB, GL_COLOR_ATTACHMENT0_EXT);
      _pFboB->attachTexture2D(_p2TexB, GL_COLOR_ATTACHMENT1_EXT);
#endif
   } // end PottsLabeling3_Relaxed::allocate()

   void
   PottsLabeling3_Relaxed::deallocate()
   {
      _uBufA->deallocate();
      _uBufA->deallocate();

#if defined(GPU_POTTS_LABELING_USE_PACKED_P)
      _pBufA->deallocate();
      _pBufB->deallocate();
#else
      _p1TexA.deallocateID();
      _p2TexA.deallocateID();
      _p1TexB.deallocateID();
      _p2TexB.deallocateID();

      _pFboA->deallocate();
      _pFboB->deallocate();
#endif
   } // end PottsLabeling3_Relaxed::deallocate()

   void
   PottsLabeling3_Relaxed::setZero()
   {
      glClearColor(0, 0, 0, 0);
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
      _uBufA->activate();
      glClear(GL_COLOR_BUFFER_BIT);
#if defined(GPU_POTTS_LABELING_USE_PACKED_P)
      _pBufA->activate();
      glClear(GL_COLOR_BUFFER_BIT);
#else
      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
      _pFboA->activate();
      glDrawBuffersARB(1, buffers+1);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+0);
      glClear(GL_COLOR_BUFFER_BIT);
#endif
   } // PottsLabeling3_Relaxed::setZero()

   void
   PottsLabeling3_Relaxed::copyBuffersFrom(PottsLabeling3_Relaxed& src)
   {
      setupNormalizedProjection();
      _uBufA->activate();
      src._uBufA->enableTexture(GL_TEXTURE0);
      enableTrivialTexture2DShader();
      renderNormalizedQuad();
      src._uBufA->disableTexture(GL_TEXTURE0);

#if defined(GPU_POTTS_LABELING_USE_PACKED_P)
      _pBufA->activate();
      src._pBufA->enableTexture(GL_TEXTURE0);
      renderNormalizedQuad();
      src._pBufA->disableTexture(GL_TEXTURE0);
#else
      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
      _pFboA->activate();

      glDrawBuffersARB(1, buffers+1);
      src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
      renderNormalizedQuad();
      src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

      glDrawBuffersARB(1, buffers+0);
      src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
      renderNormalizedQuad();
      src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);
#endif
      disableTrivialTexture2DShader();
   } // end PottsLabeling3_Relaxed::copyLabels()

   void
   PottsLabeling3_Relaxed::iterate(unsigned int costTexId, int nIterations)
   {
      static Cg_FragmentProgram * uShader = 0;
      static Cg_FragmentProgram * pShader = 0;

      if (uShader == 0)
      {
         uShader = new Cg_FragmentProgram("PottsLabeling3_Relaxed::uShader");
         uShader->setProgramFromFile("potts_3labeling_update_u.cg");

#if defined(GPU_POTTS_LABELING_USE_PACKED_P)
         char const * args[] = { "-DUSE_PACKED_P=1", 0 };
         uShader->compile(args);
#else
         uShader->compile();
#endif
         checkGLErrorsHere0();
      }

      if (pShader == 0)
      {
         pShader = new Cg_FragmentProgram("PottsLabeling3_Relaxed::pShader");
         pShader->setProgramFromFile("potts_3labeling_update_p.cg");
#if defined(GPU_POTTS_LABELING_USE_PACKED_P)
         char const * args[] = { "-DUSE_PACKED_P=1", 0 };
         pShader->compile(args);
#else
         pShader->compile();
#endif
         checkGLErrorsHere0();
      }

      float const ds = 1.0f/_width;
      float const dt = 1.0f/_height;

      setupNormalizedProjection();

      uShader->parameter("theta", _theta);
      pShader->parameter("timestep_over_theta", _timestep/_theta);

      for (int iter = 0; iter < nIterations; ++iter)
      {
#if defined(GPU_POTTS_LABELING_USE_PACKED_P)
         _uBufB->activate();
         //uShader->parameter("theta", theta);
         _uBufA->enableTexture(GL_TEXTURE0);
         _pBufA->enableTexture(GL_TEXTURE1);
         glActiveTexture(GL_TEXTURE2);
         glBindTexture(GL_TEXTURE_2D, costTexId);
         glEnable(GL_TEXTURE_2D);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBufA->disableTexture(GL_TEXTURE0);
         glActiveTexture(GL_TEXTURE2);
         glDisable(GL_TEXTURE_2D);

         std::swap(_uBufA, _uBufB);

         _pBufB->activate();
         //pShader->parameter("timestep_over_theta", tau/theta);
         _uBufA->enableTexture(GL_TEXTURE0);

         pShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         pShader->disable();

         _uBufA->disableTexture(GL_TEXTURE0);
         _pBufA->disableTexture(GL_TEXTURE1);

         std::swap(_pBufA, _pBufB);
#else
         _uBufB->activate();
         glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
         //checkGLErrorsHere0();
         //uShader->parameter("theta", theta);
         _uBufA->enableTexture(GL_TEXTURE0);
         _pFboA->getColorTexture(0).enable(GL_TEXTURE1);
         _pFboA->getColorTexture(1).enable(GL_TEXTURE2);
         glActiveTexture(GL_TEXTURE3);
         glBindTexture(GL_TEXTURE_2D, costTexId);
         glEnable(GL_TEXTURE_2D);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBufA->disableTexture(GL_TEXTURE0);
         glActiveTexture(GL_TEXTURE3);
         glDisable(GL_TEXTURE_2D);
//          _pBufA->getColorTexture(0).disable(GL_TEXTURE1);
//          _pBufA->getColorTexture(1).disable(GL_TEXTURE2);

         std::swap(_uBufA, _uBufB);

         //pShader->parameter("timestep_over_theta", tau/theta);

         _pFboB->activate();
         GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
         glDrawBuffersARB(2, buffers);
         //checkGLErrorsHere0();
         _uBufA->enableTexture(GL_TEXTURE0);
//          _pBufA->getColorTexture(0).enable(GL_TEXTURE1);
//          _pBufA->getColorTexture(1).enable(GL_TEXTURE2);

         pShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         pShader->disable();

         _uBufA->disableTexture(GL_TEXTURE0);
         _pFboA->getColorTexture(0).disable(GL_TEXTURE1);
         _pFboA->getColorTexture(1).disable(GL_TEXTURE2);

         std::swap(_pFboA, _pFboB);
#endif
      } // end for (iter)
   } // end PottsLabeling3_Relaxed::iterate()

//======================================================================

   void
   PottsLabeling3_LevelFun::allocate(int w, int h)
   {
      _width = w;
      _height = h;

      _uBufA->allocate(w, h);
      _uBufB->allocate(w, h);

      _p1TexA.allocateID();
      _p1TexA.reserve(w, h, TextureSpecification("rgb=16f"));
      _p2TexA.allocateID();
      _p2TexA.reserve(w, h, TextureSpecification("rgb=16f"));
      _p1TexB.allocateID();
      _p1TexB.reserve(w, h, TextureSpecification("rgb=16f"));
      _p2TexB.allocateID();
      _p2TexB.reserve(w, h, TextureSpecification("rgb=16f"));

      _qTexA.allocateID();
      _qTexA.reserve(w, h, TextureSpecification("rgb=16f"));
      _qTexB.allocateID();
      _qTexB.reserve(w, h, TextureSpecification("rgb=16f"));

      _pFboA->allocate();
      _pFboA->makeCurrent();
      _pFboA->attachTexture2D(_p1TexA, GL_COLOR_ATTACHMENT0_EXT);
      _pFboA->attachTexture2D(_p2TexA, GL_COLOR_ATTACHMENT1_EXT);
      _pFboA->attachTexture2D(_qTexA, GL_COLOR_ATTACHMENT2_EXT);

      _pFboB->allocate();
      _pFboB->makeCurrent();
      _pFboB->attachTexture2D(_p1TexB, GL_COLOR_ATTACHMENT0_EXT);
      _pFboB->attachTexture2D(_p2TexB, GL_COLOR_ATTACHMENT1_EXT);
      _pFboB->attachTexture2D(_qTexB, GL_COLOR_ATTACHMENT2_EXT);
   } // end PottsLabeling3_LevelFun::allocate()

   void
   PottsLabeling3_LevelFun::deallocate()
   {
      _uBufA->deallocate();
      _uBufA->deallocate();

      _p1TexA.deallocateID();
      _p2TexA.deallocateID();
      _qTexA.deallocateID();
      _p1TexB.deallocateID();
      _p2TexB.deallocateID();
      _qTexB.deallocateID();

      _pFboA->deallocate();
      _pFboB->deallocate();
   } // end PottsLabeling3_LevelFun::deallocate()

   void
   PottsLabeling3_LevelFun::setZero()
   {
      glClearColor(0, 0, 0, 0);
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
      _uBufA->activate();
      glClear(GL_COLOR_BUFFER_BIT);
      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT };
      _pFboA->activate();
      glDrawBuffersARB(1, buffers+2);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+1);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+0);
      glClear(GL_COLOR_BUFFER_BIT);
   } // PottsLabeling3_LevelFun::setZero()

   void
   PottsLabeling3_LevelFun::copyBuffersFrom(PottsLabeling3_LevelFun& src)
   {
      setupNormalizedProjection();
      _uBufA->activate();
      src._uBufA->enableTexture(GL_TEXTURE0);
      enableTrivialTexture2DShader();
      renderNormalizedQuad();
      src._uBufA->disableTexture(GL_TEXTURE0);

      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT };
      _pFboA->activate();

      glDrawBuffersARB(1, buffers+2);
      src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
      renderNormalizedQuad();
      src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

      glDrawBuffersARB(1, buffers+1);
      src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
      renderNormalizedQuad();
      src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

      glDrawBuffersARB(1, buffers+0);
      src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
      renderNormalizedQuad();
      src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

      disableTrivialTexture2DShader();
   } // end PottsLabeling3_LevelFun::copyLabels()

   void
   PottsLabeling3_LevelFun::iterate(unsigned int costTexId, int nIterations)
   {
      static Cg_FragmentProgram * uShader = 0;
      static Cg_FragmentProgram * pqShader = 0;

      if (uShader == 0)
      {
         uShader = new Cg_FragmentProgram("PottsLabeling3_LevelFun::uShader");
         uShader->setProgramFromFile("potts_3labeling_levelfun_update_u.cg");

         uShader->compile();
         checkGLErrorsHere0();
      }

      if (pqShader == 0)
      {
         pqShader = new Cg_FragmentProgram("PottsLabeling3_LevelFun::pqShader");
         pqShader->setProgramFromFile("potts_3labeling_levelfun_update_pq.cg");
         pqShader->compile();
         checkGLErrorsHere0();
      }

      float const ds = 1.0f/_width;
      float const dt = 1.0f/_height;

      setupNormalizedProjection();

      uShader->parameter("tau", _tau_primal);
      pqShader->parameter("tau", _tau_dual);

      for (int iter = 0; iter < nIterations; ++iter)
      {
         _uBufB->activate();
         glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

         _uBufA->enableTexture(GL_TEXTURE0);
         _pFboA->getColorTexture(0).enable(GL_TEXTURE1);
         _pFboA->getColorTexture(1).enable(GL_TEXTURE2);
         _pFboA->getColorTexture(2).enable(GL_TEXTURE3);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBufA->disableTexture(GL_TEXTURE0);

         std::swap(_uBufA, _uBufB);

         _pFboB->activate();
         GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT };
         glDrawBuffersARB(3, buffers);
         _uBufA->enableTexture(GL_TEXTURE0);

         glActiveTexture(GL_TEXTURE4);
         glBindTexture(GL_TEXTURE_2D, costTexId);
         glEnable(GL_TEXTURE_2D);

         pqShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         pqShader->disable();

         _uBufA->disableTexture(GL_TEXTURE0);
         _pFboA->getColorTexture(0).disable(GL_TEXTURE1);
         _pFboA->getColorTexture(1).disable(GL_TEXTURE2);
         _pFboA->getColorTexture(2).disable(GL_TEXTURE3);

         glActiveTexture(GL_TEXTURE4);
         glDisable(GL_TEXTURE_2D);

         std::swap(_pFboA, _pFboB);
      } // end for (iter)
   } // end PottsLabeling3_LevelFun::iterate()

//======================================================================

   void
   PottsLabeling4_LevelFun::allocate(int w, int h)
   {
      _width = w;
      _height = h;

      _uBufA->allocate(w, h);
      _uBufB->allocate(w, h);

      _p1TexA.allocateID();
      _p1TexA.reserve(w, h, TextureSpecification("rgba=16f"));
      _p2TexA.allocateID();
      _p2TexA.reserve(w, h, TextureSpecification("rgba=16f"));
      _p1TexB.allocateID();
      _p1TexB.reserve(w, h, TextureSpecification("rgba=16f"));
      _p2TexB.allocateID();
      _p2TexB.reserve(w, h, TextureSpecification("rgba=16f"));

      _qTexA.allocateID();
      _qTexA.reserve(w, h, TextureSpecification("rgba=16f"));
      _qTexB.allocateID();
      _qTexB.reserve(w, h, TextureSpecification("rgba=16f"));

      _pFboA->allocate();
      _pFboA->makeCurrent();
      _pFboA->attachTexture2D(_p1TexA, GL_COLOR_ATTACHMENT0_EXT);
      _pFboA->attachTexture2D(_p2TexA, GL_COLOR_ATTACHMENT1_EXT);
      _pFboA->attachTexture2D(_qTexA, GL_COLOR_ATTACHMENT2_EXT);

      _pFboB->allocate();
      _pFboB->makeCurrent();
      _pFboB->attachTexture2D(_p1TexB, GL_COLOR_ATTACHMENT0_EXT);
      _pFboB->attachTexture2D(_p2TexB, GL_COLOR_ATTACHMENT1_EXT);
      _pFboB->attachTexture2D(_qTexB, GL_COLOR_ATTACHMENT2_EXT);
   } // end PottsLabeling4_LevelFun::allocate()

   void
   PottsLabeling4_LevelFun::deallocate()
   {
      _uBufA->deallocate();
      _uBufA->deallocate();

      _p1TexA.deallocateID();
      _p2TexA.deallocateID();
      _qTexA.deallocateID();
      _p1TexB.deallocateID();
      _p2TexB.deallocateID();
      _qTexB.deallocateID();

      _pFboA->deallocate();
      _pFboB->deallocate();
   } // end PottsLabeling4_LevelFun::deallocate()

   void
   PottsLabeling4_LevelFun::setZero()
   {
      glClearColor(0, 0, 0, 0);
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
      _uBufA->activate();
      glClear(GL_COLOR_BUFFER_BIT);
      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT };
      _pFboA->activate();
      glDrawBuffersARB(1, buffers+2);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+1);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+0);
      glClear(GL_COLOR_BUFFER_BIT);
   } // PottsLabeling4_LevelFun::setZero()

   void
   PottsLabeling4_LevelFun::copyBuffersFrom(PottsLabeling4_LevelFun& src)
   {
      setupNormalizedProjection();
      _uBufA->activate();
      src._uBufA->enableTexture(GL_TEXTURE0);
      enableTrivialTexture2DShader();
      renderNormalizedQuad();
      src._uBufA->disableTexture(GL_TEXTURE0);

      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT };
      _pFboA->activate();

      glDrawBuffersARB(1, buffers+2);
      src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
      renderNormalizedQuad();
      src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

      glDrawBuffersARB(1, buffers+1);
      src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
      renderNormalizedQuad();
      src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

      glDrawBuffersARB(1, buffers+0);
      src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
      renderNormalizedQuad();
      src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

      disableTrivialTexture2DShader();
   } // end PottsLabeling4_LevelFun::copyLabels()

   void
   PottsLabeling4_LevelFun::iterate(unsigned int costTexId, int nIterations)
   {
      static Cg_FragmentProgram * uShader = 0;
      static Cg_FragmentProgram * pqShader = 0;

      if (uShader == 0)
      {
         uShader = new Cg_FragmentProgram("PottsLabeling4_LevelFun::uShader");
         uShader->setProgramFromFile("potts_4labeling_levelfun_update_u.cg");

         uShader->compile();
         checkGLErrorsHere0();
      }

      if (pqShader == 0)
      {
         pqShader = new Cg_FragmentProgram("PottsLabeling4_LevelFun::pqShader");
         pqShader->setProgramFromFile("potts_4labeling_levelfun_update_pq.cg");
         pqShader->compile();
         checkGLErrorsHere0();
      }

      float const ds = 1.0f/_width;
      float const dt = 1.0f/_height;

      setupNormalizedProjection();

      uShader->parameter("tau", _tau_primal);
      pqShader->parameter("tau", _tau_dual);

      for (int iter = 0; iter < nIterations; ++iter)
      {
         _uBufB->activate();
         glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

         _uBufA->enableTexture(GL_TEXTURE0);
         _pFboA->getColorTexture(0).enable(GL_TEXTURE1);
         _pFboA->getColorTexture(1).enable(GL_TEXTURE2);
         _pFboA->getColorTexture(2).enable(GL_TEXTURE3);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBufA->disableTexture(GL_TEXTURE0);

         std::swap(_uBufA, _uBufB);

         _pFboB->activate();
         GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT };
         glDrawBuffersARB(3, buffers);
         _uBufA->enableTexture(GL_TEXTURE0);

         glActiveTexture(GL_TEXTURE4);
         glBindTexture(GL_TEXTURE_2D, costTexId);
         glEnable(GL_TEXTURE_2D);

         pqShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         pqShader->disable();

         _uBufA->disableTexture(GL_TEXTURE0);
         _pFboA->getColorTexture(0).disable(GL_TEXTURE1);
         _pFboA->getColorTexture(1).disable(GL_TEXTURE2);
         _pFboA->getColorTexture(2).disable(GL_TEXTURE3);

         glActiveTexture(GL_TEXTURE4);
         glDisable(GL_TEXTURE_2D);

         std::swap(_pFboA, _pFboB);
      } // end for (iter)
   } // end PottsLabeling4_LevelFun::iterate()

} // end namespace V3D_GPU

#endif
