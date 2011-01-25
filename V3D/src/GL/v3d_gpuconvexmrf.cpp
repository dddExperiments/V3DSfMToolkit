#include "v3d_gpuconvexmrf.h"

#if defined(V3DLIB_GPGPU_ENABLE_CG)

#include <GL/glew.h>
#include <iostream>

using namespace std;

namespace V3D_GPU
{

   void
   ConvexMRF_3Labels_Generic::allocate(int w, int h)
   {
      _width = w;
      _height = h;

      _uBufs[0]->allocate(w, h);
      _uBufs[1]->allocate(w, h);

      char const * texSpec = "rgb=16f";

      for (int i = 0; i < 2; ++i)
      {
         _p1Texs[i].allocateID(); _p1Texs[i].reserve(w, h, TextureSpecification(texSpec));
         _p2Texs[i].allocateID(); _p2Texs[i].reserve(w, h, TextureSpecification(texSpec));
         _r1Texs[i].allocateID(); _r1Texs[i].reserve(w, h, TextureSpecification(texSpec));
         _r2Texs[i].allocateID(); _r2Texs[i].reserve(w, h, TextureSpecification(texSpec));
         _s1Texs[i].allocateID(); _s1Texs[i].reserve(w, h, TextureSpecification(texSpec));
         _s2Texs[i].allocateID(); _s2Texs[i].reserve(w, h, TextureSpecification(texSpec));
         _qTexs[i].allocateID(); _qTexs[i].reserve(w, h, TextureSpecification(texSpec));

         _pqFbos[i]->allocate();
         _pqFbos[i]->makeCurrent();
         _pqFbos[i]->attachTexture2D(_p1Texs[i], GL_COLOR_ATTACHMENT0_EXT);
         _pqFbos[i]->attachTexture2D(_p2Texs[i], GL_COLOR_ATTACHMENT1_EXT);
         _pqFbos[i]->attachTexture2D(_qTexs[i], GL_COLOR_ATTACHMENT2_EXT);

         _rsFbos[i]->allocate();
         _rsFbos[i]->makeCurrent();
         _rsFbos[i]->attachTexture2D(_r1Texs[i], GL_COLOR_ATTACHMENT0_EXT);
         _rsFbos[i]->attachTexture2D(_r2Texs[i], GL_COLOR_ATTACHMENT1_EXT);
         _rsFbos[i]->attachTexture2D(_s1Texs[i], GL_COLOR_ATTACHMENT2_EXT);
         _rsFbos[i]->attachTexture2D(_s2Texs[i], GL_COLOR_ATTACHMENT3_EXT);
      } // end for (i)
   } // end ConvexMRF_3Labels_Generic::allocate()

   void
   ConvexMRF_3Labels_Generic::deallocate()
   {
      _uBufs[0]->deallocate();
      _uBufs[1]->deallocate();

      for (int i = 0; i < 2; ++i)
      {
         _p1Texs[i].deallocateID();
         _p2Texs[i].deallocateID();
         _r1Texs[i].deallocateID();
         _r2Texs[i].deallocateID();
         _s1Texs[i].deallocateID();
         _s2Texs[i].deallocateID();
         _qTexs[i].deallocateID();

         _pqFbos[i]->deallocate();
         _rsFbos[i]->deallocate();
      }
   } // end ConvexMRF_3Labels_Generic::deallocate()

   void
   ConvexMRF_3Labels_Generic::setZero()
   {
      glClearColor(0, 0, 0, 0);
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
      _uBufs[0]->activate();
      glClear(GL_COLOR_BUFFER_BIT);
      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
                                 GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT };
      _pqFbos[0]->activate();
      glDrawBuffersARB(1, buffers+2);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+1);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+0);
      glClear(GL_COLOR_BUFFER_BIT);

      _rsFbos[0]->activate();
      glDrawBuffersARB(1, buffers+3);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+2);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+1);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+0);
      glClear(GL_COLOR_BUFFER_BIT);
   } // ConvexMRF_3Labels_Generic::setZero()

//    void
//    ConvexMRF_3Labels_Generic::copyBuffersFrom(ConvexMRF_3Labels_Generic& src)
//    {
//       setupNormalizedProjection();
//       _uBufA->activate();
//       src._uBufA->enableTexture(GL_TEXTURE0);
//       enableTrivialTexture2DShader();
//       renderNormalizedQuad();
//       src._uBufA->disableTexture(GL_TEXTURE0);

//       GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT };
//       _pFboA->activate();

//       glDrawBuffersARB(1, buffers+2);
//       src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
//       renderNormalizedQuad();
//       src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

//       glDrawBuffersARB(1, buffers+1);
//       src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
//       renderNormalizedQuad();
//       src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

//       glDrawBuffersARB(1, buffers+0);
//       src._pFboA->getColorTexture(0).enable(GL_TEXTURE0);
//       renderNormalizedQuad();
//       src._pFboA->getColorTexture(0).disable(GL_TEXTURE0);

//       disableTrivialTexture2DShader();
//    } // end ConvexMRF_3Labels_Generic::copyLabels()

   void
   ConvexMRF_3Labels_Generic::iterate(unsigned int costTexId, int nIterations)
   {
      static Cg_FragmentProgram * uShader = 0;
      static Cg_FragmentProgram * pqShader = 0;
      static Cg_FragmentProgram * rsShader = 0;

      if (uShader == 0)
      {
         uShader = new Cg_FragmentProgram("ConvexMRF_3Labels_Generic::uShader");
         uShader->setProgramFromFile("convex_mrf_3labeling_update_u_generic.cg");

         uShader->compile();
         checkGLErrorsHere0();
      }

      if (pqShader == 0)
      {
         pqShader = new Cg_FragmentProgram("ConvexMRF_3Labels_Generic::pqShader");
         pqShader->setProgramFromFile("convex_mrf_3labeling_update_pq_generic.cg");
         pqShader->compile();
         checkGLErrorsHere0();
      }

      if (rsShader == 0)
      {
         rsShader = new Cg_FragmentProgram("ConvexMRF_3Labels_Generic::rsShader");
         rsShader->setProgramFromFile("convex_mrf_3labeling_update_rs_generic.cg");
         rsShader->compile();
         checkGLErrorsHere0();
      }

      float const ds = 1.0f/_width;
      float const dt = 1.0f/_height;

      setupNormalizedProjection();

      uShader->parameter("tau", _tau_primal);
      pqShader->parameter("tau", _tau_dual);
      pqShader->parameter("c1", _c1);
      rsShader->parameter("tau", _tau_dual);
      rsShader->parameter("c2", _c2);
      checkGLErrorsHere0();

      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
                                 GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT };

      for (int iter = 0; iter < nIterations; ++iter)
      {
         // Update u
         //cout << "u..." << endl;
         _uBufs[1]->activate();
         glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

         _uBufs[0]->enableTexture(GL_TEXTURE0);
         _pqFbos[0]->getColorTexture(0).enable(GL_TEXTURE1);
         _pqFbos[0]->getColorTexture(1).enable(GL_TEXTURE2);
         _pqFbos[0]->getColorTexture(2).enable(GL_TEXTURE3);
         _rsFbos[0]->getColorTexture(0).enable(GL_TEXTURE4);
         _rsFbos[0]->getColorTexture(1).enable(GL_TEXTURE5);
         _rsFbos[0]->getColorTexture(2).enable(GL_TEXTURE6);
         _rsFbos[0]->getColorTexture(3).enable(GL_TEXTURE7);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBufs[0]->disableTexture(GL_TEXTURE0);
         _rsFbos[0]->getColorTexture(0).disable(GL_TEXTURE4);
         _rsFbos[0]->getColorTexture(1).disable(GL_TEXTURE5);
         _rsFbos[0]->getColorTexture(2).disable(GL_TEXTURE6);
         _rsFbos[0]->getColorTexture(3).disable(GL_TEXTURE7);

         std::swap(_uBufs[0], _uBufs[1]);

         // Update p and q
         //cout << "p,q..." << endl;
         _pqFbos[1]->activate();
         glDrawBuffersARB(3, buffers);
         _uBufs[0]->enableTexture(GL_TEXTURE0);

         glActiveTexture(GL_TEXTURE4);
         glBindTexture(GL_TEXTURE_2D, costTexId);
         glEnable(GL_TEXTURE_2D);

         pqShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         pqShader->disable();

         _uBufs[0]->disableTexture(GL_TEXTURE0);
         _pqFbos[0]->getColorTexture(0).disable(GL_TEXTURE1);
         _pqFbos[0]->getColorTexture(1).disable(GL_TEXTURE2);
         _pqFbos[0]->getColorTexture(2).disable(GL_TEXTURE3);

         glActiveTexture(GL_TEXTURE4);
         glDisable(GL_TEXTURE_2D);

         std::swap(_pqFbos[0], _pqFbos[1]);

         // Update r and s
         //cout << "r,s..." << endl;
         _rsFbos[1]->activate();
         glDrawBuffersARB(4, buffers);
         _uBufs[0]->enableTexture(GL_TEXTURE0);
         _rsFbos[0]->getColorTexture(0).enable(GL_TEXTURE1);
         _rsFbos[0]->getColorTexture(1).enable(GL_TEXTURE2);
         _rsFbos[0]->getColorTexture(2).enable(GL_TEXTURE3);
         _rsFbos[0]->getColorTexture(3).enable(GL_TEXTURE4);

         rsShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         rsShader->disable();

         _uBufs[0]->disableTexture(GL_TEXTURE0);
         _rsFbos[0]->getColorTexture(0).disable(GL_TEXTURE1);
         _rsFbos[0]->getColorTexture(1).disable(GL_TEXTURE2);
         _rsFbos[0]->getColorTexture(2).disable(GL_TEXTURE3);
         _rsFbos[0]->getColorTexture(3).disable(GL_TEXTURE4);

         std::swap(_rsFbos[0], _rsFbos[1]);
      } // end for (iter)

      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
   } // end ConvexMRF_3Labels_Generic::iterate()

//**********************************************************************

   void
   ConvexMRF_3Labels_Opt::allocate(int w, int h)
   {
      _width = w;
      _height = h;

      _uBufs[0]->allocate(w, h);
      _uBufs[1]->allocate(w, h);
      _rsBufs[0]->allocate(w, h);
      _rsBufs[1]->allocate(w, h);

      char const * texSpec = "rgb=16f";

      for (int i = 0; i < 2; ++i)
      {
         _p1Texs[i].allocateID(); _p1Texs[i].reserve(w, h, TextureSpecification(texSpec));
         _p2Texs[i].allocateID(); _p2Texs[i].reserve(w, h, TextureSpecification(texSpec));
         _qTexs[i].allocateID(); _qTexs[i].reserve(w, h, TextureSpecification(texSpec));

         _pqFbos[i]->allocate();
         _pqFbos[i]->makeCurrent();
         _pqFbos[i]->attachTexture2D(_p1Texs[i], GL_COLOR_ATTACHMENT0_EXT);
         _pqFbos[i]->attachTexture2D(_p2Texs[i], GL_COLOR_ATTACHMENT1_EXT);
         _pqFbos[i]->attachTexture2D(_qTexs[i], GL_COLOR_ATTACHMENT2_EXT);
      } // end for (i)
   } // end ConvexMRF_3Labels_Opt::allocate()

   void
   ConvexMRF_3Labels_Opt::deallocate()
   {
      _uBufs[0]->deallocate();
      _uBufs[1]->deallocate();
      _rsBufs[0]->deallocate();
      _rsBufs[1]->deallocate();

      for (int i = 0; i < 2; ++i)
      {
         _p1Texs[i].deallocateID();
         _p2Texs[i].deallocateID();
         _qTexs[i].deallocateID();
         _pqFbos[i]->deallocate();
      }
   } // end ConvexMRF_3Labels_Opt::deallocate()

   void
   ConvexMRF_3Labels_Opt::setZero()
   {
      glClearColor(0, 0, 0, 0);
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
      _uBufs[0]->activate();
      glClear(GL_COLOR_BUFFER_BIT);

      _rsBufs[0]->activate();
      glClear(GL_COLOR_BUFFER_BIT);

      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
                                 GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT };
      _pqFbos[0]->activate();
      glDrawBuffersARB(1, buffers+2);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+1);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawBuffersARB(1, buffers+0);
      glClear(GL_COLOR_BUFFER_BIT);
   } // ConvexMRF_3Labels_Opt::setZero()

   void
   ConvexMRF_3Labels_Opt::iterate(unsigned int costTexId, int nIterations)
   {
      static Cg_FragmentProgram * uShader = 0;
      static Cg_FragmentProgram * pqShader = 0;
      static Cg_FragmentProgram * rsShader = 0;

      if (uShader == 0)
      {
         uShader = new Cg_FragmentProgram("ConvexMRF_3Labels_Opt::uShader");
         uShader->setProgramFromFile("convex_mrf_3labeling_update_u_opt.cg");

         uShader->compile();
         checkGLErrorsHere0();
      }

      if (pqShader == 0)
      {
         pqShader = new Cg_FragmentProgram("ConvexMRF_3Labels_Opt::pqShader");
         pqShader->setProgramFromFile("convex_mrf_3labeling_update_pq_opt.cg");
         pqShader->compile();
         checkGLErrorsHere0();
      }

      if (rsShader == 0)
      {
         rsShader = new Cg_FragmentProgram("ConvexMRF_3Labels_Opt::rsShader");
         rsShader->setProgramFromFile("convex_mrf_3labeling_update_rs_opt.cg");
         rsShader->compile();
         checkGLErrorsHere0();
      }

      float const ds = 1.0f/_width;
      float const dt = 1.0f/_height;

      setupNormalizedProjection();

      uShader->parameter("tau", _tau_primal);
      pqShader->parameter("tau", _tau_dual);
      pqShader->parameter("c1", _c1);
      pqShader->parameter("c2", _c2);
      rsShader->parameter("tau", _tau_dual);
      rsShader->parameter("c2", _c2);
      checkGLErrorsHere0();

      GLenum const buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
                                 GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT };

      for (int iter = 0; iter < nIterations; ++iter)
      {
         // Update u
         //cout << "u..." << endl;
         _uBufs[1]->activate();
         glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

         _uBufs[0]->enableTexture(GL_TEXTURE0);
         _pqFbos[0]->getColorTexture(0).enable(GL_TEXTURE1);
         _pqFbos[0]->getColorTexture(1).enable(GL_TEXTURE2);
         _pqFbos[0]->getColorTexture(2).enable(GL_TEXTURE3);
         _rsBufs[0]->enableTexture(GL_TEXTURE4);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBufs[0]->disableTexture(GL_TEXTURE0);
         _rsBufs[0]->disableTexture(GL_TEXTURE4);

         std::swap(_uBufs[0], _uBufs[1]);

         // Update p and q
         //cout << "p,q..." << endl;
         _pqFbos[1]->activate();
         glDrawBuffersARB(3, buffers);
         _uBufs[0]->enableTexture(GL_TEXTURE0);

         glActiveTexture(GL_TEXTURE4);
         glBindTexture(GL_TEXTURE_2D, costTexId);
         glEnable(GL_TEXTURE_2D);

         pqShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         pqShader->disable();

         _uBufs[0]->disableTexture(GL_TEXTURE0);
         _pqFbos[0]->getColorTexture(0).disable(GL_TEXTURE1);
         _pqFbos[0]->getColorTexture(1).disable(GL_TEXTURE2);
         _pqFbos[0]->getColorTexture(2).disable(GL_TEXTURE3);

         glActiveTexture(GL_TEXTURE4);
         glDisable(GL_TEXTURE_2D);

         std::swap(_pqFbos[0], _pqFbos[1]);

         // Update r and s
         //cout << "r,s..." << endl;
         _rsBufs[1]->activate();
         glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
         _uBufs[0]->enableTexture(GL_TEXTURE0);
         _rsBufs[0]->enableTexture(GL_TEXTURE1);

         rsShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         rsShader->disable();

         _uBufs[0]->disableTexture(GL_TEXTURE0);
         _rsBufs[0]->disableTexture(GL_TEXTURE1);

         std::swap(_rsBufs[0], _rsBufs[1]);
      } // end for (iter)
   } // end ConvexMRF_3Labels_Opt::iterate()

} // end namespace V3D_GPU

#endif
