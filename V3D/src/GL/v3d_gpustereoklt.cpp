#include "v3d_gpustereoklt.h"

#include <GL/glew.h>

#include <algorithm>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace V3D_GPU;


void
KLT_StereoTracker::allocate(int width, int height, int pointsWidth, int pointsHeight)
{
   _width = width;
   _height = height;
   _featureWidth = pointsWidth;
   _featureHeight = pointsHeight;

   _pointsBufferA.allocate(pointsWidth, pointsHeight);
   _pointsBufferB.allocate(pointsWidth, pointsHeight);
   checkGLErrorsHere0();

   _featuresBuffer0 = &_pointsBufferA;
   _featuresBuffer1 = &_pointsBufferB;
}

void
KLT_StereoTracker::deallocate()
{
   checkGLErrorsHere0();
   _pointsBufferA.deallocate();
   _pointsBufferB.deallocate();
   checkGLErrorsHere0();
}

void
KLT_StereoTracker::setProjections(V3D::Matrix3x4d const& leftP, V3D::Matrix3x4d const& rightP)
{
   // Transform the given projection matrices to texture coordinate space.
   V3D::Matrix3x3d S;
   V3D::makeIdentityMatrix(S);

   S[0][0] = 1.0/_width;
   S[1][1] = 1.0/_height;
   S[0][2] = 0.5/_width;
   S[1][2] = 0.5/_height;

   V3D::multiply_A_B(S, leftP, _leftP);
   V3D::multiply_A_B(S, rightP, _rightP);
}

void
KLT_StereoTracker::providePoints(V3D::Vector4f const * points)
{
//    if (_featuresBuffer1->isCurrent())
//       FrameBufferObject::disableFBORendering();

   _featuresBuffer1->bindTexture();
   glTexSubImage2D(_featuresBuffer1->textureTarget(),
                   0, 0, 0, _featureWidth, _featureHeight,
                   GL_RGBA, GL_FLOAT, points);
   glBindTexture(_featuresBuffer1->textureTarget(), 0);
}

void
KLT_StereoTracker::readPoints(V3D::Vector4f * points)
{
   _featuresBuffer1->activate();
   glReadPixels(0, 0, _featureWidth, _featureHeight, GL_RGBA, GL_FLOAT, points);
}

void
KLT_StereoTracker::trackPoints(unsigned int pyrTexLeft0, unsigned int pyrTexRight0,
                               unsigned int pyrTexLeft1, unsigned int pyrTexRight1)
{
   if (_trackingShader == 0)
   {
      _trackingShader = new Cg_FragmentProgram("KLT_StereoTracker::_trackingShader");
      _trackingShader->setProgramFromFile("klt_stereotracker.cg");

      vector<string> args;
      //args.push_back("-unroll"); args.push_back("all");
      //char const * args[] = { "-profile", "gp4fp", 0 };
      char str[512];
      sprintf(str, "-DNITERATIONS=%i", _nIterations);
      args.push_back(str);
      sprintf(str, "-DN_LEVELS=%i", _nLevels);
      args.push_back(str);
      sprintf(str, "-DLEVEL_SKIP=%i", _levelSkip);
      args.push_back(str);
      sprintf(str, "-DHALF_WIDTH=%i", _windowWidth/2);
      args.push_back(str);
      _trackingShader->compile(args);
      //cout << shader->getCompiledString() << endl;
      checkGLErrorsHere0();
   } // end if (shader == 0)

   setupNormalizedProjection();
   _featuresBuffer1->activate();

   float const ds = 1.0f / _width;
   float const dt = 1.0f / _height;

   _featuresBuffer0->enableTexture(GL_TEXTURE0_ARB);

   glActiveTexture(GL_TEXTURE1_ARB);
   glBindTexture(GL_TEXTURE_2D, pyrTexLeft0);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
   glEnable(GL_TEXTURE_2D);

   glActiveTexture(GL_TEXTURE2_ARB);
   glBindTexture(GL_TEXTURE_2D, pyrTexRight0);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
   glEnable(GL_TEXTURE_2D);

   glActiveTexture(GL_TEXTURE3_ARB);
   glBindTexture(GL_TEXTURE_2D, pyrTexLeft1);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
   glEnable(GL_TEXTURE_2D);

   glActiveTexture(GL_TEXTURE4_ARB);
   glBindTexture(GL_TEXTURE_2D, pyrTexRight1);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
   glEnable(GL_TEXTURE_2D);

   _trackingShader->parameter("ds", ds, dt);
   _trackingShader->parameter("wh", _width, _height);
   _trackingShader->parameter("sqrConvergenceThreshold", _convergenceThreshold*_convergenceThreshold);
   _trackingShader->parameter("SSD_Threshold", _SSD_Threshold);
   _trackingShader->parameter("validRegion", _margin/_width, _margin/_height,
                              1.0f - _margin/_width, 1.0f - _margin/_height);
   _trackingShader->matrixParameterR("PL", 3, 4, &_leftP[0][0]);
   _trackingShader->matrixParameterR("PR", 3, 4, &_rightP[0][0]);
   _trackingShader->enable();
   renderNormalizedQuad();
   _trackingShader->disable();

   _featuresBuffer0->disableTexture(GL_TEXTURE0_ARB);
   glActiveTexture(GL_TEXTURE1_ARB);
   glDisable(GL_TEXTURE_2D);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glActiveTexture(GL_TEXTURE2_ARB);
   glDisable(GL_TEXTURE_2D);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glActiveTexture(GL_TEXTURE3_ARB);
   glDisable(GL_TEXTURE_2D);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glActiveTexture(GL_TEXTURE4_ARB);
   glDisable(GL_TEXTURE_2D);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
} // end KLT_StereoTracker::trackPoints()
