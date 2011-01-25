#include "Geometry/v3d_tripletutilities.h"

using namespace std;
using namespace V3D;

namespace
{

   inline double
   medianQuantile(std::vector<double> const& vs)
   {
#if 0
      return vs[vs.size()/2]; // Median of local ratios
#else
      // Take the mean of the central 20% quantile
      float avg = 0.0;
      int const radius = std::max(1, int(0.1f * vs.size()));
      for (int k = vs.size()/2-radius; k <= vs.size()/2+radius; ++k) avg += vs[k];
      avg /= (2*radius+1);
      return avg;
#endif
   }

} // end namespace <>

void
V3D::computeTripletLengthRatios(float cosAngleThreshold,
                                Matrix3x3d const& R01, Vector3d const& T01,
                                Matrix3x3d const& R12, Vector3d const& T12,
                                Matrix3x3d const& R20, Vector3d const& T20,
                                std::vector<PointCorrespondence> const& corrs01,
                                std::vector<PointCorrespondence> const& corrs12,
                                std::vector<PointCorrespondence> const& corrs02,
                                double& s012, int& n012, double& s120, int& n120, double& s201, int& n201)
{
   Matrix3x3d I;  makeIdentityMatrix(I);
   Matrix3x4d P0; makeIdentityMatrix(P0);

   CameraMatrix cam01, cam12, cam20;

   int const view0 = corrs01[0].left.view;
   int const view1 = corrs01[0].right.view;
   int const view2 = corrs12[0].right.view;

   cam01.setIntrinsic(I);
   cam12.setIntrinsic(I);
   cam20.setIntrinsic(I);

   cam01.setRotation(R01);
   cam12.setRotation(R12);
   cam20.setRotation(R20);

   cam01.setTranslation(T01);
   cam12.setTranslation(T12);
   cam20.setTranslation(T20);

   Matrix3x4d const P01 = cam01.getProjection();
   Matrix3x4d const P12 = cam12.getProjection();
   Matrix3x4d const P20 = cam20.getProjection();

   std::vector<PointCorrespondence> allCorrs;
   allCorrs.reserve(corrs01.size() + corrs12.size() + corrs02.size());
   for (size_t k = 0; k < corrs01.size(); ++k) allCorrs.push_back(corrs01[k]);
   for (size_t k = 0; k < corrs12.size(); ++k) allCorrs.push_back(corrs12[k]);
   for (size_t k = 0; k < corrs02.size(); ++k) allCorrs.push_back(corrs02[k]);

   std::vector<TriangulatedPoint> model;
   TriangulatedPoint::connectTracks(allCorrs, model, 3);

   vector<double> ratios012, ratios120, ratios201;
   ratios012.reserve(model.size());
   ratios120.reserve(model.size());
   ratios201.reserve(model.size());

   for (size_t j = 0; j < model.size(); ++j)
   {
      vector<PointMeasurement> const& ms = model[j].measurements;
      if (ms.size() != 3) continue;

      bool foundView0 = false;
      bool foundView1 = false;
      bool foundView2 = false;

      PointMeasurement m0, m1, m2;

      for (int l = 0; l < ms.size(); ++l)
      {
         if (ms[l].view == view0)
         {
            foundView0 = true; m0 = ms[l];
         }
         else if (ms[l].view == view1)
         {
            foundView1 = true; m1 = ms[l];
         }
         else if (ms[l].view == view2)
         {
            foundView2 = true; m2 = ms[l];
         }
      } // end for (l)

      if (!foundView0 || !foundView1 || !foundView2) continue;

      // Found a point visible in all 3 views.

      // Check, if the pairwise triangulation angles are sufficient

      int nGood = 0;

      Vector3d ray0, ray1;
      ray0[0] = m0.pos[0]; ray0[1] = m0.pos[1]; ray0[2] = 1.0f;
      normalizeVector(ray0);
      ray1 = cam01.getRay(m1.pos);
      bool good01 = (innerProduct(ray0, ray1) > cosAngleThreshold);
      if (good01) ++nGood;

      ray0[0] = m1.pos[0]; ray0[1] = m1.pos[1]; ray0[2] = 1.0f;
      normalizeVector(ray0);
      ray1 = cam12.getRay(m2.pos);
      bool good12 = (innerProduct(ray0, ray1) > cosAngleThreshold);
      if (good12) ++nGood;

      ray0[0] = m2.pos[0]; ray0[1] = m2.pos[1]; ray0[2] = 1.0f;
      normalizeVector(ray0);
      ray1 = cam20.getRay(m0.pos);
      bool good20 = (innerProduct(ray0, ray1) > cosAngleThreshold);
      if (good20) ++nGood;

      if (nGood < 2) continue;

      // Compute scale ratios
      PointCorrespondence corr01, corr12, corr20;

      corr01.left = m0; corr01.right = m1;
      corr12.left = m1; corr12.right = m2;
      corr20.left = m2; corr20.right = m0;

      Vector3d X01 = triangulateLinear(P0, P01, corr01);
      good01 = good01 && (X01[2] > 0.0);
      Vector3d X12 = triangulateLinear(P0, P12, corr12);
      good12 = good12 && (X12[2] > 0.0);
      Vector3d X20 = triangulateLinear(P0, P20, corr20);
      good20 = good20 && (X20[2] > 0.0);

      if (good01 && good12) ratios012.push_back(distance_L2(X01, cam01.cameraCenter()) / norm_L2(X12));
      if (good12 && good20) ratios120.push_back(distance_L2(X12, cam12.cameraCenter()) / norm_L2(X20));
      if (good20 && good01) ratios201.push_back(distance_L2(X20, cam20.cameraCenter()) / norm_L2(X01));
   } // end for (j)

   std::sort(ratios012.begin(), ratios012.end());
   std::sort(ratios120.begin(), ratios120.end());
   std::sort(ratios201.begin(), ratios201.end());

   s012 = medianQuantile(ratios012);
   s120 = medianQuantile(ratios120);
   s201 = medianQuantile(ratios201);

   n012 = ratios012.size();
   n120 = ratios120.size();
   n201 = ratios201.size();
} // end V3D::computeTripletLengthRatios()
