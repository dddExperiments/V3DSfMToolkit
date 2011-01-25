#include "Base/v3d_timer.h"
#include "Base/v3d_utilities.h"
#include "Math/v3d_sparseeig.h"
#include "Math/v3d_optimization.h"
#include "Geometry/v3d_poseutilities.h"
#include "Geometry/v3d_mviewinitialization.h"

#include "reconstruction_common.h"

#include <list>

using namespace std;
using namespace V3D;

//----------------------------------------------------------------------

CalibrationDatabase::CalibrationDatabase(char const * calibDbName)
{
   ifstream is(calibDbName);
   if (!is) throwV3DErrorHere("Could not open calibration database file.");

   int nViews;
   is >> nViews;
   _imageDimensions.resize(nViews);
   _intrinsics.resize(nViews);
   _distortions.resize(nViews);

   for (int i = 0; i < nViews; ++i)
   {
      Matrix3x3d& K = _intrinsics[i];
      makeIdentityMatrix(K);
      is >> K[0][0] >> K[0][1] >> K[0][2];
      is >> K[1][1] >> K[1][2];

      is >> _distortions[i][0] >> _distortions[i][1] >> _distortions[i][2] >> _distortions[i][3];
      is >> _imageDimensions[i].first >> _imageDimensions[i].second;
   } // end for (i)
} // end CalibrationDatabase::CalibrationDatabase()

//----------------------------------------------------------------------

namespace
{

   inline int
   getShuffleCount(int N, int nRounds, int minSampleSize)
   {
      return std::max(1, (int)((float)nRounds*(float)minSampleSize/(float)N + 0.5));
   }

} // end namespace <>

namespace V3D
{

   void
   computeRobustOrientationMLE(std::vector<PointCorrespondence> const& corrs,
                               Matrix3x3d const& K1, Matrix3x3d const& K2,
                               double inlierThreshold, int nSamples,
                               RobustOrientationResult& res,
                               bool reportInliers, RobustOrientationMode mode)
   {
      res.inliers.clear();

      int const N = corrs.size();
      int const minSize = 5;

      double const sqrInlierThreshold = inlierThreshold*inlierThreshold;

      if (N < minSize) throwV3DErrorHere("At least 5 point correspondences are required.");

      Matrix3x3d const invK1   = invertedMatrix(K1);
      Matrix3x3d const invK2   = invertedMatrix(K2);
      Matrix3x3d const invK2_t = invK2.transposed();

      vector<Matrix3x3d> Es;
      Matrix3x3d bestEssential;

      vector<PointCorrespondence> normalizedCorrs(corrs);

      vector<Vector2d> leftSamples, rightSamples;

      // normalize points by multiplication with inverse affine matrix
      for (int i = 0; i < N; ++i)
      {
         multiply_A_v_projective(invK1, corrs[i].left.pos, normalizedCorrs[i].left.pos);
         multiply_A_v_projective(invK2, corrs[i].right.pos, normalizedCorrs[i].right.pos);
      }

      unsigned int const nShuffles = getShuffleCount(N, nSamples, minSize);

      double minError = 1e30;

      double outlierFraction = 1.0;

      // vector to hold the indices of the sample points
      vector<unsigned int> indices(N);
      for (int i = 0; i < N; ++i) indices[i] = i;

      double x1[5], y1[5], x2[5], y2[5];

      vector<int> inliers;
      if (mode.iterativeRefinement) inliers.resize(N);

      unsigned int drawnSamples = 0;      
      for (unsigned int s = 0; s < nShuffles; ++s)
      {
         if (drawnSamples > nSamples) break;
         // shuffle indices 
         random_shuffle(indices.begin(), indices.end());

         for (int j = 0; j < N-minSize; j += minSize)
         {
            if (drawnSamples > nSamples) break;
            Es.clear();

            x1[0] = normalizedCorrs[indices[j+0]].left.pos[0]; y1[0] = normalizedCorrs[indices[j+0]].left.pos[1];
            x1[1] = normalizedCorrs[indices[j+1]].left.pos[0]; y1[1] = normalizedCorrs[indices[j+1]].left.pos[1];
            x1[2] = normalizedCorrs[indices[j+2]].left.pos[0]; y1[2] = normalizedCorrs[indices[j+2]].left.pos[1];
            x1[3] = normalizedCorrs[indices[j+3]].left.pos[0]; y1[3] = normalizedCorrs[indices[j+3]].left.pos[1];
            x1[4] = normalizedCorrs[indices[j+4]].left.pos[0]; y1[4] = normalizedCorrs[indices[j+4]].left.pos[1];

            x2[0] = normalizedCorrs[indices[j+0]].right.pos[0]; y2[0] = normalizedCorrs[indices[j+0]].right.pos[1];
            x2[1] = normalizedCorrs[indices[j+1]].right.pos[0]; y2[1] = normalizedCorrs[indices[j+1]].right.pos[1];
            x2[2] = normalizedCorrs[indices[j+2]].right.pos[0]; y2[2] = normalizedCorrs[indices[j+2]].right.pos[1];
            x2[3] = normalizedCorrs[indices[j+3]].right.pos[0]; y2[3] = normalizedCorrs[indices[j+3]].right.pos[1];
            x2[4] = normalizedCorrs[indices[j+4]].right.pos[0]; y2[4] = normalizedCorrs[indices[j+4]].right.pos[1];

            Es.clear();
            try
            {
               computeEssentialsFromFiveCorrs(x1, y1, x2, y2, Es);
            }
            catch (std::exception exn)
            {
               Es.clear();
               cerr << "Exception caught from computeEssentialsFromFiveCorrs(): " << exn.what() << endl;
            }
            catch (std::string s)
            {
               Es.clear();
               cerr << "Exception caught from computeEssentialsFromFiveCorrs(): " << s << endl;
            }
            catch (...)
            {
               Es.clear();
               cerr << "Unknown exception from computeEssentialsFromFiveCorrs()." << endl;
            }

            ++drawnSamples;

            for (int r = 0; r < Es.size(); ++r)
            {
               Matrix3x3d fund = invK2_t * Es[r] * invK1;

               int nInliers = 0;
               double curError = 0;
               for (int i = 0; i < N; ++i)
               {
                  double const dist = sampsonEpipolarError(corrs[i], fund);
                  curError += std::min(dist, sqrInlierThreshold);
                  if (dist < sqrInlierThreshold) ++nInliers;
               }

               if (curError < minError)
               {
                  Matrix3x3d R;
                  Vector3d t;
                  bool const status = relativePoseFromEssential(Es[r], 5, x1, y1, x2, y2, R, t);
                  if (!status) continue;

                  if (mode.iterativeRefinement)
                  {
                     Matrix3x3d const R_orig = R;
                     Vector3d const t_orig = t;

                     /// get inlier index
                     inliers.clear();
                     for (int i = 0; i < N; ++i)
                     {
                        double const dist = sampsonEpipolarError(corrs[i], fund);
                        if (dist < sqrInlierThreshold) inliers.push_back(i);
                     }

                     vector<Vector2d> left(inliers.size());
                     vector<Vector2d> right(inliers.size());

                     for (int i = 0; i < inliers.size(); ++i)
                     {
                        int const ix = inliers[i];
                        left[i][0] = corrs[ix].left.pos[0];
                        left[i][1] = corrs[ix].left.pos[1];
                        right[i][0] = corrs[ix].right.pos[0];
                        right[i][1] = corrs[ix].right.pos[1];
                     }

                     Matrix3x3d const E_orig = computeEssentialFromRelativePose(R, t);

                     refineRelativePose(left, right, K1, K2, R, t, inlierThreshold);

                     Matrix3x3d E_refined = computeEssentialFromRelativePose(R, t);
                     Matrix3x3d F_refined = invK2_t * E_refined * invK1;

                     int const nInliers_orig = nInliers;

                     /// re-estimate inliers/error
                     double curError_refined = 0;
                     nInliers = 0;
                     for (int i = 0; i < N; ++i)
                     {
                        double const dist = sampsonEpipolarError(corrs[i], F_refined);
                        curError_refined += std::min(dist, sqrInlierThreshold);
                        if (dist < sqrInlierThreshold) ++nInliers;
                     }

                     // if (curError_refined > curError)
                     // {
                     //    cout << "computeRobustOrientationMLE(): curError_refined (" << curError_refined
                     //         << ") > curError (" << curError << ")." << endl;
                     //    cout << "inliers.size() = " << inliers.size() << endl;
                     //    cout << "nInliers_orig = " << nInliers_orig << ", nInliers = " << nInliers << endl;
                     //    cout << "R_orig = "; displayMatrix(R_orig);
                     //    cout << "R_refined = "; displayMatrix(R);
                     //    cout << "t_orig = "; displayVector(t_orig);
                     //    cout << "t_refined = "; displayVector(t);
                     // }

                     fund  = F_refined;
                     Es[r] = E_refined;
                     curError = std::min(curError, curError_refined);
                  } // end if (mode.iterativeRefinement)

                  minError        = curError;
                  res.error       = minError / N;
                  res.essential   = Es[r];
                  res.fundamental = fund;
                  res.rotation    = R;
                  res.translation = t;

                  /// adaptive number of samples computation 
                  outlierFraction = 1.0 - float(nInliers - minSize) / float(corrs.size() - minSize);
                  nSamples = std::min(nSamples, ransacNSamples(minSize, outlierFraction, 1.0 - 10e-10));
               } // end if (curError < minError)
            } // end for (r)
         } // end for (j)
      } // end for (s)

      if (reportInliers)
      {
         res.inliers.reserve(N);
         for (int i = 0; i < N; ++i)
         {
            double const dist = sampsonEpipolarError(corrs[i], res.fundamental);
            if (dist < sqrInlierThreshold) res.inliers.push_back(i);
         }
      } // end if (reportInliers)
   } // end computeRobustOrientationMLE()

//**********************************************************************

   void
   computeScaleRatios(float cosAngleThreshold,
                      Matrix3x3d const& R01, Vector3d const& T01,
                      Matrix3x3d const& R12, Vector3d const& T12,
                      Matrix3x3d const& R20, Vector3d const& T20,
                      std::vector<PointCorrespondence> const& corrs01,
                      std::vector<PointCorrespondence> const& corrs12,
                      std::vector<PointCorrespondence> const& corrs02,
                      double& s012, double& s120, double& s201, double& weight)
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

//       for (int j = 0; j < model.size(); ++j)
//       {
//          cout << "X" << j << ": ";
//          TriangulatedPoint& X = model[j];
//          for (int l = 0; l < X.measurements.size(); ++l)
//          {
//             PointMeasurement const& m = X.measurements[l];
//             cout << "(" << m.view << ": " << m.id << ") ";
//          }
//          cout << endl;
//       }

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
               foundView0 = true;
               m0 = ms[l];
            }
            else if (ms[l].view == view1)
            {
               foundView1 = true;
               m1 = ms[l];
            }
            else if (ms[l].view == view2)
            {
               foundView2 = true;
               m2 = ms[l];
            }
         }

         if (!foundView0) continue;
         if (!foundView1) continue;
         if (!foundView2) continue;

         // Found a point visible in all 3 views.

         // Check, if the pairwise triangulation angles are sufficient
         Vector3d ray0, ray1;
         ray0[0] = m0.pos[0]; ray0[1] = m0.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam01.getRay(m1.pos);
         if (innerProduct(ray0, ray1) > cosAngleThreshold) continue;

         ray0[0] = m1.pos[0]; ray0[1] = m1.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam12.getRay(m2.pos);
         if (innerProduct(ray0, ray1) > cosAngleThreshold) continue;

         ray0[0] = m2.pos[0]; ray0[1] = m2.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam20.getRay(m0.pos);
         if (innerProduct(ray0, ray1) > cosAngleThreshold) continue;

         // Compute scale ratios
         PointCorrespondence corr01, corr12, corr20;

         corr01.left = m0; corr01.right = m1;
         corr12.left = m1; corr12.right = m2;
         corr20.left = m2; corr20.right = m0;

         Vector3d X01 = triangulateLinear(P0, P01, corr01);
         if (X01[2] <= 0.0) continue;
         Vector3d X12 = triangulateLinear(P0, P12, corr12);
         if (X12[2] <= 0.0) continue;
         Vector3d X20 = triangulateLinear(P0, P20, corr20);
         if (X20[2] <= 0.0) continue;

         ratios012.push_back(distance_L2(X01, cam01.cameraCenter()) / norm_L2(X12));
         ratios120.push_back(distance_L2(X12, cam12.cameraCenter()) / norm_L2(X20));
         ratios201.push_back(distance_L2(X20, cam20.cameraCenter()) / norm_L2(X01));
      } // end for (j)

      //cout << "cam01.cameraCenter() = "; displayVector(cam01.cameraCenter());
      //cout << "cam12.cameraCenter() = "; displayVector(cam12.cameraCenter());
      //cout << "cam20.cameraCenter() = "; displayVector(cam20.cameraCenter());
      //displayVector(ratios012);
      //displayVector(ratios120);
      //displayVector(ratios201);

      //cout << "ratios012.size() = " << ratios012.size() << endl;
      if (ratios012.size() < 10) // There should be probably more tests
      {
         weight = -1.0f;
         return;
      }

      weight = ratios012.size();
      //weight = 1.0f;

      std::sort(ratios012.begin(), ratios012.end());
      std::sort(ratios120.begin(), ratios120.end());
      std::sort(ratios201.begin(), ratios201.end());

      s012 = medianQuantile(ratios012);
      s120 = medianQuantile(ratios120);
      s201 = medianQuantile(ratios201);
   } // end computeScaleRatios()

//**********************************************************************

   void
   computeScaleRatiosGeneralized(Matrix3x3d const& R01, Vector3d const& T01,
                                 Matrix3x3d const& R12, Vector3d const& T12,
                                 Matrix3x3d const& R20, Vector3d const& T20,
                                 std::vector<PointCorrespondence> const& corrs01,
                                 std::vector<PointCorrespondence> const& corrs12,
                                 std::vector<PointCorrespondence> const& corrs02,
                                 double& s012, double& s120, double& s201, double& weight, float const cosAngleThreshold)
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
               foundView0 = true;
               m0 = ms[l];
            }
            else if (ms[l].view == view1)
            {
               foundView1 = true;
               m1 = ms[l];
            }
            else if (ms[l].view == view2)
            {
               foundView2 = true;
               m2 = ms[l];
            }
         }

         if (!foundView0) continue;
         if (!foundView1) continue;
         if (!foundView2) continue;

         // Found a point visible in all 3 views.

         // Check, if the pairwise triangulation angles are sufficient
         Vector3d ray0, ray1;
         ray0[0] = m0.pos[0]; ray0[1] = m0.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam01.getRay(m1.pos);
         bool const goodAngle01 = (innerProduct(ray0, ray1) < cosAngleThreshold);

         ray0[0] = m1.pos[0]; ray0[1] = m1.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam12.getRay(m2.pos);
         bool const goodAngle12 = (innerProduct(ray0, ray1) < cosAngleThreshold);

         ray0[0] = m2.pos[0]; ray0[1] = m2.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam20.getRay(m0.pos);
         bool const goodAngle20 = (innerProduct(ray0, ray1) < cosAngleThreshold);

         int nGoodAngles = 0;
         if (goodAngle01) ++nGoodAngles;
         if (goodAngle12) ++nGoodAngles;
         if (goodAngle20) ++nGoodAngles;
         if (nGoodAngles < 2) continue;

         // Compute scale ratios
         PointCorrespondence corr01, corr12, corr20;

         corr01.left = m0; corr01.right = m1;
         corr12.left = m1; corr12.right = m2;
         corr20.left = m2; corr20.right = m0;

         // Initialize 3D points as very distance to handle close-to panoramic image pairs correctly
         Vector3d X01(0.0, 0.0, 1e6), X12(0.0, 0.0, 1e6), X20(0.0, 0.0, 1e6);

         if (goodAngle01) X01 = triangulateLinear(P0, P01, corr01);
         if (X01[2] <= 0.0) continue;
         if (goodAngle12) X12 = triangulateLinear(P0, P12, corr12);
         if (X12[2] <= 0.0) continue;
         if (goodAngle20) X20 = triangulateLinear(P0, P20, corr20);
         if (X20[2] <= 0.0) continue;

         ratios012.push_back(distance_L2(X01, cam01.cameraCenter()) / norm_L2(X12));
         ratios120.push_back(distance_L2(X12, cam12.cameraCenter()) / norm_L2(X20));
         ratios201.push_back(distance_L2(X20, cam20.cameraCenter()) / norm_L2(X01));
      } // end for (j)

      //cout << "cam01.cameraCenter() = "; displayVector(cam01.cameraCenter());
      //cout << "cam12.cameraCenter() = "; displayVector(cam12.cameraCenter());
      //cout << "cam20.cameraCenter() = "; displayVector(cam20.cameraCenter());
      //displayVector(ratios012);
      //displayVector(ratios120);
      //displayVector(ratios201);

      //cout << "ratios012.size() = " << ratios012.size() << endl;
      if (ratios012.size() < 10) // There should be probably more tests
      {
         weight = -1.0f;
         return;
      }

      weight = ratios012.size();
      //weight = 1.0f;

      std::sort(ratios012.begin(), ratios012.end());
      std::sort(ratios120.begin(), ratios120.end());
      std::sort(ratios201.begin(), ratios201.end());

      s012 = medianQuantile(ratios012);
      s120 = medianQuantile(ratios120);
      s201 = medianQuantile(ratios201);
   } // end computeScaleRatiosGeneralized()

} // end namespace V3D

//**********************************************************************

void
extractConnectedComponent(std::map<ViewPair, std::set<int> > const& pairThirdViewMap,
                          std::set<ViewTripletKey>& unhandledTriples,
                          std::set<ViewTripletKey>& connectedTriples,
                          std::set<ViewPair>& handledEdges)
{
   // Breadth-first search for connected components

   connectedTriples.clear();
   handledEdges.clear();

   list<ViewPair> edgeQueue;

   ViewTripletKey startTriple = *unhandledTriples.begin();
   unhandledTriples.erase(unhandledTriples.begin());
   connectedTriples.insert(startTriple);
   edgeQueue.push_back(ViewPair(startTriple.views[0], startTriple.views[1]));
   edgeQueue.push_back(ViewPair(startTriple.views[0], startTriple.views[2]));
   edgeQueue.push_back(ViewPair(startTriple.views[1], startTriple.views[2]));

   while (!edgeQueue.empty())
   {
      ViewPair curPair = edgeQueue.front();
      edgeQueue.pop_front();

      handledEdges.insert(curPair);

      map<ViewPair, set<int> >::const_iterator p = pairThirdViewMap.find(curPair);
      assert(p != pairThirdViewMap.end());
      set<int> const& thirdViews = (*p).second;

      for (set<int>::const_iterator q = thirdViews.begin(); q != thirdViews.end(); ++q)
      {
         int i0 = curPair.view0;
         int i1 = curPair.view1;
         int i2 = *q;
         sort3(i0, i1, i2);
         ViewTripletKey key(i0, i1, i2);

         if (connectedTriples.find(key) != connectedTriples.end()) continue;
         if (unhandledTriples.find(key) == unhandledTriples.end()) continue;

         connectedTriples.insert(key);
         unhandledTriples.erase(key);

         ViewPair pair01(i0, i1);
         ViewPair pair02(i0, i2);
         ViewPair pair12(i1, i2);

         if (handledEdges.find(pair01) == handledEdges.end()) edgeQueue.push_back(pair01);
         if (handledEdges.find(pair02) == handledEdges.end()) edgeQueue.push_back(pair02);
         if (handledEdges.find(pair12) == handledEdges.end()) edgeQueue.push_back(pair12);
      } // end for (q)
      //cout << "edgeQueue.size() = " << edgeQueue.size() << endl;
   } // end while
} // end computeConntectedComponent()

//**********************************************************************

SubmodelReconstruction::SubmodelReconstruction(std::set<int> const& viewIds,
                                               std::set<ViewTripletKey> const& collectedTriples)
   : _nViews(viewIds.size()), _viewIdBackMap(viewIds.size())
{
   using namespace std;

   // Map view ids to the range [0, N-1]
   for (set<int>::const_iterator p = viewIds.begin(); p != viewIds.end(); ++p)
   {
      int newId = _viewIdMap.size();
      _viewIdMap.insert(make_pair(*p, newId));
      _viewIdBackMap[newId] = *p;
   } // end for (p)

   // Filter relevant triplets
   for (set<ViewTripletKey>::const_iterator p = collectedTriples.begin(); p != collectedTriples.end(); ++p)
   {
      int const v0 = (*p).views[0];
      int const v1 = (*p).views[1];
      int const v2 = (*p).views[2];

      if (viewIds.find(v0) != viewIds.end() &&
          viewIds.find(v1) != viewIds.end() &&
          viewIds.find(v2) != viewIds.end())
      {
         _triplets.push_back(*p);
         _viewPairs.insert(ViewPair(v0, v1));
         _viewPairs.insert(ViewPair(v0, v2));
         _viewPairs.insert(ViewPair(v1, v2));
      }
   } // end for (p)

   for (set<ViewPair>::const_iterator p = _viewPairs.begin(); p != _viewPairs.end(); ++p)
   {
      int const i0 = (*_viewIdMap.find((*p).view0)).second;
      int const i1 = (*_viewIdMap.find((*p).view1)).second;
      _viewPairVecPosMap.insert(make_pair(ViewPair(i0, i1), _viewPairVec.size()));
      _viewPairVec.push_back(ViewPair(i0, i1));
   } // end for (p)
} // end SubmodelReconstruction::SubmodelReconstruction()

void
SubmodelReconstruction::computeConsistentRotations(std::map<ViewPair, V3D::Matrix3x3d> const& relRotations)
{
   using namespace std;
   using namespace V3D;

   cout << "Computing consistent rotations..." << endl;
   {
      vector<Matrix3x3d> tmpRelRotations;
            
      for (set<ViewPair>::const_iterator p = _viewPairs.begin(); p != _viewPairs.end(); ++p)
         tmpRelRotations.push_back(relRotations.find(*p)->second);

      Timer t("computeConsistentRotations()");
      t.start();
      int const method = V3D_CONSISTENT_ROTATION_METHOD_SPARSE_EIG;
      vector<pair<int, int> > viewPairVec(_viewPairVec.size());
      for (size_t i = 0; i < _viewPairVec.size(); ++i)
         viewPairVec[i] = make_pair(_viewPairVec[i].view0, _viewPairVec[i].view1);

      V3D::computeConsistentRotations(_nViews, tmpRelRotations, viewPairVec, _rotations, method);
      t.stop();
      t.print();
   }
   cout << "done." << endl;
} // end SubmodelReconstruction::computeConsistentRotations()

void
SubmodelReconstruction::computeConsistentTranslations_L1(V3D::CachedStorage<TripletDataTable>& tripletDataCache,
                                                         std::map<ViewTripletKey, int> const& tripletOIDMap)
{
   _translations.resize(_nViews);

   int const nViewPairs = _viewPairs.size();

   vector<Vector3d> c_ji, c_jk;
   vector<Vector3i> ijks;

   for (vector<ViewTripletKey>::const_iterator p = _triplets.begin(); p != _triplets.end(); ++p)
   {
      int const v0 = (*p).views[0];
      int const v1 = (*p).views[1];
      int const v2 = (*p).views[2];

      int const i0 = _viewIdMap[v0];
      int const i1 = _viewIdMap[v1];
      int const i2 = _viewIdMap[v2];

      int const oid = tripletOIDMap.find(*p)->second;

      TripleReconstruction const * tripletData = tripletDataCache[oid];

      assert(tripletData->views[0] == v0);
      assert(tripletData->views[1] == v1);
      assert(tripletData->views[2] == v2);

      Matrix3x4d RT01 = getRelativeOrientation(tripletData->orientations[0], tripletData->orientations[1]);
      Matrix3x4d RT02 = getRelativeOrientation(tripletData->orientations[0], tripletData->orientations[2]);
      Matrix3x4d RT12 = getRelativeOrientation(tripletData->orientations[1], tripletData->orientations[2]);

      Matrix3x4d RT10 = getRelativeOrientation(tripletData->orientations[1], tripletData->orientations[0]);
      Matrix3x4d RT20 = getRelativeOrientation(tripletData->orientations[2], tripletData->orientations[0]);
      Matrix3x4d RT21 = getRelativeOrientation(tripletData->orientations[2], tripletData->orientations[1]);

      Vector3d T01 = RT01.col(3), T02 = RT02.col(3), T12 = RT12.col(3);
      Vector3d T10 = RT10.col(3), T20 = RT20.col(3), T21 = RT21.col(3);

      // Recall: c_j - c_i = -R_j^T T_j + R_i^T T_i = -R_j^T (T_j - R_ij T_i) = -R_j^T T_ij
      Vector3d c01 = _rotations[i1].transposed() * (-T01);
      Vector3d c02 = _rotations[i2].transposed() * (-T02);
      Vector3d c12 = _rotations[i2].transposed() * (-T12);

      Vector3d c10 = _rotations[i0].transposed() * (-T10);
      Vector3d c20 = _rotations[i0].transposed() * (-T20);
      Vector3d c21 = _rotations[i1].transposed() * (-T21);

      ijks.push_back(makeVector3(i0, i1, i2)); c_ji.push_back(c10); c_jk.push_back(c12);
      ijks.push_back(makeVector3(i2, i0, i1)); c_ji.push_back(c02); c_jk.push_back(c01);
      ijks.push_back(makeVector3(i1, i2, i0)); c_ji.push_back(c21); c_jk.push_back(c20);
   } // end for (p)

   vector<Vector3d> centers(_nViews);

   {
      MultiViewInitializationParams_BOS params;
      params.verbose = true;
      params.nIterations = 10000;

      Timer t("computeConsistentCameraCenters()");
      t.start();
      //computeConsistentCameraCenters_L1(c_ji, c_jk, ijks, centers, true);
      computeConsistentCameraCenters_L2_BOS(c_ji, c_jk, ijks, centers, params);
      t.stop();
      t.print();
   }

   for (int i = 0; i < _nViews; ++i) _translations[i] = _rotations[i] * (-centers[i]);
} // end SubmodelReconstruction::computeConsistentTranslations_L1()

void
SubmodelReconstruction::generateSparseReconstruction(std::vector<V3D::PointCorrespondence> const& allCorrs)
{
   using namespace std;
   using namespace V3D;

   vector<PointCorrespondence> corrs;
   for (size_t k = 0; k < allCorrs.size(); ++k)
   {
      PointCorrespondence c = allCorrs[k];

      map<int, int>::const_iterator p1 = _viewIdMap.find(c.left.view);
      map<int, int>::const_iterator p2 = _viewIdMap.find(c.right.view);
               

      if (p1 != _viewIdMap.end() && p2 != _viewIdMap.end())
      {
         // Already bring view ids to the [0..N-1] range
         c.left.view = p1->second;
         c.right.view = p2->second;
         corrs.push_back(c);
      }
   } // end for (k)

   _sparseReconstruction.clear();
   int const nRequiredViews = 3;
   TriangulatedPoint::connectTracks(corrs, _sparseReconstruction, nRequiredViews);
   cout << "sparse reconstruction (before logical filtering) has " << _sparseReconstruction.size() << " 3D points." << endl;
   filterConsistentSparsePoints(_sparseReconstruction);
   cout << "sparse reconstruction (after logical filtering) has " << _sparseReconstruction.size() << " 3D points." << endl;

   _cameras.resize(_nViews);
   Matrix3x3d K; makeIdentityMatrix(K);

   for (int i = 0; i < _nViews; ++i)
   {
      _cameras[i].setIntrinsic(K);
      _cameras[i].setRotation(_rotations[i]);
      _cameras[i].setTranslation(_translations[i]);
   }

   for (int i = 0; i < _sparseReconstruction.size(); ++i)
      _sparseReconstruction[i].pos = triangulateLinear(_cameras, _sparseReconstruction[i].measurements);
} // end SubmodelReconstruction::generateSparseReconstruction()
