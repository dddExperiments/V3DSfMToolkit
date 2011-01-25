#include "reconstruction_common.h"

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_timer.h"
#include "Math/v3d_sparseeig.h"
#include "Geometry/v3d_mviewutilities.h"
#include "Geometry/v3d_metricbundle.h"

#include <iostream>
#include <sstream>
#include <list>

using namespace std;
using namespace V3D;

namespace
{

   inline double
   computeDiameter(vector<Vector3d> const& points)
   {
#if 0
      Vector3d bboxMin(1e30, 1e30, 1e30);
      Vector3d bboxMax(-1e30, -1e30, -1e30);
      for (int i = 0; i < points.size(); ++i)
      {
         bboxMin[0] = std::min(bboxMin[0], points[i][0]);
         bboxMin[1] = std::min(bboxMin[1], points[i][1]);
         bboxMin[2] = std::min(bboxMin[2], points[i][2]);
         bboxMax[0] = std::max(bboxMax[0], points[i][0]);
         bboxMax[1] = std::max(bboxMax[1], points[i][1]);
         bboxMax[2] = std::max(bboxMax[2], points[i][2]);
      }
      return norm_L2(bboxMax - bboxMin);
#else
      Vector3d center(0, 0, 0);
      for (int i = 0; i < points.size(); ++i)
         addVectorsIP(points[i], center);
      scaleVectorIP(1.0 / points.size(), center);
      vector<double> dists(points.size());
      for (int i = 0; i < points.size(); ++i)
         dists[i] = sqrNorm_L2(points[i] - center);
      std::sort(dists.begin(), dists.end());
      // take 75% quantile
      return 2*sqrt(3*dists[points.size() / 4]);
#endif
   } // end computeDiameter();

   void
   computeRobustSimilarityTransformation(std::vector<Vector3d> const& left, std::vector<Vector3d> const& right,
                                         double inlierThresholdL, double inlierThresholdR, int nTrials,
                                         Matrix3x3d& R, Vector3d& T, double& scale, std::vector<int> &inliers)
   {
      inliers.clear();

      vector<int> curInliers;
 
      int const N = left.size();
 
      if (N < 3) throwV3DErrorHere("computeRobustSimilarityTransform(): at least 3 point correspondences required.");

      cout << "computeRobustSimilarityTransformation(): N = " << N << endl;

      // vector to hold the indices of the sample points
      vector<Vector3d> ptsLeftTrans(N), ptsRightTrans(N);
      vector<Vector3d> left_pts(3), right_pts(3);

      for (int trial = 0; trial < nTrials; ++trial)
      {
         int j0 = int((double(N) * rand() / (RAND_MAX + 1.0)));
         int j1 = int((double(N) * rand() / (RAND_MAX + 1.0)));
         int j2 = int((double(N) * rand() / (RAND_MAX + 1.0)));

         //cout << "trial = " << trial << " j0 = " << j0 << " j1 = " << j1 << " j2 = " << j2 << endl;

         if (j0 == j1 || j0 == j2 || j1 == j2) continue;

         left_pts[0]  = left[j0];  left_pts[1]  = left[j1];  left_pts[2]  = left[j2];
         right_pts[0] = right[j0]; right_pts[1] = right[j1]; right_pts[2] = right[j2];

         Matrix3x3d R0, R0t;
         Vector3d T0;
         double scale0, rcpScale0;
         getSimilarityTransformation(left_pts, right_pts, R0, T0, scale0);

         rcpScale0 = 1.0 / scale0;
         R0t = R0.transposed();

         for (int i = 0; i < N; ++i) ptsLeftTrans[i]  = scale0 * (R0 * left[i] + T0); 
         for (int i = 0; i < N; ++i) ptsRightTrans[i] = R0t * (rcpScale0 * right[i] - T0);

         curInliers.clear();

         for (int i = 0; i < N; ++i)
         {
            double const distL = distance_L2(left[i], ptsRightTrans[i]);
            double const distR = distance_L2(right[i], ptsLeftTrans[i]);

            if (distL < inlierThresholdL && distR < inlierThresholdR) curInliers.push_back(i);
         } // end for (i)
 
         if (curInliers.size() > inliers.size())
         {
            inliers = curInliers;
            R = R0;
            T = T0;
            scale = scale0;
         }
      } // end for (trial)

      cout << "computeRobustSimilarityTransformation(): inliers.size() = " << inliers.size() << " (out of " << N << " 3d-3d correspondences)" << endl;
   } // end computeRobustSimilarityTransformation()

   inline int
   getViewIdIntersectionSize(std::set<int> const& subcomp1Views, std::set<int> const& subcomp2Views)
   {
      using namespace std;

      int res = 0;

      set<int>::const_iterator p1 = subcomp1Views.begin();
      set<int>::const_iterator p2 = subcomp2Views.begin();

      while (p1 != subcomp1Views.end() && p2 != subcomp2Views.end())
      {
         if (*p1 > *p2)
            ++p2;
         else if (*p1 < *p2)
            ++p1;
         else
         {
            ++res;
            ++p1; ++p2;
         }
      } // end while
      return res;
   } // end getViewIdIntersectionSize()

   inline std::set<int>
   getViewIdIntersection(std::set<int> const& subcomp1Views, std::set<int> const& subcomp2Views)
   {
      using namespace std;

      set<int> res;
      set<int>::const_iterator p1 = subcomp1Views.begin();
      set<int>::const_iterator p2 = subcomp2Views.begin();

      while (p1 != subcomp1Views.end() && p2 != subcomp2Views.end())
      {
         if (*p1 > *p2)
            ++p2;
         else if (*p1 < *p2)
            ++p1;
         else
         {
            res.insert(*p1);
            ++p1; ++p2;
         }
      } // end while
      return res;
   } // end getViewIdIntersection()

   void
   computeTransformationBetweenSubModels(SubmodelReconstruction const& model1, set<int> const& views1,
                                         SubmodelReconstruction const& model2, set<int> const& views2,
                                         Matrix3x3d& R, Vector3d& T, double& scale, double& weight,
                                         int nRequiredCommonPoints = 30, bool verbose = false)
   {
      set<int> const commonViews = getViewIdIntersection(views1, views2);

#if 0
      if (commonViews.size() < 2)
      {
         if (verbose) cout << "Too few common views." << endl;
         weight = 0;
         return;
      }
#else
      if (commonViews.empty())
      {
         weight = 0;
         return;
      }
#endif

      map<pair<int, int>, int> measurementPointMap;

      for (int j = 0; j < model1._sparseReconstruction.size(); ++j)
      {
         TriangulatedPoint const& X1 = model1._sparseReconstruction[j];
         for (int k = 0; k < X1.measurements.size(); ++k)
         {
            PointMeasurement const& m = X1.measurements[k];
            int const view = model1._viewIdBackMap[m.view];
            if (commonViews.find(view) != commonViews.end())
               measurementPointMap.insert(make_pair(make_pair(view, m.id), j));
         } // end for (k)
      } // end for (j)

      vector<Vector3d> Xs1, Xs2;

      for (int j = 0; j < model2._sparseReconstruction.size(); ++j)
      {
         TriangulatedPoint const& X2 = model2._sparseReconstruction[j];
         for (int k = 0; k < X2.measurements.size(); ++k)
         {
            PointMeasurement const& m = X2.measurements[k];
            int const view2 = model2._viewIdBackMap[m.view];

            map<pair<int, int>, int>::const_iterator p = measurementPointMap.find(make_pair(view2, m.id));
            if (p != measurementPointMap.end())
            {
               Xs1.push_back(model1._sparseReconstruction[p->second].pos);
               Xs2.push_back(X2.pos);
               break;
            }
         } // end for (k)
      } // end for (j)

      if (Xs1.size() < nRequiredCommonPoints)
      {
         if (verbose) cout << "Too few 3D-3D correspondences." << endl;
         weight = 0;
         return;
      }

      double const diameter1 = computeDiameter(Xs1);
      double const diameter2 = computeDiameter(Xs2);
      double const inlierThresholdL = 0.02 * diameter1;
      double const inlierThresholdR = 0.02 * diameter2;

      vector<int> inlierIndices;
      //computeRobustSimilarityTransformation(Xs1, Xs2, inlierThresholdL, inlierThresholdR, 100, R, T, scale, inlierIndices);
      computeRobustSimilarityTransformationMLE(Xs1, Xs2, inlierThresholdL, inlierThresholdR, 100,
                                               R, T, scale, inlierIndices);

      if (inlierIndices.size() < nRequiredCommonPoints)
      {
         if (verbose) cout << "Too few inlier 3D-3D correspondences." << endl;
         weight = 0;
         return;
      }

      double const minPointRatio = 0.5;
      if (double(inlierIndices.size()) < minPointRatio * Xs1.size())
      {
         if (verbose) cout << "Rejected because the number of 3D-3D correspondences dropped too much." << endl;
         weight = 0;
         return;
      }

      if (0)
      {
         cout << "diameter(Xs2) = " << computeDiameter(Xs2) << endl;

         vector<double> distances(inlierIndices.size());
         for (size_t i = 0; i < inlierIndices.size(); ++i)
         {
            int const ix = inlierIndices[i];
            Vector3d const& X1 = Xs1[ix];
            Vector3d const& X2 = Xs2[ix];
            Vector3d const XX2 = scale*(R*X1 + T);
            distances[i] = distance_L2(X2, XX2);
         }
         std::sort(distances.begin(), distances.end());
         cout << "d = "; displayVector(distances);
      } // end scope

      vector<Vector3d> inlierXs1(inlierIndices.size());
      vector<Vector3d> inlierXs2(inlierIndices.size());

      for (size_t i = 0; i < inlierIndices.size(); ++i)
      {
         inlierXs1[i] = Xs1[inlierIndices[i]];
         inlierXs2[i] = Xs2[inlierIndices[i]];
      }

//       for (size_t i = 0; i < inlierXs1.size(); ++i)
//       {
//          Vector3d const& X1 = inlierXs1[i];
//          Vector3d const& X2 = inlierXs2[i];
//          cout << "[" << X1[0] << " " << X1[1] << " " << X1[2] << "] <-> [" << X2[0] << " " << X2[1] << " " << X2[2] << "]" << endl;
//       }

      cout << "calling getSimilarityTransformation() with " << inlierXs1.size() << " inliers (out of "
           << Xs1.size() << " 3d-3d correspondences)." << endl;
      getSimilarityTransformation(inlierXs1, inlierXs2, R, T, scale);

      if (scale < 1e-3 || scale > 1e3)
      {
         if (verbose) cout << "Scale ratio is too large." << endl;
         weight = 0;
         return;
      }

      weight = sqrt(double(inlierIndices.size()));
      //weight = 1;

      if (verbose) cout << "Found " << inlierIndices.size() << " 3D-3D correspondences." << endl;
   } // end computeTransformationBetweenSubModels()

} // end namespace <>

int
main(int argc, char * argv[])
{
   if (argc != 2)
   {
      cerr << "Usage: " << argv[0] << " <config file>" << endl;
      return -1;
   }

   try
   {
      cout << "Reading config file..." << endl;
      ConfigurationFile cf(argv[1]);

      int const nRequiredAlignmentPoints = cf.get("REQUIRED_ALIGNMENT_POINTS", 500);

      cout << "done." << endl;
      cout << "Connecting to DBs..." << endl;

      SQLite3_Database submodelsDB("submodels.db");

      SubmodelsTable submodelsTable = submodelsDB.getTable<SubmodelReconstruction>("submodels_data");
      CachedStorage<SubmodelsTable> submodelsCache(submodelsTable, 100);
      cout << "done." << endl;

      int const nAllSubModels = submodelsTable.size();
      cout << "nAllSubModels = " << nAllSubModels << endl;

      vector<Matrix3x3d> allRelRotations; // between submodels
      vector<Vector3d>   allRelTranslations;
      vector<double>     allRelScales;
      vector<double>     allRelWeights;
      vector<int>        allSubModelPairs;

      vector<set<int> > submodelViewSets(nAllSubModels);

      for (int j = 0; j < nAllSubModels; ++j)
      {
         SubmodelReconstruction const& subModel = *submodelsCache[j];
         set<int>& viewSet = submodelViewSets[j];

         for (size_t k = 0; k < subModel._viewIdBackMap.size(); ++k) viewSet.insert(subModel._viewIdBackMap[k]);
      } // end for (j)

      vector<pair<int, int> > putativeAlignments;

      for (int j1 = 0; j1 < nAllSubModels; ++j1)
      {
         set<int> const& views1 = submodelViewSets[j1];

         for (int j2 = j1+1; j2 < nAllSubModels; ++j2)
         {
            set<int> const& views2 = submodelViewSets[j2];
            if (getViewIdIntersectionSize(views1, views2) >= 1)
               putativeAlignments.push_back(make_pair(j1, j2));
         }
      }
      cout << "Found " << putativeAlignments.size() << " potential alignments between submodels." << endl;

      for (size_t k = 0; k < putativeAlignments.size(); ++k)
      {
         int const j1 = putativeAlignments[k].first;
         int const j2 = putativeAlignments[k].second;

         cout << "Determining similarity transform between submodel " << j1 << " and " << j2 << endl;

         // Make a copy, since the pointer might be invalidated by later cache accesses
         SubmodelReconstruction const  subModel1 = *submodelsCache[j1];
         set<int> const& views1 = submodelViewSets[j1];

         SubmodelReconstruction const& subModel2 = *submodelsCache[j2];
         set<int> const& views2 = submodelViewSets[j2];

         Matrix3x3d R_rel;
         Vector3d T_rel;
         double scale, weight;
         computeTransformationBetweenSubModels(subModel1, views1, subModel2, views2, R_rel, T_rel,
                                               scale, weight, nRequiredAlignmentPoints);

         if (weight > 0)
         {
            allRelRotations.push_back(R_rel);
            allRelTranslations.push_back(T_rel);
            allRelScales.push_back(scale);
            allRelWeights.push_back(weight);
            int const pair = j1 + (j2 << 16);
            allSubModelPairs.push_back(pair);
            cout << "weight = " << weight << ", scale = " << scale << endl;
         }
      } // end for (k)

      ofstream os("subcomp_alignment.bin", ios::binary);
      BinaryOStreamArchive ar(os);

      serializeVector(allRelRotations, ar);
      serializeVector(allRelTranslations, ar);
      serializeVector(allRelScales, ar);
      serializeVector(allRelWeights, ar);
      serializeVector(allSubModelPairs, ar);
   }
   catch (std::string s)
   {
      cerr << "Exception caught: " << s << endl;
   }
   catch (std::exception exn)
   {
      cerr << "Exception caught: " << exn.what() << endl;
   }
   catch (...)
   {
      cerr << "Unhandled exception." << endl;
   }

   return 0;
} // end main()
