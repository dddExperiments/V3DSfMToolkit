#include "reconstruction_common.h"
#include "inference_common.h"

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_timer.h"
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
   } // end computeDiameter();

   inline bool
   areConnectedTriplets(ViewTripletKey const& t1, ViewTripletKey const& t2)
   {
      int nOverlappingViews = 0;
      if (t1.views[0] == t2.views[0] || t1.views[0] == t2.views[1] || t1.views[0] == t2.views[2]) ++nOverlappingViews;
      if (t1.views[1] == t2.views[0] || t1.views[1] == t2.views[1] || t1.views[1] == t2.views[2]) ++nOverlappingViews;
      if (t1.views[2] == t2.views[0] || t1.views[2] == t2.views[1] || t1.views[2] == t2.views[2]) ++nOverlappingViews;

      return nOverlappingViews >= 2;
      //return nOverlappingViews >= 1;
   }

//    inline std::vector<int>
//    getViewIdIntersection(TripleReconstruction const& t1, TripleReconstruction const& t2)
//    {
//       vector<int> res;
//       if (t1.views[0] == t2.views[0] || t1.views[0] == t2.views[1] || t1.views[0] == t2.views[2]) res.push_back(t1.views[0]);
//       if (t1.views[1] == t2.views[0] || t1.views[1] == t2.views[1] || t1.views[1] == t2.views[2]) res.push_back(t1.views[1]);
//       if (t1.views[2] == t2.views[0] || t1.views[2] == t2.views[1] || t1.views[2] == t2.views[2]) res.push_back(t1.views[2]);
//       return res;
//    } // end getViewIdIntersection()

   inline std::set<int>
   getViewIdIntersection(TripleReconstruction const& t1, TripleReconstruction const& t2)
   {
      set<int> res;
      if (t1.views[0] == t2.views[0] || t1.views[0] == t2.views[1] || t1.views[0] == t2.views[2]) res.insert(t1.views[0]);
      if (t1.views[1] == t2.views[0] || t1.views[1] == t2.views[1] || t1.views[1] == t2.views[2]) res.insert(t1.views[1]);
      if (t1.views[2] == t2.views[0] || t1.views[2] == t2.views[1] || t1.views[2] == t2.views[2]) res.insert(t1.views[2]);
      return res;
   } // end getViewIdIntersection()

   void
   computeRobustSimilarityTransformation(CameraMatrix const& leftCam, CameraMatrix const& rightCam,
                                         std::vector<Vector3d> const& Xs1, std::vector<Vector3d> const& Xs2,
                                         std::vector<Vector2d> const& left2, std::vector<Vector2d> const& right2,
                                         double inlierThreshold, int nTrials,
                                         Matrix3x3d& R, Vector3d& T, double& scale, std::vector<int> &inliers)
   {
      inliers.clear();
 
      int const N = Xs1.size();
 
      double const inlierThreshold2 = inlierThreshold*inlierThreshold;

      if (N < 3) throwV3DErrorHere("computeRobustSimilarityTransform(): at least 3 point correspondences required.");

      //cout << "computeRobustSimilarityTransformation(): N = " << N << endl;

      int bestInlierCount = 0;

      // vector to hold the indices of the sample points
      vector<Vector3d> ptsLeftTrans(N);
      vector<Vector3d> left_pts(3), right_pts(3);

      vector<int> curInliers;

      for (int trial = 0; trial < nTrials; ++trial)
      {
         int j0 = int((double(N) * rand() / (RAND_MAX + 1.0)));
         int j1 = int((double(N) * rand() / (RAND_MAX + 1.0)));
         int j2 = int((double(N) * rand() / (RAND_MAX + 1.0)));

         //cout << "trial = " << trial << " j0 = " << j0 << " j1 = " << j1 << " j2 = " << j2 << endl;

         if (j0 == j1 || j0 == j2 || j1 == j2) continue;

         left_pts[0]  = Xs1[j0]; left_pts[1]  = Xs1[j1]; left_pts[2]  = Xs1[j2];
         right_pts[0] = Xs2[j0]; right_pts[1] = Xs2[j1]; right_pts[2] = Xs2[j2];

         Matrix3x3d R0;
         Vector3d T0;
         double scale0;
         getSimilarityTransformation(left_pts, right_pts, R0, T0, scale0);

         for (int i = 0; i < N; ++i) ptsLeftTrans[i] = scale0 * (R0 * Xs1[i] + T0);

         curInliers.clear();

         for (int i = 0; i < N; ++i)
         {
            Vector2d const pL = leftCam.projectPoint(ptsLeftTrans[i]);
            Vector2d const pR = rightCam.projectPoint(ptsLeftTrans[i]);

            double distL = sqrNorm_L2(left2[i] - pL);
            double distR = sqrNorm_L2(right2[i] - pR);
            if (distL < inlierThreshold2 && distR < inlierThreshold2)
               curInliers.push_back(i);
         } // end for (i)
 
         if (curInliers.size() > inliers.size())
         {
            inliers = curInliers;
            R = R0;
            T = T0;
            scale = scale0;
         }
      } // end for (trial)
      //cout << "computeRobustSimilarityTransformation(): inliers.size() = " << inliers.size() << endl;
   } // end computeRobustSimilarityTransformation()

#if 0
   void
   computeTransformationBetweenTriplets(TripleReconstruction const& triplet1, TripleReconstruction const& triplet2,
                                        Matrix3x3d const& K,
                                        Matrix3x3d& R, Vector3d& T, double& scale, double& weight,
                                        int nRequiredCommonPoints, double inlierThreshold, bool verbose = false)
   {
      weight = 0.0;

      set<int> const commonViews = getViewIdIntersection(triplet1, triplet2);

      if (commonViews.size() < 1)
      {
         weight = 0;
         return;
      }

      int const leftView = commonViews[0];
      int const rightView = commonViews[1];

      int leftView1 = -1, leftView2 = -1, rightView1 = -1, rightView2 = -1;

      for (int i = 0; i < 3; ++i)
      {
         if (triplet1.views[i] == leftView)
            leftView1 = i;
         else if (triplet1.views[i] == rightView)
            rightView1 = i;

         if (triplet2.views[i] == leftView)
            leftView2 = i;
         else if (triplet2.views[i] == rightView)
            rightView2 = i;
      }
      //cout << "leftView1 = " << leftView1 << ", rightView1 = " << rightView1 << ", leftView2 = " << leftView2 << ", rightView2 = " << rightView2 << endl;

      CameraMatrix leftCam2, rightCam2;
      leftCam2.setIntrinsic(K);
      leftCam2.setOrientation(triplet2.orientations[leftView2]);

      rightCam2.setIntrinsic(K);
      rightCam2.setOrientation(triplet2.orientations[rightView2]);

      map<int, int> measurementPointMap;

      for (int j = 0; j < triplet1.model.size(); ++j)
      {
         TriangulatedPoint const& X1 = triplet1.model[j];

         if (X1.measurements.size() != 3) continue; // Should not happen in theory

         for (int k = 0; k < X1.measurements.size(); ++k)
         {
            PointMeasurement const& m = X1.measurements[k];
            int const view = triplet1.views[m.view];
            if (view == leftView) measurementPointMap.insert(make_pair(m.id, j));
         } // end for (k)
      } // end for (j)

      //cout << "measurementPointMap.size() = " << measurementPointMap.size() << endl;

      vector<pair<int, int> > indexPairs;
      for (int j = 0; j < triplet2.model.size(); ++j)
      {
         TriangulatedPoint const& X2 = triplet2.model[j];

         if (X2.measurements.size() != 3) continue; // Should not happen in theory

         for (int k = 0; k < X2.measurements.size(); ++k)
         {
            PointMeasurement const& m = X2.measurements[k];
            int const view = triplet2.views[m.view];
            if (view != leftView) continue;

            map<int, int>::const_iterator p = measurementPointMap.find(m.id);
            if (p != measurementPointMap.end())
            {
               indexPairs.push_back(make_pair(p->second, j));
               break;
            }
         } // end for (k)
      } // end for (j)

      vector<Vector3d> Xs1, Xs2;
      //vector<Vector2f> left1, right1;
      vector<Vector2d> left2, right2;

      for (int i = 0; i < indexPairs.size(); ++i)
      {
         int const j1 = indexPairs[i].first;
         int const j2 = indexPairs[i].second;

         TriangulatedPoint const& X1 = triplet1.model[j1];
         TriangulatedPoint const& X2 = triplet2.model[j2];

         bool foundLeft = false, foundRight = false;

         Vector2d pL, pR;

         // De-normalize point measurements to pixels
         for (int k = 0; k < X2.measurements.size(); ++k)
         {
            PointMeasurement const& m = X2.measurements[k];
            if (m.view == leftView2)
            {
               foundLeft = true;
               multiply_A_v_projective(K, m.pos, pL);
            }
            else if (m.view == rightView2)
            {
               foundRight = true;
               multiply_A_v_projective(K, m.pos, pR);
            }
         }

         if (foundLeft && foundRight)
         {
            Xs1.push_back(X1.pos);
            Xs2.push_back(X2.pos);
            left2.push_back(pL);
            right2.push_back(pR);
         }
      } // end for (k)

      int const nPoints = Xs1.size();
      if (left2.size() != nPoints || right2.size() != nPoints)
      {
         cout << "Number of 3D points and projections do not match, nPoints = " << nPoints
              << ", left2.size() = " << left2.size() << ", right2.size() = " << right2.size() << endl;
         weight = 0;
         return;
      }

      if (nPoints < nRequiredCommonPoints)
      {
         if (verbose) cout << "Too few 3D-3D correspondences (" << nPoints << ")." << endl;
         weight = 0;
         return;
      }

      vector<int> inlierIndices;
      computeRobustSimilarityTransformation(leftCam2, rightCam2, Xs1, Xs2, left2, right2, inlierThreshold, 20, R, T, scale, inlierIndices);

      if (inlierIndices.size() < nRequiredCommonPoints)
      {
         if (verbose) cout << "Too few inlier 3D-3D correspondences (" << inlierIndices.size() << ")." << endl;
         weight = 0;
         return;
      }

      vector<Vector3d> inlierXs1(inlierIndices.size());
      vector<Vector3d> inlierXs2(inlierIndices.size());

      for (size_t i = 0; i < inlierIndices.size(); ++i)
      {
         inlierXs1[i] = Xs1[inlierIndices[i]];
         inlierXs2[i] = Xs2[inlierIndices[i]];
      }

      //cout << "calling getSimilarityTransformation() with " << inlierXs1.size() << " inliers." << endl;
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
   } // end computeTransformationBetweenTriplets()
#endif

//**********************************************************************

   void
   computeRobustSimilarityTransformation(std::vector<Vector3d> const& left, std::vector<Vector3d> const& right,
                                         int nTrials, double inlierThreshold,
                                         Matrix3x3d& R, Vector3d& T, double& scale, std::vector<int> &inliers)
   {
      inliers.clear();
 
      int const N = left.size();
 
      if (N < 3) throwV3DErrorHere("computeRobustSimilarityTransform(): at least 3 point correspondences required.");

      //cout << "computeRobustSimilarityTransformation(): N = " << N << endl;

      int bestInlierCount = 0;

      // vector to hold the indices of the sample points
      vector<Vector3d> ptsLeftTrans(N);
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

         Matrix3x3d R0;
         Vector3d T0;
         double scale0;
         getSimilarityTransformation(left_pts, right_pts, R0, T0, scale0);

         for (int i = 0; i < N; ++i) ptsLeftTrans[i] = scale0 * (R0 * left[i] + T0); 

         unsigned int inlcount = 0;
         for (int i = 0; i < N; ++i)
         {
            double dist = distance_L2(right[i], ptsLeftTrans[i]); 
            if (dist < inlierThreshold) ++inlcount;
         } // end for (i)
 
         if (inlcount > bestInlierCount)
         {
            bestInlierCount = inlcount;
            R = R0;
            T = T0;
            scale = scale0;
         }
      } // end for (trial)

      for (int i = 0; i < N; ++i)
         ptsLeftTrans[i] = scale * (R * left[i] + T); 

      for (int i = 0; i < N; ++i)
      {
         double dist = distance_L2(right[i], ptsLeftTrans[i]); 
         if (dist < inlierThreshold)
            inliers.push_back(i);
      } // end for (i)
      //cout << "computeRobustSimilarityTransformation(): inliers.size() = " << inliers.size() << endl;
   } // end computeRobustSimilarityTransformation()

   void
   computeTransformationBetweenTriplets(TripleReconstruction const& triplet1, TripleReconstruction const& triplet2,
                                        Matrix3x3d& R, Vector3d& T, double& scale, double& weight,
                                        int nRequiredCommonPoints, bool verbose = false)
   {
      weight = 0.0;

      std::set<int> const commonViews = getViewIdIntersection(triplet1, triplet2);

      if (commonViews.size() < 1)
      {
         weight = 0;
         return;
      }

      map<pair<int, int>, int> measurementPointMap;

      for (int j = 0; j < triplet1.model.size(); ++j)
      {
         TriangulatedPoint const& X1 = triplet1.model[j];

         if (X1.measurements.size() != 3) continue; // Should not happen in theory

         for (int k = 0; k < X1.measurements.size(); ++k)
         {
            PointMeasurement const& m = X1.measurements[k];
            int const view = triplet1.views[m.view];
            if (commonViews.find(view) != commonViews.end())
               measurementPointMap.insert(make_pair(make_pair(view, m.id), j));
         } // end for (k)
      } // end for (j)

      //cout << "measurementPointMap.size() = " << measurementPointMap.size() << endl;

      vector<Vector3d> Xs1, Xs2;
      for (int j = 0; j < triplet2.model.size(); ++j)
      {
         TriangulatedPoint const& X2 = triplet2.model[j];

         if (X2.measurements.size() != 3) continue; // Should not happen in theory

         for (int k = 0; k < X2.measurements.size(); ++k)
         {
            PointMeasurement const& m = X2.measurements[k];
            int const view = triplet2.views[m.view];

            map<pair<int, int>, int>::const_iterator p = measurementPointMap.find(make_pair(view, m.id));
            if (p != measurementPointMap.end())
            {
               Xs1.push_back(triplet1.model[p->second].pos);
               Xs2.push_back(X2.pos);
               break;
            }
         } // end for (k)
      } // end for (j)

      int const nPoints = Xs1.size();

      if (nPoints < nRequiredCommonPoints)
      {
         if (verbose) cout << "Too few 3D-3D correspondences (" << nPoints << ")." << endl;
         weight = 0;
         return;
      }

      double const diameter2 = computeDiameter(Xs2);
      //double const inlierThreshold = 0.002 * diameter2;
      double const inlierThreshold = 0.005 * diameter2;

      vector<int> inlierIndices;
      computeRobustSimilarityTransformation(Xs1, Xs2, inlierThreshold, 20, R, T, scale, inlierIndices);

      if (inlierIndices.size() < nRequiredCommonPoints)
      {
         if (verbose) cout << "Too few inlier 3D-3D correspondences (" << inlierIndices.size() << ")." << endl;
         weight = 0;
         return;
      }

      vector<Vector3d> inlierXs1(inlierIndices.size());
      vector<Vector3d> inlierXs2(inlierIndices.size());

      for (size_t i = 0; i < inlierIndices.size(); ++i)
      {
         inlierXs1[i] = Xs1[inlierIndices[i]];
         inlierXs2[i] = Xs2[inlierIndices[i]];
      }

      //cout << "calling getSimilarityTransformation() with " << inlierXs1.size() << " inliers." << endl;
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
   } // end computeTransformationBetweenTriplets()

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
      ConfigurationFile cf(argv[1]);

      double maxReproError = cf.get("MAX_REPROJECTION_ERROR_SUBMODEL", -1.0);
      if (maxReproError < 0) maxReproError = cf.get("MAX_REPROJECTION_ERROR_TRIPLET", -1.0);

      int nRequiredTriplePoints = cf.get("REQUIRED_TRIPLE_POINTS_SUBMODEL", -1);
      if (nRequiredTriplePoints < 0) nRequiredTriplePoints = cf.get("REQUIRED_TRIPLE_POINTS", 50);

      int const nRequiredVisiblePoints = cf.get("REQUIRED_VISIBLE_POINTS", 30);

//       SQLite3_Database matchesDB("pairwise_matches.db");
//       MatchDataTable matchDataTable = matchesDB.getTable<PairwiseMatch>("matches_data");
//       CachedStorage<MatchDataTable> matchDataCache(matchDataTable, 100);

      SQLite3_Database tripletDB("triplets.db");
      TripletDataTable tripletDataTable = tripletDB.getTable<TripleReconstruction>("triplet_data");
      CachedStorage<TripletDataTable> tripletDataCache(tripletDataTable, 100);

      set<int>           allViews;
      set<ViewTripletKey> allTriplets;
      map<ViewTripletKey, int> tripletOIDMap;
      map<int, ViewTripletKey> OID_tripletMap;
      //map<ViewPair, set<int> > pairThirdViewMap;
      {
         typedef SQLite3_Database::Table<TripletListItem> TripletListTable;
         TripletListTable tripletListTable = tripletDB.getTable<TripletListItem>("triplet_list");

         for (TripletListTable::const_iterator p = tripletListTable.begin(); bool(p); ++p)
         {
            int const oid        = (*p).first;
            TripletListItem item = (*p).second;
            ViewTripletKey   key  = item.views;

            if (item.nTriangulatedPoints < nRequiredTriplePoints) continue;

            allTriplets.insert(key);
            tripletOIDMap.insert(make_pair(key, oid));
            OID_tripletMap.insert(make_pair(oid, key));

            int const i0 = key.views[0];
            int const i1 = key.views[1];
            int const i2 = key.views[2];

            allViews.insert(i0); allViews.insert(i1); allViews.insert(i2);

            //cout << "oid: " << oid << ": (" << i0 << ", " << i1 << ", " << i2 << ")" << endl;
         }
      } // end scope

      cout << "Considering = " << tripletOIDMap.size() << " triplets." << endl;

      SerializableVector<SerializableVector<int> > submodelsTripletOIDs;
      serializeDataFromFile("submodel_triplet_lists.bin", submodelsTripletOIDs);

      int const nRequiredCommonPoints = 20;
      double const inlierThreshold = 20.0;

      cout << "Going to check " << submodelsTripletOIDs.size() << " submodels." << endl;

      SerializableVector<SerializableVector<TripletAlignment> > allTripletAlignments;
      map<pair<int, int>, TripletAlignment> cachedAlignments;
      int cacheHits = 0;

      for (int submodel = 0; submodel < submodelsTripletOIDs.size(); ++submodel)
      {
         vector<int> const& curTripletOIDs = submodelsTripletOIDs[submodel];
         cout << "Checking submodel " << submodel << " with " << curTripletOIDs.size() << " triplets:" << endl;

         vector<pair<int, int> > tripletOIDPairs;
         set<int> submodelViews;

         for (int k1 = 0; k1 < curTripletOIDs.size(); ++k1)
         {
            int const oid1 = curTripletOIDs[k1];

            if (OID_tripletMap.find(oid1) == OID_tripletMap.end()) continue;

            ViewTripletKey const triplet1 = OID_tripletMap.find(oid1)->second;

            submodelViews.insert(triplet1.views[0]);
            submodelViews.insert(triplet1.views[1]);
            submodelViews.insert(triplet1.views[2]);

            for (int k2 = k1+1; k2 < curTripletOIDs.size(); ++k2)
            {
               int const oid2 = curTripletOIDs[k2];
               if (OID_tripletMap.find(oid2) == OID_tripletMap.end()) continue;

               ViewTripletKey const triplet2 = OID_tripletMap.find(oid2)->second;

//                cout << "oid1 = " << oid1 << ", oid2 = " << oid2 << endl;
//                cout << "t1 = (" << triplet1.views[0] << ", " << triplet1.views[1] << ", " << triplet1.views[2] << ")" << endl;
//                cout << "t2 = (" << triplet2.views[0] << ", " << triplet2.views[1] << ", " << triplet2.views[2] << ")" << endl;

               if (areConnectedTriplets(triplet1, triplet2))
                  tripletOIDPairs.push_back(make_pair(oid1, oid2));
            } // end for (k2)
         } // end for (k1)

         cout << tripletOIDPairs.size() << " transformations between triplets need to be computed." << endl;

         allTripletAlignments.push_back(SerializableVector<TripletAlignment>());

         int nSuccesses = 0;
         set<int> remainingTriplets;

         for (int k = 0; k < tripletOIDPairs.size(); ++k)
         {
            int const oid1 = tripletOIDPairs[k].first;
            int const oid2 = tripletOIDPairs[k].second;

            map<pair<int, int>, TripletAlignment>::const_iterator q = cachedAlignments.find(make_pair(oid1, oid2));
            if (q != cachedAlignments.end())
            {
               ++nSuccesses;
               ++cacheHits;
               allTripletAlignments.back().push_back(q->second);
               remainingTriplets.insert(oid1);
               remainingTriplets.insert(oid2);
            }
            else
            {
               TripleReconstruction const& triplet1 = *tripletDataCache[oid1];
               TripleReconstruction const& triplet2 = *tripletDataCache[oid2];

               Matrix3x3d R;
               Vector3d   T;
               double scale, weight;

#if 0
               computeTransformationBetweenTriplets(triplet1, triplet2, intrinsic, R, T, scale, weight,
                                                    nRequiredCommonPoints, inlierThreshold, false);
#else
               computeTransformationBetweenTriplets(triplet1, triplet2, R, T, scale, weight, nRequiredCommonPoints, false);
#endif
               if (weight > 0)
               {
                  ++nSuccesses;

                  TripletAlignment alignment;
                  alignment.oids[0] = oid1;
                  alignment.oids[1] = oid2;
                  alignment.weight  = weight;
                  alignment.scale   = scale;
                  alignment.R       = R;
                  alignment.T       = T;
                  allTripletAlignments.back().push_back(alignment);
                  cachedAlignments.insert(make_pair(make_pair(oid1, oid2), alignment));

                  remainingTriplets.insert(oid1);
                  remainingTriplets.insert(oid2);
               }
            } // end if
         } // end for (k)
         cout << nSuccesses << " transformations between triplets in this submodel estimated." << endl;
         cout << remainingTriplets.size() << " triplets are remaining." << endl;

#if 1
         if (!remainingTriplets.empty())
         {
            // MST construction scope
            SerializableVector<TripletAlignment> const curAlignments(allTripletAlignments.back());

            allTripletAlignments.back().clear();

            int const nMST_Trees = 5;

            set<pair<int, int> > insertedPairs;

            vector<pair<int, int> > edges;
            vector<double> weights;

            for (int k = 0; k < curAlignments.size(); ++k)
            {
               edges.push_back(make_pair(curAlignments[k].oids[0], curAlignments[k].oids[1]));
               weights.push_back(curAlignments[k].weight);
            } // end for (k)

            for (int tree = 0; tree < nMST_Trees; ++tree)
            {
               map<int, int> parentNodes;
               computeMST(edges, weights, parentNodes);
               set<pair<int, int> > mstEdges;
               for (map<int, int>::const_iterator p = parentNodes.begin(); p != parentNodes.end(); ++p)
               {
                  mstEdges.insert(make_pair(p->first, p->second));
                  mstEdges.insert(make_pair(p->second, p->first));
               }

               for (int k = 0; k < curAlignments.size(); ++k)
               {
                  pair<int, int> tripletPair(curAlignments[k].oids[0], curAlignments[k].oids[1]);

                  if (mstEdges.find(tripletPair) != mstEdges.end())
                  {
                     if (insertedPairs.find(tripletPair) == insertedPairs.end())
                     {
                        allTripletAlignments.back().push_back(curAlignments[k]);
                        insertedPairs.insert(tripletPair);
                        std::swap(tripletPair.first, tripletPair.second);
                        insertedPairs.insert(tripletPair);
                     }
                     weights[k] *= 10;
                  }
               } // end for (k)
            } // end for (tree)

            cout << allTripletAlignments.back().size() << "/" << curAlignments.size() << " alignments are still present." << endl;

            remainingTriplets.clear();
            for (int k = 0; k < allTripletAlignments.back().size(); ++k)
            {
               remainingTriplets.insert(allTripletAlignments.back()[k].oids[0]);
               remainingTriplets.insert(allTripletAlignments.back()[k].oids[1]);
            }
         } // end scope
#endif

         set<int> remainingViews;
         for (set<int>::const_iterator p = remainingTriplets.begin(); p != remainingTriplets.end(); ++p)
         {
            ViewTripletKey const& key = OID_tripletMap.find(*p)->second;
            remainingViews.insert(key.views[0]);
            remainingViews.insert(key.views[1]);
            remainingViews.insert(key.views[2]);
         }
         cout << "Remaining views (" << remainingViews.size() << "): [ ";
         for (set<int>::const_iterator p = remainingViews.begin(); p != remainingViews.end(); ++p)
            cout << *p << " ";
         cout << "]" << endl;

         if (remainingViews.size() != submodelViews.size())
         {
            cout << "Something is wrong with this submodel, clearing alignment transformations." << endl;
            allTripletAlignments.back().clear();
         }
      } // end for (model)

      int nEstimated = 0;
      for (int k = 0; k < allTripletAlignments.size(); ++k) nEstimated += allTripletAlignments[k].size();
      cout << nEstimated << " transformations between triplets estimated, " << cacheHits << " were cache hits. "<< endl;

      serializeDataToFile("triplets_alignment.bin", allTripletAlignments);
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
}
