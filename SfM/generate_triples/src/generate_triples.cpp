#include "reconstruction_common.h"

#include <iostream>
#include <sstream>

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_storage.h"
#include "Math/v3d_mathutilities.h"
#include "Geometry/v3d_poseutilities.h"
#include "Geometry/v3d_metricbundle.h"

using namespace std;
using namespace V3D;

#ifdef WIN32 //HA
#include <math.h>
#endif

namespace
{

   inline bool
   comparePointMeasurements(PointMeasurement const& a, PointMeasurement const& b)
   {
      if (a.id < b.id) return true;
      if (a.id > b.id) return false;
      if (a.view < b.view) return true;
      if (a.view > b.view) return false;
      return false;
   }

   inline bool
   comparePointCorrespondences(PointCorrespondence const& a, PointCorrespondence const& b)
   {
      if (comparePointMeasurements(a.left, b.left)) return true;
      if (a.left == b.left)
      {
         return comparePointMeasurements(a.right, b.right);
      }
      else
         return false;
   }

   void
   filterTripleModel(CalibrationDatabase const& calibDb, int v0, int v1, int v2,
                     double maxError, vector<CameraMatrix> const& cameras,
                     vector<TriangulatedPoint>& model)
   {
      int views[3];
      views[0] = v0; views[1] = v1; views[2] = v2;

      for (int j = 0; j < model.size(); ++j)
      {
         TriangulatedPoint& X = model[j];

         vector<PointMeasurement> ms;

         for (int k = 0; k < X.measurements.size(); ++k)
         {
            PointMeasurement const& m = X.measurements[k];
            int const i = m.view;
            Vector2d p = cameras[i].projectPoint(X.pos);
            Vector2f pp = makeVector2<float>(p[0], p[1]);

            double const f = calibDb.getAvgFocalLength(views[i]);

            double err = f * norm_L2(pp - X.measurements[k].pos);
            if (err < maxError) ms.push_back(m);
         }
         X.measurements = ms;
      } // end for (j)

      vector<TriangulatedPoint> const origModel(model);
      model.clear();

      for (int j = 0; j < origModel.size(); ++j)
      {
         TriangulatedPoint const& X = origModel[j];
         if (X.measurements.size() >= 3) model.push_back(X);
      }
   } // end filterTripleModel()

} // end namespace <>


int
main(int argc, char * argv[])
{
   if (argc != 4 && argc != 5)
   {
      cerr << "Usage: " << argv[0] << " <config file> <first view no> <last view no> [<edges black list file>]" << endl;
      return -1;
   }

   try
   {
      ConfigurationFile cf(argv[1]);

      double const maxReproError = cf.get("MAX_REPROJECTION_ERROR_TRIPLET", 1.0);
      int    const nMinCorrs = cf.get("MINIMUM_CORRESPONDENCES_TRIPLET", 0);
      int    const nRequiredTriplePoints = cf.get("REQUIRED_TRIPLET_POINTS", 50);
      double const minPointRatio = cf.get("MINIMUM_TRIPLET_POINT_RATIO", 0.5);

      float  const cosAngleThreshold = cosf(cf.get("MIN_TRIANGULATION_ANGLE", 5.0f) * M_PI/180.0f);
      cout << "cosAngleThreshold = " << cosAngleThreshold << endl;

      bool const filterCorrespondences = cf.get("FILTER_TRIPLET_CORRESPONDENCES", true);

      CalibrationDatabase calibDb("calibration_db.txt");

      set<pair<int, int> > edgesBlackList;
      if (argc == 5)
      {
         cout << "Reading black list file..." << endl;
         ifstream is(argv[4]);
         if (is)
         {
            while (is)
            {
               if (is.eof()) break;
               int v1, v2;
               is >> v1 >> v2;
               edgesBlackList.insert(make_pair(v1, v2));
            }
         }
         else
            cout << "Cannot open " << argv[2] << endl;
         cout << "Read " << edgesBlackList.size() << " entries in the black list."<< endl;
      }


      int const firstView = atoi(argv[2]);
      int const lastView = atoi(argv[3]);
      int const nViews = lastView - firstView + 1;

      SQLite3_Database matchesDB("pairwise_matches.db");
      SQLite3_Database tripletDB("triplets.db");

      map<ViewPair, int> viewPairOIDMap;

      {
         typedef SQLite3_Database::Table<ViewPair> Table;
         Table table = matchesDB.getTable<ViewPair>("matches_list");
         for (Table::const_iterator p = table.begin(); bool(p); ++p)
         {
            int const oid = (*p).first;
            ViewPair pair = (*p).second;
            int const view1 = pair.view0;
            int const view2 = pair.view1;

            if (view1 < firstView || view1 > lastView) continue;
            if (view2 < firstView || view2 > lastView) continue;

            if (edgesBlackList.find(make_pair(view1, view2)) != edgesBlackList.end() ||
                edgesBlackList.find(make_pair(view2, view1)) != edgesBlackList.end()) continue;

            viewPairOIDMap.insert(make_pair(pair, oid));
         }
      } // end scope
      cout << "Considering = " << viewPairOIDMap.size() << " view pairs." << endl;

      Matrix3x3d I; makeIdentityMatrix(I);

      int nCheckedTriples = 0;
      int nValidTriples = 0;

      tripletDB.createTable("triplet_data", true);
      tripletDB.createTable("triplet_list", true);

      typedef SQLite3_Database::Table<PairwiseMatch> MatchDataTable;
      MatchDataTable matchDataTable = matchesDB.getTable<PairwiseMatch>("matches_data");

      typedef SQLite3_Database::Table<TripleReconstruction> TripletDataTable;
      TripletDataTable tripletDataTable = tripletDB.getTable<TripleReconstruction>("triplet_data");

      typedef SQLite3_Database::Table<TripletListItem> TripletListTable;
      TripletListTable tripletListTable = tripletDB.getTable<TripletListItem>("triplet_list");

      CachedStorage<MatchDataTable> matchDataCache(matchDataTable, 100);

      set<ViewPair> goodViewPairs; // the ones that are appearing in a triplet

      int nDropped = 0;

      // Enumerate all found tripels
      for (map<ViewPair, int>::const_iterator p01 = viewPairOIDMap.begin(); p01 != viewPairOIDMap.end(); ++p01)
      {
         int const v0    = (*p01).first.view0;
         int const v1    = (*p01).first.view1;
         int const oid01 = (*p01).second;

         PairwiseMatch * match01Ptr = matchDataCache[oid01];

         if (!match01Ptr || match01Ptr->corrs.size() < nMinCorrs) continue;

         // Make a copy to handle possible removal from the LRU cache
         PairwiseMatch const match01(*match01Ptr);

         Matrix3x3d const& R01_orig = match01.rotation;

         for (int v2 = v1+1; v2 <= lastView; ++v2)
         {
            //cout << "(" << v0 << ", " << v1 << ", " << v2 << ")" << endl;

            Vector3d T01 = match01.translation;

            map<ViewPair, int>::const_iterator p02 = viewPairOIDMap.find(ViewPair(v0, v2));
            if (p02 == viewPairOIDMap.end()) continue;

            map<ViewPair, int>::const_iterator p12 = viewPairOIDMap.find(ViewPair(v1, v2));
            if (p12 == viewPairOIDMap.end()) continue;

            int const oid02 = (*p02).second;
            int const oid12 = (*p12).second;

            PairwiseMatch * match02Ptr = matchDataCache[oid02];
            if (match02Ptr->corrs.size() < nMinCorrs) continue;

            PairwiseMatch * match12Ptr = matchDataCache[oid12];
            if (match12Ptr->corrs.size() < nMinCorrs) continue;

#if 0
            PairwiseMatch match02(*match02Ptr);
            PairwiseMatch match12(*match12Ptr);
#else
            PairwiseMatch& match02 = *match02Ptr;
            PairwiseMatch& match12 = *match12Ptr;
#endif
            //cout << "(" << v0 << ", " << v1 << ", " << v2 << ")" << endl;
            //cout << match01.corrs.size() << " " << match02.corrs.size() << " " << match12.corrs.size() << endl;

            ++nCheckedTriples;
            if ((nCheckedTriples % 100) == 0) cout << "nCheckedTriples = " << nCheckedTriples << "..." << endl;

            // Build a local reconstruction for this triple and geometrically verify this triple

            Matrix3x3d const& R02_orig = match02.rotation;
            Matrix3x3d const& R12_orig = match12.rotation;

            Vector3d T02 = match02.translation;
            Vector3d T12 = match12.translation;

            vector<Matrix3x3d> tripleRotations(3);

            {
               vector<Matrix3x3d>      tripleRelRotations;
               vector<pair<int, int> > tripleViewPairs;

               tripleRelRotations.push_back(R01_orig);
               tripleRelRotations.push_back(R02_orig);
               tripleRelRotations.push_back(R12_orig);

               tripleViewPairs.push_back(make_pair(0, 1));
               tripleViewPairs.push_back(make_pair(0, 2));
               tripleViewPairs.push_back(make_pair(1, 2));

               computeConsistentRotations(3, tripleRelRotations, tripleViewPairs, tripleRotations);
            } // end scope

            Matrix3x3d const& R0 = tripleRotations[0];
            Matrix3x3d const& R1 = tripleRotations[1];
            Matrix3x3d const& R2 = tripleRotations[2];

            Matrix3x3d const R01 = R1 * R0.transposed();
            Matrix3x3d const R02 = R2 * R0.transposed();
            Matrix3x3d const R12 = R2 * R1.transposed();
            Matrix3x3d const R20 = R02.transposed();

            Vector3d const T20 = -(R20 * T02);

            vector<PointCorrespondence> const& corrs01 = match01.corrs;
            vector<PointCorrespondence> const& corrs02 = match02.corrs;
            vector<PointCorrespondence> const& corrs12 = match12.corrs;

            // Compute the lengths of the bases first
            double s012, s120, s201, weight;
#if 0
            computeScaleRatios(cosAngleThreshold, R01, T01, R12, T12, R20, T20,
                               corrs01, corrs12, corrs02, s012, s120, s201, weight);
#else
            computeScaleRatiosGeneralized(R01, T01, R12, T12, R20, T20,
                                          corrs01, corrs12, corrs02, s012, s120, s201, weight);
#endif
            //cout << "weight = " << weight << endl;
            //cout << "scales = " << s012 << " " << s120 << " " << s201 << endl;
            if (weight <= 0) continue;

            Matrix<double> B(3, 3, 0.0);

            B[0][2] = weight;
            B[0][0] = -weight*s012;
            B[1][0] = weight;
            B[1][1] = -weight*s201;
            B[2][1] = weight;
            B[2][2] = -weight*s120;

            Vector<double> baseLengths(3);
            {
               SVD<double> svd(B);
               svd.getV().getColumnSlice(0, 3, 2, baseLengths);
            }
            //cout << "l = "; displayVector(baseLengths);
            if (baseLengths[0] < 0.0) baseLengths *= -1.0f;
            baseLengths *= sqrt((double)3); //HA

            normalizeVector(T01); scaleVectorIP(baseLengths[0], T01);
            normalizeVector(T02); scaleVectorIP(baseLengths[1], T02);
            normalizeVector(T12); scaleVectorIP(baseLengths[2], T12);

            Matrix<double> A(3*3, 3*3, 0.0);
            Vector<double> rhs(A.num_rows());

            //cout << "computeConsistentTranslations():" << endl;
            copyMatrixSlice(I,            0, 0, 3, 3, A, 3*0, 3*1);
            copyMatrixSlice((-1.0) * R01, 0, 0, 3, 3, A, 3*0, 3*0);
            copyMatrixSlice(I,            0, 0, 3, 3, A, 3*1, 3*2);
            copyMatrixSlice((-1.0) * R02, 0, 0, 3, 3, A, 3*1, 3*0);
            copyMatrixSlice(I,            0, 0, 3, 3, A, 3*2, 3*2);
            copyMatrixSlice((-1.0) * R12, 0, 0, 3, 3, A, 3*2, 3*1);

            rhs[3*0+0] = T01[0]; rhs[3*0+1] = T01[1]; rhs[3*0+2] = T01[2];
            rhs[3*1+0] = T02[0]; rhs[3*1+1] = T02[1]; rhs[3*1+2] = T02[2];
            rhs[3*2+0] = T12[0]; rhs[3*2+1] = T12[1]; rhs[3*2+2] = T12[2];

            Matrix<double> Aplus;
            makePseudoInverse(A, Aplus);

            Vector<double> X = Aplus * rhs;

            vector<Vector3d> tripleTranslations(3);

            for (int i = 0; i < 3; ++i)
            {
               tripleTranslations[i][0] = X[3*i+0];
               tripleTranslations[i][1] = X[3*i+1];
               tripleTranslations[i][2] = X[3*i+2];
               //cout << "T" << i << " = "; displayVector(tripleTranslations[i]);
            }

            vector<CameraMatrix> tripleCams(3);
            tripleCams[0].setIntrinsic(I);
            tripleCams[0].setRotation(R0);
            tripleCams[0].setTranslation(tripleTranslations[0]);

            tripleCams[1].setIntrinsic(I);
            tripleCams[1].setRotation(R1);
            tripleCams[1].setTranslation(tripleTranslations[1]);

            tripleCams[2].setIntrinsic(I);
            tripleCams[2].setRotation(R2);
            tripleCams[2].setTranslation(tripleTranslations[2]);

            vector<PointCorrespondence> tripleCorrs;
            tripleCorrs.reserve(corrs01.size() + corrs02.size() + corrs12.size());

            for (int k = 0; k < corrs01.size(); ++k)
            {
               PointCorrespondence corr = corrs01[k];
               corr.left.view  = 0;
               corr.right.view = 1;
               tripleCorrs.push_back(corr);
            }

            for (int k = 0; k < corrs02.size(); ++k)
            {
               PointCorrespondence corr = corrs02[k];
               corr.left.view  = 0;
               corr.right.view = 2;
               tripleCorrs.push_back(corr);
            }

            for (int k = 0; k < corrs12.size(); ++k)
            {
               PointCorrespondence corr = corrs12[k];
               corr.left.view  = 1;
               corr.right.view = 2;
               tripleCorrs.push_back(corr);
            }

            vector<TriangulatedPoint> tripleModel;
            TriangulatedPoint::connectTracks(tripleCorrs, tripleModel, 3);
            //cout << " tripleModel.size() = " << tripleModel.size() << endl;

            int const nPointsBefore = tripleModel.size();

            for (int j = 0; j < tripleModel.size(); ++j)
            {
               tripleModel[j].pos = triangulateLinear(tripleCams, tripleModel[j].measurements);
            }

            filterTripleModel(calibDb, v0, v1, v2, maxReproError, tripleCams, tripleModel);
            //cout << " tripleModel.size() = " << tripleModel.size() << endl;

            if (tripleModel.size() < nRequiredTriplePoints) continue;
            if (double(tripleModel.size()) < minPointRatio * nPointsBefore)
            {
               ++nDropped;
               continue;
            }

            int const nTripletPointsBefore = tripleModel.size();

            {
               BundlePointStructure bundleStruct(tripleModel);

               V3D::StdMetricBundleOptimizer opt(1.0, tripleCams, bundleStruct.points3d, bundleStruct.measurements,
                                                 bundleStruct.correspondingView, bundleStruct.correspondingPoint);

               V3D::optimizerVerbosenessLevel = 0;
               opt.tau = 1e-3;
               opt.maxIterations = 5;
               opt.minimize();

               bundleStruct.createPointStructure(tripleModel);
            }

            int const nTripletPointsAfter = tripleModel.size();

            if (nTripletPointsAfter != nTripletPointsBefore)
            {
               cout << "nTripletPointsBefore = " << nTripletPointsBefore << ", nTripletPointsAfter = " << nTripletPointsAfter << endl;
            }

            TripleReconstruction reconstruction;
            reconstruction.views[0] = v0;
            reconstruction.views[1] = v1;
            reconstruction.views[2] = v2;
            reconstruction.intrinsics[0] = tripleCams[0].getIntrinsic();
            reconstruction.intrinsics[1] = tripleCams[1].getIntrinsic();
            reconstruction.intrinsics[2] = tripleCams[2].getIntrinsic();
            reconstruction.orientations[0] = tripleCams[0].getOrientation();
            reconstruction.orientations[1] = tripleCams[1].getOrientation();
            reconstruction.orientations[2] = tripleCams[2].getOrientation();
            reconstruction.model = tripleModel;

            TripletListItem listItem;
            listItem.views = ViewTripletKey(v0, v1, v2);
            listItem.nTriangulatedPoints = tripleModel.size();

            tripletDataTable.updateObject(nValidTriples, reconstruction);
            tripletListTable.updateObject(nValidTriples, listItem);

            goodViewPairs.insert(ViewPair(v0, v1));
            goodViewPairs.insert(ViewPair(v0, v2));
            goodViewPairs.insert(ViewPair(v1, v2));

            ++nValidTriples;
         } // end for (v2)
      } // end for (i)

      cout << nCheckedTriples << " triplets were checked." << endl;
      cout << nDropped << " triplets rejected because the number of 3D points dropped too much." << endl;
      cout << nValidTriples << " triplets survived the geometric test." << endl;

      // Determine the set of correspondences used in the later stages

      map<ViewTripletKey, int> tripletOIDMap;
      {
//          typedef SQLite3_Database::Table<TripletListItem> TripletListTable;
//          TripletListTable tripletListTable = tripletDB.getTable<TripletListItem>("triplet_list");

         for (TripletListTable::const_iterator p = tripletListTable.begin(); bool(p); ++p)
         {
            int const oid        = (*p).first;
            TripletListItem item = (*p).second;
            ViewTripletKey   key  = item.views;

            if (key.views[0] < firstView || key.views[0] > lastView) continue;
            if (key.views[1] < firstView || key.views[1] > lastView) continue;
            if (key.views[2] < firstView || key.views[2] > lastView) continue;

            if (item.nTriangulatedPoints < nRequiredTriplePoints) continue;
//             if (item.nTriangulatedPoints < nRequiredTriplePoints)
//             {
//                cout << "nRequiredTriplePoints = " << nRequiredTriplePoints << ", item.nTriangulatedPoints = " << item.nTriangulatedPoints << endl;
//                continue;
//             }

            tripletOIDMap.insert(make_pair(key, oid));
         }
      } // end scope

      SerializableVector<PointCorrespondence> allCorrs;

      if (!filterCorrespondences)
      {
         for (set<ViewPair>::const_iterator p = goodViewPairs.begin(); p != goodViewPairs.end(); ++p)
         {
            ViewPair const& key = *p;
            map<ViewPair, int>::const_iterator q = viewPairOIDMap.find(key);
            if (q == viewPairOIDMap.end())
            {
               cout << "Cannot find OID for view pair " << key;
               continue;
            }
            int const oid = (*q).second;
            PairwiseMatch * matchData = matchDataCache[oid];

            for (size_t k = 0; k < matchData->corrs.size(); ++k)
            {
               PointCorrespondence c = matchData->corrs[k];
               allCorrs.push_back(c);
            } // end for (k)
         } // end for (p)
      }
      else
      {
         for (set<ViewPair>::const_iterator p = goodViewPairs.begin(); p != goodViewPairs.end(); ++p)
         {
            ViewPair const& key = *p;
            map<ViewPair, int>::const_iterator q = viewPairOIDMap.find(key);
            if (q == viewPairOIDMap.end())
            {
               cout << "Cannot find OID for view pair " << key;
               continue;
            }
            int const oid = (*q).second;
            PairwiseMatch * matchData = matchDataCache[oid];
         } // end for (p)

         cout << "Going to collect correspondences from " << tripletOIDMap.size() << " triplets..." << endl;

         CachedStorage<TripletDataTable> tripletDataCache(tripletDataTable, 100);

         // Use only pairwise correspondences still available in the triplets.
         int count = 0, nTripletCorrs = 0;
         for (map<ViewTripletKey, int>::const_iterator p = tripletOIDMap.begin();
              p != tripletOIDMap.end(); ++p, ++count)
         {
            int const oid = p->second;
            TripleReconstruction const * triplet = tripletDataCache[oid];
            for (int j = 0; j < triplet->model.size(); ++j)
            {
               TriangulatedPoint const& X = triplet->model[j];
               if (X.measurements.size() != 3) continue; // Shouldn't happen
               PointMeasurement m0 = X.measurements[0];
               PointMeasurement m1 = X.measurements[1];
               PointMeasurement m2 = X.measurements[2];

               m0.view = triplet->views[m0.view];
               m1.view = triplet->views[m1.view];
               m2.view = triplet->views[m2.view];

               PointCorrespondence c01(m0, m1);
               PointCorrespondence c02(m0, m2);
               PointCorrespondence c12(m1, m2);

               // Should not be necessary, just to be on the safe side
               if (c01.left.view < c01.right.view) std::swap(c01.left, c01.right);
               if (c02.left.view < c02.right.view) std::swap(c02.left, c02.right);
               if (c12.left.view < c12.right.view) std::swap(c12.left, c12.right);

               allCorrs.push_back(c01);
               allCorrs.push_back(c02);
               allCorrs.push_back(c12);
               nTripletCorrs += 3;
            } // end for (j)

            if ((count % 10000) == 0)
            {
               cout << count << " triplets handled." << endl;
               std::sort(allCorrs.begin(), allCorrs.end(), comparePointCorrespondences);
               SerializableVector<PointCorrespondence> allCorrsCopy(allCorrs);
               allCorrs.clear();
               std::unique_copy(allCorrsCopy.begin(), allCorrsCopy.end(),
                                back_insert_iterator<SerializableVector<PointCorrespondence> >(allCorrs));
            } // end if
         } // end for (p)

         cout << "allCorrs.size() before call to unique = " << nTripletCorrs << endl;
         std::sort(allCorrs.begin(), allCorrs.end(), comparePointCorrespondences);
         SerializableVector<PointCorrespondence> allCorrsCopy(allCorrs);
         allCorrs.clear();
         std::unique_copy(allCorrsCopy.begin(), allCorrsCopy.end(),
                          back_insert_iterator<SerializableVector<PointCorrespondence> >(allCorrs));
      } // end if

      cout << "allCorrs.size() = " << allCorrs.size() << endl;
      serializeDataToFile("allcorrs.bin", allCorrs);
   }
   catch (std::string s)
   {
      cerr << "Exception caught: " << s << endl;
   }
   catch (...)
   {
      cerr << "Unhandled exception." << endl;
   }

   return 0;
}
