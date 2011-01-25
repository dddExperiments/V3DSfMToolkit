#include "reconstruction_common.h"

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_timer.h"
#include "Base/v3d_utilities.h"
#include "Math/v3d_sparseeig.h"
#include "Geometry/v3d_mviewutilities.h"
#include "Geometry/v3d_metricbundle.h"

#include <iostream>
#include <sstream>
#include <list>

using namespace std;
using namespace V3D;

int
main(int argc, char * argv[])
{
   if (argc != 4 && argc != 5)
   {
      cerr << "Usage: " << argv[0] << " <config file> <first view no> <last view no> [<EG edges black list file>]" << endl;
      return -1;
   }

   try
   {
      ConfigurationFile cf(argv[1]);

      double maxReproError = cf.get("MAX_REPROJECTION_ERROR_SUBMODEL", -1.0);
      if (maxReproError < 0) maxReproError = cf.get("MAX_REPROJECTION_ERROR_TRIPLET", -1.0);

      int nRequiredTriplePoints = cf.get("REQUIRED_TRIPLE_POINTS_SUBMODEL", -1);
      if (nRequiredTriplePoints < 0) nRequiredTriplePoints = cf.get("REQUIRED_TRIPLET_POINTS", 50);

      int const nRequiredVisiblePoints = cf.get("REQUIRED_VISIBLE_POINTS", 30);

      int const viewFrequency = cf.get("SUBMODEL_VIEW_FREQUENCY", 10);
      int const submodelsize = cf.get("SUBMODEL_MAX_SIZE", 50);
      int const nIterationsBA = cf.get("SUBMODEL_BUNDLE_ITERATIONS", 10);
      double const minFilteredRatio = cf.get("MIN_FILTERED_POINT_COUNT_RATIO", 0.75);

      bool const useRandomSubmodelGrowing = cf.get("USE_RANDOM_SUBMODEL_GROWING", false);

      int const firstView = atoi(argv[2]);
      int const lastView = atoi(argv[3]);

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

      SQLite3_Database matchesDB("pairwise_matches.db");
      SQLite3_Database tripletDB("triplets.db");
      SQLite3_Database submodelsDB("submodels.db");

      submodelsDB.createTable("submodels_data", true);
      submodelsDB.createTable("models_data", true);

      MatchDataTable matchDataTable = matchesDB.getTable<PairwiseMatch>("matches_data");
      CachedStorage<MatchDataTable> matchDataCache(matchDataTable, 100);

      TripletDataTable tripletDataTable = tripletDB.getTable<TripleReconstruction>("triplet_data");
      CachedStorage<TripletDataTable> tripletDataCache(tripletDataTable, 100);

      SubmodelsTable submodelsTable = submodelsDB.getTable<SubmodelReconstruction>("submodels_data");

      map<ViewPair, int> viewPairOIDMap;
      map<ViewPair, double> viewPairWeightMap;

      {
         // Read in the nr. of correspondences for all pairwise matches
         typedef SQLite3_Database::Table<PairwiseMatch> Table;
         Table table = matchesDB.getTable<PairwiseMatch>("matches_data");
         for (Table::const_iterator p = table.begin(); bool(p); ++p)
         {
            int const oid = (*p).first;
            PairwiseMatch matchData = (*p).second;
            int const view1 = matchData.views.view0;
            int const view2 = matchData.views.view1;

            if (view1 < firstView || view1 > lastView) continue;
            if (view2 < firstView || view2 > lastView) continue;

            viewPairOIDMap.insert(make_pair(matchData.views, oid));
            viewPairWeightMap.insert(make_pair(matchData.views, matchData.corrs.size()));
         } // end for (p)
      } // end scope
      cout << "Considering = " << viewPairOIDMap.size() << " view pairs." << endl;

      set<int>           allViews;
      set<ViewPair>      allViewPairs;
      set<ViewTripletKey> allTriplets;
      map<ViewTripletKey, int> tripletOIDMap;
      map<ViewPair, set<int> > pairThirdViewMap;
      {
         typedef SQLite3_Database::Table<TripletListItem> TripletListTable;
         TripletListTable tripletListTable = tripletDB.getTable<TripletListItem>("triplet_list");

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

            int const i0 = key.views[0];
            int const i1 = key.views[1];
            int const i2 = key.views[2];

            if (edgesBlackList.find(make_pair(i0, i1)) != edgesBlackList.end()) continue;
            if (edgesBlackList.find(make_pair(i0, i2)) != edgesBlackList.end()) continue;
            if (edgesBlackList.find(make_pair(i1, i2)) != edgesBlackList.end()) continue;

            allTriplets.insert(key);
            tripletOIDMap.insert(make_pair(key, oid));

            allViews.insert(i0); allViews.insert(i1); allViews.insert(i2);

            allViewPairs.insert(ViewPair(i0, i1));
            allViewPairs.insert(ViewPair(i0, i2));
            allViewPairs.insert(ViewPair(i1, i2));

            pairThirdViewMap[ViewPair(i0, i1)].insert(i2);
            pairThirdViewMap[ViewPair(i0, i2)].insert(i1);
            pairThirdViewMap[ViewPair(i1, i2)].insert(i0);
         }
      } // end scope

      map<ViewPair, Matrix3x3d> relRotations;
      map<ViewPair, Vector3d>   relTranslations;

      for (set<ViewPair>::const_iterator p = allViewPairs.begin(); p != allViewPairs.end(); ++p)
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

         relRotations.insert(make_pair(key, matchData->rotation));
         relTranslations.insert(make_pair(key, matchData->translation));
      } // end for (p)

      SerializableVector<PointCorrespondence> allCorrs;
      serializeDataFromFile("allcorrs.bin", allCorrs);
      cout << "allCorrs.size() = " << allCorrs.size() << endl;

      int const nAllViews = allViews.size();
      vector<int> viewIdBackMap(nAllViews);
      {
         int i = 0;
         for (set<int>::const_iterator p = allViews.begin(); p != allViews.end(); ++p, ++i)
            viewIdBackMap[i] = *p;
      }

      cout << "Computing models from " << allTriplets.size() << " triplets." << endl;

      int const nSubmodels = (viewFrequency * allViews.size()) / submodelsize;
      cout << "Going to draw " << nSubmodels << " submodels." << endl;

      vector<set<int> > connComponents;
      map<int, set<int> > mstAdjacencyMap;
      if (!useRandomSubmodelGrowing)
      {
         // Compute the MSTs and number of components for the epipolar/camera graph

         vector<pair<int, int> > edges;
         vector<double> weights;

         for (set<ViewPair>::const_iterator p = allViewPairs.begin(); p != allViewPairs.end(); ++p)
         {
            edges.push_back(make_pair(p->view0, p->view1));
            weights.push_back(viewPairWeightMap.find(*p)->second);
         }

         vector<pair<int, int> > mstEdges;

         getMinimumSpanningForest(edges, weights, mstEdges, connComponents);
         cout << "Camera graph has " << connComponents.size() << " connected component(s)." << endl;

         for (size_t i = 0; i < mstEdges.size(); ++i)
         {
            mstAdjacencyMap[mstEdges[i].first].insert(mstEdges[i].second);
            mstAdjacencyMap[mstEdges[i].second].insert(mstEdges[i].first);
         }
      } // end scope

      vector<set<int> > submodelsViews;

      for (int subModel = 0; subModel < nSubmodels; ++subModel)
      {
         int const startView = viewIdBackMap[int((double(nAllViews) * (rand() / (RAND_MAX + 1.0))))];

         std::set<int> submodelViews;
         if (useRandomSubmodelGrowing)
            growModel(allTriplets, startView, submodelsize, submodelViews);
         else
            growModelMST(mstAdjacencyMap, allTriplets, startView, submodelsize, submodelViews);

         if (submodelViews.empty()) continue;

         submodelsViews.push_back(submodelViews);

         cout << "Views in submodel " << subModel << ": ";
         for (set<int>::const_iterator p = submodelViews.begin(); p != submodelViews.end(); ++p)
            cout << *p << " ";
         cout << endl;
      } // end for (subModel)

      int nGoodSubmodels = 0;

      for (size_t subModelId = 0; subModelId < submodelsViews.size(); ++subModelId)
      {
         cout << "----------------------------------------------------------------------" << endl;
         cout << "Computing sub-model " << subModelId << ":" << endl;

         SubmodelReconstruction subModel(submodelsViews[subModelId], allTriplets);

         subModel.computeConsistentRotations(relRotations);

         subModel.computeConsistentTranslations_L1(tripletDataCache, tripletOIDMap);
         subModel.generateSparseReconstruction(allCorrs);

         int const nPointsBefore = subModel._sparseReconstruction.size();

         int const nRequiredViews = 3;
         filterInlierSparsePoints(calibDb, subModel._viewIdBackMap, maxReproError, nRequiredViews, subModel._cameras, subModel._sparseReconstruction);
         cout << "sparse reconstruction (after geometric filtering) has "
              << subModel._sparseReconstruction.size() << " 3D points." << endl;

         int const nPointsAfter = subModel._sparseReconstruction.size();

         bool success = true;

         {
            vector<int> camCounts(subModel._cameras.size(), 0);
            for (int j = 0; j < subModel._sparseReconstruction.size(); ++j)
            {
               TriangulatedPoint const& X = subModel._sparseReconstruction[j];
               for (int k = 0; k < X.measurements.size(); ++k)
                  ++camCounts[X.measurements[k].view];
            }
            cout << "camCounts = "; displayVector(camCounts);

            for (int i = 0; i < camCounts.size(); ++i)
            {
               if (camCounts[i] < nRequiredVisiblePoints)
               {
                  cout << "Number of 3D points visible in camera " << i << " is too small." << endl;
                  success = false;
                  break;
               }
            } // end for (i)

         } // end scope

         showAccuracyInformation(calibDb, subModel._viewIdBackMap, subModel._cameras, subModel._sparseReconstruction);

         if (!success) continue;

         BundlePointStructure bundleStruct(subModel._sparseReconstruction);

         char wrlName[200];
         //sprintf(wrlName, "points3d-submodel-%i.wrl", subModelId);
         //writePointsToVRML(bundleStruct.points3d, wrlName);

         if (1)
         {
            {
#if 0
               SerializableVector<CameraMatrix> savedCameras(subModel._cameras);
               vector<Vector3d> savedPoints(bundleStruct.points3d);
               ScopedBundleExtrinsicNormalizer extNormalizer(savedCameras, savedPoints);

               V3D::StdMetricBundleOptimizer opt(1.0, savedCameras, savedPoints, bundleStruct.measurements,
                                                 bundleStruct.correspondingView, bundleStruct.correspondingPoint);

               V3D::optimizerVerbosenessLevel = 0;
               opt.tau = 1e-3;
               opt.maxIterations = nIterationsBA;
               opt.minimize();
               cout << "optimizer status = " << opt.status << endl;

               // Keep the normalized positions, but scale everything
               double const scale = subModel._cameras.size() + savedPoints.size();
               for (int i = 0; i < savedCameras.size(); ++i)
               {
                  Vector3d c = savedCameras[i].cameraCenter();
                  savedCameras[i].setCameraCenter(scale * c);
               }
               for (int j = 0; j < savedPoints.size(); ++j)
                  scaleVectorIP(scale, savedPoints[j]);

               subModel._cameras = savedCameras;
               bundleStruct.points3d = savedPoints;
#else

               V3D::StdMetricBundleOptimizer opt(1.0, subModel._cameras, bundleStruct.points3d, bundleStruct.measurements,
                                                 bundleStruct.correspondingView, bundleStruct.correspondingPoint);

               V3D::optimizerVerbosenessLevel = 0;
               opt.tau = 1e-3;
               opt.maxIterations = nIterationsBA;
               opt.minimize();
               cout << "optimizer status = " << opt.status << endl;
#endif
            } // end scope

#if 0
            vector<float> norms(bundleStruct.points3d.size());
            for (size_t i = 0; i < bundleStruct.points3d.size(); ++i)
            {
               Vector3d& X = bundleStruct.points3d[i];
               norms[i] = norm_L2(X);
            }
            std::sort(norms.begin(), norms.end());
            float distThr = norms[int(norms.size() * 0.9f)];
            //cout << "90% quantile distance: " << distThr << endl;
            for (size_t i = 0; i < bundleStruct.points3d.size(); ++i)
            {
               Vector3d& X = bundleStruct.points3d[i];
               if (norm_L2(X) > 3*distThr) makeZeroVector(X);
            }            
            sprintf(wrlName, "ba-points3d-submodel-%i.wrl", int(subModelId));
            cout << "Writing " << wrlName << endl;
            writePointsToVRML(bundleStruct.points3d, wrlName);
#endif
         } // end if

         bundleStruct.createPointStructure(subModel._sparseReconstruction, true);
         showAccuracyInformation(calibDb, subModel._viewIdBackMap, subModel._cameras, subModel._sparseReconstruction);

         // If the number of surviving 3D points is much smaller after geometric filtering,
         // assume that something went wrong...
         if (double(nPointsAfter) < double(nPointsBefore)*minFilteredRatio)
         {
            cout << "Number of surviving 3D points dropped substantially." << endl;
            success = false;
         }
//          if (success)
//          {
//             vector<int> camCounts(subModel._cameras.size(), 0);
//             for (int j = 0; j < subModel._sparseReconstruction.size(); ++j)
//             {
//                TriangulatedPoint const& X = subModel._sparseReconstruction[j];
//                for (int k = 0; k < X.measurements.size(); ++k)
//                   ++camCounts[X.measurements[k].view];
//             }
//             for (int i = 0; i < camCounts.size(); ++i)
//             {
//                if (camCounts[i] < nRequiredVisiblePoints)
//                {
//                   cout << "Number of 3D points visible in camera " << i << " is too small." << endl;
//                   success = false;
//                   break;
//                }
//             } // end for (i)
//          }

         if (success)
         {
            // Use the number of good submodels to have finally a compact range for submodel IDs.
            // This greatly simplifies the following processing steps.
            cout << "Using OID = " << nGoodSubmodels << " for submodel " << subModelId << endl;
            unsigned long const oid = nGoodSubmodels;
            submodelsTable.updateObject(oid, subModel);
            ++nGoodSubmodels;
         }
         else
            cout << "Submodel " << subModelId << " is a bad reconstruction and ignored." << endl;
      } // end for (subModelId)
      cout << nGoodSubmodels << "(out of " << submodelsViews.size() << ") submodels were generated." << endl;
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
