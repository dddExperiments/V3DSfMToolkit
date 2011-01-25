#include "reconstruction_common.h"

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_storage.h"
#include "Base/v3d_timer.h"
#include "Math/v3d_linear.h"
#include "Math/v3d_sparseeig.h"
#include "Math/v3d_optimization.h"
#include "Geometry/v3d_poseutilities.h"
#include "Geometry/v3d_mviewutilities.h"
#include "Geometry/v3d_mviewinitialization.h"
#include "Geometry/v3d_metricbundle.h"

#include <iostream>
#include <sstream>
#include <list>

using namespace std;
using namespace V3D;

namespace
{

   void writeCameraFrustumToVRML(CameraMatrix const& cam, float width, float height, double scale,
                                 Vector3f const& color, const char *filename, bool append)
   {
      using namespace std;

      ofstream os;

      os.open(filename, append ? ios::app : ios::out);
      os.precision(15);

      if (!append) os <<"#VRML V2.0 utf8" << endl;
      os << " Shape {" << endl;
      os << " appearance Appearance {" << endl;
      os << "  material Material { emissiveColor 0.2 0.2 0.2" << endl;
      os << "  diffuseColor " << color[0]/255 << " " << color[1]/255 << " " << color[2]/255 << endl;
      os << "  }" << endl;
      os << " }" << endl;

      os <<" geometry IndexedFaceSet {" << endl;
      os <<" solid FALSE " << endl;
      os <<" ccw TRUE " << endl;
      os <<" coord Coordinate {" << endl;
      os <<" point [" << endl;

      Vector3d const center = cam.cameraCenter();
      Vector3d const ll = cam.unprojectPixel(makeVector2(0.0, double(height)), scale);
      Vector3d const lr = cam.unprojectPixel(makeVector2(double(width), double(height)), scale);
      Vector3d const ul = cam.unprojectPixel(makeVector2(0.0, 0.0), scale);
      Vector3d const ur = cam.unprojectPixel(makeVector2(double(width), 0.0), scale);

      // write point coordinates
      os << ll[0] << " " << ll[1] << " " << ll[2] << "," << endl;
      os << lr[0] << " " << lr[1] << " " << lr[2] << "," << endl;
      os << ul[0] << " " << ul[1] << " " << ul[2] << "," << endl;
      os << ur[0] << " " << ur[1] << " " << ur[2] << "," << endl;
      os << center[0] << " " << center[1] << " " << center[2] << "," << endl;

      os << " ]\n } "<< endl;

      os << "  coordIndex [\n";
      os << "  0,1,2,3,-1" << endl;
      os << "  0,1,4,-1" << endl;
      os << "  1,3,4,-1" << endl;
      os << "  3,2,4,-1" << endl;
      os << "  2,0,4,-1\n  ]" << endl;
      os << " }\n}" << endl;
   } // end writeCameraFrustumToVRML()

} // end namespace <>

int
main(int argc, char * argv[])
{
   if (argc != 4 && argc != 5)
   {
      cerr << "Usage: " << argv[0] << " <config file> <first view no> <last view no> [<edges black list>]" << endl;
      return -1;
   }

   try
   {
      ConfigurationFile cf(argv[1]);

      bool   const applyBA = cf.get("APPLY_BUNDLE", true);
      int const bundleMode = cf.get("BUNDLE_MODE", FULL_BUNDLE_METRIC);

      double maxReproError = cf.get("MAX_REPROJECTION_ERROR_RECONSTRUCT", -1.0);
      if (maxReproError < 0) maxReproError = cf.get("MAX_REPROJECTION_ERROR", 10.0);

      int nRequiredTriplePoints = cf.get("REQUIRED_TRIPLE_POINTS_RECONSTRUCT", -1);
      if (nRequiredTriplePoints < 0) nRequiredTriplePoints = cf.get("REQUIRED_TRIPLE_POINTS", 50);

      double const sigma = cf.get("SIGMA", 1.0);

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

      map<ViewPair, int> viewPairOIDMap;
      {
         // Read in all view pairs and build a mapping between view pairs and oids.
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

      set<ViewTripletKey> unhandledTriples;
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

            unhandledTriples.insert(key);
            tripletOIDMap.insert(make_pair(key, oid));

            int const i0 = key.views[0];
            int const i1 = key.views[1];
            int const i2 = key.views[2];

            pairThirdViewMap[ViewPair(i0, i1)].insert(i2);
            pairThirdViewMap[ViewPair(i0, i2)].insert(i1);
            pairThirdViewMap[ViewPair(i1, i2)].insert(i0);
         }
      } // end scope

      cout << "Computing models from " << unhandledTriples.size() << " triplets." << endl;

      typedef SQLite3_Database::Table<PairwiseMatch> MatchDataTable;
      MatchDataTable matchDataTable = matchesDB.getTable<PairwiseMatch>("matches_data");
      CachedStorage<MatchDataTable> matchDataCache(matchDataTable, 100);

      typedef SQLite3_Database::Table<TripleReconstruction> TripletDataTable;
      TripletDataTable tripletDataTable = tripletDB.getTable<TripleReconstruction>("triplet_data");
      CachedStorage<TripletDataTable> tripletDataCache(tripletDataTable, 100);

      int componentId = 0;

      while (!unhandledTriples.empty())
      {
         cout << "Starting a new component." << endl;

         // Breadth-first search for connected components

         set<ViewTripletKey> collectedTriples;
         set<ViewPair> handledEdges;

         extractConnectedComponent(pairThirdViewMap, unhandledTriples, collectedTriples, handledEdges);

         cout << "Component has " << collectedTriples.size() << " triples." << endl;

         // Collect views and view pairs relevant for this component.
         set<int> compViewIds;
         set<ViewPair> compViewPairs;
         for (set<ViewTripletKey>::const_iterator p = collectedTriples.begin(); p != collectedTriples.end(); ++p)
         {
            int const v0 = (*p).views[0];
            int const v1 = (*p).views[1];
            int const v2 = (*p).views[2];
            compViewIds.insert(v0);
            compViewIds.insert(v1);
            compViewIds.insert(v2);

            compViewPairs.insert(ViewPair(v0, v1));
            compViewPairs.insert(ViewPair(v0, v2));
            compViewPairs.insert(ViewPair(v1, v2));
         }
         cout << "This component has " << compViewIds.size() << " views: ";
         for (set<int>::const_iterator p = compViewIds.begin(); p != compViewIds.end(); ++p)
            cout << (*p) << " ";
         cout << endl;
         cout << "This component uses " << compViewPairs.size() << " view pairs." << endl;

         int const nCompViews = compViewIds.size();

         // Map view ids to the range [0, N-1]
         map<int, int> viewIdMap;
         vector<int> viewIdBackMap(nCompViews);
         for (set<int>::const_iterator p = compViewIds.begin(); p != compViewIds.end(); ++p)
         {
            int newId = viewIdMap.size();
            viewIdMap.insert(make_pair(*p, newId));
            viewIdBackMap[newId] = *p;
         }         

         vector<PointCorrespondence> allCompCorrs;
         map<ViewPair, Matrix3x3d>   relRotations;
         map<ViewPair, Vector3d>     relTranslations;

         // Read in all relevant pairwise data
         for (set<ViewPair>::const_iterator p = compViewPairs.begin(); p != compViewPairs.end(); ++p)
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
               // Already bring view ids to the [0..N-1] range
               c.left.view = viewIdMap[c.left.view];
               c.right.view = viewIdMap[c.right.view];
               allCompCorrs.push_back(c);
            } // end for (k)
            relRotations.insert(make_pair(key, matchData->rotation));
            relTranslations.insert(make_pair(key, matchData->translation));
         } // end for (p)

         cout << allCompCorrs.size() << " correspondences are remaining." << endl;

         vector<Matrix3x3d> rotations;
         vector<Vector3d>   translations;

         vector<pair<int, int> >  viewPairVec;
         map<pair<int, int>, int> viewPairVecPosMap;

         cout << "Computing consistent rotations..." << endl;
         {
            vector<Matrix3x3d> tmpRelRotations;
            
            for (set<ViewPair>::const_iterator p = compViewPairs.begin(); p != compViewPairs.end(); ++p)
            {
               int const i0 = viewIdMap[(*p).view0];
               int const i1 = viewIdMap[(*p).view1];
               viewPairVecPosMap.insert(make_pair(make_pair(i0, i1), viewPairVec.size()));
               viewPairVec.push_back(make_pair(i0, i1));
               tmpRelRotations.push_back(relRotations[*p]);
            }

            Timer t("computeConsistentRotations()");
            t.start();
            int const method = V3D_CONSISTENT_ROTATION_METHOD_SPARSE_EIG;
            V3D::computeConsistentRotations(nCompViews, tmpRelRotations, viewPairVec, rotations, method);
            t.stop();
            t.print();
         }
         cout << "done." << endl;

         translations.resize(nCompViews, Vector3d(0, 0, 0));

         vector<Vector3d> c_ji, c_jk;
         vector<Vector3i> ijks;

         for (set<ViewTripletKey>::const_iterator p = collectedTriples.begin(); p != collectedTriples.end(); ++p)
         {
            int const v0 = (*p).views[0];
            int const v1 = (*p).views[1];
            int const v2 = (*p).views[2];

            int const i0 = viewIdMap[v0];
            int const i1 = viewIdMap[v1];
            int const i2 = viewIdMap[v2];

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
            Vector3d c01 = rotations[i1].transposed() * (-T01);
            Vector3d c02 = rotations[i2].transposed() * (-T02);
            Vector3d c12 = rotations[i2].transposed() * (-T12);

            Vector3d c10 = rotations[i0].transposed() * (-T10);
            Vector3d c20 = rotations[i0].transposed() * (-T20);
            Vector3d c21 = rotations[i1].transposed() * (-T21);

            ijks.push_back(makeVector3(i0, i1, i2)); c_ji.push_back(c10); c_jk.push_back(c12);
            ijks.push_back(makeVector3(i2, i0, i1)); c_ji.push_back(c02); c_jk.push_back(c01);
            ijks.push_back(makeVector3(i1, i2, i0)); c_ji.push_back(c21); c_jk.push_back(c20);
         } // end for (p)

         vector<Vector3d> centers(nCompViews);

         {
            MultiViewInitializationParams_BOS params;
            params.verbose = true;
            params.nIterations = 50000;

            Timer t("computeConsistentCameraCenters()");
            t.start();
            //computeConsistentCameraCenters_L1(c_ji, c_jk, ijks, centers, true);
            computeConsistentCameraCenters_L2_BOS(c_ji, c_jk, ijks, centers, params);
            t.stop();
            t.print();
         }

         for (int i = 0; i < nCompViews; ++i) translations[i] = rotations[i] * (-centers[i]);

         vector<TriangulatedPoint> sparseReconstruction;
         int const nRequiredViews = std::min(3, nCompViews);
         TriangulatedPoint::connectTracks(allCompCorrs, sparseReconstruction, nRequiredViews);
         cout << "sparse reconstruction (before logical filtering) has " << sparseReconstruction.size() << " 3D points." << endl;

         filterConsistentSparsePoints(sparseReconstruction);
         cout << "sparse reconstruction (after logical filtering) has " << sparseReconstruction.size() << " 3D points." << endl;

//          for (int j = 0; j < sparseReconstruction.size(); ++j)
//          {
//             cout << "X" << j << ": ";
//             VRTriangulatedPoint& X = sparseReconstruction[j];
//             for (int l = 0; l < X.measurements.size(); ++l)
//             {
//                VRImageMeasurement const& m = X.measurements[l];
//                cout << "(" << m.view << ": " << m.id << ") ";
//             }
//             cout << endl;
//          }

         vector<CameraMatrix> cameras(nCompViews);
         Matrix3x3d K; makeIdentityMatrix(K);

         for (int i = 0; i < nCompViews; ++i)
         {
            cameras[i].setIntrinsic(K);
            cameras[i].setRotation(rotations[i]);
            cameras[i].setTranslation(translations[i]);
         }

         for (int i = 0; i < sparseReconstruction.size(); ++i)
            sparseReconstruction[i].pos = triangulateLinear(cameras, sparseReconstruction[i].measurements);

         filterInlierSparsePoints(calibDb, viewIdBackMap, maxReproError, nRequiredViews, cameras, sparseReconstruction);
         cout << "sparse reconstruction (after geometric filtering) has " << sparseReconstruction.size() << " 3D points." << endl;

         {
            vector<int> camCounts(cameras.size(), 0);
            for (int j = 0; j < sparseReconstruction.size(); ++j)
            {
               TriangulatedPoint const& X = sparseReconstruction[j];
               for (int k = 0; k < X.measurements.size(); ++k)
                  ++camCounts[X.measurements[k].view];
            }
            cout << "camCounts = "; displayVector(camCounts);
         } // end scope

         showAccuracyInformation_Linf(calibDb, viewIdBackMap, cameras, sparseReconstruction, sigma*1.02);
         //showAccuracyInformation(K, cameras, sparseReconstruction, sigma);
         //showAccuracyInformation(focalLength, cameras, sparseReconstruction);

         BundlePointStructure bundleStruct(sparseReconstruction);

         char wrlName[200];
         sprintf(wrlName, "points3d-%i.wrl", componentId);

         writePointsToVRML(bundleStruct.points3d, wrlName);

         sprintf(wrlName, "cams-%i.wrl", componentId);
         Vector3f camColor = makeVector3(100.0f, 50.0f, 50.0f);
         for (int i = 0; i < cameras.size(); ++i)
            writeCameraFrustumToVRML(cameras[i], 0.5, 0.5, 0.25, camColor, wrlName, i > 0);

         if (applyBA)
         {
            cout << "cameras.size() = " << cameras.size() << endl;
            cout << "bundleStruct.points3d.size() = " << bundleStruct.points3d.size() << endl;
            cout << "bundleStruct.measurements.size() = " << bundleStruct.measurements.size() << endl;

            {
#if 1
               vector<CameraMatrix> savedCameras(cameras);
               vector<Vector3d> savedPoints(bundleStruct.points3d);
               ScopedBundleExtrinsicNormalizer extNormalizer(savedCameras, savedPoints);

               V3D::optimizerVerbosenessLevel = 1;

# if 0
               V3D::StdMetricBundleOptimizer opt(1.0, savedCameras, savedPoints, bundleStruct.measurements,
                                                 bundleStruct.correspondingView, bundleStruct.correspondingPoint);
# else
               if (0) {
                  StdDistortionFunction distortion;
                  V3D::CommonInternalsMetricBundleOptimizer opt(FULL_BUNDLE_NO_ROTATIONS, 1.0, K, distortion,
                                                                savedCameras, savedPoints, bundleStruct.measurements,
                                                                bundleStruct.correspondingView, bundleStruct.correspondingPoint);
                  opt.tau = 1e-3;
                  opt.maxIterations = 20;
                  opt.minimize();
                  cout << "optimizer status = " << opt.status << endl;
               }

               StdDistortionFunction distortion;
               V3D::CommonInternalsMetricBundleOptimizer opt(bundleMode, 1.0, K, distortion,
                                                             savedCameras, savedPoints, bundleStruct.measurements,
                                                             bundleStruct.correspondingView, bundleStruct.correspondingPoint);
# endif
               opt.tau = 1e-3;
               opt.maxIterations = 100;
               opt.minimize();
               cout << "optimizer status = " << opt.status << endl;
               if (bundleMode > 0) cout << "New intrinsic: "; displayMatrix(K);

               // Keep the normalized positions, but scale everything
               double const scale = cameras.size() + savedPoints.size();
               for (int i = 0; i < savedCameras.size(); ++i)
               {
                  Vector3d c = savedCameras[i].cameraCenter();
                  savedCameras[i].setCameraCenter(scale * c);
               }
               for (int j = 0; j < savedPoints.size(); ++j)
                  scaleVectorIP(scale, savedPoints[j]);

               cameras = savedCameras;
               bundleStruct.points3d = savedPoints;
#else
               V3D::StdMetricBundleOptimizer opt(1.0, cameras, bundleStruct.points3d, bundleStruct.measurements,
                                                 bundleStruct.correspondingView, bundleStruct.correspondingPoint);
               V3D::optimizerVerbosenessLevel = 1;
               opt.tau = 1e-3;
               opt.maxIterations = 100;
               opt.minimize();
               cout << "optimizer status = " << opt.status << endl;
               if (bundleMode > 0) cout << "New intrinsic: "; displayMatrix(K);
#endif
            } // end scope

            vector<float> norms(bundleStruct.points3d.size());

            for (size_t i = 0; i < bundleStruct.points3d.size(); ++i)
            {
               Vector3d& X = bundleStruct.points3d[i];
               norms[i] = norm_L2(X);
            }
            std::sort(norms.begin(), norms.end());
            float distThr = norms[int(norms.size() * 0.9f)];
            cout << "90% quantile distance: " << distThr << endl;

            for (size_t i = 0; i < bundleStruct.points3d.size(); ++i)
            {
               Vector3d& X = bundleStruct.points3d[i];
               if (norm_L2(X) > 3*distThr) makeZeroVector(X);
            }            

            sprintf(wrlName, "ba-points3d-%i.wrl", componentId);
            writePointsToVRML(bundleStruct.points3d, wrlName);

            sprintf(wrlName, "ba-cams-%i.wrl", componentId);
            for (int i = 0; i < cameras.size(); ++i)
               writeCameraFrustumToVRML(cameras[i], 0.5, 0.5, 10.25, camColor, wrlName, i > 0);
         } // end if

         SerializableVector<TriangulatedPoint> finalReconstruction;
         bundleStruct.createPointStructure(finalReconstruction, true);

         //showAccuracyInformation(focalLength, cameras, finalReconstruction);
         showAccuracyInformation_Linf(calibDb, viewIdBackMap, cameras, finalReconstruction, sigma*1.02);

         {
            char buf[1024];
            sprintf(buf, "sparse-model-%i.bin", componentId);
            ofstream os(buf, ios::binary);
            BinaryOStreamArchive ar(os);
            serializeVector(viewIdBackMap, ar);
            serializeVector(cameras, ar);
            serializeVector(finalReconstruction, ar);
         } // end scope

         {
            char name[200];
            sprintf(name, "model-%i-cams.txt", componentId);
            ofstream os(name);
            os << nCompViews << endl;
            for (int i = 0; i < nCompViews; ++i)
            {
               os << viewIdBackMap[i] << " ";
               Matrix3x4d RT = cameras[i].getOrientation();
               os << RT[0][0] << " " << RT[0][1] << " " << RT[0][2] << " " << RT[0][3] << endl;
               os << RT[1][0] << " " << RT[1][1] << " " << RT[1][2] << " " << RT[1][3] << endl;
               os << RT[2][0] << " " << RT[2][1] << " " << RT[2][2] << " " << RT[2][3] << endl;
            }
         } // end scope

         {
            char name[200];
            sprintf(name, "model-%i-points.txt", componentId);
            ofstream os(name);

            os << finalReconstruction.size() << endl;
            os.precision(10);

            Vector3f p;

            for (size_t i = 0; i < finalReconstruction.size(); ++i)
            {
               TriangulatedPoint const& X = finalReconstruction[i];
               os << X.pos[0] << " " << X.pos[1] << " " << X.pos[2] << " " << X.measurements.size() << " ";
               for (int k = 0; k < X.measurements.size(); ++k)
               {
                  PointMeasurement m = X.measurements[k];

                  Matrix3x3d const K = calibDb.getIntrinsic(viewIdBackMap[m.view]);
                  multiply_A_v_affine(K, m.pos, p);

                  m.pos[0] = p[0]; m.pos[1] = p[1];
                  os << m.view << " " << m.id << " " << m.pos[0] << " " << m.pos[1] << " ";
               }
               os << endl;
            }
         } // end scope

         ++componentId;
      } // end while
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
