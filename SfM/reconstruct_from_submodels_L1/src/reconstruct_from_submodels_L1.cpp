#include "reconstruction_common.h"

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_timer.h"
#include "Base/v3d_utilities.h"
#include "Geometry/v3d_mviewutilities.h"
#include "Geometry/v3d_mviewinitialization.h"
#include "Geometry/v3d_metricbundle.h"

#include <queue>
#include <iostream>
#include <sstream>
#include <list>

using namespace std;
using namespace V3D;

namespace
{

   void
   extractConnectedComponent(std::map<int, std::set<int> > const& edgeMap,
                             std::set<std::pair<int, int> >& unhandledEdges,
                             std::set<std::pair<int, int> >& connectedEdges,
                             std::set<int>& handledNodes)
   {
      // Breadth-first search for connected components
      using namespace std;

      connectedEdges.clear();
      handledNodes.clear();

      list<int> nodeQueue;

      pair<int, int> startEdge = *unhandledEdges.begin();
      unhandledEdges.erase(unhandledEdges.begin());
      connectedEdges.insert(startEdge);
      nodeQueue.push_back(startEdge.first);
      nodeQueue.push_back(startEdge.second);

      while (!nodeQueue.empty())
      {
         int curNode = nodeQueue.front();
         nodeQueue.pop_front();

         handledNodes.insert(curNode);

         map<int, set<int> >::const_iterator p = edgeMap.find(curNode);
         assert(p != edgeMap.end());
         set<int> const& otherNodes = (*p).second;

         for (set<int>::const_iterator q = otherNodes.begin(); q != otherNodes.end(); ++q)
         {
            int i0 = curNode;
            int i1 = *q;
            sort2(i0, i1);
            pair<int, int> key(i0, i1);

            if (connectedEdges.find(key) != connectedEdges.end()) continue;
            if (unhandledEdges.find(key) == unhandledEdges.end()) continue;

            connectedEdges.insert(key);
            unhandledEdges.erase(key);

            if (handledNodes.find(i0) == handledNodes.end()) nodeQueue.push_back(i0);
            if (handledNodes.find(i1) == handledNodes.end()) nodeQueue.push_back(i1);
         } // end for (q)
      } // end while
   } // end computeConntectedComponent()

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

      bool   const applyBA     = cf.get("APPLY_BUNDLE", true);
      int    const bundleMode  = cf.get("BUNDLE_MODE", FULL_BUNDLE_METRIC);

      double maxReproError = cf.get("MAX_REPROJECTION_ERROR_RECONSTRUCT", -1.0);
      if (maxReproError < 0) maxReproError = cf.get("MAX_REPROJECTION_ERROR_TRIPLET", 10.0);

      int nRequiredAlignmentPoints = cf.get("REQUIRED_ALIGNMENT_POINTS_RECONSTRUCT", -1);
      if (nRequiredAlignmentPoints < 0) nRequiredAlignmentPoints = cf.get("REQUIRED_ALIGNMENT_POINTS", 50);
      cout << "done." << endl;

      CalibrationDatabase calibDb("calibration_db.txt");

      set<pair<int, int> > edgesBlackList;
      if (argc >= 3)
      {
         cout << "Reading alignment black list file..." << endl;
         ifstream is(argv[2]);
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
         cout << "Read " << edgesBlackList.size() << " entries in the submodel alignment black list."<< endl;
      }

      set<pair<int, int> > membershipBlackList;
      if (argc == 4)
      {
         cout << "Reading alignment black list file..." << endl;
         ifstream is(argv[3]);
         if (is)
         {
            while (is)
            {
               if (is.eof()) break;
               int m, v;
               is >> m >> v;
               membershipBlackList.insert(make_pair(m, v));
            }
         }
         else
            cout << "Cannot open " << argv[3] << endl;
         cout << "Read " << membershipBlackList.size() << " entries in the membership black list."<< endl;
      }

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
      vector<pair<int, int> >  allSubModelPairs;
      map<pair<int, int>, int> subModelPairPosMap;

      vector<vector<int> >          allSubModelViews;
      vector<vector<CameraMatrix> > allSubModelCameras;

      SerializableVector<PointCorrespondence> allCorrs;
      serializeDataFromFile("allcorrs.bin", allCorrs);
      cout << "allCorrs.size() = " << allCorrs.size() << endl;

      for (int j1 = 0; j1 < nAllSubModels; ++j1)
      {
         SubmodelReconstruction const& subModel1 = *submodelsCache[j1];
         allSubModelViews.push_back(subModel1._viewIdBackMap);
         allSubModelCameras.push_back(subModel1._cameras);
      } // end for (j1)

      {
         double const minEdgeWeight = sqrt(double(nRequiredAlignmentPoints));

         ifstream is("subcomp_alignment.bin", ios::binary);
         BinaryIStreamArchive ar(is);

         vector<int> allSubModelPairsTmp;

         vector<Matrix3x3d> allRelRotationsTmp;
         vector<Vector3d>   allRelTranslationsTmp;
         vector<double>     allRelScalesTmp;
         vector<double>     allRelWeightsTmp;

         serializeVector(allRelRotationsTmp, ar);
         serializeVector(allRelTranslationsTmp, ar);
         serializeVector(allRelScalesTmp, ar);
         serializeVector(allRelWeightsTmp, ar);
         serializeVector(allSubModelPairsTmp, ar);

         cout << "Read in " << allSubModelPairsTmp.size() << " pairwise transformations between submodels." << endl;

         for (size_t k = 0; k < allSubModelPairsTmp.size(); ++k)
         {
            if (allRelWeightsTmp[k] < minEdgeWeight) continue;

            int const pair = allSubModelPairsTmp[k];
            int const j1 = pair & 0xffff;
            int const j2 = (pair >> 16);

            if (edgesBlackList.find(make_pair(j1, j2)) != edgesBlackList.end() ||
                edgesBlackList.find(make_pair(j2, j1)) != edgesBlackList.end()) continue;

            subModelPairPosMap.insert(make_pair(make_pair(j1, j2), allSubModelPairs.size()));
            allSubModelPairs.push_back(make_pair(j1, j2));
            allRelRotations.push_back(allRelRotationsTmp[k]);
            allRelTranslations.push_back(allRelTranslationsTmp[k]);
            allRelScales.push_back(allRelScalesTmp[k]);
            allRelWeights.push_back(allRelWeightsTmp[k]);
         } // end for (k)
         cout << allSubModelPairs.size() << " are remaining." << endl;
      } // end scope

      vector<set<int> > connSubModels;

      {
         map<int, set<int> > subcompEdgeMap;
         set<pair<int, int> > subcompEdges;
         for (int k = 0; k < allSubModelPairs.size(); ++k)
         {
            int const i0 = allSubModelPairs[k].first;
            int const i1 = allSubModelPairs[k].second;
            subcompEdges.insert(allSubModelPairs[k]);
            subcompEdgeMap[i0].insert(i1);
            subcompEdgeMap[i1].insert(i0);
         } // end for (k)

         while (!subcompEdges.empty())
         {
            set<int>             connectedNodes;
            set<pair<int, int> > connectedEdges;
            extractConnectedComponent(subcompEdgeMap, subcompEdges, connectedEdges, connectedNodes);
            connSubModels.push_back(connectedNodes);
         }
      } // end scope

      cout << "Total number of components/models: " << connSubModels.size() << endl;

      for (int componentId = 0; componentId < connSubModels.size(); ++componentId)
      {
         set<int> const& connectedNodes = connSubModels[componentId];
         set<pair<int, int> > connectedEdges; // edges between submodels in this component
         for (size_t k = 0; k < allSubModelPairs.size(); ++k)
         {
            int const m0 = allSubModelPairs[k].first;
            int const m1 = allSubModelPairs[k].second;
            if (connectedNodes.find(m0) != connectedNodes.end() &&
                connectedNodes.find(m1) != connectedNodes.end())
               connectedEdges.insert(allSubModelPairs[k]);
         }

         int const nSubModels = connectedNodes.size();

         cout << "This component has " << connectedNodes.size() << " sub-components: ";
         for (set<int>::const_iterator p = connectedNodes.begin(); p != connectedNodes.end(); ++p)
            cout << (*p) << " ";
         cout << endl;

         vector<vector<int> >          subModelViews;
         vector<vector<CameraMatrix> > subModelCameras;

         // Map submodel ids to the range [0, N-1]
         map<int, int> subcompIdMap;
         vector<int> subcompIdBackMap(nSubModels);
         for (set<int>::const_iterator p = connectedNodes.begin(); p != connectedNodes.end(); ++p)
         {
            int newId = subcompIdMap.size();
            subcompIdMap.insert(make_pair(*p, newId));
            subcompIdBackMap[newId] = *p;

            subModelViews.push_back(allSubModelViews[*p]);
            subModelCameras.push_back(allSubModelCameras[*p]);
         }

         // Map all views in this component to [0, K-1]
         CompressedRangeMapping compViewRange;
         set<int> allCompViews;
         for (int i = 0; i < subModelViews.size(); ++i)
            for (int j = 0; j < subModelViews[i].size(); ++j)
            {
               compViewRange.addElement(subModelViews[i][j]);
               allCompViews.insert(subModelViews[i][j]);
            }

         int const nCompViews = compViewRange.size();

         vector<Matrix3x3d> relRotations; // between submodels
         vector<pair<int, int> > subModelPairs;

         for (set<pair<int, int> >::const_iterator p = connectedEdges.begin(); p != connectedEdges.end(); ++p)
         {
            pair<int, int> const& pair = *p;
            int const pos = subModelPairPosMap.find(pair)->second;

            int const i0 = subcompIdMap.find(pair.first)->second;
            int const i1 = subcompIdMap.find(pair.second)->second;

            subModelPairs.push_back(make_pair(i0, i1));
            relRotations.push_back(allRelRotations[pos]);
         }
         int const nPairs = subModelPairs.size();

         vector<Matrix3x3d> rotations;
         computeConsistentRotations(nSubModels, relRotations, subModelPairs, rotations);

         map<ViewPair, int> viewPairOccurrences;
         for (int t = 0; t < nSubModels; ++t)
         {
            vector<int> const& views = subModelViews[t];
            for (int k1 = 0; k1 < views.size(); ++k1)
            {
               int const i = compViewRange.toCompressed(views[k1]);

               for (int k2 = k1+1; k2 < views.size(); ++k2)
               {
                  int const j = compViewRange.toCompressed(views[k2]);

                  ViewPair vp = (i < j) ? ViewPair(i, j) : ViewPair(j, i);
                  viewPairOccurrences[vp] += 1;
               } // end for (k2)
            } // end for (k1)
         } // end for (t)


         vector<Vector3d> c_ij;
         vector<Vector2i> ijs;
         vector<int> submodelIndices;
         vector<double> weights;
         vector<Vector3d> centers(nCompViews);

         for (int t = 0; t < nSubModels; ++t)
         {
            // Rotation from submodel frame to world system
            Matrix3x3d const Rw = rotations[t].transposed();

            vector<int> const& views = subModelViews[t];
            vector<CameraMatrix> const& cameras = subModelCameras[t];

            // Determine the approximate scale of this submodel first
            Vector3d meanCenter(0, 0, 0);
            for (int i = 0; i < cameras.size(); ++i) addVectorsIP(cameras[i].cameraCenter(), meanCenter);
            scaleVectorIP(1.0/cameras.size(), meanCenter);
            double scale = 0;
            for (int i = 0; i < cameras.size(); ++i)
               scale += sqrNorm_L2(cameras[i].cameraCenter() - meanCenter);
            scale /= cameras.size();
            scale = sqrt(scale);

            for (int k1 = 0; k1 < views.size(); ++k1)
            {
               int const i = compViewRange.toCompressed(views[k1]);
               Vector3d const Ci = cameras[k1].cameraCenter();

               for (int k2 = k1+1; k2 < views.size(); ++k2)
               {
                  int const j = compViewRange.toCompressed(views[k2]);
                  Vector3d const Cj = cameras[k2].cameraCenter();

                  double weight = (1.0/scale);
                  ViewPair vp = (i < j) ? ViewPair(i, j) : ViewPair(j, i);
                  weight /= sqrt(double(viewPairOccurrences[vp]));

                  c_ij.push_back(Rw * (Cj - Ci));
                  ijs.push_back(Vector2i(i, j));
                  submodelIndices.push_back(t);
                  weights.push_back(weight);
               } // end for (k2)
            } // end for (k1)
         } // end for (t)

         MultiViewInitializationParams_BOS params;
         params.alpha = 1.0;
         params.verbose = true;
         params.nIterations = 10000;

         Timer t("computeConsistentCameraCenters()");
         t.start();
         computeConsistentCameraCenters_L2_SDMM(nSubModels, c_ij, ijs, submodelIndices, weights, centers, params);
         t.stop();
         t.print();

         vector<CameraMatrix> compCameras(nCompViews);
         vector<int>          cameraAssignmentCount(nCompViews, 0);

         for (int k = 0; k < nSubModels; ++k)
         {
            Matrix3x3d const& R_xform = rotations[k];

            vector<CameraMatrix>& cameras = subModelCameras[k];

            for (int i = 0; i < cameras.size(); ++i)
            {
               int const subModelId = subcompIdBackMap[k]; // This is the global submodel ID
               int const viewId = subModelViews[k][i]; // This is the global view ID

               if (membershipBlackList.find(make_pair(subModelId, viewId)) != membershipBlackList.end())
                  continue;

               int const compViewId = compViewRange.toCompressed(viewId);

               Matrix3x3d const R_cam = cameras[i].getRotation();
               Matrix3x3d const R_new = R_cam * R_xform;

               CameraMatrix const& srcCam = cameras[i];
               CameraMatrix&       dstCam = compCameras[compViewId];

               if (cameraAssignmentCount[compViewId] == 0)
               {
                  dstCam = srcCam;
                  dstCam.setCameraCenter(centers[compViewId]);
                  cameraAssignmentCount[compViewId] = 1;
               }
               else
               {
                  double const N = cameraAssignmentCount[compViewId];
                  dstCam.setRotation(R_new);
                  dstCam.setCameraCenter(centers[compViewId]);
                  ++cameraAssignmentCount[compViewId];
               }
            } // end for (i)
         } // end for (k)

         SerializableVector<PointCorrespondence> compCorrs;

         for (size_t k = 0; k < allCorrs.size(); ++k)
         {
            PointCorrespondence c = allCorrs[k];

            int const v0 = c.left.view;
            int const v1 = c.right.view;

            if (allCompViews.find(v0) == allCompViews.end() || allCompViews.find(v1) == allCompViews.end()) continue;

            // Bring view ids to the [0..N-1] range
            c.left.view = compViewRange.toCompressed(c.left.view);
            c.right.view = compViewRange.toCompressed(c.right.view);
            compCorrs.push_back(c);
         } // end for (k)
         cout << compCorrs.size() << " correspondences in this component." << endl;

         vector<TriangulatedPoint> sparseReconstruction;
         int const nRequiredViews = 3;
         TriangulatedPoint::connectTracks(compCorrs, sparseReconstruction, nRequiredViews);
         cout << "sparse reconstruction (before logical filtering) has " << sparseReconstruction.size() << " 3D points." << endl;
         filterConsistentSparsePoints(sparseReconstruction);
         cout << "sparse reconstruction (after logical filtering) has " << sparseReconstruction.size() << " 3D points." << endl;

         for (int i = 0; i < sparseReconstruction.size(); ++i)
            sparseReconstruction[i].pos = triangulateLinear(compCameras, sparseReconstruction[i].measurements);

         filterInlierSparsePoints(calibDb, compViewRange.bwdMap(), maxReproError, nRequiredViews, compCameras, sparseReconstruction);
         cout << "sparse reconstruction (after geometric filtering) has " << sparseReconstruction.size() << " 3D points." << endl;

         {
            vector<int> camCounts(compCameras.size(), 0);
            for (int j = 0; j < sparseReconstruction.size(); ++j)
            {
               TriangulatedPoint const& X = sparseReconstruction[j];
               for (int k = 0; k < X.measurements.size(); ++k)
                  ++camCounts[X.measurements[k].view];
            }
            cout << "camCounts = "; displayVector(camCounts);
         } // end scope

         showAccuracyInformation(calibDb, compViewRange.bwdMap(), compCameras, sparseReconstruction);

         BundlePointStructure bundleStruct(sparseReconstruction);

         if (bundleStruct.points3d.size() == 0) continue;

         char wrlName[200];
         sprintf(wrlName, "points3d-%i.wrl", componentId);
         writePointsToVRML(bundleStruct.points3d, wrlName);

         if (applyBA)
         {
            cout << "compCameras.size() = " << compCameras.size() << endl;
            cout << "bundleStruct.points3d.size() = " << bundleStruct.points3d.size() << endl;
            cout << "bundleStruct.measurements.size() = " << bundleStruct.measurements.size() << endl;

            {
#if 1
               vector<CameraMatrix> savedCameras(compCameras);
               vector<Vector3d> savedPoints(bundleStruct.points3d);
               ScopedBundleExtrinsicNormalizer extNormalizer(savedCameras, savedPoints);

               StdDistortionFunction distortion;
               Matrix3x3d K; makeIdentityMatrix(K);
               V3D::CommonInternalsMetricBundleOptimizer opt(bundleMode, 1.0, K, distortion,
                                                             savedCameras, savedPoints, bundleStruct.measurements,
                                                             bundleStruct.correspondingView, bundleStruct.correspondingPoint);
               V3D::optimizerVerbosenessLevel = 1;
               opt.tau = 1e-3;
               opt.maxIterations = 100;
               opt.minimize();
               cout << "optimizer status = " << opt.status << endl;
               if (bundleMode > 0) cout << "New intrinsic: "; displayMatrix(K);

               // Keep the normalized positions, but scale everything
               double const scale = compCameras.size() + savedPoints.size();
               for (int i = 0; i < savedCameras.size(); ++i)
               {
                  Vector3d c = savedCameras[i].cameraCenter();
                  savedCameras[i].setCameraCenter(scale * c);
               }
               for (int j = 0; j < savedPoints.size(); ++j)
                  scaleVectorIP(scale, savedPoints[j]);

               compCameras = savedCameras;
               bundleStruct.points3d = savedPoints;
#else
               StdDistortionFunction distortion;
               Matrix3x3d K; makeIdentityMatrix(K);
               V3D::CommonInternalsMetricBundleOptimizer opt(bundleMode, 1.0, K, distortion,
                                                             compCameras, bundleStruct.points3d, bundleStruct.measurements,
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
            cout << "Writing " << wrlName << endl;
            writePointsToVRML(bundleStruct.points3d, wrlName);
         } // end if (applyBA)

         SerializableVector<TriangulatedPoint> finalReconstruction;
         bundleStruct.createPointStructure(finalReconstruction, true);

         showAccuracyInformation(calibDb, compViewRange.bwdMap(), compCameras, finalReconstruction);

         {
            char name[200];
            sprintf(name, "model-%i-cams.txt", componentId);
            ofstream os(name);
            os << nCompViews << endl;
            for (int i = 0; i < nCompViews; ++i)
            {
               os << compViewRange.toOrig(i) << " ";
               Matrix3x4d RT = compCameras[i].getOrientation();
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

                  Matrix3x3d const K = calibDb.getIntrinsic(compViewRange.toOrig(m.view));
                  multiply_A_v_affine(K, m.pos, p);

                  m.pos[0] = p[0]; m.pos[1] = p[1];
                  os << m.view << " " << m.id << " " << m.pos[0] << " " << m.pos[1] << " ";
               }
               os << endl;
            }
         } // end scope
      } // end for (componentId)
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
