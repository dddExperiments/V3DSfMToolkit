#include "reconstruction_common.h"
#include "inference_common.h"
#include "cycle_inference.h"

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_timer.h"
#include "Geometry/v3d_mviewutilities.h"
#include "Geometry/v3d_poseutilities.h"

#include <iostream>
#include <sstream>
#include <list>

using namespace std;
using namespace V3D;

namespace
{

   struct ViewPairEdgeData
   {
         int tripletOID;
         Matrix3x3d R;
         double s;
         Vector3d v0;
   }; // end struct ViewPairEdgeData

   inline Matrix3x3d
   createRotationForVectorAlignment(Vector3d const& src, Vector3d const& dst)
   {
      Vector4d q;
      Matrix3x3d R;

      createQuaternionForVectorAlignment(src, dst, q);
      createRotationMatrixFromQuaternion(q, R);
      return R;
   } // end createRotationForVectorAlignment()

} // end namespace <>

int
main(int argc, char * argv[])
{
   if (argc != 1 && argc != 2)
   {
      cerr << "Usage: " << argv[0] << " [<config file>]" << endl;
      return -1;
   }

   int viewFrequency = 5;
   int submodelsize = 20;
   bool useRandomSubmodelGrowing = false;

   bool const verbose = 0;

   try
   {
      if (argc == 2)
      {
         ConfigurationFile cf(argv[1]);
         viewFrequency = cf.get("SUBMODEL_VIEW_FREQUENCY", viewFrequency);
         submodelsize = cf.get("SUBMODEL_MAX_SIZE", submodelsize);
         useRandomSubmodelGrowing = cf.get("USE_RANDOM_SUBMODEL_GROWING", useRandomSubmodelGrowing);
      } // end if

      SQLite3_Database matchesDB("pairwise_matches.db");
      MatchDataTable matchDataTable = matchesDB.getTable<PairwiseMatch>("matches_data");
      CachedStorage<MatchDataTable> matchDataCache(matchDataTable, 1000);

      SQLite3_Database tripletDB("triplets.db");
      TripletDataTable tripletDataTable = tripletDB.getTable<TripleReconstruction>("triplet_data");
      CachedStorage<TripletDataTable> tripletDataCache(tripletDataTable, 1000);

      map<ViewPair, int> viewPairOIDMap;
      map<int, ViewPair> OID_viewPairMap;
      map<ViewPair, double> viewPairWeightMap;
      map<int, double> viewPairOIDWeightMap;

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

            ViewPair key(view1, view2);

            viewPairOIDMap.insert(make_pair(matchData.views, oid));
            OID_viewPairMap.insert(make_pair(oid, matchData.views));
            viewPairWeightMap.insert(make_pair(matchData.views, matchData.corrs.size()));
            viewPairOIDWeightMap.insert(make_pair(oid, matchData.corrs.size()));
         } // end for (p)
      } // end scope
      cout << "Considering = " << viewPairOIDMap.size() << " view pairs." << endl;

      set<int>                 allViews;
      set<ViewPair>            allViewPairs;
      map<ViewTripletKey, int> tripletOIDMap;
      map<int, ViewTripletKey> OID_tripletMap;
      set<ViewTripletKey>      allTriplets;

      {
         typedef SQLite3_Database::Table<TripletListItem> TripletListTable;
         TripletListTable tripletListTable = tripletDB.getTable<TripletListItem>("triplet_list");

         for (TripletListTable::const_iterator p = tripletListTable.begin(); bool(p); ++p)
         {
            int const oid        = (*p).first;
            TripletListItem item = (*p).second;
            ViewTripletKey   key  = item.views;

            allTriplets.insert(key);
            tripletOIDMap.insert(make_pair(key, oid));
            OID_tripletMap.insert(make_pair(oid, key));

            int const i0 = key.views[0];
            int const i1 = key.views[1];
            int const i2 = key.views[2];

            allViews.insert(i0); allViews.insert(i1); allViews.insert(i2);

            allViewPairs.insert(ViewPair(i0, i1));
            allViewPairs.insert(ViewPair(i0, i2));
            allViewPairs.insert(ViewPair(i1, i2));
         }
      } // end scope
      cout << "Considering = " << tripletOIDMap.size() << " triplets." << endl;

      int const nAllViews = allViews.size();
      vector<int> viewIdBackMap(nAllViews);
      {
         int i = 0;
         for (set<int>::const_iterator p = allViews.begin(); p != allViews.end(); ++p, ++i)
            viewIdBackMap[i] = *p;
      }

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


      vector<double> allErrors;

      char name[1000];

      set<ViewPair> blackList;

      for (int subModel = 0; subModel < nSubmodels; ++subModel)
      {
         int const startView = viewIdBackMap[int((double(nAllViews) * (rand() / (RAND_MAX + 1.0))))];

         std::set<int> submodelViews;
         if (useRandomSubmodelGrowing)
            growModel(allTriplets, startView, submodelsize, submodelViews);
         else
            growModelMST(mstAdjacencyMap, allTriplets, startView, submodelsize, submodelViews);

         vector<int> curViews(submodelViews.size());
         std::copy(submodelViews.begin(), submodelViews.end(), curViews.begin());

         set<int> curTripletOIDs;
         for (int i1 = 0; i1 < curViews.size()-2; ++i1)
         {
            int const v1 = curViews[i1];
            for (int i2 = i1+1; i2 < curViews.size()-1; ++i2)
            {
               int const v2 = curViews[i2];
               if (allViewPairs.find(ViewPair(v1, v2)) != allViewPairs.end())
               {
                  for (int i3 = i2+1; i3 < curViews.size(); ++i3)
                  {
                     int const v3 = curViews[i3];
                     map<ViewTripletKey, int>::const_iterator p = tripletOIDMap.find(ViewTripletKey(v1, v2, v3));
                     if (p != tripletOIDMap.end())
                        curTripletOIDs.insert(p->second);
                  } // end for (i3)
               } // end if
            } // end for (i2)
         } // end for (i1)
         cout << "Checking submodel with " << curTripletOIDs.size() << " triplets:" << endl;

         // Nodes in this graph are view pairs, and edges are induced via triplets
         // We use the OIDs of view pairs as their IDs

         vector<pair<int, int> >  edges;
         vector<double>           edgeWeights;
         vector<ViewPairEdgeData> edgesData;
         map<pair<int, int>, int> edgePosMap;

         for (set<int>::const_iterator p = curTripletOIDs.begin(); p != curTripletOIDs.end(); ++p)
         {
            int const oid = *p;
            if (OID_tripletMap.find(oid) == OID_tripletMap.end()) continue;
            ViewTripletKey const triplet = OID_tripletMap.find(oid)->second;

            int const v0 = triplet.views[0];
            int const v1 = triplet.views[1];
            int const v2 = triplet.views[2];

            // We add 1 to avoid oid==0. We use negative oid values to indicate reversed edges (view pairs).
            int const oid01 = viewPairOIDMap.find(ViewPair(v0, v1))->second + 1;
            int const oid02 = viewPairOIDMap.find(ViewPair(v0, v2))->second + 1;
            int const oid12 = viewPairOIDMap.find(ViewPair(v1, v2))->second + 1;

            TripleReconstruction const& tripletModel = *tripletDataCache[oid];

            assert(tripletModel.views[0] == v0);
            assert(tripletModel.views[1] == v1);
            assert(tripletModel.views[2] == v2);

            // We can assume v0 < v1 < v2
            if (v1 < v0) cerr << "v0 (" << v0 << ") > v1 (" << v1 << ")!!!" << endl;
            if (v2 < v0) cerr << "v0 (" << v0 << ") > v2 (" << v2 << ")!!!" << endl;
            if (v2 < v1) cerr << "v1 (" << v1 << ") > v2 (" << v2 << ")!!!" << endl;

            CameraMatrix cam0(tripletModel.intrinsics[0], tripletModel.orientations[0]);
            CameraMatrix cam1(tripletModel.intrinsics[1], tripletModel.orientations[1]);
            CameraMatrix cam2(tripletModel.intrinsics[2], tripletModel.orientations[2]);

            Vector3d const c0 = cam0.cameraCenter();
            Vector3d const c1 = cam1.cameraCenter();
            Vector3d const c2 = cam2.cameraCenter();

            double const d01 = distance_L2(c0, c1);
            double const d02 = distance_L2(c0, c2);
            double const d12 = distance_L2(c1, c2);

            {
               // Does one of the pairwise relations look like a pure homography?
               double const maxRatio = 10;

               double d1 = d01, d2 = d02, d3 = d12;
               sort3(d1, d2, d3);

               double const ratio = d3 / d1;
               if (ratio > maxRatio) continue;
            }
            
            double const s12 = d02 / d01;
            double const s02 = d12 / d01;
            double const s01 = d12 / d02;

#if 1
            Vector3d const c1_0 = cam0.transformPointIntoCameraSpace(c1);
            Vector3d const c2_0 = cam0.transformPointIntoCameraSpace(c2);

            Vector3d const c0_1 = cam1.transformPointIntoCameraSpace(c0);
            Vector3d const c2_1 = cam1.transformPointIntoCameraSpace(c2);

            Vector3d const c0_2 = cam2.transformPointIntoCameraSpace(c0);
            Vector3d const c1_2 = cam2.transformPointIntoCameraSpace(c1);

            Matrix3x3d const R12 = createRotationForVectorAlignment(c1_0, c2_0);
            Matrix3x3d const R02 = createRotationForVectorAlignment(c0_1, c2_1);
            Matrix3x3d const R01 = createRotationForVectorAlignment(c0_2, c1_2);
#else
            Matrix3x3d const R01 = getRelativeRotation(cam0.getRotation(), cam1.getRotation());
            Matrix3x3d const R02 = getRelativeRotation(cam0.getRotation(), cam2.getRotation());
            Matrix3x3d const R12 = getRelativeRotation(cam1.getRotation(), cam2.getRotation());
#endif

            double const weight = tripletModel.model.size();

            ViewPairEdgeData edgeData;
            pair<int, int> e;
            edgeData.tripletOID = oid;

            // (0,1) -> (0,2)
            edgeData.R = R12;
            edgeData.s = s12;
            edgeData.v0 = c1_0; normalizeVector(edgeData.v0);
            e = make_pair(oid01, oid02);
            edgePosMap.insert(make_pair(e, edges.size()));
            edges.push_back(e);
            edgeWeights.push_back(weight);
            edgesData.push_back(edgeData);

            // (2,0) -> (2,1)
            edgeData.R = R01;
            edgeData.s = s01;
            edgeData.v0 = c0_2; normalizeVector(edgeData.v0);
            e = make_pair(-oid02, -oid12);
            edgePosMap.insert(make_pair(e, edges.size()));
            edges.push_back(e);
            edgeWeights.push_back(weight);
            edgesData.push_back(edgeData);

            // (1,0) -> (1,2)
            edgeData.R = R02;
            edgeData.s = s02;
            edgeData.v0 = c0_1; normalizeVector(edgeData.v0);
            e = make_pair(-oid01, oid12);
            edgePosMap.insert(make_pair(e, edges.size()));
            edges.push_back(e);
            edgeWeights.push_back(weight);
            edgesData.push_back(edgeData);
         } // end for (k)

         std::vector<std::pair<int, bool> > loopEdges;

         LoopSamplerParams loopParams;
         loopParams.nTrees = 10; // We are not sampling trees, since MST does not support multi-edges
         loopParams.maxLoopLength = 4;
         drawLoops(edges, edgeWeights, loopEdges, loopParams);

         PathInference inference;
         set<int> visitedViewPairOIDs;

         for (size_t pos = 0; pos < loopEdges.size(); )
         {
            Matrix3x3d accumRot;
            makeIdentityMatrix(accumRot);
            double accumScale = 1.0;

            vector<int> cycle;

            int const len = loopEdges[pos].first;
            ++pos;

            // Check if this loop is not induced by just one triplet...
            int headTripletOID = edgesData[loopEdges[pos+0].first].tripletOID;
            bool properLoop = false;
            for (int i = 1; i < len; ++i)
            {
               ViewPairEdgeData const& edgeData = edgesData[loopEdges[pos+i].first];
               if (edgeData.tripletOID != headTripletOID)
               {
                  properLoop = true;
                  break;
               }
            } // end for (i)

            // bool isMyLoop = false;
            // {
            //    //int const oid = viewPairOIDMap.find(ViewPair(34, 35))->second;
            //    //int const oid = viewPairOIDMap.find(ViewPair(450, 451))->second;
            //    int const oid = viewPairOIDMap.find(ViewPair(183, 184))->second;
            //    for (int i = 0; i < len; ++i)
            //    {
            //       pair<int, int> const& edge = edges[loopEdges[pos+i].first];
            //       int const oidSrc = abs(edge.first) - 1;
            //       int const oidDst = abs(edge.second) - 1;

            //       if (oidSrc == oid || oidDst == oid)
            //       {
            //          isMyLoop = true;
            //          break;
            //       }
            //    } // end for (i)
            // } // end scope
            
            if (properLoop)
            {
               Vector3d const v0 = edgesData[loopEdges[pos].first].v0;

               //bool const reportLoop = verbose && (len >= 3) && isMyLoop;
               bool const reportLoop = verbose && (len >= 3);

               if (reportLoop)
                  cout << endl << "cycle of length " << len << ": ";

               for (int i = 0; i < len; ++i)
               {
                  pair<int, int> const& edge = edges[loopEdges[pos+i].first];
                  ViewPairEdgeData const& edgeData = edgesData[loopEdges[pos+i].first];

                  bool const reverse = loopEdges[pos+i].second;

                  bool const srcFlipped = edge.first < 0;
                  bool const dstFlipped = edge.second < 0;

                  int const oidSrc = abs(edge.first) - 1;
                  int const oidDst = abs(edge.second) - 1;

                  ViewPair const src = OID_viewPairMap[oidSrc];
                  ViewPair const dst = OID_viewPairMap[oidDst];

                  visitedViewPairOIDs.insert(oidSrc);
                  visitedViewPairOIDs.insert(oidDst);

                  double const curRatio = reverse ? (1.0 / edgeData.s) : edgeData.s;
                  Matrix3x3d const curRot = reverse ? edgeData.R.transposed() : edgeData.R;

                  if (reportLoop)
                  {
                     if (!srcFlipped)
                        cout << "(" << src.view0 << "--" << src.view1 << " --> ";
                     else
                        cout << "(" << src.view1 << "--" << src.view0 << " --> ";
                     if (!dstFlipped)
                        cout << dst.view0 << "--" << dst.view1;
                     else
                        cout << dst.view1 << "--" << dst.view0;
                     cout << ", reverse = " << (reverse ? "yes" : "no") << ", curRatio = " << curRatio << "); ";
                     cout << endl << "curRot = "; displayMatrix(curRot);
                     cout << endl;
                  }

                  cycle.push_back(reverse ? oidDst : oidSrc);

                  accumRot = curRot * accumRot;
                  accumScale *= curRatio;
               } // end for (i)

#if 0
               // For better symmetry, invert the scale if less than 1.
               if (accumScale < 1.0) accumScale = 1.0 / accumScale;
               if (reportLoop)
               {
                  cout.precision(10);
                  if (accumScale < 2)
                     cout << "accumScale = " << accumScale << endl;
                  else
                     cout << "accumScale = " << accumScale << "!!!" << endl;
                  displayMatrix(accumRot);
               }
               scaleMatrixIP(accumScale, accumRot);
               //displayMatrix(accumRot); cout << endl;
               accumRot[0][0] -= 1;
               accumRot[1][1] -= 1;
               accumRot[2][2] -= 1;
               double const err = matrixNormFrobenius(accumRot);

               if (reportLoop && err > 2.0) cout << "!!!" << endl;

               double const lambdaGood = 2.0;
               double const lambdaBad = 0.5;
               double const p_good = lambdaGood * exp(-lambdaGood*err);
               double const p_bad = lambdaBad * exp(-lambdaBad*err);
               //double const p_bad = 1.0/10;
#else
               Vector3d const v1 = accumScale * accumRot * v0;
               if (reportLoop)
               {
                  cout << "v0 = "; displayVector(v0);
                  cout << "v1 = "; displayVector(v1);
               }
               double const err = distance_L2(v0, v1);
               if (reportLoop) cout << "err = " << err << endl;
               if (reportLoop && err > 0.5)
               {
                  cout << "!!!" << endl;
                  cout << "err = " << err << endl;
                  cout << "accumScale = " << accumScale << endl;
                  cout << "accumRot = "; displayMatrix(accumRot);
               }

               double const lambdaGood = 1/0.05;
               double const lambdaBad = 1.0;
               double const p_good = lambdaGood * exp(-lambdaGood*err);
               double const p_bad = lambdaBad * exp(-lambdaBad*err);
               //double const p_bad = 0.01;
#endif
               allErrors.push_back(err);
               inference.addPath(cycle, p_good, p_bad);
            } // end if (properLoop)
            pos += len;
         } // end for (pos)

         cout << "Inference based on " << inference.factorCount() << " loops." << endl;

         // Add unary priors
         for (set<int>::const_iterator p = visitedViewPairOIDs.begin(); p != visitedViewPairOIDs.end(); ++p)
         {
            int const nCorrs = viewPairOIDWeightMap.find(*p)->second;
            double const lambda = 1.0 / 100;
            double const p_prior = lambda * exp(-lambda*nCorrs);
            inference.addPrior(*p, 1-p_prior);
         }

         double const rejectThreshold = 0.2;

         if (inference.factorCount() > 0)
         {
            PathInference::Result g;
            if (0) g = inference.runInferenceLP();
            if (0) g = inference.runInferenceLP(true, 60); // run at most one minute
            if (1) g = inference.runInferenceBP(1000);

            for (PathInference::Result::const_iterator p = g.begin(); p != g.end(); ++p)
            {
               double const belief = p->second;

               ViewPair const viewPair = OID_viewPairMap.find(p->first)->second;
               //cout << "(" << viewPair.view0 << ", " << viewPair.view1 << "): b = " << belief << endl;

               if (belief > rejectThreshold)
               {
                  blackList.insert(viewPair);
                  cout << "(" << viewPair.view0 << ", " << viewPair.view1 << ") added to black list." << endl;
               }
            }
         } // end if
      } // end for (submodel)

      {
         ofstream os("blacklist_EG_stage2.txt");
         for (set<ViewPair>::const_iterator p = blackList.begin(); p != blackList.end(); ++p)
            os << p->view0 << " " << p->view1 << endl;
      }

      {
         ofstream os("all_EG_stage2_errors.m");
         os << "v = [";
         for (int k = 0; k < allErrors.size(); ++k) os << allErrors[k] << " ";
         os << "];" << endl;
      }
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
