#include "reconstruction_common.h"
#include "cycle_inference.h"
#include "inference_common.h"

#include <iostream>
#include <sstream>

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_storage.h"
#include "Math/v3d_mathutilities.h"

using namespace std;
using namespace V3D;

namespace
{

   template <typename Mat>
   inline typename Mat::value_type
   matrixSqrNorm_Frob(Mat const& A)
   {
      typename Mat::value_type res(0);
      for (int i = 0; i < A.num_rows(); ++i)
         for (int j = 0; j < A.num_cols(); ++j)
            res += A[i][j]*A[i][j];
      return res;
   }

} // end namespace <>


int
main(int argc, char * argv[])
{
   int const nMinCorrs = 20;

   try
   {
      SQLite3_Database matchesDB("pairwise_matches.db");
      SQLite3_Database tripletDB("triplets.db");

      map<ViewPair, int> viewPairOIDMap;

      int nViews = 0;

      {
         typedef SQLite3_Database::Table<ViewPair> Table;
         Table table = matchesDB.getTable<ViewPair>("matches_list");
         for (Table::const_iterator p = table.begin(); bool(p); ++p)
         {
            int const oid = (*p).first;
            ViewPair pair = (*p).second;
            int const view1 = pair.view0;
            int const view2 = pair.view1;

            nViews = std::max(nViews, view1 + 1);
            nViews = std::max(nViews, view2 + 1);

            viewPairOIDMap.insert(make_pair(pair, oid));
         }
      } // end scope
      cout << "Considering = " << viewPairOIDMap.size() << " view pairs." << endl;

      Matrix3x3d I; makeIdentityMatrix(I);

      typedef SQLite3_Database::Table<PairwiseMatch> MatchDataTable;
      MatchDataTable matchDataTable = matchesDB.getTable<PairwiseMatch>("matches_data");

      CachedStorage<MatchDataTable> matchDataCache(matchDataTable, 100);

      ofstream loopDataFile("loopdata-EG.txt");

      EdgeCycleInference inference("EG");
//       for (map<ViewPair, int>::const_iterator p01 = viewPairOIDMap.begin(); p01 != viewPairOIDMap.end(); ++p01)
//       {
//          int const v0    = (*p01).first.view0;
//          int const v1    = (*p01).first.view1;

//          double const prior = 0.95;
//          inference.addEdgePrior(make_pair(v0, v1), prior);
//       }

      vector<double> allErrors;

      vector<pair<int, int> > edges;
      vector<double> initialWeights;
      for (map<ViewPair, int>::const_iterator p01 = viewPairOIDMap.begin(); p01 != viewPairOIDMap.end(); ++p01)
      {
         int const v0    = (*p01).first.view0;
         int const v1    = (*p01).first.view1;
         int const oid01 = (*p01).second;

         PairwiseMatch * match01Ptr = matchDataCache[oid01];
         if (!match01Ptr || match01Ptr->corrs.size() < nMinCorrs) continue;

         edges.push_back(make_pair(v0, v1));
         initialWeights.push_back(-1.0 * match01Ptr->corrs.size());
      } // end for (p01)

      vector<std::pair<int, bool> > loopEdges;

      drawLoops(edges, initialWeights, loopEdges);

      for (size_t pos = 0; pos < loopEdges.size(); )
      {
         Matrix3x3d accumRot;
         makeIdentityMatrix(accumRot);

         vector<int> cycle;

         int const len = loopEdges[pos].first;
         ++pos;
         for (int i = 0; i < len; ++i)
         {
            pair<int, int> const& edge = edges[loopEdges[pos+i].first];
            bool const reverse = loopEdges[pos+i].second;

            cycle.push_back(reverse ? edge.second : edge.first);

            map<ViewPair, int>::const_iterator pOid = viewPairOIDMap.find(ViewPair(edge.first, edge.second));
            int const oid = (*pOid).second;
            PairwiseMatch const& matchData = *matchDataCache[oid];

            Matrix3x3d Rrel = matchData.rotation;
            if (reverse) Rrel = Rrel.transposed();

            accumRot = Rrel * accumRot;
         } // end for (i)
         pos += len;

         double const angle012 = 180 / M_PI * getRotationMagnitude(accumRot);

         loopDataFile << len << " ";
         for (int i = 0; i < len; ++i) loopDataFile << cycle[i] << " ";
         loopDataFile << angle012 << endl;

         double const lambda = 0.5;
         double const p_good = lambda * exp(-lambda*angle012);
         double const p_bad = 1.0 / 180.0;
         allErrors.push_back(angle012);
         inference.addEdgeCycle(cycle, p_good, p_bad);
      } // end for (pos)

      double const rejectThreshold = 0.2;

      inference.emitAvgErrorToDot();

      char const * infNames[] = { "lp", "bnb", "bp" };
      int  const nInferenceIterations[] = { 2, 1, 3 };

      int const infMethod = 2;
      //for (int infMethod = 0; infMethod < 3; ++infMethod)
      {
         inference.clearBlackList();

         char blackListName[200];
         char dotName[200];

         for (int iter = 0; iter < nInferenceIterations[infMethod]; ++iter)
         {
            sprintf(blackListName, "blacklist_EG_%s-%i.txt", infNames[infMethod], iter);
            sprintf(dotName, "%s-%i", infNames[infMethod], iter);

            Graph g;
            switch (infMethod)
            {
               case 0:
                  g = inference.runInferenceLP();
                  break;
               case 1:
                  g = inference.runInferenceLP(true);
                  break;
               case 2:
                  g = inference.runInferenceBP(1000);
                  break;
            } // end switch

            inference.emitBeliefToDot(g, dotName);
            inference.writeBlackList(g, rejectThreshold, blackListName);

            set<pair<int, int> > blackList;
            for (Graph::const_iterator p = g.begin(); p != g.end(); ++p)
            {
               if (p->second > rejectThreshold)
                  blackList.insert(p->first);
            }
            inference.extendBlackList(blackList);
         } // end for (iter)
      } // end for (infMethod)

      {
         ofstream os("all_EG_errors.m");
         os << "v = [";
         for (int k = 0; k < allErrors.size(); ++k) os << allErrors[k] << " ";
         os << "];" << endl;
      }
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
