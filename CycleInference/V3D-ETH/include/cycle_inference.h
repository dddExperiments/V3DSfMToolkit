// -*- C++ -*-
#ifndef CYCLE_INFERENCE_H
#define CYCLE_INFERENCE_H

#include "graphutils.h"

#include "Base/v3d_utilities.h"

#include <vector>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <string>

struct PathInference
{
   public:
      typedef std::map<int, double> Result;

      PathInference()
      { }

      void addPrior(int var, double priorProb)
      {
         int const id = _varRange.addElement(var);
         _priorNodes.push_back(id);
         _priorProbs.push_back(priorProb);
      }

      void addPath(std::vector<int> const& origPath, double probGood, double probBad)
      {
         using namespace std;

         probGood = std::max(1e-300, probGood);
         probBad  = std::max(1e-300, probBad);

         std::list<int> path;
         for (int k = 0; k < origPath.size(); ++k)
         {
            int const var = origPath[k];
            int const id  = _varRange.addElement(var);

            path.push_back(id);

            if (_pathCounts.find(var) == _pathCounts.end())
               _pathCounts.insert(make_pair(var, 1));
            else
               ++_pathCounts[var];
         }
         _paths.push_back(path);
         _probsPos.push_back(probGood);
         _probsNeg.push_back(probBad);
      }

      void clearBlackList() { _blackList.clear(); }
      void extendBlackList(std::set<int> const& blackList)
      {
         for (std::set<int>::const_iterator p = blackList.begin(); p != blackList.end(); ++p)
         {
            int const id = _varRange.addElement(*p);
            _blackList.insert(id);
         }
      }

      Result runInferenceLP(bool useBnB = false, double timeout = 0) const;
      Result runInferenceBP(size_t maxIter = 10000) const;

      Result getAvgError(double const maxRatio = 3.0) const;

      int factorCount() const { return _paths.size() + _priorNodes.size(); }

      std::map<int, int> const& getPathCounts() const { return _pathCounts; }

      friend struct EdgeCycleInference;

      typedef std::pair<int, double> SimpleFactor;

      struct ComplexFactor
      {
            std::list<int> vars;
            double probPos, probNeg;
      };

   protected:
      void generateFactors(std::vector<SimpleFactor>& simpleFactors, std::vector<ComplexFactor>& complexFactors) const;

      void fillObjectiveCoeffs(std::vector<SimpleFactor> const& simpleFactors, std::vector<ComplexFactor> const& complexFactors,
                               std::vector<double>& obj) const;

      std::vector<int>    _priorNodes;
      std::vector<double> _priorProbs;

      std::vector<std::list<int> > _paths;
      std::vector<double>          _probsPos;
      std::vector<double>          _probsNeg;
      std::map<int, int>           _pathCounts;

      std::set<int> _blackList;

      std::map<std::pair<int, int>, int> _edgeID_Map;
      std::vector<std::pair<int, int> >  _ID_edgeMap;

      V3D::CompressedRangeMapping _varRange;
}; // end struct PathInference

struct EdgeCycleInference
{
   public:
      typedef V3D::Graph Result;

      EdgeCycleInference(char const * name)
         : _name(name)
      { }

      void addEdgePrior(std::pair<int, int> const& e, double priorProb)
      {
         int const id = this->registerEdge(e);
         _inference.addPrior(id, priorProb);
      }

      void addEdgeCycle(std::vector<int> const& cycle, double probGood, double probBad)
      {
         using namespace std;

         vector<int> path;

         for (int k = 0; k < cycle.size(); ++k)
         {
            int v1 = cycle[k];
            int v2 = (k < cycle.size() - 1) ? cycle[k+1] : cycle[0];
            if (v1 > v2) std::swap(v1, v2);

            pair<int, int> const e = make_pair(v1, v2);

            int const id = this->registerEdge(e);
            path.push_back(id);
         }
         _inference.addPath(path, probGood, probBad);
      }

      Result runInferenceLP(bool useBnB = false, double timeout = 0) const
      {
         using namespace std;

         PathInference::Result const belief = _inference.runInferenceLP(useBnB, timeout);
         Result g;
         for (PathInference::Result::const_iterator p = belief.begin(); p != belief.end(); ++p)
         {
            pair<int, int> const& e = _ID_edgeMap[p->first];
            g.insert(make_pair(e, p->second));
         }
         return g;
      }

      Result runInferenceBP(size_t maxIter = 10000) const
      {
         using namespace std;

         PathInference::Result const belief = _inference.runInferenceBP(maxIter);
         Result g;
         for (PathInference::Result::const_iterator p = belief.begin(); p != belief.end(); ++p)
         {
            pair<int, int> const& e = _ID_edgeMap[p->first];
            g.insert(make_pair(e, p->second));
         }
         return g;
      }

      void clearBlackList() { _inference.clearBlackList(); }
      void extendBlackList(std::set<std::pair<int, int> > const& blackList)
      {
         std::set<int> idBlackList;
         for (std::set<std::pair<int, int> >::const_iterator p = blackList.begin(); p != blackList.end(); ++p)
         {
            int const id = this->registerEdge(*p);
            idBlackList.insert(id);
         }
         _inference.extendBlackList(idBlackList);
      }

      void emitAvgErrorToDot(double const maxRatio = 3.0) const;
      void emitBeliefToDot(V3D::Graph const& g, char const * spec) const;

      static void writeBlackList(V3D::Graph const& g, double rejectThreshold, char const * filename);

      int factorCount() const { return _inference.factorCount(); }

   protected:
      int registerEdge(std::pair<int, int> e)
      {
         using namespace std;

         if (e.first > e.second) std::swap(e.first, e.second);

         int id = -1;

         map<pair<int, int>, int>::const_iterator p = _edgeID_Map.find(e);
         if (p != _edgeID_Map.end())
            id = p->second;
         else
         {
            id = _edgeID_Map.size();
            _edgeID_Map.insert(make_pair(e, id));
            _ID_edgeMap.push_back(e);
         }
         return id;
      } // end registerEdge()

      std::string _name;

      PathInference _inference;

      std::map<std::pair<int, int>, int> _edgeID_Map;
      std::vector<std::pair<int, int> >  _ID_edgeMap;
}; // end struct EdgeCycleInference

#endif
