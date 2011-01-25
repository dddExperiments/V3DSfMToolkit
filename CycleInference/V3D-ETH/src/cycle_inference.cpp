#include "Base/v3d_exception.h"
#include "cycle_inference.h"

#include "Math/v3d_linear.h"

#include "dai/alldai.h"
#include "dai/bp.h"
#include "dai/jtree.h"
#include "dai/exactinf.h"

extern "C"
{
#include "lpsolve/lp_lib.h"
}

#include <cstdio>
#include <fstream>
#include <set>

using namespace std;
using namespace V3D;

namespace
{

   typedef PathInference::ComplexFactor ComplexFactor;

   // A branch & bound node for 0-1 integer problems
   struct BB_Node
   {
         BB_Node(vector<ComplexFactor> const& complexFactors)
            : _complexFactors(complexFactors), _nLoops(complexFactors.size())
         { }

         unsigned long id;
         std::set<int> vars0, vars1; // Ids for variables clamped to 0 or 1, respectively.

         vector<ComplexFactor> const& _complexFactors;
         int const _nLoops;

         double evalRelaxedCost(vector<double> const& costs) const
         {
            // We forget about the constraints and utilize a greedy 0-1 assignment on the variables

            int const nVars = costs.size() - 1; // Note that vars start at 1
            int const nNodes = nVars - _nLoops;

            int const xStart = 1;
            int const yStart = xStart + nNodes;

            double res = 0;
            for (int i = xStart; i < yStart; ++i)
            {
               if (vars0.find(i) != vars0.end())
               {
                  // Nothing
               }
               else if (vars1.find(i) != vars1.end())
                  res += costs[i];
               else
                  res += std::min(0.0, costs[i]);
            }

            for (int i = yStart; i < costs.size(); ++i)
               res += std::min(0.0, costs[i]);

            return res;
         }

         int getFirstConflictingVar(vector<double> const& costs) const
         {
            int const nVars = costs.size() - 1; // Note that vars start at 1
            int const nNodes = nVars - _nLoops;

            int const xStart = 1;
            int const yStart = xStart + nNodes;

            vector<bool> solution(nVars+1, false);
            for (int i = xStart; i < yStart; ++i)
            {
               if (vars0.find(i) != vars0.end())
               {
                  // Nothing
               }
               else if (vars1.find(i) != vars1.end())
                  solution[i] = true;
               else
                  solution[i] = (costs[i] > 0) ? true : false;
            }

            for (int i = yStart; i < costs.size(); ++i)
               solution[i] = (costs[i] > 0) ? true : false;

            for (int k = 0; k < _complexFactors.size(); ++k)
            {
               ComplexFactor const& factor = _complexFactors[k];
               bool const loopStatus = solution[yStart + k];
               bool accumStatus = false;

               for (list<int>::const_iterator p = factor.vars.begin(); p != factor.vars.end(); ++p)
               {
                  accumStatus = accumStatus || solution[(*p) + 1];
                  if (accumStatus && !loopStatus) return (*p) + 1;
               } // end for (p)

               if (accumStatus != loopStatus) return factor.vars.front()+1;
            } // end for (k)

            return -1;
         } // end getFirstConflictingVar()
   }; // end struct BB_Node

} // end namespace <>

//**********************************************************************

void
PathInference::generateFactors(vector<SimpleFactor>& simpleFactors, vector<ComplexFactor>& complexFactors) const
{
   simpleFactors.clear();
   complexFactors.clear();

   set<int> handledPriors;

   for (int i = 0; i < _priorNodes.size(); ++i)
   {
      int const id = _priorNodes[i];
      handledPriors.insert(id);

      if (_blackList.find(id) == _blackList.end())
      {
         double const p0 = _priorProbs[i];
         simpleFactors.push_back(make_pair(id, p0));
      }
      else
         simpleFactors.push_back(make_pair(id, 0.0));
   } // end for (i)

   for (set<int>::const_iterator p = _blackList.begin(); p != _blackList.end(); ++p)
   {
      if (handledPriors.find(*p) == handledPriors.end())
         simpleFactors.push_back(make_pair(*p, 0.0));
   }

   for (int k = 0; k < _paths.size(); ++k)
   {
      bool inBlackList = false;

      list<int> const& path = _paths[k];
      for (list<int>::const_iterator p = path.begin(); p != path.end(); ++p)
         if (_blackList.find(*p) != _blackList.end())
         {
            inBlackList = true;
            break;
         }
      if (!inBlackList)
      {
         ComplexFactor f;
         f.probPos = _probsPos[k];
         f.probNeg = _probsNeg[k];
         f.vars    = path;
         complexFactors.push_back(f);
      }
   } // end for (k)   
   cout << "simpleFactors.size() = " << simpleFactors.size() << ", complexFactors.size() = " << complexFactors.size() << endl;
} // end PathInference::generateFactors()


void
PathInference::fillObjectiveCoeffs(vector<SimpleFactor> const& simpleFactors, vector<ComplexFactor> const& complexFactors,
                                   vector<double>& obj) const
{
   int const nPaths = complexFactors.size();
   int const nNodes = _varRange.size();
   int const nVars = nNodes + nPaths;
   obj.resize(nVars+1, 0.0);

   int const xStart = 1;
   int const yStart = xStart + nNodes;

#define XVAR(i) (xStart+(i))
#define YVAR(i) (yStart+(i))

   // Set the objective function
   for (int i = 0; i < simpleFactors.size(); ++i)
   {
      int const id = simpleFactors[i].first;
      double const p0 = simpleFactors[i].second;
      double const p1 = 1.0 - p0;
      double bias = log(p0) - log(p1);
      bias = std::max(-1000.0, std::min(1000.0, bias));
      obj[XVAR(id)] = bias;
   }

   for (int k = 0; k < nPaths; ++k)
   {
      double const p0 = complexFactors[k].probPos;
      double const p1 = complexFactors[k].probNeg;

      double bias = log(p0) - log(p1);
      bias = std::max(-1000.0, std::min(1000.0, bias));
      obj[YVAR(k)] = bias;
   } // end for (k)
} // end PathInference::fillObjectiveCoeffs()

PathInference::Result
PathInference::runInferenceLP(bool useBnB, double timeout) const
{
   vector<SimpleFactor> simpleFactors;
   vector<ComplexFactor> complexFactors;
   this->generateFactors(simpleFactors, complexFactors);

   int const nPaths = complexFactors.size();
   int const nNodes = _varRange.size();
   int const nVars = nNodes + nPaths;

   vector<double> X_lp(nVars);
   vector<double> obj(nVars+1, 0.0);

   int const xStart = 1;
   int const yStart = xStart + nNodes;

#define XVAR(i) (xStart+(i))
#define YVAR(i) (yStart+(i))

   lprec * lp = make_lp(0, nVars);
   set_add_rowmode(lp, TRUE);

   if (timeout > 0) set_timeout(lp, timeout);

   {
      this->fillObjectiveCoeffs(simpleFactors, complexFactors, obj);
      set_obj_fn(lp, &obj[0]);

      if (!useBnB)
      {
         for (int i = 0; i < nNodes; ++i) set_bounds(lp, XVAR(i), 0.0, 1.0);
      }
      else
      {
         for (int i = 0; i < nNodes; ++i) set_binary(lp, XVAR(i), 1);
      }
      for (int k = 0; k < nPaths; ++k) set_bounds(lp, YVAR(k), 0.0, 1.0);
   } // end scope

   {
      // Add y_l >= x_i constraints for i in the loop l

      double row[2];
      int    colno[2];

      for (int k = 0; k < nPaths; ++k)
      {
         list<int> const& vars = complexFactors[k].vars;
         for (list<int>::const_iterator p = vars.begin(); p != vars.end(); ++p)
         {
            row[0] = 1.0; colno[0] = YVAR(k);
            row[1] = -1.0; colno[1] = XVAR(*p);
            add_constraintex(lp, 2, row, colno, GE, 0.0);
         }
      } // end for (k)
   }

   {
      // Add sum x_i >= y_l constraints for i in the loop l
      vector<double> row(nNodes+1);
      vector<int>    colno(nNodes+1);

      for (int k = 0; k < nPaths; ++k)
      {
         list<int> const& vars = complexFactors[k].vars;
         int pos = 0;
         for (list<int>::const_iterator p = vars.begin(); p != vars.end(); ++p, ++pos)
         {
            row[pos] = 1.0; colno[pos] = XVAR(*p);
         }
         row[pos] = -1.0; colno[pos] = YVAR(k);
         add_constraintex(lp, vars.size()+1, &row[0], &colno[0], GE, 0.0);
      } // end for (k)

   }

   set_add_rowmode(lp, FALSE);

   set_verbose(lp, 4);
   int res = solve(lp);
   cout << "lp return code = " << res << endl;

   double * Y;
   get_ptr_primal_solution(lp, &Y);
   int m = get_Nrows(lp);

   for (int i = 0; i < nNodes; ++i) X_lp[i] = Y[m + XVAR(i)];
   for (int i = 0; i < nPaths; ++i) X_lp[i+nNodes] = Y[m + YVAR(i)];
   //cout << "X_lp = "; displayVector(X_lp);

   delete_lp(lp);

   if (res != OPTIMAL)
   {
      cout << "PathInference::runInferenceLP(): non-optimal result, probably due to timeout." << endl;
      return Result();
   }

   {
      double val = 0.0;
      for (int i = 0; i < X_lp.size(); ++i) val += obj[i+1]*X_lp[i];
      cout << "Objective value for LP solution = " << val << endl;

      int nViolations = 0;
      for (int k = 0; k < nPaths; ++k)
      {
         list<int> const& vars = complexFactors[k].vars;
         double y_L = 0.0;

         for (list<int>::const_iterator p = vars.begin(); p != vars.end(); ++p)
            y_L = std::max(y_L, X_lp[*p]);
         if (y_L + 1e-6 < X_lp[k+nNodes])
         {
            ++nViolations;
            //cout << "max(x_e) = " << y_L << ", x_L = " << X[k+nEdges] << endl;
         }
      } // end for (k)
      cout << "nViolations = " << nViolations << endl;
   } // end scope

   Result result;

   for (int i = 0; i < nNodes; ++i)
      result.insert(make_pair(_varRange.toOrig(i), X_lp[i]));

   return result;
} // end PathInference::runInferenceLP()

PathInference::Result
PathInference::runInferenceBP(size_t maxIter) const
{
   cout << "Running loopy BP..." << endl;

   vector<SimpleFactor> simpleFactors;
   vector<ComplexFactor> complexFactors;
   this->generateFactors(simpleFactors, complexFactors);

   int const nPaths = complexFactors.size();
   int const nNodes = _varRange.size();

   // All variables are binary
   std::vector<dai::Factor> allFactors;
   std::vector<dai::Var> allVars(nNodes);

   for (int i = 0; i < nNodes; ++i)
   {
      allVars[i].label() = i;
      allVars[i].states() = 2;
   }

   double const eps = 1e-10;

   // Add loop factors
   for (int k = 0; k < nPaths; ++k)
   {
      list<int> const& varList = complexFactors[k].vars;

      double const p0 = complexFactors[k].probPos + eps;
      double const p1 = complexFactors[k].probNeg + eps;

      vector<size_t> vars(varList.size());
      std::copy(varList.begin(), varList.end(), vars.begin());
      std::sort(vars.begin(), vars.end());

      int const tableSize = 1 << vars.size();

#if 0
      // Explicit normalization
      double const np0 = p0 / ((tableSize-1.0)*p1 + p0);
      double const np1 = p1 / ((tableSize-1.0)*p1 + p0);
#else
      double const np0 = p0;
      double const np1 = p1;
#endif

      vector<double> probs(tableSize, np1);
      probs.back() = np0;

      //displayVector(vars); cout << "rho = " << rho << " p0 = " << p0 << " p1 = " << p1 << endl;

      std::vector<dai::Var> factorVars(vars.size());
      for (int i = 0; i < vars.size(); ++i)
      {
         factorVars[i].label() = vars[i];
         factorVars[i].states() = 2;
      }

      allFactors.push_back(dai::Factor(dai::VarSet(factorVars.begin(), factorVars.end()), &probs[0]));
   }

   // Add unary priors
   for (int i = 0; i < simpleFactors.size(); ++i)
   {
      int const id = simpleFactors[i].first;
      //double probs[2] = { _priorProbs[i], 1.0 - _priorProbs[i] };
      double probs[2] = { 1.0 - simpleFactors[i].second + eps, simpleFactors[i].second + eps };
      allFactors.push_back(dai::Factor(allVars[id], probs));
   }

   using namespace dai;
   FactorGraph fg(allFactors.begin(), allFactors.end(), allVars.begin(), allVars.end());

   //size_t  maxiter = 100000;
   double  tol = 1e-6;
   size_t  verb = 1;

   PropertySet opts;
   opts.set("maxiter", maxIter);
   opts.set("tol", tol);
   opts.set("verbose", verb);
   opts.set("inference", string("MAXPROD"));

#if 1
   opts.set("updates", string("SEQFIX"));
   opts.set("logdomain", true);
   //opts.Set("damping", 0.1);

   //BP bp(fg, opts("updates", string("SEQFIX"))("logdomain", true));
   BP bp(fg, opts);
#elif 0
   opts.set("updates", string("HUGIN"));
   JTree bp(fg, opts);
#else
   ExactInf bp(fg, opts);
#endif
   bp.init();

//    if (initializeBP_fromLP)
//    {
//             FactorGraph& fg = bp.fg();
//             double c = props.logdomain ? 0.0 : 1.0;
//             for (int i = 0; i < nEdges; ++i)
//             {
//                double const c = bp.props.logdomain ? X_lp[i] : X_lp[i];
//                foreach( const dai::Neighbor &I, fg.nbV(i) )
//                {
//                   message( i, I.iter ).fill( c );
//                   newMessage( i, I.iter ).fill( c );
//                }
//             }
//    } // end if (initializeBP_fromLP)

   bp.run();

   Result result;

   for (int i = 0; i < nNodes; ++i)
   {
      Factor const& b = bp.belief(fg.var(i));
      result.insert(make_pair(_varRange.toOrig(i), b[0]));
   } // end scope

   {
      vector<double> obj;
      this->fillObjectiveCoeffs(simpleFactors, complexFactors, obj);

      int const nPaths = complexFactors.size();
      int const nNodes = _varRange.size();
      int const nVars = nNodes + nPaths;

      vector<double> X_bp(nVars, 0.0);

      for (int i = 0; i < nNodes; ++i)
      {
         Factor const& b = bp.belief(fg.var(i));
         X_bp[i] = b[0];
         //X_bp[i] = (b[0] > 0.5) ? 1.0 : 0.0;
      } // end scope

      for (int k = 0; k < nPaths; ++k)
      {
         list<int> const& vars = complexFactors[k].vars;
         int const yVar = k + nNodes;

         for (list<int>::const_iterator p = vars.begin(); p != vars.end(); ++p)
            X_bp[yVar] = std::max(X_bp[yVar], X_bp[*p]);
      } // end for (k)

      //cout << "obj = "; displayVector(obj);
      //cout << "X_bp = "; displayVector(X_bp);

      double val = 0.0;
      for (int i = 0; i < X_bp.size(); ++i) val += obj[i+1]*X_bp[i];
      cout << "Objective value for BP solution = " << val << endl;
   } // end scope

   return result;
} // end PathInference::runInferenceBP()

PathInference::Result
PathInference::getAvgError(double const maxRatio) const
{
   Result g;
   for (int k = 0; k < _paths.size(); ++k)
   {
      list<int> const& path = _paths[k];

      double ratio = -(log(_probsPos[k]) - log(_probsNeg[k]));
      ratio = std::max(-maxRatio, std::min(maxRatio, ratio));

      for (list<int>::const_iterator p = path.begin(); p != path.end(); ++p)
      {
         int const id = _varRange.toOrig(*p);
         if (g.find(id) == g.end())
            g.insert(make_pair(id, ratio));
         else
            g[id] += ratio;
      }
   } // end for (k)

   for (Result::iterator p = g.begin(); p != g.end(); ++p)
   {
      int const count = _pathCounts.find(p->first)->second;
      p->second /= std::max(1, count);
   }

   return g;
} // end PathInference::getAvgError()

//**********************************************************************

void
EdgeCycleInference::emitAvgErrorToDot(double const maxRatio) const
{
   Graph g;
   for (int k = 0; k < _inference._paths.size(); ++k)
   {
      list<int> const& path = _inference._paths[k];

      double ratio = -(log(_inference._probsPos[k]) - log(_inference._probsNeg[k]));
      ratio = std::max(-maxRatio, std::min(maxRatio, ratio));

      for (list<int>::const_iterator p = path.begin(); p != path.end(); ++p)
      {
         int const id = _inference._varRange.toOrig(*p);
         pair<int, int> const& e = _ID_edgeMap[id];
         if (g.find(e) == g.end())
            g.insert(make_pair(e, ratio));
         else
            g[e] += ratio;
      }
   } // end for (k)

   for (Graph::iterator p = g.begin(); p != g.end(); ++p)
   {
      int const id = _edgeID_Map.find(p->first)->second;
      int const count = _inference._pathCounts.find(id)->second;
      p->second /= std::max(1, count);
   }

   char name[200];
   sprintf(name, "avg_error_%s.dot", _name.c_str());

   ofstream os(name);
   emitGraph(g, os);
} // end EdgeCycleInference::emitAvgErrorToDot()

void
EdgeCycleInference::emitBeliefToDot(Graph const& g, char const * spec) const
{
   char name[200];
   sprintf(name, "belief_%s_%s.dot", _name.c_str(), spec);

   ofstream os(name);
   emitGraph(g, os);
}

void
EdgeCycleInference::writeBlackList(Graph const& g, double rejectThreshold, char const * filename)
{
   ofstream os(filename);

   for (Graph::const_iterator p = g.begin(); p != g.end(); ++p)
   {
      double const belief = p->second;
      if (belief > rejectThreshold)
         os << p->first.first << " " << p->first.second << endl;
   }
}
