// -*- C++ -*-
#ifndef GRAPH_UTILS
#define GRAPH_UTILS

#include <vector>
#include <map>
#include <iostream>
#include <cassert>

namespace V3D
{
   using namespace std;

   typedef map<pair<int, int>, double> Graph;

   inline void
   displayGraph(Graph const& g, ostream& os = std::cout)
   {
      for (Graph::const_iterator p = g.begin(); p != g.end(); ++p)
         os << p->first.first << " -- " << p->first.second << ": " << p->second << endl;
   }

   inline void
   emitGraph(Graph const& g, ostream& os)
   {
      os << "graph {" << endl;

      double minVal = 1e300;
      double maxVal = -1e300;

      for (Graph::const_iterator p = g.begin(); p != g.end(); ++p)
      {
         minVal = std::min(minVal, p->second);
         maxVal = std::max(maxVal, p->second);
      }
      double const range = std::max(0.001, maxVal - minVal);

      for (Graph::const_iterator p = g.begin(); p != g.end(); ++p)
      {
         os << p->first.first << " -- " << p->first.second;
         //double const intensity = std::max(0.0, (p->second - minVal) / range);
         //double const intensity = 0.3 + 0.7*(p->second - minVal) / range;
         double const intensity = 0.4 + 0.6*(p->second - minVal) / range;
         os << " [color = \"0, 0, " << 1.0-intensity*intensity << "\"];" << endl;
      }

      os << "}" << endl;
   }

} // end namespace V3D

#endif
