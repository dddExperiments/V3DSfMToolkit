#include "inference_common.h"
#include "graphutils.h"

#include "Base/v3d_utilities.h"

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/property_map.hpp>
#include <boost/graph/connected_components.hpp>

#include <iterator>

using namespace std;
using namespace V3D;

namespace
{

   template <typename T>
   struct PropertyMapFromVector
   {
         typedef T value_type;
         typedef T& reference;
         typedef int key_type;
         typedef boost::read_write_property_map_tag category;

         PropertyMapFromVector(std::vector<T>& vec_)
            : vec(vec_)
         { }

         std::vector<T>& vec;
   }; // end struct PropertyMapFromVector

   template <typename T>
   inline T&
   get(PropertyMapFromVector<T>& pmap, int key) { return pmap.vec[key]; }

   template <typename T>
   inline void
   put(PropertyMapFromVector<T>& pmap, int key, T val) { pmap.vec[key] = val; }

} // end namespace <>

void
computeMST(int const nNodes, std::vector<std::pair<int, int> > const& edges, std::vector<double> const& weights,
           std::vector<int>& parentNodes)
{
   using namespace boost;
   typedef adjacency_list < vecS, vecS, undirectedS,
      property<vertex_distance_t, int>, property<edge_weight_t, double> > Graph;
   typedef std::pair<int, int> E;

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
   Graph g(nNodes);
   property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g); 
   for (std::size_t j = 0; j < sizeof(edges) / sizeof(E); ++j) {
      graph_traits<Graph>::edge_descriptor e; bool inserted;
      tie(e, inserted) = add_edge(edges[j].first, edges[j].second, g);
      weightmap[e] = weights[j];
   }
#else
   Graph g(edges.begin(), edges.end(), weights.begin(), nNodes);
   property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
#endif
   std::vector<graph_traits< Graph >::vertex_descriptor> p(num_vertices(g));

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
   property_map<Graph, vertex_distance_t>::type distance = get(vertex_distance, g);
   property_map<Graph, vertex_index_t>::type indexmap = get(vertex_index, g);
   prim_minimum_spanning_tree(g, *vertices(g).first, &p[0], distance, weightmap, indexmap, 
                              default_dijkstra_visitor());
#else
   prim_minimum_spanning_tree(g, &p[0]);
#endif

//       for (std::size_t i = 0; i != p.size(); ++i)
//          if (p[i] != i)
//             std::cout << "parent[" << i << "] = " << p[i] << std::endl;
//          else
//             std::cout << "parent[" << i << "] = no parent" << std::endl;

   parentNodes.resize(nNodes);

   for (std::size_t i = 0; i != p.size(); ++i)
   {
      if (p[i] != i)
         parentNodes[i] = p[i];
      else
         parentNodes[i] = -1;
   }

//    vector<int> heightHist(num_nodes, 0);
//    for (size_t i = 0; i < p.size(); ++i)
//    {
//       int h = 0;
//       int n = p[i];
//       while (n != p[n]) { ++h; n = p[n]; }
//       ++heightHist[h];
//    }
//    cout << "height histogram: "; displayVector(heightHist);
} // end computeMST()

void
computeMST(vector<pair<int, int> > const& edges, vector<double> const& weights, map<int, int>& parentNodes)
{
   parentNodes.clear();
   V3D::CompressedRangeMapping range;

   vector<pair<int, int> > newEdges(edges.size());

   for (int k = 0; k < edges.size(); ++k)
   {
      int const node1 = range.addElement(edges[k].first);
      int const node2 = range.addElement(edges[k].second);
      newEdges[k] = make_pair(node1, node2);
   }
   int const nNodes = range.size();
   vector<int> parents;
   computeMST(nNodes, newEdges, weights, parents);

   for (int k = 0; k < parents.size(); ++k)
   {
      if (parents[k] >= 0)
      {
         int const c = range.toOrig(k);
         int const p = range.toOrig(parents[k]);
         parentNodes.insert(make_pair(c, p));
      }
   } // end for (k)
} // end computeMST()

//----------------------------------------------------------------------

namespace
{

   struct NormalizedLoop
   {
         NormalizedLoop(vector<int> const& loop)
         {
            _loop.reserve(loop.size());

            back_insert_iterator<vector<int> > ii(_loop);

            vector<int>::const_iterator minPos = min_element(loop.begin(), loop.end());
            std::copy(minPos, loop.end(), ii);
            std::copy(loop.begin(), minPos, ii);
         }

         vector<int> _loop;
   }; // end struct NormalizedLoop

   bool
   operator<(NormalizedLoop const& a, NormalizedLoop const& b)
   {
      return a._loop < b._loop;
   }

   bool
   operator==(NormalizedLoop const& a, NormalizedLoop const& b)
   {
      return a._loop == b._loop;
   }

} // end namespace <>

void
drawLoops(vector<pair<int, int> > const& edges_, vector<double> const& initialWeights,
          vector<pair<int, bool> >& loopEdges, LoopSamplerParams const& params)
{
   loopEdges.clear();

   set<NormalizedLoop> drawnLoops;

   set<int> nodeSet;
   for (int i = 0; i < edges_.size(); ++i)
   {
      int const v0 = edges_[i].first;
      int const v1 = edges_[i].second;

      nodeSet.insert(v0);
      nodeSet.insert(v1);
   }

   // Bring every node id to a compressed range
   CompressedRangeMapping idMap;

   // Add nodes in sorted order to keep monotonicity of ids
   for (set<int>::const_iterator p = nodeSet.begin(); p != nodeSet.end(); ++p) idMap.addElement(*p);

   vector<pair<int, int> > edges(edges_.size());
   map<pair<int, int>, int > edgePosMap;

   for (int i = 0; i < edges.size(); ++i)
   {
      int const v0 = idMap.toCompressed(edges_[i].first);
      int const v1 = idMap.toCompressed(edges_[i].second);

      edgePosMap.insert(make_pair(make_pair(v0, v1), i));
      edges[i] = make_pair(v0, v1);
      //cout << "[" << v0 << ", " << v1 << "]" << endl;
   } // end for (i)

   int const nNodes = idMap.size();

   vector<int> curLoop;
   int nDuplicates = 0;

   // Generate all 3-loops first
   for (map<pair<int, int>, int>::const_iterator p01 = edgePosMap.begin(); p01 != edgePosMap.end(); ++p01)
   {
      int const v0 = (*p01).first.first;
      int const v1 = (*p01).first.second;

      int const vMax = std::max(v0, v1);

      for (int v2 = vMax+1; v2 < nNodes; ++v2)
      {
         //cout << "(" << v0 << ", " << v1 << ", " << v2 << ")" << endl;

         map<pair<int, int>, int>::const_iterator p02 = edgePosMap.find(make_pair(v0, v2));
         if (p02 == edgePosMap.end()) continue;

         map<pair<int, int>, int>::const_iterator p12 = edgePosMap.find(make_pair(v1, v2));
         if (p12 == edgePosMap.end()) continue;

         curLoop.resize(3);
         curLoop[0] = p01->second;
         curLoop[1] = p12->second;
         curLoop[2] = p02->second;

         if (drawnLoops.find(NormalizedLoop(curLoop)) != drawnLoops.end())
         {
            ++nDuplicates;
            continue;
         }

         loopEdges.push_back(make_pair(3, false)); // Length first
         loopEdges.push_back(make_pair(p01->second, false));
         loopEdges.push_back(make_pair(p12->second, false));
         loopEdges.push_back(make_pair(p02->second, true));
      } // end for (i2)
   } // end for (p01)

   if (params.nTrees == 0) return;

   vector<double> weights(initialWeights);

   for (int tree = 0; tree < params.nTrees; ++ tree)
   {
      //cout << "tree = " << tree << "..." << endl;

      vector<pair<int, int> > mstEdges;
      vector<std::set<int> > connComponents;

      getMinimumSpanningForest(edges, weights, mstEdges, connComponents);

      vector<set<int> > neighborNodes(nNodes);
      for (size_t i = 0; i < mstEdges.size(); ++i)
      {
         int const v0 = mstEdges[i].first;
         int const v1 = mstEdges[i].second;
         neighborNodes[v0].insert(v1);
         neighborNodes[v1].insert(v0);
      } // end for (i)

      //cout << "mstEdges.size() = " << mstEdges.size() << ", connComponents.size() = " << connComponents.size() << endl;

//       for (int i = 0; i < mstEdges.size(); ++i)
//          cout << "[" << mstEdges[i].first << ", " << mstEdges[i].second << "]" << endl;

      for (int component = 0; component < connComponents.size(); ++component)
      {
         set<int> const& connComponent = connComponents[component];

         vector<int> nodeDepths(nNodes, -1);
         vector<int> parentNodes(nNodes);
         // Traverse (breadth first) MST starting with node 0
         list<int> nodeQueue;

         int const vStart = *connComponent.begin(); // root node
         nodeQueue.push_back(vStart);
         nodeDepths[vStart] = 0;
         parentNodes[vStart] = vStart;
         while (!nodeQueue.empty())
         {
            int const v = nodeQueue.front();
            nodeQueue.pop_front();

            set<int> const& neighbors = neighborNodes[v];
            for (set<int>::const_iterator p = neighbors.begin(); p != neighbors.end(); ++p)
            {
               if (nodeDepths[*p] < 0)
               {
                  parentNodes[*p] = v;
                  nodeDepths[*p] = nodeDepths[v] + 1;
                  nodeQueue.push_back(*p);
               }
            } // end for (p)
         } // end while()

//          for (int i = 0; i < nNodes; ++i)
//             cout << "parentNodes[" << i << "] = " << parentNodes[i] << ", nodeDepth = " << nodeDepths[i] << endl;

         vector<int> branch0, branch1;

         // Sample loops defined over this MST
         for (set<int>::const_iterator p1 = connComponent.begin(); p1 != connComponent.end(); ++p1)
         {
            if (p1 == connComponent.end()) break;

            int const v0 = *p1;

            set<int>::const_iterator p2 = p1;
            ++p2;
            for (; p2 != connComponent.end(); ++p2)
            {
               int const v1 = *p2;

               if (edgePosMap.find(make_pair(v0, v1)) == edgePosMap.end()) continue;

               //cout << "--------------------" << endl;

               int d0 = nodeDepths[v0];
               int node0 = v0;

               int d1 = nodeDepths[v1];
               int node1 = v1;

               branch0.clear();
               branch1.clear();

               while (node0 != node1)
               {
                  //cout << "d0 = " << d0 << ", d1 = " << d1 << endl;
                  if (d0 > d1)
                  {
                     branch0.push_back(node0);
                     //cout << "Adding branch0 link " << node0 << " -> " << parentNodes[node0] << endl;
                     node0 = parentNodes[node0];
                     --d0;
                  }
                  else
                  {
                     branch1.push_back(node1);
                     //cout << "Adding branch1 link " << node1 << " -> " << parentNodes[node1] << endl;
                     node1 = parentNodes[node1];
                     --d1;
                  } // end if
               } // end while
               branch0.push_back(node0);

               //cout << "branch0 = "; displayVector(branch0);
               //cout << "branch1 = "; displayVector(branch1);

               for (int i = branch1.size() - 1; i >= 0; --i)
                  branch0.push_back(branch1[i]);

               int const loopLength = branch0.size();
               //cout << "v0 = " << v0 << ", v1 = " << v1 << ", loopLength = " << loopLength << endl;

               if (loopLength <= 3 || loopLength > params.maxLoopLength) continue;
               //displayVector(branch0);

               if (drawnLoops.find(NormalizedLoop(branch0)) != drawnLoops.end())
               {
                  ++nDuplicates;
                  continue;
               }

               // emit the loop
               loopEdges.push_back(make_pair(loopLength, false));
               for (int i = 0; i < loopLength; ++i)
               {
                  int const v0 = branch0[i];
                  int const v1 = (i < loopLength-1) ? branch0[i+1] : branch0[0];

                  map<pair<int, int>, int>::const_iterator p;
                  p = edgePosMap.find(make_pair(v0, v1));
                  if (p != edgePosMap.end())
                     loopEdges.push_back(make_pair(p->second, false));
                  else
                  {
                     p = edgePosMap.find(make_pair(v1, v0));
                     if (p != edgePosMap.end())
                        loopEdges.push_back(make_pair(p->second, true));
                     else
                        cout << "Oops, did not find edge index for (" << v0 << ", " << v1 << ")." << endl;
                  }
               } // end for (i)
            } // end for (p2)
         } // end for (p1)
      } // end for (component)

      for (int i = 0; i < mstEdges.size(); ++i)
      {
         int const v0 = mstEdges[i].first;
         int const v1 = mstEdges[i].second;

         map<pair<int, int>, int>::const_iterator p = edgePosMap.find(make_pair(v0, v1));
         if (p != edgePosMap.end())
            weights[p->second] *= params.edgeWeightFactor;
         else
         {
            p = edgePosMap.find(make_pair(v1, v0));
            if (p != edgePosMap.end())
               weights[p->second] *= params.edgeWeightFactor;
            else
               cout << "Oops, could not find MST edge." << endl;
         }
      } // end for (i)
   } // end for (tree)

   cout << nDuplicates << " loops were duplicates." << endl;
} // end drawLoops()
