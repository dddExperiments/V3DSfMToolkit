#include "reconstruction_common.h"

#include "Base/v3d_utilities.h"
#include "Math/v3d_mathutilities.h"

using namespace std;
using namespace V3D;

namespace
{

   template <typename T> inline T sqr(T x) { return x*x; }

#define INF 1E20

   // dt of 1d function using sqrd distance
   inline void
   dt1d(float const * f, int n, float * d)
   {
      int *v = new int[n];
      float *z = new float[n+1];
      int k = 0;
      v[0] = 0;
      z[0] = -INF;
      z[1] = +INF;
      for (int q = 1; q <= n-1; q++)
      {
         float s  = ((f[q]+sqr(q))-(f[v[k]]+sqr(v[k])))/(2*q-2*v[k]);
         while (s <= z[k])
         {
            --k;
            s = ((f[q]+sqr(q))-(f[v[k]]+sqr(v[k])))/(2*q-2*v[k]);
         }
         ++k;
         v[k] = q;
         z[k] = s;
         z[k+1] = +INF;
      }

      k = 0;
      for (int q = 0; q <= n-1; q++)
      {
         while (z[k+1] < q)
            k++;
         d[q] = sqr(q-v[k]) + f[v[k]];
      }

      delete [] v;
      delete [] z;
   }

   // dt of 1d function using sqrd distance
   inline void
   dt1d(float const * f, int n, float spacing, float * d)
   {
      float const sqrSpacing = sqr(spacing);

      int   * v = new int[n];
      float * z = new float[n+1];

      float num, denom;

      int k = 0;
      v[0] = 0;
      z[0] = -INF;
      z[1] = +INF;
      for (int q = 1; q <= n-1; ++q)
      {
         num   = (f[q]+sqr(q))-(f[v[k]]+sqr(v[k]));
         denom = 2*(q-v[k]);
         float s = num / denom;
         while (s <= z[k])
         {
            --k;
            num   = (f[q]+sqr(q))-(f[v[k]]+sqr(v[k]));
            denom = 2*(q-v[k]);
            s = num / denom;
         }
         ++k;
         v[k] = q;
         z[k] = s;
         z[k+1] = +INF;
      }

      k = 0;
      for (int q = 0; q <= n-1; ++q)
      {
         while (z[k+1] < q) ++k;
         d[q] = sqrSpacing*sqr(q-v[k]) + f[v[k]];
      }

      delete [] v;
      delete [] z;
   }

   // dt of 2d function using sqrd distance
   inline void
   dt2d(Image<float>& im)
   {
      int const width = im.width();
      int const height = im.height();

      float * f = new float[std::max(width,height)];
      float * d = new float[std::max(width,height)];

      // transform along columns
      for (int x = 0; x < width; x++)
      {
         for (int y = 0; y < height; y++)
            f[y] = im(x, y);

         dt1d(f, height, d);

         for (int y = 0; y < height; y++)
            im(x, y) = d[y];
      }

      // transform along rows
      for (int y = 0; y < height; y++)
      {
         for (int x = 0; x < width; x++)
            f[x] = im(x, y);

         dt1d(f, width, d);

         for (int x = 0; x < width; x++)
            im(x, y) = d[x];
      }

      delete [] d;
      delete [] f;
   }

} // end namespace <>

void
computeSqrDistanceTransform1D(float const * src, int n, float * dest)
{
   dt1d(src, n, dest);
}

void
computeSqrDistanceTransform1D(float const * src, int n, float spacing, float * dest)
{
   dt1d(src, n, spacing, dest);
}

void
computeEuclideanSqrDistanceTransform(Image<float>& distImg)
{
   int const width = distImg.width();
   int const height = distImg.height();

   for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
      {
         if (distImg(x, y) != 0)
            distImg(x, y) = INF;
      }

  dt2d(distImg);
} // end computeEuclideanSqrDistanceTransform()

//**********************************************************************

namespace
{

   inline bool
   areConnectedTriplets(ViewTripletKey const& t1, ViewTripletKey const& t2)
   {
      int nOverlappingViews = 0;
      if (t1.views[0] == t2.views[0] || t1.views[0] == t2.views[1] || t1.views[0] == t2.views[2]) ++nOverlappingViews;
      if (t1.views[1] == t2.views[0] || t1.views[1] == t2.views[1] || t1.views[1] == t2.views[2]) ++nOverlappingViews;
      if (t1.views[2] == t2.views[0] || t1.views[2] == t2.views[1] || t1.views[2] == t2.views[2]) ++nOverlappingViews;

      return nOverlappingViews >= 2;
   }

} // end namespace <>

// Random growing from the given starting view
void
growModel(std::set<ViewTripletKey> const& modelTriples, int startView, int maxSize,
          std::set<int>& submodelViews)
{
   using namespace std;

   submodelViews.clear();

   set<ViewTripletKey> queue, handledTriplets;

   for (set<ViewTripletKey>::const_iterator p = modelTriples.begin(); p != modelTriples.end(); ++p)
   {
      ViewTripletKey const& triple = *p;

      if (triple.views[0] == startView || triple.views[1] == startView || triple.views[2] == startView)
      {
         queue.insert(triple);
         handledTriplets.insert(triple);
      }
   } // end for (p)

   while (!queue.empty() && submodelViews.size() < maxSize)
   {
      int const n = int((double(queue.size()) * (rand() / (RAND_MAX + 1.0))));
      //cout << "n = " << n << ", queue.size() = " << queue.size() << endl;
      set<ViewTripletKey>::iterator p = queue.begin();
      for (int i = 0; i < n; ++i) ++p;
      ViewTripletKey const nextTriplet = *p;

      queue.erase(p);

      submodelViews.insert(nextTriplet.views[0]);
      submodelViews.insert(nextTriplet.views[1]);
      submodelViews.insert(nextTriplet.views[2]);

      bool found = false;
      // Insert all triplets connected to the recently drawn one
      for (set<ViewTripletKey>::const_iterator q = modelTriples.begin(); q != modelTriples.end(); ++q)
      {
         if (handledTriplets.find(*q) == handledTriplets.end() && areConnectedTriplets(*q, nextTriplet))
         {
            queue.insert(*q);
            handledTriplets.insert(*q);
            found = true;
         } // end if
      } // end for (q)
      //if (!found) break;
   } // end while
} // end growModel()

// MST-based growing from the given starting view
void
growModelMST(map<int, set<int> > const& mstAdjacencyMap, std::set<ViewTripletKey> const& modelTriples,
             int startView, int maxSize, std::set<int>& submodelViews)
{
   using namespace std;

   submodelViews.clear();

   list<int> currentFrontier;
   currentFrontier.push_back(startView);

   while (!currentFrontier.empty() && submodelViews.size() < maxSize)
   {
      int const nextView = currentFrontier.front();
      currentFrontier.pop_front();
      submodelViews.insert(nextView);
      set<int> const& neighbors = mstAdjacencyMap.find(nextView)->second;
      for (set<int>::const_iterator p = neighbors.begin(); p != neighbors.end(); ++p)
      {
         if (submodelViews.find(*p) == submodelViews.end()) // Already inserted?
            currentFrontier.push_back(*p);
      }
   } // end while

   // Now we have the potential views in a submodel, but we have to check if all are connected via triplets.
   // Remove view not triplet-reachable.

   set<ViewTripletKey> submodelTriplets;

   for (set<ViewTripletKey>::const_iterator p = modelTriples.begin(); p != modelTriples.end(); ++p)
   {
      ViewTripletKey const& triple = *p;

      if (submodelViews.find(triple.views[0]) == submodelViews.end()) continue;
      if (submodelViews.find(triple.views[1]) == submodelViews.end()) continue;
      if (submodelViews.find(triple.views[2]) == submodelViews.end()) continue;

      submodelTriplets.insert(triple);
   } // end for (p)

   vector<pair<int, int> > tripletEdges;
   vector<double> weights;
   int n1 = 0;
   for (set<ViewTripletKey>::const_iterator p1 = submodelTriplets.begin(); p1 != submodelTriplets.end(); ++p1, ++n1)
   {
      ViewTripletKey const& t1 = *p1;

      int n2 = 0;
      for (set<ViewTripletKey>::const_iterator p2 = submodelTriplets.begin(); p2 != p1; ++p2, ++n2)
      {
         ViewTripletKey const& t2 = *p2;
         if (areConnectedTriplets(t1, t2))
         {
            tripletEdges.push_back(make_pair(n1, n2));
            weights.push_back(1);
         }
      } // end for (p2, n2)
   } // end for (p1, n1)

   if (tripletEdges.empty())
   {
      submodelViews.clear();
      return;
   }

   int const nNodes = submodelTriplets.size();

   vector<pair<int, int> > mstEdges;
   vector<set<int> > connComponents;

   getMinimumSpanningForest(tripletEdges, weights, mstEdges, connComponents);

   int largestComponent = 0;
   int largestSize = -1;
   for (size_t i = 0; i < connComponents.size(); ++i)
   {
      if (connComponents[i].size() > largestSize)
      {
         largestSize = connComponents[i].size();
         largestComponent = i;
      }
   } // end for (i)

   submodelViews.clear();

   n1 = 0;
   for (set<ViewTripletKey>::const_iterator p = submodelTriplets.begin(); p != submodelTriplets.end(); ++p, ++n1)
   {
      if (connComponents[largestComponent].find(n1) != connComponents[largestComponent].end())
      {
         submodelViews.insert(p->views[0]);
         submodelViews.insert(p->views[1]);
         submodelViews.insert(p->views[2]);
      }
   } // end for (p1, n1)
} // end growModel()

//**********************************************************************

namespace V3D
{

   double
   sampleSimilarityTransform(std::vector<Vector3d> const& left, std::vector<Vector3d> const& right,
                             int const minSample[3],
                             double inlierThresholdL, double inlierThresholdR,
                             Matrix3x3d& R, Vector3d& T, double& scale, std::vector<int> &inliers)
   {
      inliers.clear();

      int const N = left.size();
 
      if (N < 3) throwV3DErrorHere("computeRobustSimilarityTransform(): at least 3 point correspondences required.");

      vector<Vector3d> ptsLeftTrans(N), ptsRightTrans(N);
      vector<Vector3d> left_pts(3), right_pts(3);

      int const j0 = minSample[0];
      int const j1 = minSample[1];
      int const j2 = minSample[2];

      left_pts[0]  = left[j0];  left_pts[1]  = left[j1];  left_pts[2]  = left[j2];
      right_pts[0] = right[j0]; right_pts[1] = right[j1]; right_pts[2] = right[j2];

      Matrix3x3d R0, R0t;
      Vector3d T0;
      double scale0, rcpScale0;
      getSimilarityTransformation(left_pts, right_pts, R0, T0, scale0);

      rcpScale0 = 1.0 / scale0;
      R0t = R0.transposed();

      for (int i = 0; i < N; ++i) ptsLeftTrans[i]  = scale0 * (R0 * left[i] + T0); 
      for (int i = 0; i < N; ++i) ptsRightTrans[i] = R0t * (rcpScale0 * right[i] - T0);

      double score = 0;

      for (int i = 0; i < N; ++i)
      {
         double const distL = distance_L2(left[i], ptsRightTrans[i]);
         double const distR = distance_L2(right[i], ptsLeftTrans[i]);

         score += std::min(distL, inlierThresholdL);
         score += std::min(distR, inlierThresholdR);

         if (distL < inlierThresholdL && distR < inlierThresholdR) inliers.push_back(i);
      } // end for (i)
      return score;
   } // end sampleSimilarityTransform()

   double
   computeRobustSimilarityTransformationMLE(std::vector<Vector3d> const& left, std::vector<Vector3d> const& right,
                                            double inlierThresholdL, double inlierThresholdR, int nTrials,
                                            Matrix3x3d& R, Vector3d& T, double& scale, std::vector<int> &inliers)
   {
      vector<int> indices(left.size());
      for (int i = 0; i < indices.size(); ++i) indices[i] = i;
      random_shuffle(indices.begin(), indices.end());

      int nRuns = 1;
      int sampleOffset = 3;

      float inlierRatio;
      int expectedRuns, remainingRuns;

      double const confidence = log(1-0.99);

      double score = sampleSimilarityTransform(left, right, &indices[0], inlierThresholdL, inlierThresholdR,
                                               R, T, scale, inliers);
      inlierRatio = float(inliers.size()) / left.size();
      expectedRuns = ransacNSamples<3>(inlierRatio, confidence);

      vector<int> curInliers;

      while (1)
      {
         remainingRuns = expectedRuns - nRuns;
         if (remainingRuns <= 0) break;

         if (sampleOffset + 3 >= left.size())
         {
            random_shuffle(indices.begin(), indices.end());
            sampleOffset = 0;
         }

         Matrix3x3d R1;
         Vector3d T1;
         double scale1;
         double score1 = sampleSimilarityTransform(left, right, &indices[sampleOffset],
                                                   inlierThresholdL, inlierThresholdR,
                                                   R1, T1, scale1, curInliers);

         ++nRuns;
         //cout << "nRuns = " << nRuns << ", score = " << score << ", inliers.size() = " << inliers.size() << endl;
         sampleOffset += 3;

         if (score1 < score)
         {
            score        = score;
            inliers      = curInliers;
            inlierRatio  = float(inliers.size()) / left.size();
            expectedRuns = ransacNSamples<3>(inlierRatio, confidence);
            R = R1;
            T = T1;
            scale = scale1;
         }
      } // end while
      return double(inliers.size()) / left.size();
   } // end computeRobustSimilarityTransformationMLE()

} // end namespace V3D
