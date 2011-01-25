// -*- C++ -*-
#ifndef V3D_K_MEANS_H
#define V3D_K_MEANS_H

#include <Math/v3d_linear.h>

namespace V3D
{

   template <typename Vec>
   inline double
   computeNearestCenters_K_Means(std::vector<Vec> const& points,
                                 std::vector<Vec> const& centers, std::vector<int>& nearestIndex)
   {
      int const N = points.size();
      int const K = centers.size();
      nearestIndex.resize(N);

      double E = 0.0;
      for (int i = 0; i < N; ++i)
      {
         int nearestCenter = -1;
         float nearestDistance = 1e30f;
         for (int j = 0; j < K; ++j)
         {
            float const d = distance_L2(points[i], centers[j]);
            if (d < nearestDistance)
            {
               nearestDistance = d;
               nearestCenter   = j;
            }
         } // end for (j)
         nearestIndex[i] = nearestCenter;
         E += nearestDistance*nearestDistance;
      } // end for (i)
      return E;
   } // end computeNearestCenters_K_Means()

   template <typename Vec, typename T>
   inline void
   computeClusterSizes_K_Means(std::vector<Vec> const& points, std::vector<Vec> const& centers, std::vector<T>& counts)
   {
      int const N = points.size();
      int const K = centers.size();
      counts.resize(K);
      std::fill(counts.begin(), counts.end(), 0);

      std::vector<int> nearestIndex;
      computeNearestCenters_K_Means(points, centers, nearestIndex);

      for (int i = 0; i < N; ++i)
         counts[nearestIndex[i]] += 1;
   } // end computeClusterSizes_K_Means()

   template <typename Vec>
   inline double
   refine_K_MeansClustering(std::vector<Vec> const& points, std::vector<Vec>& centers)
   {
      using namespace std;

      int const N = points.size();
      int const K = centers.size();

      vector<int> nearestIndex;
      double E_old = computeNearestCenters_K_Means(points, centers, nearestIndex);

      vector<int> counts(K);

      while (1)
      {
         for (int j = 0; j < K; ++j) V3D::makeZeroVector(centers[j]);

         std::fill(counts.begin(), counts.end(), 0);
         for (int i = 0; i < N; ++i)
         {
            int const j = nearestIndex[i];
            V3D::addVectorsIP(points[i], centers[j]);
            ++counts[j];
         }
         for (int j = 0; j < K; ++j)
         {
            if (counts[j] > 0)
               V3D::scaleVectorIP(1.0f / counts[j], centers[j]);
         }

         double const E_new = computeNearestCenters_K_Means(points, centers, nearestIndex);
         if (E_old <= E_new) break;
         E_old = E_new;
      } // end while
      return E_old;
   } // end run_K_MeansClustering()

   template <typename Vec>
   inline void
   run_K_MeansClustering(std::vector<Vec> const& points, std::vector<Vec>& centers)
   {
      int const N = points.size();
      int const K = centers.size();

      // Take point with max. L2 norm as first cluster center
      float maxDist = -1e30;
      for (int i = 0; i < N; ++i)
      {
         float const d = V3D::norm_L2(points[i]);
         if (d > maxDist)
         {
            maxDist = d;
            V3D::copyVector(points[i], centers[0]);
         }
      }

      for (int j = 1; j < K; ++j)
      {
         maxDist = -1e30;
         for (int i = 0; i < N; ++i)
         {
            float d = 1e30;
            for (int j1 = 0; j1 < j; ++j1)
               d = std::min(d, distance_L2(points[i], centers[j1]));

            if (d > maxDist)
            {
               maxDist = d;
               V3D::copyVector(points[i], centers[j]);
            }
         } // end for (i)
      } // end for (j)

      refine_K_MeansClustering(points, centers);
   } // end run_K_MeansClustering()

} // end namespace V3D

#endif
