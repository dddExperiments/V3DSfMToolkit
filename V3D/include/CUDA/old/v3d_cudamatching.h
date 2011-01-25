// -*- C++ -*-
#ifndef V3D_CUDA_MATCHING_H
#define V3D_CUDA_MATCHING_H

#if defined(V3DLIB_ENABLE_CUDA)

#include <vector>
#include <algorithm>

#include "Math/v3d_linear.h"

namespace V3D_CUDA
{

   struct SIFT_Matching
   {
         SIFT_Matching()
            : _warnOnRounding(true),
              _useReductionPath(false),
              _nMaxLeftFeatures(0), _nMaxRightFeatures(0), _nLeftFeatures(0), _nRightFeatures(0)
         { }

         void enableRoundingWarning(bool flag = true)
         {
            _warnOnRounding = flag;
         }

         void enableReductionPath(bool flag = true)
         {
            _useReductionPath = flag;
         }

         void allocate(int nMaxLeftFeatures, int nMaxRightFeatures, bool useGuidedMatching = false);
         void allocate(int nMaxFeatures, bool useGuidedMatching = false)
         {
            this->allocate(nMaxFeatures, nMaxFeatures, useGuidedMatching);
         }

         void deallocate();

         void setLeftFeatures(int nFeatures, float const * descriptors);
         void setRightFeatures(int nFeatures, float const * descriptors);

         // Ratio test is disabled if minRatio < 0.
         void findPutativeMatches(std::vector<std::pair<int, int> >& matches,
                                  float const minScore = 0.9f, float const minRatio = 0.8f);

         void runGuidedMatching(V3D::Matrix3x3d const& F, float const distanceThreshold,
                                V3D::Vector2f const * leftPositions,
                                V3D::Vector2f const * rightPositions,
                                std::vector<std::pair<int, int> >& matches,
                                float const minScore = 0.9f, float const minRatio = 0.8f);

         void runGuidedHomographyMatching(V3D::Matrix3x3d const& H, float const distanceThreshold,
                                          V3D::Vector2f const * leftPositions,
                                          V3D::Vector2f const * rightPositions,
                                          std::vector<std::pair<int, int> >& matches,
                                          float const minScore = 0.9f, float const minRatio = 0.8f);

      protected:
         bool   _warnOnRounding;
         bool   _useReductionPath;
         bool   _useGuidedMatching;
         int    _nMaxLeftFeatures, _nMaxRightFeatures, _nLeftFeatures, _nRightFeatures;
         float *_d_leftFeatures, *_d_rightFeatures, *_d_scores;
         float *_d_leftPositions, *_d_rightPositions;
   }; // end struct SIFT_Matching

   template <int BranchFactor, int Depth>
   struct VocabularyTreeTraversal
   {
         VocabularyTreeTraversal<BranchFactor, Depth>();

         void allocate(int nMaxFeatures);
         void deallocate();

         void setNodes(float const * nodes);

         void traverseTree(int nFeatures, float const * features, int * leaveIds);

         static inline int getNodeCount()
         {
            int res = 0;
            int len = 1;
            for (int d = 0; d < Depth; ++d)
            {
               len *= BranchFactor;
               res += len;
            }
            return res;
         }

         static inline int getLeaveCount()
         {
            int res = 1;
            for (int d = 0; d < Depth; ++d)
               res *= BranchFactor;
            return res;
         }

      protected:
         int    _nMaxFeatures;
         float *_d_nodes, *_d_features;
         int   *_d_leaveIds;
   }; // end struct VocabularyTreeTraversal

} // end namespace V3D_CUDA

#endif // defined(V3DLIB_ENABLE_CUDA)

#endif
