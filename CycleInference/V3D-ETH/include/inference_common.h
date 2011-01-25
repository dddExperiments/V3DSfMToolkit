// -*- C++ -*-
#ifndef INFERENCE_COMMON_H
#define INFERENCE_COMMON_H

#include "Math/v3d_mathutilities.h"

#include <vector>
#include <map>

// This version requires that the node IDs are in [0..nNodes-1].
void computeMST(int const nNodes, std::vector<std::pair<int, int> > const& edges, std::vector<double> const& weights,
                std::vector<int>& parentNodes);

// This version allows arbitrary node IDs
void computeMST(std::vector<std::pair<int, int> > const& edges, std::vector<double> const& weights,
                std::map<int, int>& parentNodes);


struct LoopSamplerParams
{
      LoopSamplerParams()
         : nTrees(10), maxLoopLength(6), edgeWeightFactor(0.1)
      { }

      int nTrees, maxLoopLength;
      double edgeWeightFactor;
}; // end struct LoopSamplerParams

void drawLoops(std::vector<std::pair<int, int> > const& edges, std::vector<double> const& initialWeights,
               std::vector<std::pair<int, bool> >& loopEdges, LoopSamplerParams const& params = LoopSamplerParams());


namespace V3D
{

   inline void
   accumulateTransformations(vector<SimilarityTransform> const& xforms, SimilarityTransform& accumXform, double& sigmaT)
   {
      SimilarityTransform accum;
      sigmaT = 0;

      for (int i = xforms.size()-1; i >= 0; --i)
      {
         accum   = accum * xforms[i];
         //sigmaT += accum.scale * sqrNorm_L2(xforms[i].T);
         sigmaT += accum.scale;
      }
      accumXform = accum;
      sigmaT     = sqrt(sigmaT);
   } // end accumulateTransformations()

} // end namespace V3D

#endif
