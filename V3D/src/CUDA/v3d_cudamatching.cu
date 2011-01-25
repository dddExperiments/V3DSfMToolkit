//#include "CUDA/v3d_cudamatching.h"
#include "CUDA/v3d_cudamatching.cuh"

//#if defined(V3DLIB_ENABLE_CUDA)

#include <cuda.h>
#include <cublas.h>
//#include <iostream>

//using namespace std;
//using namespace V3D;

texture<float, 1, cudaReadModeElementType> rightPos_tex;

// Note: CUBLAS uses column-major matrix storage, but in the kernels we use the C-convention of row-major storage.

static __global__ void
_cuda_computeColumnMaxima(int nRows, int nCols, float threshold, float const * Scores, int * ArgMax)
{
   int const col = blockIdx.x * blockDim.x + threadIdx.x;

   float maxVal = -1e30;
   int   argMax = -1;

   for (int row = 0; row < nRows; ++row)
   {
      float const val = Scores[row * nCols + col];
      argMax = (val > maxVal) ? row : argMax;
      maxVal = max(maxVal, val);
   }

   ArgMax[col] = (maxVal > threshold) ? argMax : -1;
} // end _cuda_computeColumnMaxima()

__global__ void
_cuda_computeRowMaxima(int nRows, int nCols, float threshold, float const * Scores, int * ArgMax)
{
   __shared__ float scores[NTHREADS][NTHREADS];

   int const row0 = blockIdx.x * blockDim.x;
   int const row  = row0 + threadIdx.x;
   int const startPos0 = nCols * row0;

   float maxVal = -1e30;
   int   argMax = -1;

   for (int col0 = 0; col0 < nCols; col0 += NTHREADS)
   {
      // Read NTHREADSxNTHREADS block from the matrix
#pragma unroll
      for (int k = 0; k < NTHREADS; ++k)
      {
         scores[k][threadIdx.x] = Scores[startPos0 + k*nCols + col0 + threadIdx.x];
         //__syncthreads();
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < NTHREADS; ++k)
      {
         float const val = scores[threadIdx.x][k];
         argMax = (val > maxVal) ? (col0+k) : argMax;
         maxVal = max(maxVal, val);
      }
      __syncthreads();
   } // end for (col0)

   ArgMax[row] = (maxVal > threshold) ? argMax : -1;
} // end _cuda_computeRowMaxima()

static __global__ void
_cuda_computeRowMaxima_Reduce(int nRows, int nCols, float threshold, float const * Scores, int * ArgMax)
{
   __shared__ float scores_sh[NTHREADS];
   __shared__ int   argmax_sh[NTHREADS];

   int const tidx = threadIdx.x;

   int const row = blockIdx.y*gridDim.x + blockIdx.x;
   int const startPos0 = nCols * row;

   float maxVal  = -1e30;
   int   argMax  = -1;

   for (int col0 = 0; col0 < nCols; col0 += 2*NTHREADS)
   {
      float score0 = Scores[startPos0 + col0 + threadIdx.x];
      float score1 = Scores[startPos0 + col0 + threadIdx.x + NTHREADS];
      argmax_sh[tidx] = (score0 > score1) ? (col0 + tidx) : (col0 + tidx + NTHREADS);
      scores_sh[tidx] = max(score0, score1);

      argmax_sh[tidx] = (scores_sh[tidx] > scores_sh[tidx+16]) ? argmax_sh[tidx] : argmax_sh[tidx+16];
      scores_sh[tidx] = max(scores_sh[tidx], scores_sh[tidx+16]);

      argmax_sh[tidx] = (scores_sh[tidx] > scores_sh[tidx+8]) ? argmax_sh[tidx] : argmax_sh[tidx+8];
      scores_sh[tidx] = max(scores_sh[tidx], scores_sh[tidx+8]);

      argmax_sh[tidx] = (scores_sh[tidx] > scores_sh[tidx+4]) ? argmax_sh[tidx] : argmax_sh[tidx+4];
      scores_sh[tidx] = max(scores_sh[tidx], scores_sh[tidx+4]);

      argmax_sh[tidx] = (scores_sh[tidx] > scores_sh[tidx+2]) ? argmax_sh[tidx] : argmax_sh[tidx+2];
      scores_sh[tidx] = max(scores_sh[tidx], scores_sh[tidx+2]);

      argmax_sh[tidx] = (scores_sh[tidx] > scores_sh[tidx+1]) ? argmax_sh[tidx] : argmax_sh[tidx+1];
      scores_sh[tidx] = max(scores_sh[tidx], scores_sh[tidx+1]);

      argMax = (scores_sh[tidx] > maxVal) ? (argmax_sh[tidx]) : argMax;
      maxVal = max(maxVal, scores_sh[tidx]);
      //__syncthreads();
   } // end for (col0)

   if (tidx == 0)
      ArgMax[row] = (maxVal > threshold) ? argMax : -1;
} // end _cuda_computeRowMaxima_Reduce()


static __global__ void
_cuda_computeColumnMaximaWithRatioTest(int nRows, int nCols, float threshold, float minRatio, float const * Scores, int * ArgMax)
{
   int const col = blockIdx.x * blockDim.x + threadIdx.x;

   float maxVal  = -1e30;
   float nextVal = -1e30;
   int   argMax  = -1;

   for (int row = 0; row < nRows; ++row)
   {
      float const val = Scores[row * nCols + col];
      argMax = (val > maxVal) ? row : argMax;
      nextVal = (val > maxVal) ? maxVal : max(val, nextVal);
      maxVal = max(maxVal, val);
   }

#ifdef USE_ARCCOS_RATIO
   float const nextDist = acos(min(1.0, nextVal));
   float const bestDist = acos(min(1.0, maxVal));
#else
   float const nextDist = max(0.001, 2 - 2*nextVal);
   float const bestDist = 2 - 2*maxVal;
#endif


   float ratio = bestDist/nextDist;

   ArgMax[col] = (maxVal > threshold && ratio < minRatio) ? argMax : -1;
} // end _cuda_computeColumnMaximaWithRatioTest()

static __global__ void
_cuda_computeRowMaximaWithRatioTest(int nRows, int nCols, float threshold, float minRatio, float const * Scores, int * ArgMax)
{
   __shared__ float scores[NTHREADS][NTHREADS];

   int const row0 = blockIdx.x * blockDim.x;
   int const row  = row0 + threadIdx.x;
   int const startPos0 = nCols * row0;

   float maxVal  = -1e30;
   float nextVal = -1e30;
   int   argMax  = -1;

   for (int col0 = 0; col0 < nCols; col0 += NTHREADS)
   {
      // Read NTHREADSxNTHREADS block from the matrix
#pragma unroll
      for (int k = 0; k < NTHREADS; ++k)
      {
         scores[k][threadIdx.x] = Scores[startPos0 + k*nCols + col0 + threadIdx.x];
         //__syncthreads();
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < NTHREADS; ++k)
      {
         float const val = scores[threadIdx.x][k];
         argMax = (val > maxVal) ? (col0+k) : argMax;
         nextVal = (val > maxVal) ? maxVal : max(val, nextVal);
         maxVal = max(maxVal, val);
      }
      __syncthreads();
   } // end for (col0)

#ifdef USE_ARCCOS_RATIO
   float const nextDist = acos(min(1.0, nextVal));
   float const bestDist = acos(min(1.0, maxVal));
#else
   float const nextDist = max(0.001, 2 - 2*nextVal);
   float const bestDist = 2 - 2*maxVal;
#endif

   float ratio = bestDist/nextDist;

   ArgMax[row] = (maxVal > threshold && ratio < minRatio) ? argMax : -1;
} // end _cuda_computeRowMaximaWithRatioTest()

static __global__ void
_cuda_computeRowMaximaWithRatioTest_Reduce(int nRows, int nCols, float threshold, float minRatio, float const * Scores, int * ArgMax)
{
   __shared__ float scores_sh[NTHREADS];
   __shared__ float nextval_sh[NTHREADS];
   __shared__ int   argmax_sh[NTHREADS];

   int const tidx = threadIdx.x;

   int const row = blockIdx.y*gridDim.x + blockIdx.x;
   int const startPos0 = nCols * row;

   float maxVal  = -1e30;
   float nextVal = -1e30;
   int   argMax  = -1;

   float tmp;

#define REDUCTION_UPDATE(OFS)                                           \
   score0 = scores_sh[tidx];                                            \
   score1 = scores_sh[tidx+OFS];                                        \
   tmp = min(score0, score1);                                           \
   nextval_sh[tidx] = max(tmp, max(nextval_sh[tidx], nextval_sh[tidx+OFS])); \
   argmax_sh[tidx] = (score0 > score1) ? argmax_sh[tidx] : argmax_sh[tidx+OFS]; \
   scores_sh[tidx] = max(score0, score1);

   for (int col0 = 0; col0 < nCols; col0 += 2*NTHREADS)
   {
      float score0 = Scores[startPos0 + col0 + threadIdx.x];
      float score1 = Scores[startPos0 + col0 + threadIdx.x + NTHREADS];
      nextval_sh[tidx] = min(score0, score1);
      argmax_sh[tidx] = (score0 > score1) ? (col0 + tidx) : (col0 + tidx + NTHREADS);
      scores_sh[tidx] = max(score0, score1);

      REDUCTION_UPDATE(16);
      REDUCTION_UPDATE(8);
      REDUCTION_UPDATE(4);
      REDUCTION_UPDATE(2);
      REDUCTION_UPDATE(1);

      argMax = (scores_sh[tidx] > maxVal) ? argmax_sh[tidx] : argMax;
      tmp = min(scores_sh[tidx], maxVal);
      nextVal = max(tmp, max(nextVal, nextval_sh[tidx]));
      maxVal  = max(maxVal, scores_sh[tidx]);
      //__syncthreads();
   } // end for (col0)

#undef REDUCTION_UPDATE

#ifdef USE_ARCCOS_RATIO
   float const nextDist = acos(min(1.0, nextVal));
   float const bestDist = acos(min(1.0, maxVal));
#else
   float const nextDist = max(0.001, 2 - 2*nextVal);
   float const bestDist = 2 - 2*maxVal;
#endif

   float ratio = bestDist/nextDist;

   if (tidx == 0)
      ArgMax[row] = (maxVal > threshold && ratio < minRatio) ? argMax : -1;
} // end _cuda_computeRowMaximaWithRatioTest_Reduce()

static __global__ void
_cuda_eliminateNonEpipolarScores(int nRows, int nCols, float sqrThreshold,
                                 float3 const F1, float3 const F2, float3 const F3,
                                 float const * leftPos, float * Scores)
{
   int const col = blockIdx.x * blockDim.x + threadIdx.x;

   float2 p, q;
   p.x = leftPos[col];
   p.y = leftPos[col+nCols];

   float3 Fp, Ftq;
   Fp.x = F1.x*p.x + F1.y*p.y + F1.z;
   Fp.y = F2.x*p.x + F2.y*p.y + F2.z;
   Fp.z = F3.x*p.x + F3.y*p.y + F3.z;

   for (int row = 0; row < nRows; ++row)
   {
      q.x = tex1Dfetch(rightPos_tex, row);
      q.y = tex1Dfetch(rightPos_tex, row+nRows);

      Ftq.x = F1.x*q.x + F2.x*q.y + F3.x;
      Ftq.y = F1.y*q.x + F2.y*q.y + F3.y;
      Ftq.z = F1.z*q.x + F2.z*q.y + F3.z;

      float num = q.x*Fp.x + q.y*Fp.y + Fp.z;
      num *= num;
      float const denom = Fp.x*Fp.x + Fp.y*Fp.y + Ftq.x*Ftq.x + Ftq.y*Ftq.y;
      float const sampsonDist = num / denom;

      float val = Scores[row * nCols + col];
      val = (sampsonDist > sqrThreshold) ? -1e20f : val;
      Scores[row * nCols + col] = val;
   }
} // end _cuda_eliminateNonEpipolarScores()

static __global__ void
_cuda_eliminateNonHomographyScores(int nRows, int nCols, float sqrThreshold,
                                   float3 const H1, float3 const H2, float3 const H3,
                                   float const * leftPos, float * Scores)
{
   int const col = blockIdx.x * blockDim.x + threadIdx.x;

   float2 p, q;
   p.x = leftPos[col];
   p.y = leftPos[col+nCols];

   float3 Hp;
   Hp.x = H1.x*p.x + H1.y*p.y + H1.z;
   Hp.y = H2.x*p.x + H2.y*p.y + H2.z;
   Hp.z = H3.x*p.x + H3.y*p.y + H3.z;

   Hp.x /= Hp.z;
   Hp.y /= Hp.z;

   for (int row = 0; row < nRows; ++row)
   {
      q.x = tex1Dfetch(rightPos_tex, row);
      q.y = tex1Dfetch(rightPos_tex, row+nRows);

      float const sqrDist = (Hp.x-q.x)*(Hp.x-q.x) + (Hp.y-q.y)*(Hp.y-q.y);

      float val = Scores[row * nCols + col];
      val = (sqrDist > sqrThreshold) ? -1e20f : val;
      Scores[row * nCols + col] = val;
   }
} // end _cuda_eliminateNonHomographyScores()


extern "C" void computeColumnMaxima(int nRows, int nCols, float threshold, float const * d_scores, int * argMax)
{
  dim3 gridDim(nCols/NTHREADS, 1, 1);
  dim3 blockDim(NTHREADS, 1, 1);

  int const size = nCols * sizeof(int);

  int * d_argMax;

  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_argMax, size) );
  _cuda_computeColumnMaxima<<< gridDim, blockDim, 0 >>>(nRows, nCols, threshold, d_scores, d_argMax);
  CUDA_SAFE_CALL( cudaMemcpy( argMax, d_argMax, size, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaFree(d_argMax) );
}

extern "C" void computeRowMaxima(bool useReduction, int nRows, int nCols, float threshold, float const * d_scores, int * argMax)
{
  int const size = nRows * sizeof(int);
  int * d_argMax;

  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_argMax, size) );

  if (useReduction)
  {
     dim3 gridDim(nRows/32, 32, 1);
     dim3 blockDim(NTHREADS, 1, 1);
     _cuda_computeRowMaxima_Reduce<<< gridDim, blockDim, 0 >>>(nRows, nCols, threshold, d_scores, d_argMax);
  }
  else
  {
     dim3 gridDim(nRows/NTHREADS, 1, 1);
     dim3 blockDim(NTHREADS, 1, 1);
     _cuda_computeRowMaxima<<< gridDim, blockDim, 0 >>>(nRows, nCols, threshold, d_scores, d_argMax);
  }

  CUDA_SAFE_CALL( cudaMemcpy( argMax, d_argMax, size, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaFree(d_argMax) );
}

extern "C" void computeColumnMaximaWithRatioTest(int nRows, int nCols, float threshold, float minRatio, float const * d_scores, int * argMax)
{
  dim3 gridDim(nCols/NTHREADS, 1, 1);
  dim3 blockDim(NTHREADS, 1, 1);

  int const size = nCols * sizeof(int);

  int * d_argMax;

  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_argMax, size) );
  _cuda_computeColumnMaximaWithRatioTest<<< gridDim, blockDim, 0 >>>(nRows, nCols, threshold, minRatio, d_scores, d_argMax);
  CUDA_SAFE_CALL( cudaMemcpy( argMax, d_argMax, size, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaFree(d_argMax) );
}

extern "C" void computeRowMaximaWithRatioTest(bool useReduction, int nRows, int nCols, float threshold, float minRatio, float const * d_scores, int * argMax)
{
  int const size = nRows * sizeof(int);

  int * d_argMax;

  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_argMax, size) );

  if (useReduction)
  {
     dim3 gridDim(nRows/32, 32, 1);
     dim3 blockDim(NTHREADS, 1, 1);
     _cuda_computeRowMaximaWithRatioTest_Reduce<<< gridDim, blockDim, 0 >>>(nRows, nCols, threshold, minRatio, d_scores, d_argMax);
  }
  else
  {
     dim3 gridDim(nRows/NTHREADS, 1, 1);
     dim3 blockDim(NTHREADS, 1, 1);
     _cuda_computeRowMaximaWithRatioTest<<< gridDim, blockDim, 0 >>>(nRows, nCols, threshold, minRatio, d_scores, d_argMax);
  }

  CUDA_SAFE_CALL( cudaMemcpy( argMax, d_argMax, size, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaFree(d_argMax) );
}

extern "C" void eliminateNonEpipolarScores(float* _d_rightPositions, float* _d_leftPositions, float* _d_scores, int rightSize, int _nRightFeatures, int _nLeftFeatures, float distanceThreshold, float3 F1, float3 F2, float3 F3)
{
	CUDA_SAFE_CALL( cudaBindTexture(0, rightPos_tex, _d_rightPositions, rightSize) );

	dim3 gridDim(_nLeftFeatures/NTHREADS, 1, 1);
	dim3 blockDim(NTHREADS, 1, 1);

	float const sqrThreshold = distanceThreshold*distanceThreshold;

	_cuda_eliminateNonEpipolarScores<<< gridDim, blockDim, 0 >>>(_nRightFeatures, _nLeftFeatures, sqrThreshold, F1, F2, F3, _d_leftPositions, _d_scores);
	CUDA_SAFE_CALL( cudaUnbindTexture(rightPos_tex) );
}

extern "C" void eliminateNonHomographyScores(float* _d_rightPositions, float* _d_leftPositions, float* _d_scores, int rightSize, int _nRightFeatures, int _nLeftFeatures, float distanceThreshold, float3 H1, float3 H2, float3 H3)
{
	CUDA_SAFE_CALL( cudaBindTexture(0, rightPos_tex, _d_rightPositions, rightSize) );

	dim3 gridDim(_nLeftFeatures/NTHREADS, 1, 1);
	dim3 blockDim(NTHREADS, 1, 1);

	float const sqrThreshold = distanceThreshold*distanceThreshold;

	_cuda_eliminateNonHomographyScores<<< gridDim, blockDim, 0 >>>(_nRightFeatures, _nLeftFeatures, sqrThreshold, H1, H2, H3, _d_leftPositions, _d_scores);
	CUDA_SAFE_CALL( cudaUnbindTexture(rightPos_tex) );
}

//----------------------------------------------------------------------

template <int BranchFactor, int Depth>
__global__ void
_cuda_traverseTree(float const * keys, float const * tree, int * leaves)
{
   int const tidx = threadIdx.x;
   int const keyId = gridDim.x*blockIdx.y + blockIdx.x;
   int const keyStart0 = 128 * keyId;

   __shared__ float score_sh[NTHREADS_VOCTREE];
#if defined(USE_FULL_KEY)
   __shared__ float key_sh[128];
#endif
   __shared__ int ofs_sh;

#if defined(USE_FULL_KEY)
# if 0
#pragma unroll
   for (int l = 0; l < NCHUNKS_VOCTREE; ++l)
      key_sh[NTHREADS_VOCTREE*l + tidx] = keys[keyStart0 + NTHREADS_VOCTREE*l + tidx];
# else
#pragma unroll
   for (int l = 0; l < NCHUNKS_VOCTREE; ++l)
   {
      key_sh[NTHREADS_VOCTREE*(2*l+0) + tidx] = keys[keyStart0 + NTHREADS_VOCTREE*(2*l+0) + tidx];
      key_sh[NTHREADS_VOCTREE*(2*l+1) + tidx] = keys[keyStart0 + NTHREADS_VOCTREE*(2*l+1) + tidx];
   }
# endif
#endif

   ofs_sh = 0;

   int leafId = 0;
   int nextLeafId = 0;
   //int factor = 1;

   float score, maxScore;

#pragma unroll
   for (int d = 0; d < Depth; ++d)
   {
      float bestBranch = -1;
      maxScore = -1e30;

#pragma unroll
      for (int k = 0; k < BranchFactor; ++k)
      {
         score = 0.0f;

#pragma unroll
         for (int l = 0; l < NCHUNKS_VOCTREE; ++l)
         {
#if defined(USE_FULL_KEY)
# if 0
            float node = tree[ofs_sh + 128*k + NTHREADS_VOCTREE*l + tidx];
            score_sh[tidx] = key_sh[NTHREADS_VOCTREE*l + tidx]*node;
# else
            float node = tree[ofs_sh + 128*k + NTHREADS_VOCTREE*(2*l+0) + tidx];
            score_sh[tidx] = key_sh[NTHREADS_VOCTREE*(2*l+0) + tidx]*node;
            node = tree[ofs_sh + 128*k + NTHREADS_VOCTREE*(2*l+1) + tidx];
            score_sh[tidx] += key_sh[NTHREADS_VOCTREE*(2*l+1) + tidx]*node;
# endif
#else
            // Read the respective fraction of the feature descriptor.
            float key = keys[keyStart0 + NTHREADS_VOCTREE*l + tidx];
            // Read the respective fraction of the node descriptors.
            float node = tree[ofs_sh + 128*k + NTHREADS_VOCTREE*l + tidx];
            score_sh[tidx] = key*node;
#endif

            score_sh[tidx] += score_sh[tidx+16];
            score_sh[tidx] += score_sh[tidx+8];
            score_sh[tidx] += score_sh[tidx+4];
            score_sh[tidx] += score_sh[tidx+2];
            score += score_sh[tidx] + score_sh[tidx+1];
         } // end for (l)

         bestBranch = (score > maxScore) ? k : bestBranch;
         maxScore = max(score, maxScore);
      } // end for (k)
      leafId = nextLeafId + bestBranch;
      nextLeafId = (leafId + 1) * BranchFactor;
      //leafId = leafId + factor*bestBranch;
      //factor *= BranchFactor;
      if (tidx == 0) ofs_sh = 128 * nextLeafId;
      __syncthreads();
   } // end for (d)
   if (tidx == 0) leaves[keyId] = leafId;
} // end _cuda_traverseTree()

template <int BranchFactor, int Depth>
void traverseTree(int nFeatures, float* _d_features, float* _d_nodes, int* _d_leaveIds)
{
	dim3 gridDim((nFeatures+31)/32, 32, 1);
	dim3 blockDim(NTHREADS_VOCTREE, 1, 1);
	_cuda_traverseTree<BranchFactor, Depth><<<gridDim, blockDim, 0>>>(_d_features, _d_nodes, _d_leaveIds);
}

typedef void (__cdecl  *traverseTreePtr)(int,float *,float *,int *);
traverseTreePtr a = traverseTree<10,  5>;
traverseTreePtr b = traverseTree<50,  3>;
traverseTreePtr c = traverseTree<300, 2>;

//#endif // defined(V3DLIB_ENABLE_CUDA)