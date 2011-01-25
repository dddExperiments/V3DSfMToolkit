#include "CUDA/v3d_cudamatching.h"

#if defined(V3DLIB_ENABLE_CUDA)

#include <cuda.h>
#include <cublas.h>
#include <iostream>

using namespace std;
using namespace V3D;

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#define NTHREADS 32

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

texture<float, 1, cudaReadModeElementType> rightPos_tex;

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


namespace
{

   void
   computeColumnMaxima(int nRows, int nCols, float threshold, float const * d_scores, int * argMax)
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

   void
   computeRowMaxima(bool useReduction, int nRows, int nCols, float threshold, float const * d_scores, int * argMax)
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

   void
   computeColumnMaximaWithRatioTest(int nRows, int nCols, float threshold, float minRatio, float const * d_scores, int * argMax)
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

   void
   computeRowMaximaWithRatioTest(bool useReduction, int nRows, int nCols,
                                 float threshold, float minRatio, float const * d_scores, int * argMax)
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

} // end namespace <>

namespace V3D_CUDA
{

   void
   SIFT_Matching::allocate(int nMaxLeftFeatures, int nMaxRightFeatures, bool useGuidedMatching)
   {
      _useGuidedMatching = useGuidedMatching;

      if ((nMaxLeftFeatures % NTHREADS) != 0)
      {
         if(_warnOnRounding) {
            cerr << "SIFT_Matching::allocate() warning: nMaxLeftFeatures should be a multiple of " << NTHREADS << endl;
            cerr << "Rounding nMaxLeftFeatures down." << endl;
         }
         nMaxLeftFeatures = nMaxLeftFeatures - (nMaxLeftFeatures & (NTHREADS-1));
      }

      if ((nMaxRightFeatures % NTHREADS) != 0)
      {
         if(_warnOnRounding) {
            cerr << "SIFT_Matching::allocate() warning: nMaxRightFeatures should be a multiple of " << NTHREADS << endl;
            cerr << "Rounding nMaxRightFeatures down." << endl;
         }
         nMaxRightFeatures = nMaxRightFeatures - (nMaxRightFeatures & (NTHREADS-1));
      }

      _nMaxLeftFeatures  = nMaxLeftFeatures;
      _nMaxRightFeatures = nMaxRightFeatures;

      cublasAlloc(nMaxLeftFeatures, 128*sizeof(float), (void **)&_d_leftFeatures);
      cublasAlloc(nMaxRightFeatures, 128*sizeof(float), (void **)&_d_rightFeatures);
      cublasAlloc(nMaxLeftFeatures*nMaxRightFeatures, sizeof(float), (void **)&_d_scores);

      if (useGuidedMatching)
      {
         cublasAlloc(2, 2*sizeof(float)*nMaxLeftFeatures, (void **)&_d_leftPositions);
         cublasAlloc(2, 2*sizeof(float)*nMaxRightFeatures, (void **)&_d_rightPositions);
      } // end if (useGuidedMatching)
   }

   void
   SIFT_Matching::deallocate()
   {
      cublasFree(_d_leftFeatures);
      cublasFree(_d_rightFeatures);
      cublasFree(_d_scores);

      if (_useGuidedMatching)
      {
         cublasFree(_d_leftPositions);
         cublasFree(_d_rightPositions);
      }
   }

   void
   SIFT_Matching::setLeftFeatures(int nFeatures, float const * descriptors)
   {
      if ((nFeatures % NTHREADS) != 0)
      {
         if(_warnOnRounding) {
            cerr << "SIFT_Matching::setLeftFeatures() warning: nFeatures should be a multiple of " << NTHREADS << endl;
            cerr << "Rounding nFeatures down." << endl;
         }
         nFeatures = nFeatures - (nFeatures & (NTHREADS-1));
      }

      _nLeftFeatures = std::min(nFeatures, _nMaxLeftFeatures);
      cublasSetMatrix(128, _nLeftFeatures, sizeof(float), descriptors, 128, _d_leftFeatures, 128);
   }

   void
   SIFT_Matching::setRightFeatures(int nFeatures, float const * descriptors)
   {
      if ((nFeatures % NTHREADS) != 0)
      {
         if(_warnOnRounding) {
            cerr << "SIFT_Matching::setRightFeatures() warning: nFeatures should be a multiple of " << NTHREADS << endl;
            cerr << "Rounding nFeatures down." << endl;
         }
         nFeatures = nFeatures - (nFeatures & (NTHREADS-1));
      }

      _nRightFeatures = std::min(nFeatures, _nMaxRightFeatures);;
      cublasSetMatrix(128, _nRightFeatures, sizeof(float), descriptors, 128, _d_rightFeatures, 128);
   }

   void
   SIFT_Matching::findPutativeMatches(std::vector<std::pair<int, int> >& matches,
                                      float const minScore, float const minRatio)
   {
      matches.clear();
      matches.reserve(std::min(_nLeftFeatures, _nRightFeatures));

      cublasSgemm('T', 'N', _nLeftFeatures, _nRightFeatures, 128, 1.0, _d_leftFeatures, 128, _d_rightFeatures, 128,
                  0.0f, _d_scores, _nLeftFeatures);

      vector<int> colMax(_nLeftFeatures, -1);
      vector<int> rowMax(_nRightFeatures, -1);

      if (minRatio < 0)
      {
         computeColumnMaxima(_nRightFeatures, _nLeftFeatures, minScore, _d_scores, &colMax[0]);
         computeRowMaxima(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, _d_scores, &rowMax[0]);
      }
      else
      {
         computeColumnMaximaWithRatioTest(_nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &colMax[0]);
         computeRowMaximaWithRatioTest(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &rowMax[0]);
      } // end if

      for (int k = 0; k < _nLeftFeatures; ++k)
      {
         int const l = colMax[k];
         if (l >= 0 && rowMax[l] == k)
            matches.push_back(make_pair(k, l));
      } // end for (k)
   } // end SIFT_Matching::findPutativeMatches()

   void
   SIFT_Matching::runGuidedMatching(V3D::Matrix3x3d const& F, float const distanceThreshold,
                                    Vector2f const * leftPositions,
                                    Vector2f const * rightPositions,
                                    std::vector<std::pair<int, int> >& matches,
                                    float const minScore, float const minRatio)
   {
      matches.clear();
      matches.reserve(std::min(_nLeftFeatures, _nRightFeatures));

      {
         vector<float> leftPos(2*_nLeftFeatures);
         vector<float> rightPos(2*_nRightFeatures);

         for (int k = 0; k < _nLeftFeatures; ++k)
         {
            leftPos[k]                = leftPositions[k][0];
            leftPos[k+_nLeftFeatures] = leftPositions[k][1];
         }

         for (int k = 0; k < _nRightFeatures; ++k)
         {
            rightPos[k]                 = rightPositions[k][0];
            rightPos[k+_nRightFeatures] = rightPositions[k][1];
         }

         int const leftSize  = 2*sizeof(float)*_nLeftFeatures;
         int const rightSize = 2*sizeof(float)*_nRightFeatures;

         CUDA_SAFE_CALL( cudaMemcpy( _d_leftPositions, &leftPos[0], leftSize, cudaMemcpyHostToDevice) );
         CUDA_SAFE_CALL( cudaMemcpy( _d_rightPositions, &rightPos[0], rightSize, cudaMemcpyHostToDevice) );

         CUDA_SAFE_CALL( cudaBindTexture(0, rightPos_tex, _d_rightPositions, rightSize) );

         dim3 gridDim(_nLeftFeatures/NTHREADS, 1, 1);
         dim3 blockDim(NTHREADS, 1, 1);

         float const sqrThreshold = distanceThreshold*distanceThreshold;
         float3 F1, F2, F3;
         F1.x = F[0][0]; F1.y = F[0][1]; F1.z = F[0][2];
         F2.x = F[1][0]; F2.y = F[1][1]; F2.z = F[1][2];
         F3.x = F[2][0]; F3.y = F[2][1]; F3.z = F[2][2];

         _cuda_eliminateNonEpipolarScores<<< gridDim, blockDim, 0 >>>(_nRightFeatures, _nLeftFeatures, sqrThreshold,
                                                                      F1, F2, F3, _d_leftPositions, _d_scores);
         CUDA_SAFE_CALL( cudaUnbindTexture(rightPos_tex) );
      } // end scope

      vector<int> colMax(_nLeftFeatures, -1);
      vector<int> rowMax(_nRightFeatures, -1);

      if (minRatio < 0)
      {
         computeColumnMaxima(_nRightFeatures, _nLeftFeatures, minScore, _d_scores, &colMax[0]);
         computeRowMaxima(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, _d_scores, &rowMax[0]);
      }
      else
      {
         computeColumnMaximaWithRatioTest(_nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &colMax[0]);
         computeRowMaximaWithRatioTest(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &rowMax[0]);
      } // end if

      for (int k = 0; k < _nLeftFeatures; ++k)
      {
         int const l = colMax[k];
         if (l >= 0 && rowMax[l] == k)
            matches.push_back(make_pair(k, l));
      } // end for (k)

   } // end SIFT_Matching::runGuidedMatching()

   void
   SIFT_Matching::runGuidedHomographyMatching(V3D::Matrix3x3d const& H, float const distanceThreshold,
                                              Vector2f const * leftPositions,
                                              Vector2f const * rightPositions,
                                              std::vector<std::pair<int, int> >& matches,
                                              float const minScore, float const minRatio)
   {
      matches.clear();
      matches.reserve(std::min(_nLeftFeatures, _nRightFeatures));

      {
         vector<float> leftPos(2*_nLeftFeatures);
         vector<float> rightPos(2*_nRightFeatures);

         for (int k = 0; k < _nLeftFeatures; ++k)
         {
            leftPos[k]                = leftPositions[k][0];
            leftPos[k+_nLeftFeatures] = leftPositions[k][1];
         }

         for (int k = 0; k < _nRightFeatures; ++k)
         {
            rightPos[k]                 = rightPositions[k][0];
            rightPos[k+_nRightFeatures] = rightPositions[k][1];
         }

         int const leftSize  = 2*sizeof(float)*_nLeftFeatures;
         int const rightSize = 2*sizeof(float)*_nRightFeatures;

         CUDA_SAFE_CALL( cudaMemcpy( _d_leftPositions, &leftPos[0], leftSize, cudaMemcpyHostToDevice) );
         CUDA_SAFE_CALL( cudaMemcpy( _d_rightPositions, &rightPos[0], rightSize, cudaMemcpyHostToDevice) );

         CUDA_SAFE_CALL( cudaBindTexture(0, rightPos_tex, _d_rightPositions, rightSize) );

         dim3 gridDim(_nLeftFeatures/NTHREADS, 1, 1);
         dim3 blockDim(NTHREADS, 1, 1);

         float const sqrThreshold = distanceThreshold*distanceThreshold;
         float3 H1, H2, H3;
         H1.x = H[0][0]; H1.y = H[0][1]; H1.z = H[0][2];
         H2.x = H[1][0]; H2.y = H[1][1]; H2.z = H[1][2];
         H3.x = H[2][0]; H3.y = H[2][1]; H3.z = H[2][2];

         _cuda_eliminateNonHomographyScores<<< gridDim, blockDim, 0 >>>(_nRightFeatures, _nLeftFeatures, sqrThreshold,
                                                                        H1, H2, H3, _d_leftPositions, _d_scores);
         CUDA_SAFE_CALL( cudaUnbindTexture(rightPos_tex) );
      } // end scope

      vector<int> colMax(_nLeftFeatures, -1);
      vector<int> rowMax(_nRightFeatures, -1);

      if (minRatio < 0)
      {
         computeColumnMaxima(_nRightFeatures, _nLeftFeatures, minScore, _d_scores, &colMax[0]);
         computeRowMaxima(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, _d_scores, &rowMax[0]);
      }
      else
      {
         computeColumnMaximaWithRatioTest(_nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &colMax[0]);
         computeRowMaximaWithRatioTest(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &rowMax[0]);
      } // end if

      for (int k = 0; k < _nLeftFeatures; ++k)
      {
         int const l = colMax[k];
         if (l >= 0 && rowMax[l] == k)
            matches.push_back(make_pair(k, l));
      } // end for (k)

   } // end SIFT_Matching::runGuidedMatching()

} // end namespace V3D_CUDA

//----------------------------------------------------------------------

#define NTHREADS_VOCTREE 32
#define NCHUNKS_VOCTREE (128/NTHREADS_VOCTREE/2)

#define USE_FULL_KEY 1

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

namespace V3D_CUDA
{

   template <int BranchFactor, int Depth>
   VocabularyTreeTraversal<BranchFactor, Depth>::VocabularyTreeTraversal()
      : _nMaxFeatures(0), _d_nodes(0)
   { }

   template <int BranchFactor, int Depth>
   void
   VocabularyTreeTraversal<BranchFactor, Depth>::allocate(int nMaxFeatures)
   {
      if ((nMaxFeatures % NTHREADS_VOCTREE) != 0)
      {
         cerr << "VocabularyTreeTraversal::allocate() warning: nMaxFeatures should be a multiple of " << NTHREADS_VOCTREE << endl;
         cerr << "Rounding nFeatures down." << endl;
         nMaxFeatures = nMaxFeatures - (nMaxFeatures & (NTHREADS_VOCTREE-1));
      }

      _nMaxFeatures = nMaxFeatures;

      CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_features, nMaxFeatures * 128 * sizeof(float)) );
      CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_leaveIds, nMaxFeatures * sizeof(int)) );
   }

   template <int BranchFactor, int Depth>
   void
   VocabularyTreeTraversal<BranchFactor, Depth>::deallocate()
   {
      CUDA_SAFE_CALL( cudaFree(_d_features) );
      CUDA_SAFE_CALL( cudaFree(_d_leaveIds) );
      if (_d_nodes != 0)
         CUDA_SAFE_CALL( cudaFree(_d_nodes) );
   }

   template <int BranchFactor, int Depth>
   void
   VocabularyTreeTraversal<BranchFactor, Depth>::setNodes(float const * nodes)
   {
      if (_d_nodes != 0)
         CUDA_SAFE_CALL( cudaFree(_d_nodes) );

      int const nNodes = getNodeCount();

      CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_nodes, nNodes * 128 * sizeof(float)) );
      CUDA_SAFE_CALL( cudaMemcpy( _d_nodes, nodes, nNodes * 128 * sizeof(float), cudaMemcpyHostToDevice) );
   }

   template <int BranchFactor, int Depth>
   void
   VocabularyTreeTraversal<BranchFactor, Depth>::traverseTree(int nFeatures, float const * features, int * leaveIds)
   {
      if (nFeatures > _nMaxFeatures)
      {
         cerr << "VocabularyTreeTraversal::traverseTree() warning: nFeatures is larger than nMaxFeatures." << endl;
         cerr << "Reducing nFeatures." << endl;
         nFeatures = _nMaxFeatures;
      }

      dim3 gridDim((nFeatures+31)/32, 32, 1);
      dim3 blockDim(NTHREADS_VOCTREE, 1, 1);

      CUDA_SAFE_CALL( cudaMemcpy( _d_features, features, nFeatures * 128 * sizeof(float), cudaMemcpyHostToDevice) );

      int const K = BranchFactor;
      int const D = Depth;
      _cuda_traverseTree<K, D><<<gridDim, blockDim, 0>>>(_d_features, _d_nodes, _d_leaveIds);

      CUDA_SAFE_CALL( cudaMemcpy( leaveIds, _d_leaveIds, nFeatures * sizeof(int), cudaMemcpyDeviceToHost) );
   }

   // Add additional instances you need.
   template struct VocabularyTreeTraversal<10, 5>;
   template struct VocabularyTreeTraversal<50, 3>;
   template struct VocabularyTreeTraversal<300, 2>;

} // end namespace V3D_CUDA

#endif // defined(V3DLIB_ENABLE_CUDA)
