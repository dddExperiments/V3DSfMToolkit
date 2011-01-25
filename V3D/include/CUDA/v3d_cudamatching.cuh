#pragma once

#if defined(V3DLIB_ENABLE_CUDA)

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

extern "C" void computeColumnMaxima(int nRows, int nCols, float threshold, float const * d_scores, int * argMax);
extern "C" void computeRowMaxima(bool useReduction, int nRows, int nCols, float threshold, float const * d_scores, int * argMax);
extern "C" void computeColumnMaximaWithRatioTest(int nRows, int nCols, float threshold, float minRatio, float const * d_scores, int * argMax);
extern "C" void computeRowMaximaWithRatioTest(bool useReduction, int nRows, int nCols, float threshold, float minRatio, float const * d_scores, int * argMax);

extern "C" void eliminateNonEpipolarScores(float* _d_rightPositions, float* _d_leftPositions, float* _d_scores, int rightSize, int _nRightFeatures, int _nLeftFeatures, float distanceThreshold, float3 F1, float3 F2, float3 F3);
extern "C" void eliminateNonHomographyScores(float* _d_rightPositions, float* _d_leftPositions, float* _d_scores, int rightSize, int _nRightFeatures, int _nLeftFeatures, float distanceThreshold, float3 H1, float3 H2, float3 H3);

template <int BranchFactor, int Depth>
void traverseTree(int nFeatures, float* _d_features, float* _d_nodes, int* _d_leaveIds);

#define NTHREADS 32

#define NTHREADS_VOCTREE 32
#define NCHUNKS_VOCTREE (128/NTHREADS_VOCTREE/2)

#define USE_FULL_KEY 1

#endif // defined(V3DLIB_ENABLE_CUDA)