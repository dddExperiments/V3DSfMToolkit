#pragma once

#if defined(V3DLIB_ENABLE_CUDA)

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

#endif // defined(V3DLIB_ENABLE_CUDA)