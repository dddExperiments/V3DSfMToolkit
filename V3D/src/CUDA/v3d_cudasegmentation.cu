#if defined(V3DLIB_ENABLE_CUDA)

#include "v3d_cudasegmentation.h"

#include <cuda.h>

#ifdef WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <cstdio>

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                            \
      cudaError err = call;                                             \
      if( cudaSuccess != err) {                                         \
         fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",  \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
         exit(EXIT_FAILURE);                                            \
      } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                    \
      CUDA_SAFE_CALL_NO_SYNC(call);                                     \
      cudaError err = cudaThreadSynchronize();                          \
      if( cudaSuccess != err) {                                         \
         fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",  \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
         exit(EXIT_FAILURE);                                            \
      } } while (0)


#define DIM_X 16
#define DIM_Y 8

//**********************************************************************

#define SYNCTHREADS() __syncthreads()

static __global__ void
_updateUVolume_kernel(int w, int h, float tau, float * U, float const * P1, float const * P2, float const * G)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int curPos = Y*w + X;

   __shared__ float p1_sh[DIM_Y*DIM_X];
   __shared__ float p2_sh[DIM_Y*DIM_X];

   // Load u, p and q of current slice/disparity
   float u = U[curPos];
   float const g = G[curPos];
   p1_sh[ix] = P1[curPos];
   p2_sh[ix] = P2[curPos];

   SYNCTHREADS();

   // Update u
   float p1_0 = (tidx > 0) ? p1_sh[ix-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? P1[curPos - 1] : p1_0;

   float p2_0 = (tidy > 0) ? p2_sh[ix-DIM_X] : 0.0f;
   p2_0 = (tidy == 0 && Y > 0) ? P2[curPos - w] : p2_0;

   float const div_p = (((X < w-1) ? p1_sh[ix] : 0) - p1_0 +
                        ((Y < h-1) ? p2_sh[ix] : 0) - p2_0);

   float u_new = u - tau * (div_p + g);
   U[curPos] = max(0.0f, min(1.0f, u_new));
} // end updateUVolume()

static __global__ void
_updatePVolume_kernel_L1(int w, int h, float alpha, float tau, float * const U, float * P1, float * P2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const pos = __mul24(Y, w) + X;
   //int pos = Y*w + X;

   __shared__ float u_sh[DIM_Y*DIM_X];

   u_sh[ix] = U[pos];
   SYNCTHREADS();

   float const p1_cur = P1[pos];
   float const p2_cur = P2[pos];

   float       u1  = (tidx < DIM_X-1) ? u_sh[ix+1] : U[pos + 1];
   float const u_x = (X < w-1) ? (u1 - u_sh[ix]) : 0.0f;
   u1              = (tidy < DIM_Y-1) ? u_sh[ix+DIM_X] : U[pos + w];
   float const u_y = (Y < h-1) ? (u1 - u_sh[ix]) : 0.0f;

   float new_p1 = p1_cur - tau * u_x;
   float new_p2 = p2_cur - tau * u_y;

   new_p1 = max(-alpha, min(alpha, new_p1));
   new_p2 = max(-alpha, min(alpha, new_p2));

   P1[pos] = new_p1;
   P2[pos] = new_p2;
} // end _updatePVolume_kernel_L1()

static __global__ void
_updatePVolume_kernel_L2(int w, int h, float alpha, float tau, float * const U, float * P1, float * P2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const pos = __mul24(Y, w) + X;
   //int pos = Y*w + X;

   __shared__ float u_sh[DIM_Y*DIM_X];

   u_sh[ix] = U[pos];
   SYNCTHREADS();

   float const p1_cur = P1[pos];
   float const p2_cur = P2[pos];

   float       u1  = (tidx < DIM_X-1) ? u_sh[ix+1] : U[pos + 1];
   float const u_x = (X < w-1) ? (u1 - u_sh[ix]) : 0.0f;
   u1              = (tidy < DIM_Y-1) ? u_sh[ix+DIM_X] : U[pos + w];
   float const u_y = (Y < h-1) ? (u1 - u_sh[ix]) : 0.0f;

   float new_p1 = p1_cur - tau * u_x;
   float new_p2 = p2_cur - tau * u_y;

   float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2) * alpha);
   new_p1 /= norm;
   new_p2 /= norm;

   P1[pos] = new_p1;
   P2[pos] = new_p2;
} // end _updatePVolume_kernel_L2()

static __global__ void
_updatePVolume_kernel_L2_Weighted(int w, int h, float alpha, float tau, float * const U, float const * C, float * P1, float * P2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const pos = __mul24(Y, w) + X;
   //int pos = Y*w + X;

   __shared__ float u_sh[DIM_Y*DIM_X];

   u_sh[ix] = U[pos];
   SYNCTHREADS();

   float const c = C[pos];
   float const p1_cur = P1[pos];
   float const p2_cur = P2[pos];

   float       u1  = (tidx < DIM_X-1) ? u_sh[ix+1] : U[pos + 1];
   float const u_x = (X < w-1) ? (u1 - u_sh[ix]) : 0.0f;
   u1              = (tidy < DIM_Y-1) ? u_sh[ix+DIM_X] : U[pos + w];
   float const u_y = (Y < h-1) ? (u1 - u_sh[ix]) : 0.0f;

   float new_p1 = p1_cur - tau * u_x;
   float new_p2 = p2_cur - tau * u_y;

   float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2) * alpha * c);
   new_p1 /= norm;
   new_p2 /= norm;

   P1[pos] = new_p1;
   P2[pos] = new_p2;
} // end _updatePVolume_kernel_L2_weighted()

//**********************************************************************

namespace V3D_CUDA
{

   void
   BinarySegmentation::allocate(int w, int h)
   {
      // Allocate additionals row to avoid a few conditionals in the kernel
      int const size = sizeof(float) * w * (h+1);

      _w = w;
      _h = h;

      CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_u, size) );
      CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_p1, size) );
      CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_p2, size) );
      CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_g, size) );
      CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_c, size) );
   } // end BinarySegmentation::allocate()

   void
   BinarySegmentation::deallocate()
   {
      CUDA_SAFE_CALL( cudaFree(_d_u) );
      CUDA_SAFE_CALL( cudaFree(_d_p1) );
      CUDA_SAFE_CALL( cudaFree(_d_p2) );
      CUDA_SAFE_CALL( cudaFree(_d_g) );
      CUDA_SAFE_CALL( cudaFree(_d_c) );
   }

   void
   BinarySegmentation::getResult(float * uDst, float * p1Dst, float * p2Dst)
   {
      int const sz = _w*_h*sizeof(float);

      if (uDst) CUDA_SAFE_CALL( cudaMemcpy( uDst, _d_u, sz, cudaMemcpyDeviceToHost) );
      if (p1Dst) CUDA_SAFE_CALL( cudaMemcpy( p1Dst, _d_p1, sz, cudaMemcpyDeviceToHost) );
      if (p2Dst) CUDA_SAFE_CALL( cudaMemcpy( p2Dst, _d_p2, sz, cudaMemcpyDeviceToHost) );
   }

   void
   BinarySegmentation::setImageData(float const * fSrc)
   {
      int const sz = _w*_h*sizeof(float);
      CUDA_SAFE_CALL( cudaMemcpy( _d_g, fSrc, sz, cudaMemcpyHostToDevice) );
   }

   void
   BinarySegmentation::setWeightData(float const * wSrc)
   {
      int const nPixels = _w*_h;
      int const sz = nPixels*sizeof(float);
      float * rcpW = new float[nPixels];
      for (int i = 0; i < nPixels; ++i) rcpW[i] = 1.0f / wSrc[i];
      CUDA_SAFE_CALL( cudaMemcpy( _d_c, rcpW, sz, cudaMemcpyHostToDevice) );
      delete [] rcpW;
   }

   void
   BinarySegmentation::initSegmentation()
   {
      int const sz = _w*_h*sizeof(float);

      CUDA_SAFE_CALL( cudaMemset( _d_u, 0, sz) );
      CUDA_SAFE_CALL( cudaMemset( _d_p1, 0, sz) );
      CUDA_SAFE_CALL( cudaMemset( _d_p2, 0, sz) );
   }

   void
   BinarySegmentation::run(int nIterations, float alpha, float tau)
   {
      dim3 gridDim(_w/DIM_X, _h/DIM_Y, 1);
      dim3 blockDim(DIM_X, DIM_Y, 1);

      switch (_metric)
      {
         case L2_METRIC:
            for (int i = 0; i < nIterations; ++i)
            {
               _updateUVolume_kernel<<< gridDim, blockDim, 0 >>>(_w, _h, tau, _d_u, _d_p1, _d_p2, _d_g);
               _updatePVolume_kernel_L2<<< gridDim, blockDim, 0 >>>(_w, _h, alpha, tau, _d_u, _d_p1, _d_p2);
            }
            break;
         case L1_METRIC:
            for (int i = 0; i < nIterations; ++i)
            {
               _updateUVolume_kernel<<< gridDim, blockDim, 0 >>>(_w, _h, tau, _d_u, _d_p1, _d_p2, _d_g);
               _updatePVolume_kernel_L1<<< gridDim, blockDim, 0 >>>(_w, _h, alpha, tau, _d_u, _d_p1, _d_p2);
            }
            break;
         case WEIGHTED_L2_METRIC:
            for (int i = 0; i < nIterations; ++i)
            {
               _updateUVolume_kernel<<< gridDim, blockDim, 0 >>>(_w, _h, tau, _d_u, _d_p1, _d_p2, _d_g);
               _updatePVolume_kernel_L2_Weighted<<< gridDim, blockDim, 0 >>>(_w, _h, alpha, tau, _d_u, _d_c, _d_p1, _d_p2);
            }
            break;
         default:
            fprintf(stderr, "BinarySegmentation::run(): Unknown regularization metric %i\n", _metric);
      }
   } // end cuda_segmentation_CV

} // end namespace V3D_CUDA

#endif // defined(V3DLIB_ENABLE_CUDA)
