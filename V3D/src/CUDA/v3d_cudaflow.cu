#if defined(V3DLIB_ENABLE_CUDA)

#include "CUDA/v3d_cudaflow.h"

#include <cuda.h>

#ifdef WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <cstdio>

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


#define DIM_X 16
#define DIM_Y 8

#define SYNCTHREADS() __syncthreads()
//#define SYNCTHREADS()

//**********************************************************************

__global__ void
kernel_updateUVolume_theta(int w, int h, float lambda_theta, float const theta,
                           float const * A1, float const * A2, float const * C,
                           float * U, float * V, float const * PU1, float const * PU2, float const * PV1, float const * PV2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const curPos = __mul24(Y, w) + X;
   //int curPos = Y*w + X;

   __shared__ float p1_sh[DIM_Y][DIM_X];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   // Load u, p and q of current slice/disparity
   float u = U[curPos];
   float v = V[curPos];
   float const a1 = A1[curPos];
   float const a2 = A2[curPos];
   float const c  = C[curPos];

   p1_sh[tidy][tidx]   = PU1[curPos];
   p2_sh[tidy+1][tidx] = PU2[curPos];

   if (tidy == 0) p2_sh[0][tidx] = (Y > 0) ? PU2[curPos-w] : 0.0f;

   float D = c + a1*u + a2*v;

   float R2 = a1*a1 + a2*a2;
   float lam_R2 = lambda_theta * R2;
   //float step = (D + lam_R2 < 0.0f) ? lambda_theta : ((D - lam_R2 > 0.0f) ? -lambda_theta : (-D/(R2+0.001f)));
   float step = (D + lam_R2 < 0.0f) ? lambda_theta : ((D - lam_R2 > 0.0f) ? -lambda_theta : (-D/R2));

   u = u + step * a1;
   v = v + step * a2;

   SYNCTHREADS();

   // Update u
   float p1_0 = (tidx > 0) ? p1_sh[tidy][tidx-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? PU1[curPos - 1] : p1_0;
   float p2_0 = p2_sh[tidy][tidx];
   float div_p = (((X < w-1) ? p1_sh[tidy][tidx] : 0) - p1_0 +
                  ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) - p2_0);

   U[curPos] = u - theta * div_p;

   // Update v
   p1_sh[tidy][tidx]   = PV1[curPos];
   p2_sh[tidy+1][tidx] = PV2[curPos];
   if (tidy == 0) p2_sh[0][tidx] = (Y > 0) ? PV2[curPos-w] : 0.0f;
   SYNCTHREADS();

   p1_0 = (tidx > 0) ? p1_sh[tidy][tidx-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? PV1[curPos - 1] : p1_0;
   p2_0 = p2_sh[tidy][tidx];
   div_p = (((X < w-1) ? p1_sh[tidy][tidx] : 0) - p1_0 +
            ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) - p2_0);

   V[curPos] = v - theta * div_p;
} // end kernel_updateUVolume_theta()

__global__ void
kernel_updatePVolume_theta(int w, int h, float tau_over_theta, float * const U, float * P1, float * P2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int pos = Y*w + X;

   __shared__ float u_sh[DIM_Y*DIM_X];

   u_sh[ix] = U[pos];
   SYNCTHREADS();

   // Load p and q of current slice/disparity
   float const p1_cur = P1[pos];
   float const p2_cur = P2[pos];

   float       u1  = (tidx < DIM_X-1) ? u_sh[ix+1] : U[pos + 1];
   float const u_x = (X < w-1) ? (u1 - u_sh[ix]) : 0.0f;
   u1              = (tidy < DIM_Y-1) ? u_sh[ix+DIM_X] : U[pos + w];
   float const u_y = (Y < h-1) ? (u1 - u_sh[ix]) : 0.0f;

# if 1
   float new_p1 = p1_cur - tau_over_theta * u_x;
   float new_p2 = p2_cur - tau_over_theta * u_y;

   float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2));
   new_p1 /= norm;
   new_p2 /= norm;
# else
   float const tv = sqrtf(u_x*u_x + u_y*u_y);
   float const denom_p = 1.0f / (1.0f + tau * tv);
   float new_p1 = (p1_cur - tau_over_theta * u_x) * denom_p;
   float new_p2 = (p2_cur - tau_over_theta * u_y) * denom_p;
#endif

   P1[pos] = new_p1;
   P2[pos] = new_p2;
} // end kernel_updatePVolume_theta()

//**********************************************************************

#define WARP_DIM_X 16
#define WARP_DIM_Y 8

texture<float, 2, cudaReadModeElementType> I0_tex;
texture<float, 2, cudaReadModeElementType> I1_tex;

static __global__ void
kernel_warpImage(int w, int h, float const * U, float const * V, float * A1, float * A2, float * C)
{
   int const X = blockIdx.x*blockDim.x + threadIdx.x;
   int const Y = blockIdx.y*blockDim.y + threadIdx.y;

   int const pos = Y*w + X;

   float const u = U[pos];
   float const v = V[pos];

   float I0 = tex2D(I0_tex, X, Y);
   float I1 = tex2D(I1_tex, X+u, Y+v);
   float I1x = tex2D(I1_tex, X+u+0.5f, Y+v) - tex2D(I1_tex, X+u-0.5f, Y+v);
   float I1y = tex2D(I1_tex, X+u, Y+v+0.5f) - tex2D(I1_tex, X+u, Y+v-0.5f);

   I0 /= 255.0f; I1 /= 255.0f; I1x /= 255.0f; I1y /= 255.0f;

   float const eps = 0.01f;
   I1x = (I1x < 0.0f) ? min(-eps, I1x) : max(eps, I1x);
   I1y = (I1y < 0.0f) ? min(-eps, I1y) : max(eps, I1y);

   A1[pos] = I1x;
   A2[pos] = I1y;
   C[pos]  = (I1 - I1x*u - I1y*v - I0);
} // end computeCoeffs()


//**********************************************************************

static inline void
reduce4(int w, int h, float * input,  float * output)
{
   for (int y = 0; y < h; ++y)
   {
      int const yy = 2*y;
      int const yy0 = (y > 0) ? (yy-1) : 1;
      int const yy1 = yy;
      //int const yy2 = (y < h-1) ? (yy+1) : (yy-1);
      int const yy2 = yy+1;
      int const yy3 = (y < h-2) ? (yy+2) : (yy-2);

      int const rowOfs = y*w;
      float const * row0 = input + yy0*2*w; float const * row1 = input + yy1*2*w;
      float const * row2 = input + yy2*2*w; float const * row3 = input + yy3*2*w;

      for (int x = 1; x < w-1; ++x)
      {
         int const xx = 2*x;
         float value = 0;

         value += 1*row0[xx-1] + 3*row0[xx+0] + 3*row0[xx+1] + 1*row0[xx+2];
         value += 3*row1[xx-1] + 9*row1[xx+0] + 9*row1[xx+1] + 3*row1[xx+2];
         value += 3*row2[xx-1] + 9*row2[xx+0] + 9*row2[xx+1] + 3*row2[xx+2];
         value += 1*row3[xx-1] + 3*row3[xx+0] + 3*row3[xx+1] + 1*row3[xx+2];

         output[x + rowOfs] = (value+32) / (1 << 6);
      } // end for (x)
   } // end for (y)

   // Column x = 0
   for (int y = 0; y < h; ++y)
   {
      int const yy = 2*y;
      int const yy0 = (y > 0) ? (yy-1) : 1;
      int const yy1 = yy;
      //int const yy2 = (y < h-1) ? (yy+1) : (yy-1);
      int const yy2 = yy+1;
      int const yy3 = (y < h-2) ? (yy+2) : (yy-2);

      int const rowOfs = y*w;
      float const * row0 = input + yy0*2*w; float const * row1 = input + yy1*2*w;
      float const * row2 = input + yy2*2*w; float const * row3 = input + yy3*2*w;

      int const x = 0;
      int const xx = 2*x;
      float value = 0;

      value += 1*row0[xx+1] + 3*row0[xx+0] + 3*row0[xx+1] + 1*row0[xx+2];
      value += 3*row1[xx+1] + 9*row1[xx+0] + 9*row1[xx+1] + 3*row1[xx+2];
      value += 3*row2[xx+1] + 9*row2[xx+0] + 9*row2[xx+1] + 3*row2[xx+2];
      value += 1*row3[xx+1] + 3*row3[xx+0] + 3*row3[xx+1] + 1*row3[xx+2];

      output[x + rowOfs] = (value+32) / (1 << 6);
   } // end for (y)

//       // Column x = w-2
//       for (int y = 0; y < h; ++y)
//       {
//          int const yy = 2*y;
//          int const yy0 = (y > 0) ? (yy-1) : 1;
//          int const yy1 = yy;
//          //int const yy2 = (y < h-1) ? (yy+1) : (yy-1);
//          int const yy2 = yy+1;
//          int const yy3 = (y < h-2) ? (yy+2) : (yy-2);

//          int const rowOfs = y*w;
//          byte const * row0 = input + yy0*2*w; byte const * row1 = input + yy1*2*w;
//          byte const * row2 = input + yy2*2*w; byte const * row3 = input + yy3*2*w;

//          int const x = w-2;
//          int const xx = 2*x;
//          int value = 0;

//          value += 1*row0[xx-1] + 3*row0[xx+0] + 3*row0[xx+1] + 1*row0[xx+0];
//          value += 3*row1[xx-1] + 9*row1[xx+0] + 9*row1[xx+1] + 3*row1[xx+0];
//          value += 3*row2[xx-1] + 9*row2[xx+0] + 9*row2[xx+1] + 3*row2[xx+0];
//          value += 1*row3[xx-1] + 3*row3[xx+0] + 3*row3[xx+1] + 1*row3[xx+0];

//          output[x + rowOfs] = (value+32) >> 6;
//       } // end for (y)

   // Column x = w-1
   for (int y = 0; y < h; ++y)
   {
      int const yy = 2*y;
      int const yy0 = (y > 0) ? (yy-1) : 1;
      int const yy1 = yy;
      //int const yy2 = (y < h-1) ? (yy+1) : (yy-1);
      int const yy2 = yy+1;
      int const yy3 = (y < h-2) ? (yy+2) : (yy-2);

      int const rowOfs = y*w;
      float const * row0 = input + yy0*2*w; float const * row1 = input + yy1*2*w;
      float const * row2 = input + yy2*2*w; float const * row3 = input + yy3*2*w;

      int const x = w-1;
      int const xx = 2*x;
      float value = 0;

      value += 1*row0[xx-1] + 3*row0[xx+0] + 3*row0[xx+1] + 1*row0[xx-1];
      value += 3*row1[xx-1] + 9*row1[xx+0] + 9*row1[xx+1] + 3*row1[xx-1];
      value += 3*row2[xx-1] + 9*row2[xx+0] + 9*row2[xx+1] + 3*row2[xx-1];
      value += 1*row3[xx-1] + 3*row3[xx+0] + 3*row3[xx+1] + 1*row3[xx-1];

      output[x + rowOfs] = (value+32) / (1 << 6);
   } // end for (y)
} // end reduce4()

//**********************************************************************


#define UPSAMPLE_DIM_X 16

__global__ void
kernel_upsampleBuffer(int w, int h, float const factor, float const * Usrc, float * Udst)
{
   int const tidx = threadIdx.x;

   int const X0 = blockIdx.x*blockDim.x;
   int const Y  = blockIdx.y;

   __shared__ float u_sh[UPSAMPLE_DIM_X];

   u_sh[tidx] = factor * Usrc[Y*w + X0 + tidx];

   float u0 = u_sh[tidx/2];
   Udst[2*X0 + tidx + (2*Y+0)*2*w] = u0;
   Udst[2*X0 + tidx + (2*Y+1)*2*w] = u0;

   float u1 = u_sh[tidx/2 + UPSAMPLE_DIM_X/2];
   Udst[2*X0 + tidx + UPSAMPLE_DIM_X + (2*Y+0)*2*w] = u1;
   Udst[2*X0 + tidx + UPSAMPLE_DIM_X + (2*Y+1)*2*w] = u1;
}

//**********************************************************************

namespace V3D_CUDA
{

   void
   TVL1_FlowEstimationBase::allocate(int w, int h, int nLevels)
   {
      _width = w;
      _height = h;
      _nLevels = nLevels;

      _widths.resize(nLevels);
      _heights.resize(nLevels);
      _imageSizes.resize(nLevels);
      _d_coeffs.resize(nLevels);

      for (int level = 0; level < nLevels; ++level)
      {
         int const W = w / (1 << level);
         int const H = h / (1 << level);

         _widths[level]     = W;
         _heights[level]    = H;
         _imageSizes[level] = sizeof(float) * W * H;

         CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_coeffs[level], 3*_imageSizes[level]) );
      } // end for (level)
   } // end TVL1_FlowEstimationBase::allocate()

   void
   TVL1_FlowEstimationBase::deallocate()
   {
      for (int level = 0; level < _nLevels; ++level)
         CUDA_SAFE_CALL( cudaFree(_d_coeffs[level]) );
   } // end TVL1_FlowEstimationBase::deallocate()


   void
   TVL1_FlowEstimationBase::warpImage(int level, float const * I0, float const * I1, float * d_u, float * d_v)
   {
      int const w = _widths[level];
      int const h = _heights[level];

      dim3 gridDim(w/WARP_DIM_X, h/WARP_DIM_Y, 1);
      dim3 blockDim(WARP_DIM_X, WARP_DIM_Y, 1);

      int const size = _imageSizes[level];
      int const imgOfs = size / sizeof(float);

      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
      cudaArray * d_I0Array;
      cudaArray * d_I1Array;

      CUDA_SAFE_CALL( cudaMallocArray(&d_I0Array, &channelDesc, w, h));
      CUDA_SAFE_CALL( cudaMemcpyToArray(d_I0Array, 0, 0, I0, size, cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL( cudaBindTextureToArray(I0_tex, d_I0Array, channelDesc) );

      CUDA_SAFE_CALL( cudaMallocArray(&d_I1Array, &channelDesc, w, h));
      CUDA_SAFE_CALL( cudaMemcpyToArray(d_I1Array, 0, 0, I1, size, cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL( cudaBindTextureToArray(I1_tex, d_I1Array, channelDesc) );

      I0_tex.addressMode[0] = cudaAddressModeClamp;
      I0_tex.addressMode[1] = cudaAddressModeClamp;
      I0_tex.filterMode = cudaFilterModeLinear;
      I0_tex.normalized = false;

      I1_tex.addressMode[0] = cudaAddressModeClamp;
      I1_tex.addressMode[1] = cudaAddressModeClamp;
      I1_tex.filterMode = cudaFilterModeLinear;
      I1_tex.normalized = false;

      float * d_a1 = _d_coeffs[level];
      float * d_a2 = d_a1 + imgOfs;
      float * d_c  = d_a2 + imgOfs;

      kernel_warpImage<<< gridDim, blockDim, 0>>>(w, h, d_u, d_v, d_a1, d_a2, d_c);

//       V3D::Image<float> warped(w, h, 1);
//       CUDA_SAFE_CALL( cudaMemcpy( warped.begin(), d_c, size, cudaMemcpyDeviceToHost) );
//       V3D::saveImageChannel(warped, 0, -1.0f, 1.0f, "warped.png");

//       CUDA_SAFE_CALL( cudaMemcpy( warped.begin(), d_a1, size, cudaMemcpyDeviceToHost) );
//       V3D::saveImageChannel(warped, 0, -0.1f, 0.1f, "Ix.png");
//       CUDA_SAFE_CALL( cudaMemcpy( warped.begin(), d_a2, size, cudaMemcpyDeviceToHost) );
//       V3D::saveImageChannel(warped, 0, -0.1f, 0.1f, "Iy.png");

      CUDA_SAFE_CALL( cudaUnbindTexture(&I0_tex) );
      CUDA_SAFE_CALL( cudaUnbindTexture(&I1_tex) );
      CUDA_SAFE_CALL( cudaFreeArray(d_I0Array) );
      CUDA_SAFE_CALL( cudaFreeArray(d_I1Array) );
   } // end TVL1_FlowEstimationBase::warpImage()

//**********************************************************************

   void
   TVL1_FlowEstimation_Relaxed::allocate(int w, int h, int nLevels)
   {
      TVL1_FlowEstimationBase::allocate(w, h, nLevels);

      _leftPyr = new V3D::Image<float>[nLevels];
      _rightPyr = new V3D::Image<float>[nLevels];

      _bufferSizes.resize(nLevels);
      _d_u.resize(nLevels); _d_v.resize(nLevels);
      _d_pu.resize(nLevels), _d_pv.resize(nLevels);

      for (int level = 0; level < nLevels; ++level)
      {
         int const W = w / (1 << level);
         int const H = h / (1 << level);

         _leftPyr[level].resize(W, H, 1);
         _rightPyr[level].resize(W, H, 1);

         // Allocate additionals row to avoid a few conditionals in the kernel
         int const size = sizeof(float) * W * (H+1);
         _bufferSizes[level] = size;

         CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_u[level], size) );
         CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_v[level], size) );
         CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_pu[level], 2*size) );
         CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_pv[level], 2*size) );
      } // end for (level)
   } // end TVL1_FlowEstimation_Relaxed::allocate()

   void
   TVL1_FlowEstimation_Relaxed::deallocate()
   {
      delete [] _leftPyr;
      delete [] _rightPyr;

      for (int level = 0; level < _nLevels; ++level)
      {
         CUDA_SAFE_CALL( cudaFree(_d_u[level]) );
         CUDA_SAFE_CALL( cudaFree(_d_v[level]) );
         CUDA_SAFE_CALL( cudaFree(_d_pu[level]) );
         CUDA_SAFE_CALL( cudaFree(_d_pv[level]) );
      } // end for (level)

      TVL1_FlowEstimationBase::deallocate();
   } // end TVL1_FlowEstimation_Relaxed::deallocate()

   void
   TVL1_FlowEstimation_Relaxed::setImage(int index, unsigned char const * pixels)
   {
      float * floatPixels = new float[_width*_height];
      for (int k = 0; k < _width*_height; ++k)
         floatPixels[k] = float(pixels[k]);
      this->setImage(index, floatPixels);
      delete [] floatPixels;
   }

   void
   TVL1_FlowEstimation_Relaxed::setImage(int index, float const * pixels)
   {
      V3D::Image<float> * dstPyr = (index == 0) ? _leftPyr : _rightPyr;
      V3D::Image<float>& im0 = dstPyr[0];

      for (int y = 0; y < _height; ++y)
         for (int x = 0; x < _width; ++x)
            im0(x, y) = pixels[y*_width + x];

      for (size_t i = 1; i < _nLevels; ++i)
         reduce4(dstPyr[i].width(), dstPyr[i].height(), dstPyr[i-1].begin(), dstPyr[i].begin());

//       char name[200];
//       for (size_t i = 0; i < _nLevels; ++i)
//       {
//          sprintf(name, "pyr-%i-%i.png", index, i);
//          saveImageChannel(dstPyr[i], 0, 0.0f, 255.0f, name);
//       }
   } // end TVL1_FlowEstimation_Relaxed::setImage()

   void
   TVL1_FlowEstimation_Relaxed::run()
   {
      //cout << "lambda = " << _lambda << ", theta = " << _cfg._theta << ", tau = " << _cfg._tau << endl;

      for (int l = _nLevels-1; l >= _startLevel; --l)
      {
         int const W = _widths[l];
         int const H = _heights[l];

         dim3 gridDim(W/DIM_X, H/DIM_Y, 1);
         dim3 blockDim(DIM_X, DIM_Y, 1);

         float const lambdaLevel = _lambda;
         float const thetaLevel = _cfg._theta;
         //float const thetaLevel = _cfg._theta / float(1 << l);

         float const tau_theta = _cfg._tau / thetaLevel;
         float const lambda_theta = lambdaLevel * thetaLevel;
         //cout << "lambda_theta = " << lambda_theta << endl;

         if (l == _nLevels-1)
            this->clearBuffers(l);
         else
            this->upsampleBuffers(l);

         int const bufOfs = _bufferSizes[l] / sizeof(float);
         int const imgOfs = _imageSizes[l] / sizeof(float);

         float * d_u = _d_u[l];
         float * d_v = _d_v[l];
         float * d_pu1 = _d_pu[l];
         float * d_pu2 = _d_pu[l] + bufOfs;
         float * d_pv1 = _d_pv[l];
         float * d_pv2 = _d_pv[l] + bufOfs;
         float * d_a1 = _d_coeffs[l];
         float * d_a2 = _d_coeffs[l] + 1*imgOfs;
         float * d_c  = _d_coeffs[l] + 2*imgOfs;

         for (int k = 0; k < _nOuterIterations; ++k)
         {
            this->warpImage(l, _leftPyr[l].begin(), _rightPyr[l].begin(), _d_u[l], _d_v[l]);

            for (int i = 0; i < _nInnerIterations; ++i)
            {
               kernel_updatePVolume_theta<<< gridDim, blockDim, 0 >>>(W, H, tau_theta, d_u, d_pu1, d_pu2);
               kernel_updatePVolume_theta<<< gridDim, blockDim, 0 >>>(W, H, tau_theta, d_v, d_pv1, d_pv2);
               kernel_updateUVolume_theta<<< gridDim, blockDim, 0 >>>(W, H, lambda_theta, thetaLevel, d_a1, d_a2, d_c,
                                                                      d_u, d_v, d_pu1, d_pu2, d_pv1, d_pv2);
            } // end for (i)
         } // end for (k)
      } // end for (l)
   } // end TVL1_FlowEstimation_Relaxed::run()

   void
   TVL1_FlowEstimation_Relaxed::getFlowField(float * uDst, float * vDst)
   {
      int const size = _imageSizes[_startLevel];
      float * d_u = _d_u[_startLevel];
      float * d_v = _d_v[_startLevel];

      if (uDst) CUDA_SAFE_CALL( cudaMemcpy( uDst, d_u, size, cudaMemcpyDeviceToHost) );
      if (vDst) CUDA_SAFE_CALL( cudaMemcpy( vDst, d_v, size, cudaMemcpyDeviceToHost) );
   } // end TVL1_FlowEstimation_Relaxed::getFlowField()

   void
   TVL1_FlowEstimation_Relaxed::clearBuffers(int level)
   {
      int const size = _bufferSizes[level];

      float * d_u = _d_u[level];
      float * d_v = _d_v[level];
      float * d_pu = _d_pu[level];
      float * d_pv = _d_pv[level];

      CUDA_SAFE_CALL( cudaMemset( d_u, 0, size) );
      CUDA_SAFE_CALL( cudaMemset( d_v, 0, size) );
      CUDA_SAFE_CALL( cudaMemset( d_pu, 0, 2*size) );
      CUDA_SAFE_CALL( cudaMemset( d_pv, 0, 2*size) );
   }

   void
   TVL1_FlowEstimation_Relaxed::upsampleBuffers(int dstLevel)
   {
      int const w = _widths[dstLevel+1];
      int const h = _heights[dstLevel+1];

      int const dstSize = _bufferSizes[dstLevel];
      int const dstOfs = dstSize / sizeof(float);
      int const srcOfs = _bufferSizes[dstLevel+1] / sizeof(float);

      float * src_d_u = _d_u[dstLevel+1];
      float * src_d_v = _d_v[dstLevel+1];
      float * src_d_pu = _d_pu[dstLevel+1];
      float * src_d_pv = _d_pv[dstLevel+1];

      float * dst_d_u = _d_u[dstLevel];
      float * dst_d_v = _d_v[dstLevel];
      float * dst_d_pu = _d_pu[dstLevel];
      float * dst_d_pv = _d_pv[dstLevel];

      dim3 gridDim(w/UPSAMPLE_DIM_X, h, 1);
      dim3 blockDim(UPSAMPLE_DIM_X, 1, 1);
   
      kernel_upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src_d_u, dst_d_u);
      kernel_upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src_d_v, dst_d_v);

#if 1
      // Also upsample the dual variables
      kernel_upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 1.0f, src_d_pu, dst_d_pu);
      kernel_upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 1.0f, src_d_pu+srcOfs, dst_d_pu+dstOfs);
      kernel_upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 1.0f, src_d_pv, dst_d_pv);
      kernel_upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 1.0f, src_d_pv+srcOfs, dst_d_pv+dstOfs);
#else
      CUDA_SAFE_CALL( cudaMemset( dst_d_pu, 0, 2*dstSize) );
      CUDA_SAFE_CALL( cudaMemset( dst_d_pv, 0, 2*dstSize) );
#endif
   } // end TVL1_FlowEstimation_Relaxed::upsampleBuffers()

} // end namespace V3D_CUDA

#endif // defined(V3DLIB_ENABLE_CUDA)
