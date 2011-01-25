// -*- C++ -*-

#ifndef V3D_MVIEW_INITIALIZATION_H
#define V3D_MVIEW_INITIALIZATION_H

#include "Geometry/v3d_mviewutilities.h"

namespace V3D
{

#if defined(V3DLIB_ENABLE_LPSOLVE)

   void computeConsistentTranslationsOSE_L1(std::vector<Matrix3x3d> const& rotations,
                                            std::vector<float> const& weights,
                                            std::vector<Vector3d>& translations,
                                            std::vector<TriangulatedPoint>& sparseModel);

   inline void
   computeConsistentTranslationsOSE_L1(std::vector<Matrix3x3d> const& rotations,
                                       std::vector<Vector3d>& translations,
                                       std::vector<TriangulatedPoint>& sparseModel)
   {
      int K = 0;
      for (size_t i = 0; i < sparseModel.size(); ++i) K += sparseModel[i].measurements.size();

      std::vector<float> const weights(K, 1.0f);
      computeConsistentTranslationsOSE_L1(rotations, weights, translations, sparseModel);
   }

   void computeConsistentTranslationsConic_L1(float const sigma,
                                              std::vector<Matrix3x3d> const& rotations,
                                              std::vector<float> const& weights,
                                              std::vector<Vector3d>& translations,
                                              std::vector<TriangulatedPoint>& sparseModel);

   inline void
   computeConsistentTranslationsConic_L1(float const sigma,
                                         std::vector<Matrix3x3d> const& rotations,
                                         std::vector<Vector3d>& translations,
                                         std::vector<TriangulatedPoint>& sparseModel)
   {
      int K = 0;
      for (size_t i = 0; i < sparseModel.size(); ++i) K += sparseModel[i].measurements.size();

      std::vector<float> const weights(K, 1.0f);
      computeConsistentTranslationsConic_L1(sigma, rotations, weights, translations, sparseModel);
   }

   void computeConsistentTranslationsConic_L1_reduced(float const sigma,
                                                      std::vector<Matrix3x3d> const& rotations,
                                                      std::vector<Vector3d>& translations,
                                                      std::vector<TriangulatedPoint>& sparseModel);

#endif

   void computeConsistentTranslationsConic_L1_New(float const sigma,
                                                  std::vector<Matrix3x3d> const& rotations,
                                                  std::vector<float> const& weights,
                                                  std::vector<Vector3d>& translations,
                                                  std::vector<TriangulatedPoint>& sparseModel,
                                                  bool useIP = true, bool useInitialValue = false);

   inline void
   computeConsistentTranslationsConic_L1_New(float const sigma,
                                             std::vector<Matrix3x3d> const& rotations,
                                             std::vector<Vector3d>& translations,
                                             std::vector<TriangulatedPoint>& sparseModel,
                                             bool useIP = true, bool useInitialValue = false)
   {
      int K = 0;
      for (size_t i = 0; i < sparseModel.size(); ++i) K += sparseModel[i].measurements.size();

      std::vector<float> const weights(K, 1.0f);
      computeConsistentTranslationsConic_L1_New(sigma, rotations, weights, translations, sparseModel, useIP, useInitialValue);
   }

   struct TranslationRegistrationPD_Params
   {
         TranslationRegistrationPD_Params()
            : timestepRatio(16.0), timestepMultiplier(0.95), similarityThreshold(1e-6),
              nMaxIterations(10000000), reportFrequency(10000)
         { }

         double timestepRatio, timestepMultiplier, similarityThreshold;
         int nMaxIterations, reportFrequency;
   }; // end struct TranslationRegistrationPD_Params

//    void computeConsistentTranslationsConic_L1_PD(float const sigma,
//                                                  std::vector<Matrix3x3d> const& rotations,
//                                                  std::vector<Vector3d>& translations,
//                                                  std::vector<TriangulatedPoint>& sparseModel,
//                                                  TranslationRegistrationPD_Params const& params = TranslationRegistrationPD_Params());

//    void computeConsistentTranslationsConic_L1_PD2(float const sigma,
//                                                   std::vector<Matrix3x3d> const& rotations,
//                                                   std::vector<Vector3d>& translations,
//                                                   std::vector<TriangulatedPoint>& sparseModel,
//                                                   TranslationRegistrationPD_Params const& params = TranslationRegistrationPD_Params());

//    void computeConsistentTranslationsConic_L1_PD4(float const sigma,
//                                                   std::vector<Matrix3x3d> const& rotations,
//                                                   std::vector<Vector3d>& translations,
//                                                   std::vector<TriangulatedPoint>& sparseModel,
//                                                   TranslationRegistrationPD_Params const& params = TranslationRegistrationPD_Params());

   void computeConsistentTranslationsConic_Huber_PD(float const sigma,
                                                    std::vector<Matrix3x3d> const& rotations,
                                                    std::vector<Vector3d>& translations,
                                                    std::vector<TriangulatedPoint>& sparseModel,
                                                    TranslationRegistrationPD_Params const& params = TranslationRegistrationPD_Params());

   void computeConsistentTranslationsConic_Huber_PD_Popov(float const sigma,
                                                          std::vector<Matrix3x3d> const& rotations,
                                                          std::vector<Vector3d>& translations,
                                                          std::vector<TriangulatedPoint>& sparseModel,
                                                          TranslationRegistrationPD_Params const& params = TranslationRegistrationPD_Params());

//======================================================================

   typedef InlineVector<int, 3> Vector3i;
   typedef InlineVector<int, 2> Vector2i;

   bool computeConsistentCameraCenters_LP(std::vector<Vector3d> const& c_ji, std::vector<Vector3d> const& c_jk,
                                          std::vector<Vector3i> const& ijks,
                                          std::vector<Vector3d>& centers, bool verbose = false);


   struct MultiViewInitializationParams_BOS
   {
         MultiViewInitializationParams_BOS()
            : nIterations(20000), L(-1.0), alpha(0.1), stoppingThreshold(1e-6),
              verbose(false), reportFrequency(1000), checkFrequency(100)
         {  }

         int    nIterations;
         double L, alpha; // Lipschitz-constant |A|^2 (automatically computed if <= 0), primal step size
         double stoppingThreshold;
         bool   verbose;
         int    reportFrequency, checkFrequency;
   }; // end struct MultiViewInitializationParams_BOS

//    void
//    computeConsistentTranslationsConic_LP_BOS(float const sigma,
//                                              std::vector<Matrix3x3d> const& rotations,
//                                              std::vector<Vector3d>& translations,
//                                              std::vector<TriangulatedPoint>& sparseModel,
//                                              MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());

   bool
   computeConsistentCameraCenters_L2_BOS(std::vector<Vector3d> const& c_ji, std::vector<Vector3d> const& c_jk,
                                         std::vector<Vector3i> const& ijks, std::vector<Vector3d>& centers,
                                         MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());


   bool computeConsistentCameraCenters_L2_SDMM(int const nSubModels, std::vector<Vector3d> const& c_ij,
                                               std::vector<Vector2i> const& ijs, std::vector<int> const& submodelIndices,
                                               std::vector<double> const& weights, std::vector<Vector3d>& centers,
                                               MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());

# if defined(V3DLIB_ENABLE_SUITESPARSE)
   void
   computeConsistentTranslationsConic_Aniso_SDMM(float const sigma,
                                                 std::vector<Matrix3x3d> const& rotations,
                                                 std::vector<Vector3d>& translations,
                                                 std::vector<TriangulatedPoint>& sparseModel,
                                                 MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS(),
                                                 bool const strictCheirality = true);

   void
   computeConsistentTranslationsConic_Iso_SDMM(float const sigma,
                                               std::vector<Matrix3x3d> const& rotations,
                                               std::vector<Vector3d>& translations,
                                               std::vector<TriangulatedPoint>& sparseModel,
                                               MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());

   void
   computeConsistentTranslationsConic_Huber_SDMM(float const sigma,
                                                 std::vector<Matrix3x3d> const& rotations,
                                                 std::vector<Vector3d>& translations,
                                                 std::vector<TriangulatedPoint>& sparseModel,
                                                 MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());
# endif

   void
   computeConsistentTranslationsConic_Aniso_BOS(float const sigma,
                                                std::vector<Matrix3x3d> const& rotations,
                                                std::vector<Vector3d>& translations,
                                                std::vector<TriangulatedPoint>& sparseModel,
                                                MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());

   void
   computeConsistentTranslationsConic_Aniso_LS_Free(float const sigma,
                                                    std::vector<Matrix3x3d> const& rotations,
                                                    std::vector<Vector3d>& translations,
                                                    std::vector<TriangulatedPoint>& sparseModel,
                                                    MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());

   void
   computeConsistentTranslationsConic_Aniso_ADMM(float const sigma,
                                                 std::vector<Matrix3x3d> const& rotations,
                                                 std::vector<Vector3d>& translations,
                                                 std::vector<TriangulatedPoint>& sparseModel,
                                                 MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());

# if defined(V3DLIB_ENABLE_SUITESPARSE)
   void
   computeConsistentTranslationsRelaxedConic_Aniso_SDMM(float const sigma,
                                                        std::vector<Matrix3x3d> const& rotations,
                                                        std::vector<Vector3d>& translations,
                                                        std::vector<TriangulatedPoint>& sparseModel,
                                                        MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());

   void
   computeConsistentTranslationsRelaxedConic_Iso_SDMM(float const sigma,
                                                      std::vector<Matrix3x3d> const& rotations,
                                                      std::vector<Vector3d>& translations,
                                                      std::vector<TriangulatedPoint>& sparseModel,
                                                      MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());

   void
   computeConsistentTranslationsRelaxedConic_Huber_SDMM(float const sigma,
                                                        std::vector<Matrix3x3d> const& rotations,
                                                        std::vector<Vector3d>& translations,
                                                        std::vector<TriangulatedPoint>& sparseModel,
                                                        MultiViewInitializationParams_BOS const& params = MultiViewInitializationParams_BOS());
# endif

} // end namespace V3D

#endif
