#include <iostream>
#include <sstream>

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_image.h"
#include "Base/v3d_feature.h"
#include "Base/v3d_storage.h"
#include "Base/v3d_utilities.h"
#include "Geometry/v3d_poseutilities.h"
#include "CUDA/v3d_cudamatching.h"

#include "cublas.h"

#include "reconstruction_common.h"

using namespace std;
using namespace V3D;

namespace
{

   SerializableVector<SIFT_Feature> *
   loadFeatures(vector<string> const& entries, int viewNo)
   {
      ostringstream oss;
      oss << entries[viewNo] << ".features";
      cout << "reading " << oss.str() << endl;

      SerializableVector<SIFT_Feature> * curFeatures = new SerializableVector<SIFT_Feature>();

      serializeDataFromFile(oss.str().c_str(), *curFeatures);
      return curFeatures;
   }

   inline int
   getEffectiveInlierCount(vector<PointCorrespondence> const& corrs, int imageWidth, int imageHeight)
   {
      vector<Vector2f> left(corrs.size());
      vector<Vector2f> right(corrs.size());
      for (size_t i = 0; i < corrs.size(); ++i)
      {
         copyVector(corrs[i].left.pos, left[i]);
         copyVector(corrs[i].right.pos, right[i]);
      }

      float const coverage1 = estimateCoverage(left, imageWidth, imageHeight);
      float const coverage2 = estimateCoverage(right, imageWidth, imageHeight);
      float const coverage  = 0.5f*coverage1 + 0.5f*coverage2;

      return int(coverage*float(corrs.size()) + 0.5f);
   } // end getEffectiveInlierCount()

   inline void
   refinePoseFromInliers(vector<PointCorrespondence> const& corrs, Matrix3x3d const& K1, Matrix3x3d const& K2,
                         Matrix3x3d& R, Vector3d& T)
   {
      vector<Vector2d> left(corrs.size());
      vector<Vector2d> right(corrs.size());
      for (size_t i = 0; i < corrs.size(); ++i)
      {
         copyVector(corrs[i].left.pos, left[i]);
         copyVector(corrs[i].right.pos, right[i]);
      }
      cout << "refineRelativePose()..." << endl;
      refineRelativePose(left, right, K1, K2, R, T, 10.0);
      cout << "done." << endl;
   } // end refinePoseFromInliers()

} // end namespace <>


int main(int argc, char * argv[])
{
   if (argc != 4)
   {
      cerr << "Usage: " << argv[0] << " <image list file> <file.conf> <radius>" << endl;
      return -1;
   }

   cublasInit();
   float const scoreThreshold = 0.9f;

   V3D_CUDA::SIFT_Matching matcher;

   try
   {
      int const radius = atoi(argv[3]);

      ConfigurationFile cf(argv[2]);

      bool const withBackmatching = cf.get("WITH_BACKMATCHING", true);
      float const distRatio = cf.get("MATCHING_SCORE_RATIO", 0.8f);

      int const nMinRawCorrs = cf.get("MINIMUM_RAW_CORRESPONDENCES", 100);
      float const egInlierThreshold = cf.get("EG_INLIER_THRESHOLD", 2.0) / 1024; // Normalize to 1024^2 images
      int nMinInlierCorrs = cf.get("MINIMUM_INLIER_CORRESPONDENCES", -1);
      if (nMinInlierCorrs < 0) nMinInlierCorrs = nMinRawCorrs;
      double const minInlierRatio = cf.get("MINIMUM_INLIER_MATCHES_RATIO", 0.5);
      bool const useEffectiveInliers = cf.get("USE_EFFECTIVE_INLIERS", true);

      //double const minRotationError = cf.get("MIN_ROTATION_ERROR", 0.05);

      int const nMaxFeatures = cf.get("MATCHING_MAX_FEATURES", 7168);
      bool const useGuidedMatching = cf.get("USE_GUIDED_MATCHING", true);
      matcher.allocate(nMaxFeatures, useGuidedMatching);

      bool const useReductionPath = cf.get("MATCHING_USE_REDUCTION", false);
      matcher.enableReductionPath(useReductionPath);

      CalibrationDatabase calibDb("calibration_db.txt");

      std::vector<std::string> entries;

      {
         ifstream is(argv[1]);
         string name;
         while (is >> name)
            entries.push_back(name);
      }

      // Do geometric matching with putative matches

      LRU_Cache<SerializableVector<SIFT_Feature> > residentFeatures(500);

      SQLite3_Database matchesDB("pairwise_matches.db");
      matchesDB.createTable("matches_data", true);
      matchesDB.createTable("matches_list", true);

      vector<ViewPair> matchedPairs;
      vector<ViewPair> panoramaPairs;

      int nViews = entries.size();

      for (int curFrameId = 0; curFrameId < nViews; ++curFrameId)
      {
         if (!residentFeatures.has(curFrameId))
         {
            SerializableVector<SIFT_Feature> * features = loadFeatures(entries, curFrameId);
            if (features == 0) continue;
            residentFeatures.insert(curFrameId, features);
         }

         Matrix3x3d const K1 = calibDb.getIntrinsic(curFrameId);
         Matrix3x3d const invK1 = invertedMatrix(K1);

         int const imWidth1 = std::max(calibDb.getImageWidth(curFrameId), calibDb.getImageHeight(curFrameId));

         // Use copy constructor to avoid troubles if the LRU caches decides to remove this item.
         SerializableVector<SIFT_Feature> const view1Features = *residentFeatures[curFrameId];

         // Erase the least-significant bits to make the number of features divisible by 32.
         int const nFeatures1 = std::min(nMaxFeatures, int(view1Features.size() & 0xffe0));
         float * features1 = new float[128 * nFeatures1];
         for (int k = 0; k < nFeatures1; ++k)
            memcpy(features1 + 128*k, view1Features[k].descriptor, 128 * sizeof(float));

         matcher.setLeftFeatures(nFeatures1, features1);

         for (int otherFrameId = curFrameId+1; otherFrameId < curFrameId+2*radius; ++otherFrameId)
         {
            if (otherFrameId >= nViews) continue;

            if (!residentFeatures.has(otherFrameId))
            {
               SerializableVector<SIFT_Feature> * features = loadFeatures(entries, otherFrameId);
               if (features == 0) continue;
               residentFeatures.insert(otherFrameId, features);
            }

            SerializableVector<SIFT_Feature> const& view2Features = *residentFeatures[otherFrameId];

            Matrix3x3d const K2 = calibDb.getIntrinsic(otherFrameId);
            Matrix3x3d const invK2 = invertedMatrix(K2);

            int const imWidth2 = std::max(calibDb.getImageWidth(otherFrameId), calibDb.getImageHeight(otherFrameId));
            int const maxWidth = std::max(imWidth1, imWidth2);

            cout << "--------------------" << endl;
            cout << "matching view " << curFrameId << " with view " << otherFrameId << endl;

            RobustOrientationResult result;

            // Erase the least-significant bits to make the number of features divisible by 32.
            int const nFeatures2 = std::min(nMaxFeatures, int(view2Features.size() & 0xffe0));

            cout << "nFeatures1 = " << nFeatures1 << ", nFeatures2 = " << nFeatures2 << endl;

            SerializableVector<PointCorrespondence> corrs, rawCorrs;
            rawCorrs.reserve(std::min(nFeatures1, nFeatures2));

            int const view1 = curFrameId;
            int const view2 = otherFrameId;

            {
               float * features2 = new float[128 * nFeatures2];

               for (int k = 0; k < nFeatures2; ++k)
                  memcpy(features2 + 128*k, view2Features[k].descriptor, 128 * sizeof(float));

               matcher.setRightFeatures(nFeatures2, features2);

               vector<pair<int, int> > matches;
               matcher.findPutativeMatches(matches, scoreThreshold, distRatio);
               cout << "matches.size() = " << matches.size() << endl;

               for (int k = 0; k < matches.size(); ++k)
               {
                  PointCorrespondence corr;

                  int const i1 = matches[k].first;
                  int const i2 = matches[k].second;

                  corr.left.id     = view1Features[i1].id;
                  corr.left.view   = view1;
                  corr.left.pos[0] = view1Features[i1].position[0];
                  corr.left.pos[1] = view1Features[i1].position[1];

                  corr.right.id     = view2Features[i2].id;
                  corr.right.view   = view2;
                  corr.right.pos[0] = view2Features[i2].position[0];
                  corr.right.pos[1] = view2Features[i2].position[1];

                  rawCorrs.push_back(corr);
               }

               delete [] features2;
            } // end scope

            if (rawCorrs.size() < nMinRawCorrs) continue;

            cout << "RANSAC for essential matrix..." << endl;
            int const nSamples = 1000;
            double const inlierThreshold = maxWidth * egInlierThreshold;
            cout << "inlierThreshold = " << inlierThreshold << endl;
            RobustOrientationMode mode;
            mode.iterativeRefinement = true;
            computeRobustOrientationMLE(rawCorrs, K1, K2, inlierThreshold, nSamples, result, true, mode);
            cout << "done." << endl;

            cout << " this pair has " << result.inliers.size() << " inliers." << endl;

            // Ignore EG if absolute number of inliers is too small
            if (result.inliers.size() < nMinInlierCorrs)
            {
               cout << " This pair is ignored because of too few inliers." << endl;
               continue;
            }
            // Ignore EG if the number of initially surviving inliers is much smaller then the number of raw matches
            if (double(result.inliers.size()) < minInlierRatio * rawCorrs.size())
            {
               cout << " This pair is ignored because the number of matches dropped too much." << endl;
               continue;
            }

            corrs.reserve(result.inliers.size());
            for (size_t i = 0; i < result.inliers.size(); ++i)
               corrs.push_back(rawCorrs[result.inliers[i]]);

            int const imageWidth = calibDb.getImageWidth(curFrameId);
            int const imageHeight = calibDb.getImageHeight(curFrameId);

            if (!useGuidedMatching)
            {
               // if (corrs.size() < nMinInlierCorrs) continue;
               int const nInliers = (useEffectiveInliers ?
                                     getEffectiveInlierCount(corrs, imageWidth, imageHeight) : corrs.size());
               cout << "Effective inlier count is " << nInliers << endl;
               if (nInliers < nMinInlierCorrs) continue;
            }
            else
            {
               // if (corrs.size() < 5) continue;
               // if (corrs.size() < nMinInlierCorrs/2) continue;

               refinePoseFromInliers(corrs, K1, K2, result.rotation, result.translation);

               Matrix3x3d const invK1 = invertedMatrix(K1);
               Matrix3x3d const invK2 = invertedMatrix(K2);
               Matrix3x3d const invK2_t = invK2.transposed();
               Matrix3x3d const E = crossProductMatrix(result.translation) * result.rotation;
               Matrix3x3d const F = invK2_t * E * invK1;

               vector<Vector2f> leftPos(nFeatures1);
               vector<Vector2f> rightPos(nFeatures2);

               for (int k = 0; k < nFeatures1; ++k)
                  leftPos[k] = view1Features[k].position;

               for (int k = 0; k < nFeatures2; ++k)
                  rightPos[k] = view2Features[k].position;

               vector<pair<int, int> > matches;
               matcher.runGuidedMatching(F, inlierThreshold, &leftPos[0], &rightPos[0], matches, scoreThreshold, distRatio);
               cout << "matches.size() after guided matching: " << matches.size() << endl;

               corrs.clear();
               for (int k = 0; k < matches.size(); ++k)
               {
                  PointCorrespondence corr;

                  int const i1 = matches[k].first;
                  int const i2 = matches[k].second;

                  corr.left.id     = view1Features[i1].id;
                  corr.left.view   = view1;
                  corr.left.pos[0] = view1Features[i1].position[0];
                  corr.left.pos[1] = view1Features[i1].position[1];

                  corr.right.id     = view2Features[i2].id;
                  corr.right.view   = view2;
                  corr.right.pos[0] = view2Features[i2].position[0];
                  corr.right.pos[1] = view2Features[i2].position[1];

                  corrs.push_back(corr);
               } // end for (k)

               int const nInliers = (useEffectiveInliers ?
                                     getEffectiveInlierCount(corrs, imageWidth, imageHeight) : corrs.size());
               cout << "Effective inlier count is " << nInliers << endl;
               if (nInliers < nMinInlierCorrs) continue;

               cout << "refineRelativePose() after guided matching:" << endl;
               refinePoseFromInliers(corrs, K1, K2, result.rotation, result.translation);
            } // end if (useGuidedMatching)

            for (size_t i = 0; i < corrs.size(); ++i)
            {
               corrs[i].left.view = view1;
               corrs[i].right.view = view2;

               Vector3d p, q;
               multiply_A_v_affine(invK1, corrs[i].left.pos, p);
               multiply_A_v_affine(invK2, corrs[i].right.pos, q);

               corrs[i].left.pos[0]  = p[0]; corrs[i].left.pos[1]  = p[1];
               corrs[i].right.pos[0] = q[0]; corrs[i].right.pos[1] = q[1];
            }

            PairwiseMatch match;
            match.views       = ViewPair(view1, view2);
            match.rotation    = result.rotation;
            match.translation = result.translation;
            match.corrs       = corrs;

            matchesDB.updateObject("matches_data", matchedPairs.size(), match);
            matchesDB.updateObject("matches_list", matchedPairs.size(), ViewPair(view1, view2));

            matchedPairs.push_back(ViewPair(view1, view2));
         } // end for (q)
         delete [] features1;
      } // end for (p)

      // Those files are now only for information only
      {
         ofstream os("matchedPairs.txt");
         TextOStreamArchive ar(os);
         serializeVector(matchedPairs, ar);
      }

      {
         ofstream os("panoramaPairs.txt");
         TextOStreamArchive ar(os);
         serializeVector(panoramaPairs, ar);
      }
   }
   catch (std::string s)
   {
      cerr << "Exception caught: " << s << endl;
   }
   catch (...)
   {
      cerr << "Unhandled exception." << endl;
   }

   matcher.deallocate();

   cublasShutdown();

   return 0;
}
