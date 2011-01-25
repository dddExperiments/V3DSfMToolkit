// -*- C++ -*-

#include "Base/v3d_serialization.h"
#include "Base/v3d_storage.h"
#include "Base/v3d_image.h"
#include "Math/v3d_linear.h"
#include "Geometry/v3d_cameramatrix.h"
#include "Geometry/v3d_mviewutilities.h"
#include "Geometry/v3d_tripletutilities.h"

#include <iostream>
#include <set>
#include <map>
#include <list>

//----------------------------------------------------------------------

struct CalibrationDatabase
{
      CalibrationDatabase(char const * calibDbName);

      int getImageWidth(int view) const
      {
         if (view < 0 || view >= _imageDimensions.size())
            throwV3DErrorHere("view id out of range.");
         return _imageDimensions[view].first;
      }

      int getImageHeight(int view) const
      {
         if (view < 0 || view >= _imageDimensions.size())
            throwV3DErrorHere("view id out of range.");
         return _imageDimensions[view].second;
      }

      int getMaxDimension(int view) const
      {
         if (view < 0 || view >= _imageDimensions.size())
            throwV3DErrorHere("view id out of range.");
         return std::max(_imageDimensions[view].first, _imageDimensions[view].second);
      }

      V3D::Matrix3x3d getIntrinsic(int view) const
      {
         if (view < 0 || view >= _imageDimensions.size())
            throwV3DErrorHere("view id out of range.");
         return _intrinsics[view];
      }

      double getAvgFocalLength(int view) const
      {
         if (view < 0 || view >= _imageDimensions.size())
            throwV3DErrorHere("view id out of range.");
         return 0.5*(_intrinsics[view][0][0] + _intrinsics[view][1][1]);
      }

   protected:
      std::vector<std::pair<int, int> > _imageDimensions;
      std::vector<V3D::Matrix3x3d>      _intrinsics;
      std::vector<V3D::Vector4d>        _distortions;
}; // end struct CalibrationDatabase

//----------------------------------------------------------------------

struct ViewPair
{
      ViewPair()
         : view0(-1), view1(-1)
      { }

      ViewPair(int v0, int v1)
         : view0(v0), view1(v1)
      {
         if (view0 > view1)
         {
            std::cerr << "ViewPair::ViewPair(): view0 should be smaller than view1! "
                      << "This will break the SfM methods." << std::endl;
            std::cerr << "view0 = " << view0 << ", view1 = " << view1 << std::endl;
         }
      }

      bool operator<(ViewPair const& b) const
      {
         if (view0 < b.view0) return true;
         if (view0 > b.view0) return false;
         return view1 < b.view1;
      }

      template <typename Archive> void serialize(Archive& ar)
      { 
         V3D::SerializationScope<Archive> scope(ar);
         ar & view0 & view1;
      }

      V3D_DEFINE_LOAD_SAVE(ViewPair);

      int view0, view1;
};

V3D_DEFINE_IOSTREAM_OPS(ViewPair);

struct PairwiseMatch
{
      ViewPair        views;
      V3D::Matrix3x3d rotation;
      V3D::Vector3d   translation;
      V3D::SerializableVector<V3D::PointCorrespondence> corrs;

      template <typename Archive> void serialize(Archive& ar)
      { 
         V3D::SerializationScope<Archive> scope(ar);
         ar & views;
         ar & rotation;
         ar & translation;
         ar & corrs;
      }

      V3D_DEFINE_LOAD_SAVE(PairwiseMatch);
}; // end struct PairwiseMatch

//----------------------------------------------------------------------

void computeEuclideanSqrDistanceTransform(V3D::Image<float>& distImg);

template <typename Pnt>
inline float
estimateCoverage(std::vector<Pnt> const& pts, int width, int height, int const targetWidth = 256)
{
   using namespace V3D;

   float const downScaleFactor = float(width) / targetWidth;
   int const W = targetWidth;
   int const H = int(height / downScaleFactor);

   Image<float> ptImg(W, H, 1, float(W+H));

   // Estimate the minimal radius required for n uniformly distributed
   // points to cover the whole image.
   float const radius2 = float(W) * H / (pts.size() * 4);

   for (size_t i = 0; i < pts.size(); ++i)
   {
      float const X = std::max(0.0f, std::min(float(W-1), pts[i][0]/downScaleFactor));
      float const Y = std::max(0.0f, std::min(float(H-1), pts[i][1]/downScaleFactor));
      ptImg(int(X), int(Y)) = 0.0f;
   }

   unsigned int sum = 0;
   computeEuclideanSqrDistanceTransform(ptImg);

   for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
         if (ptImg(x,y) < radius2) ++sum;

   float const pi_4 = 0.785398163397448;
   return float(sum) / float(W * H * pi_4);
} // end estimateCoverage()


//----------------------------------------------------------------------

struct TripletListItem
{
      V3D::ViewTripletKey views;
      int           nTriangulatedPoints;

      template <typename Archive> void serialize(Archive& ar)
      { 
         V3D::SerializationScope<Archive> scope(ar);
         ar & views;
         ar & nTriangulatedPoints;
      }

      V3D_DEFINE_LOAD_SAVE(TripletListItem);
}; // end struct TripletListItem

V3D_DEFINE_IOSTREAM_OPS(TripletListItem);


//**********************************************************************

struct TripleReconstruction
{
      int             views[3];
      V3D::Matrix3x3d intrinsics[3];
      V3D::Matrix3x4d orientations[3];

      std::vector<V3D::TriangulatedPoint> model;

      template <typename Archive>
      void serialize(Archive& ar)
      {
         V3D::SerializationScope<Archive> scope(ar, "TripleReconstruction");
         ar & views[0] & intrinsics[0] & orientations[0];
         ar & views[1] & intrinsics[1] & orientations[1];
         ar & views[2] & intrinsics[2] & orientations[2];

         serializeVector(model, ar);
      }

      V3D_DEFINE_LOAD_SAVE(TripleReconstruction);
}; // end struct TripleReconstruction

typedef V3D::SQLite3_Database::Table<PairwiseMatch> MatchDataTable;
typedef V3D::SQLite3_Database::Table<TripleReconstruction> TripletDataTable;

//**********************************************************************

// Random growing from the given starting view
void growModel(std::set<V3D::ViewTripletKey> const& modelTriples, int startView, int maxSize,
               std::set<int>& submodelViews);

// MST-based growing from the given starting view
void growModelMST(std::map<int, std::set<int> > const& mstAdjacencyMap, std::set<V3D::ViewTripletKey> const& modelTriples,
                  int startView, int maxSize, std::set<int>& submodelViews);

//**********************************************************************

struct SubmodelReconstruction
{
      SubmodelReconstruction()
         : _nViews(0)
      { }

      SubmodelReconstruction(std::set<int> const& viewIds,
                             std::set<V3D::ViewTripletKey> const& collectedTriples);

      void computeConsistentRotations(std::map<ViewPair, V3D::Matrix3x3d> const& relRotations);
      void computeConsistentTranslationLengths(std::map<ViewPair, V3D::Vector3d> const& relTranslations,
                                               V3D::CachedStorage<MatchDataTable>& matchDataCache,
                                               std::map<ViewPair, int> const& viewPairOIDMap,
                                               V3D::CachedStorage<TripletDataTable>& tripletDataCache,
                                               std::map<V3D::ViewTripletKey, int> const& tripletOIDMap,
                                               bool reestimateScaleRations = false);
      void computeConsistentTranslations();

      void computeConsistentTranslations_L1(V3D::CachedStorage<TripletDataTable>& tripletDataCache,
                                            std::map<V3D::ViewTripletKey, int> const& tripletOIDMap);

      void generateSparseReconstruction(std::vector<V3D::PointCorrespondence> const& allCorrs);

      template <typename Archive>
      void serialize(Archive& ar)
      {
         V3D::SerializationScope<Archive> scope(ar, "SubmodelReconstruction");

         ar & _nViews;
         V3D::serializeMap(_viewIdMap, ar);
         V3D::serializeVector(_viewIdBackMap, ar);

         V3D::serializeSet(_viewPairs, ar);
         V3D::serializeVector(_triplets, ar);

         V3D::serializeVector(_rotations, ar);
         ar & _baseLengths;
         V3D::serializeVector(_relTranslationVec, ar);
         V3D::serializeVector(_translations, ar);
            
         V3D::serializeVector(_viewPairVec, ar);
         V3D::serializeMap(_viewPairVecPosMap, ar);

         ar & _cameras;
         ar & _sparseReconstruction;
      }

      V3D_DEFINE_LOAD_SAVE(SubmodelReconstruction);


      int _nViews;
      std::map<int, int> _viewIdMap;
      std::vector<int>   _viewIdBackMap;

      std::set<ViewPair>         _viewPairs;
      std::vector<V3D::ViewTripletKey> _triplets;

      std::vector<V3D::Matrix3x3d> _rotations;
      V3D::Vector<double>          _baseLengths;
      std::vector<V3D::Vector3d>   _relTranslationVec;
      std::vector<V3D::Vector3d>   _translations;

      std::vector<ViewPair>   _viewPairVec;
      std::map<ViewPair, int> _viewPairVecPosMap;

      V3D::SerializableVector<V3D::CameraMatrix>      _cameras;
      V3D::SerializableVector<V3D::TriangulatedPoint> _sparseReconstruction;
}; // end struct SubmodelReconstruction

V3D_DEFINE_IOSTREAM_OPS(SubmodelReconstruction);

typedef V3D::SQLite3_Database::Table<SubmodelReconstruction> SubmodelsTable;

//**********************************************************************

struct ModelReconstruction
{
      template <typename Archive>
      void serialize(Archive& ar)
      {
         V3D::SerializationScope<Archive> scope(ar, "ComponentData");

         ar & id;
         ar & triplets;
         ar & corrs;
         ar & submodelIDs;
      }

      V3D_DEFINE_LOAD_SAVE(ModelReconstruction);

      int id;

      V3D::SerializableVector<V3D::ViewTripletKey>            triplets;
      V3D::SerializableVector<V3D::PointCorrespondence> corrs;
      V3D::SerializableVector<unsigned long>            submodelIDs;
}; // end struct ModelReconstruction

V3D_DEFINE_IOSTREAM_OPS(ModelReconstruction);

typedef V3D::SQLite3_Database::Table<ModelReconstruction> ModelsTable;

//**********************************************************************

inline double
medianQuantile(std::vector<double> const& vs)
{
#if 0
   return vs[vs.size()/2]; // Median of local ratios
#else
   // Take the mean of the central 20% quantile
   float avg = 0.0;
   int const radius = std::max(1, int(0.1f * vs.size()));
   for (int k = vs.size()/2-radius; k <= vs.size()/2+radius; ++k) avg += vs[k];
   avg /= (2*radius+1);
   return avg;
#endif
}

// Geomtric filtering of 3D points based on their reprojection error
inline void
filterInlierSparsePoints(CalibrationDatabase const& calibDb, std::vector<int> const& viewIdBackMap,
                         double maxError, int nRequiredViews,
                         std::vector<V3D::CameraMatrix> const& cameras,
                         std::vector<V3D::TriangulatedPoint>& model)
{
   using namespace std;
   using namespace V3D;

   maxError /= 1024.0; // Given max. allowed reprojection error is w.r.t. 1024xX pixel image

   for (int j = 0; j < model.size(); ++j)
   {
      TriangulatedPoint& X = model[j];

      vector<PointMeasurement> ms;

      for (int k = 0; k < X.measurements.size(); ++k)
      {
         PointMeasurement const& m = X.measurements[k];
         int const i = m.view;

         int const w = calibDb.getMaxDimension(viewIdBackMap[i]);
         double const f = calibDb.getAvgFocalLength(viewIdBackMap[i]);

         Vector3d const XX = cameras[i].transformPointIntoCameraSpace(X.pos);
         if (XX[2] <= 0.0) continue;

         Vector2d p = cameras[i].projectPoint(X.pos);
         double err = f * distance_L2(p, m.pos);
         if (err < w * maxError) ms.push_back(m);
      }
      X.measurements = ms;
   } // end for (j)

   vector<TriangulatedPoint> const origModel(model);
   model.clear();

   for (int j = 0; j < origModel.size(); ++j)
   {
      TriangulatedPoint const& X = origModel[j];
      if (X.measurements.size() >= nRequiredViews)
         model.push_back(X);
   }
} // end filterInlierSparsePoints()

inline void
showAccuracyInformation(CalibrationDatabase const& calibDb, std::vector<int> const& viewIdBackMap,
                        std::vector<V3D::CameraMatrix> const& cameras,
                        std::vector<V3D::TriangulatedPoint> const& model, double inlierThreshold = 10.0)
{
   using namespace std;
   using namespace V3D;

   double reproError = 0.0, overallError = 0.0;
   int nMeasurements = 0;
   int nOutliers = 0;

   for (int j = 0; j < model.size(); ++j)
   {
      TriangulatedPoint const& X = model[j];
      nMeasurements += X.measurements.size();

      //cout << "j = " << j << ": ";
      for (int k = 0; k < X.measurements.size(); ++k)
      {
         int const i = X.measurements[k].view;

         double const f = calibDb.getAvgFocalLength(viewIdBackMap[i]);

         Vector2d p = cameras[i].projectPoint(X.pos);
         double err = f * distance_L2(p, X.measurements[k].pos);

         overallError += std::min(inlierThreshold, err);
         //overallError += err;

         if (err > inlierThreshold)
            ++nOutliers;
         else
            reproError += err;
      }
      //cout << endl;
   } // end for (j)

   cout << "mean inlier reprojection error = " << reproError / (nMeasurements-nOutliers) << endl;
   cout << "total mean reprojection error = " << overallError / nMeasurements << endl;
   cout << nOutliers << "/" << nMeasurements << " were outlier measurements." << endl;
} // end showAccuracyInformation()

inline void
showAccuracyInformation_Linf(CalibrationDatabase const& calibDb, std::vector<int> const& viewIdBackMap,
                             std::vector<V3D::CameraMatrix> const& cameras,
                             std::vector<V3D::TriangulatedPoint> const& model, double inlierThreshold = 10.0)
{
   using namespace std;
   using namespace V3D;

   double reproErrorL2 = 0.0, reproErrorLinf = 0.0;;
   int nMeasurements = 0;
   int nOutliers = 0;

   for (int j = 0; j < model.size(); ++j)
   {
      TriangulatedPoint const& X = model[j];
      nMeasurements += X.measurements.size();

      for (int k = 0; k < X.measurements.size(); ++k)
      {
         int const i = X.measurements[k].view;
         Vector2d p = cameras[i].projectPoint(X.pos);

         double const f = calibDb.getAvgFocalLength(viewIdBackMap[i]);

         double const errL2 = f * distance_L2(p, X.measurements[k].pos);
         double const errLinf = f * distance_Linf(p, X.measurements[k].pos);

         if (errLinf > inlierThreshold)
            ++nOutliers;
         else
         {
            reproErrorL2   += errL2;
            reproErrorLinf += errLinf;
         }
      } // end for (k)
   } // end for (j)

   cout << "mean inlier L2 reprojection error = " << reproErrorL2 / (nMeasurements-nOutliers) << endl;
   cout << "mean inlier Linf reprojection error = " << reproErrorLinf / (nMeasurements-nOutliers) << endl;
   cout << nOutliers << "/" << nMeasurements << " were outlier measurements." << endl;
} // end showAccuracyInformation()

inline void
showAccuracyInformation(V3D::Matrix3x3d const& K, std::vector<V3D::CameraMatrix> const& cameras,
                        std::vector<V3D::TriangulatedPoint> const& model, double inlierThreshold = 10.0)
{
   using namespace std;
   using namespace V3D;

   double reproError = 0.0;
   int nMeasurements = 0;
   int nOutliers = 0;

   Vector2d p, q, q0;

   for (int j = 0; j < model.size(); ++j)
   {
      TriangulatedPoint const& X = model[j];
      nMeasurements += X.measurements.size();

      //cout << "j = " << j << ": ";
      for (int k = 0; k < X.measurements.size(); ++k)
      {
         int const i = X.measurements[k].view;
         p = cameras[i].projectPoint(X.pos);
         multiply_A_v_projective(K, p, q);
         multiply_A_v_projective(K, X.measurements[k].pos, q0);

         double err = distance_L2(q, q0);
         if (err > inlierThreshold)
            ++nOutliers;
         else
            reproError += err;
      }
      //cout << endl;
   } // end for (j)

   cout << "mean inlier reprojection error = " << reproError / (nMeasurements-nOutliers) << endl;
   cout << nOutliers << "/" << nMeasurements << " were outlier measurements." << endl;
} // end showAccuracyInformation()

namespace V3D
{

   struct RobustOrientationResult
   {
         Matrix3x3d       essential, fundamental, rotation;
         Vector3d         translation;
         std::vector<int> inliers;
         double           error;
   };

   template <typename Mat>
   inline double
   sampsonEpipolarError(PointCorrespondence const& corr, Mat const& F)
   {
      assert(F.num_rows() == 3);
      assert(F.num_cols() == 3);

      Matrix3x3d const Ft = F.transposed();

      Vector3d Fp, Ftq;

      Vector2f const& p = corr.left.pos;
      Vector2f const& q = corr.right.pos;

      multiply_A_v_affine(F, p, Fp);
      multiply_A_v_affine(Ft, q, Ftq);

      double const tmp = q[0]*Fp[0] + q[1]*Fp[1] + Fp[2];
      double const num = tmp*tmp;
      double const denom = Fp[0]*Fp[0] + Fp[1]*Fp[1] + Ftq[0]*Ftq[0] + Ftq[1]*Ftq[1];
      return num/denom;
   }

   void
   computeScaleRatios(float cosAngleThreshold,
                      Matrix3x3d const& R01, Vector3d const& T01,
                      Matrix3x3d const& R12, Vector3d const& T12,
                      Matrix3x3d const& R20, Vector3d const& T20,
                      std::vector<PointCorrespondence> const& corrs01,
                      std::vector<PointCorrespondence> const& corrs12,
                      std::vector<PointCorrespondence> const& corrs02,
                      double& s012, double& s120, double& s201, double& weight);

   void
   computeScaleRatiosGeneralized(Matrix3x3d const& R01, Vector3d const& T01,
                                 Matrix3x3d const& R12, Vector3d const& T12,
                                 Matrix3x3d const& R20, Vector3d const& T20,
                                 std::vector<PointCorrespondence> const& corrs01,
                                 std::vector<PointCorrespondence> const& corrs12,
                                 std::vector<PointCorrespondence> const& corrs02,
                                 double& s012, double& s120, double& s201, double& weight,
                                 float cosAngleThreshold = 0.9994f); // about 2 degree triangulation angle

   struct RobustOrientationMode
   {
         RobustOrientationMode()
            : iterativeRefinement(false),
              earlyTerminationConfidence(1.0 - 1e-10)
         { }

         bool iterativeRefinement;
         double earlyTerminationConfidence;
   };

   inline int ransacNSamples(int s, double epsilon, double p)
   {
      double const maxN = 1e6;

      double N = log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s));
      N = std::max(0.0, std::min(maxN, N));
      return (int)N;
   }

   template<int s> inline int
   ransacNSamples(double inlierRatio, double log_p);

   template <> inline int
   ransacNSamples<3>(double inlierRatio, double log_p)
   {
      double const maxN = 1e8;
      if (inlierRatio <= 0) return int(maxN);

      double x = inlierRatio*inlierRatio*inlierRatio;

      double N = log_p / log(1-x);
      N = std::max(0.0, std::min(maxN, N));
      return int(ceil(N));
   }

   template <> inline int
   ransacNSamples<5>(double inlierRatio, double log_p)
   {
      double const maxN = 1e8;
      if (inlierRatio <= 0) return int(maxN);

      double x = inlierRatio*inlierRatio*inlierRatio*inlierRatio*inlierRatio;

      double N = log_p / log(1-x);
      N = std::max(0.0, std::min(maxN, N));
      return int(ceil(N));
   }

   void
   computeRobustOrientationMLE(std::vector<PointCorrespondence> const& corrs,
                               Matrix3x3d const& K1, Matrix3x3d const& K2,
                               double inlierThreshold,
                               int nSamples,
                               RobustOrientationResult& res,
                               bool reportInliers, RobustOrientationMode mode = RobustOrientationMode());

   double computeRobustSimilarityTransformationMLE(std::vector<Vector3d> const& left, std::vector<Vector3d> const& right,
                                                   double inlierThresholdL, double inlierThresholdR, int nTrials,
                                                   Matrix3x3d& R, Vector3d& T, double& scale, std::vector<int> &inliers);

} // end namespace V3D

template <typename T> 
void writePointsToVRML(std::vector<T> const& points, char const * filename, bool append = false)
{
   using namespace std;
   std::ofstream os;

   os.open(filename, append ? ios::app : ios::out);
   os.precision(15);

   os << "#VRML V2.0 utf8" << endl;
   os << " Shape {" << endl;

   os << " appearance Appearance {" << endl;
   os << " material Material {" << endl;
   os << "     diffuseColor 1.0 1.0 1.0" << endl; 
   os << " } } " << endl;
   os << "  geometry PointSet {" << endl;

   os <<"  coord Coordinate {" << endl;
   os <<"  point [" << endl;
   for (size_t i = 0; i < points.size(); ++i)
      os << points[i][0] <<" " << points[i][1] << " " << points[i][2] << "," << endl;
   os << " ]\n }" << endl;
   os << " }" << endl;
   os << "}" << endl;
}

inline void
writeGoodPointsToVRML(CalibrationDatabase const& calibDb, std::vector<int> const& viewIdBackMap,
                      std::vector<V3D::CameraMatrix> const& cameras,
                      std::vector<V3D::TriangulatedPoint> const& model, char const * wrlName,
                      double inlierThreshold = 10.0, int minViews = 3)
{
   using namespace std;
   using namespace V3D;

   vector<Vector3d> goodXs, goodXs2;

   Vector2d p, q, q0;

   for (int j = 0; j < model.size(); ++j)
   {
      TriangulatedPoint const& X = model[j];

      bool isGood = true;

      for (int k = 0; k < X.measurements.size(); ++k)
      {
         int const i = X.measurements[k].view;

         Matrix3x3d const& K = calibDb.getIntrinsic(viewIdBackMap[i]);

         p = cameras[i].projectPoint(X.pos);
         multiply_A_v_projective(K, p, q);
         multiply_A_v_projective(K, X.measurements[k].pos, q0);

         double err = distance_L2(q, q0);
         if (err > inlierThreshold)
         {
            isGood = false;
            continue;
         }
      } // end for (k)
      if (isGood) goodXs.push_back(X.pos);
   } // end for (j))

   vector<float> norms(goodXs.size());

   for (size_t i = 0; i < goodXs.size(); ++i) norms[i] = norm_L2(goodXs[i]);
   std::sort(norms.begin(), norms.end());
   float distThr = norms[int(norms.size() * 0.9f)];
   //cout << "90% quantile distance: " << distThr << endl;

   for (size_t i = 0; i < goodXs.size(); ++i)
   {
      Vector3d const& X = goodXs[i];
      if (norm_L2(X) < 3*distThr) goodXs2.push_back(X);
   }            
   writePointsToVRML(goodXs2, wrlName);
} // end writeGoodPointsToVRML()

void extractConnectedComponent(std::map<ViewPair, std::set<int> > const& pairThirdViewMap,
                               std::set<V3D::ViewTripletKey>& unhandledTriples,
                               std::set<V3D::ViewTripletKey>& connectedTriples,
                               std::set<ViewPair>& handledEdges);

inline void
extractConnectedComponent(std::map<ViewPair, std::set<int> > const& pairThirdViewMap,
                          std::set<V3D::ViewTripletKey>& unhandledTriples,
                          std::set<V3D::ViewTripletKey>& connectedTriples)
{
   std::set<ViewPair> handledEdges;
   extractConnectedComponent(pairThirdViewMap, unhandledTriples, connectedTriples, handledEdges);
}
