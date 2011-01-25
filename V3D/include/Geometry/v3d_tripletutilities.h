// -*- C++ -*-

#ifndef V3D_TRIPLET_UTILITIES_H
#define V3D_TRIPLET_UTILITIES_H

#include "Base/v3d_serialization.h"
#include "Geometry/v3d_mviewutilities.h"
#include "Geometry/v3d_cameramatrix.h"

#include <iostream>

namespace V3D
{

   struct ViewTripletKey
   {
         ViewTripletKey()
         {
            views[0] = views[1] = views[2] = -1;
         }

         ViewTripletKey(int v0, int v1, int v2)
         {
            if (v0 > v1 || v1 > v2 || v0 > v2)
               std::cerr << "ViewTripletKey::ViewTripletKey(): v0 < v1 < v2 failed! "
                         << "This will break the SfM methods." << std::endl;

            views[0] = v0;
            views[1] = v1;
            views[2] = v2;
         }

         bool operator<(ViewTripletKey const& b) const
         {
            if (views[0] < b.views[0]) return true;
            if (views[0] > b.views[0]) return false;
            if (views[1] < b.views[1]) return true;
            if (views[1] > b.views[1]) return false;
            return views[2] < b.views[2];
         }

         template <typename Archive> void serialize(Archive& ar)
         { 
            V3D::SerializationScope<Archive> scope(ar);
            ar & views[0] & views[1] & views[2];
         }

         V3D_DEFINE_LOAD_SAVE(ViewTripletKey);

         int views[3];
   };

   V3D_DEFINE_IOSTREAM_OPS(ViewTripletKey);

//======================================================================

   // s_ijk is the baseline length ratio s_ijk = |C_k - C_j| / |C_j - C_i|
   void computeTripletLengthRatios(float cosAngleThreshold,
                                   Matrix3x3d const& R01, Vector3d const& T01,
                                   Matrix3x3d const& R12, Vector3d const& T12,
                                   Matrix3x3d const& R20, Vector3d const& T20,
                                   std::vector<PointCorrespondence> const& corrs01,
                                   std::vector<PointCorrespondence> const& corrs12,
                                   std::vector<PointCorrespondence> const& corrs02,
                                   double& s012, int& n012, double& s120, int& n120, double& s201, int& n201);

} // end namespace V3D

#endif
