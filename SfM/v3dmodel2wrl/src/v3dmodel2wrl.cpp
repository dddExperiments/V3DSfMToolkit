#include "Base/v3d_image.h"
#include "Base/v3d_imageprocessing.h"
#include "Geometry/v3d_mviewutilities.h"
#include "Geometry/v3d_cameramatrix.h"

#include "reconstruction_common.h"

#include <fstream>
#include <iostream>
#include <string>
#include <cstdio>

using namespace std;
using namespace V3D;

namespace
{
   template <typename T> 
   void writeColoredPointsToVRML(std::vector<T> const& points, std::vector<V3D::Vector3f> const& colors,
                                 char const * filename, bool append = false)
   {
      using namespace std;
      std::ofstream os;

      os.open(filename, append ? ios::app : ios::out);
      os.precision(15);

      if (!append) os <<"#VRML V2.0 utf8" << endl;
      os << "Shape {" << endl;

      os << " appearance Appearance {" << endl;
      os << "   material Material {" << endl;
      os << "        ambientIntensity 0.1" << endl;
      os << "        diffuseColor 1.0 0.7 0.2" << endl;
      os << " } } " << endl;
      os << "  geometry PointSet {" << endl;

      // write point coordinates
      os << "  coord Coordinate {" << endl;
      os << "  point [" << endl;
      for (size_t i = 0; i < points.size(); ++i)
         os << points[i][0] <<" " << points[i][1] << " " << points[i][2] << "," << endl;
      os << " ]\n } # end points"<<endl;

      os << " color Color { color [" << endl;
      for (size_t i = 0; i < points.size(); ++i)
      {
         double r = colors[i][0] / 255.0;
         double g = colors[i][1] / 255.0;
         double b = colors[i][2] / 255.0;
         os << min(max(r, 0.0), 1.0) << " " << min(max(g, 0.0), 1.0) << " " << min(max(b, 0.0), 1.0) << " ," << endl;
      }
      os << " ] } # end color" << endl;
      os << " }" << endl;
      os << "}" << endl;
   } // end writeColoredPointsToVRML()

   void writeCameraFrustumToVRML(CameraMatrix const& cam, int width, int height, double scale,
                                 Vector3f const& color, const char *filename, bool append)
   {
      using namespace std;

      ofstream os;

      os.open(filename, append ? ios::app : ios::out);
      os.precision(15);

      if (!append) os <<"#VRML V2.0 utf8" << endl;
      os << " Shape {" << endl;
      os << " appearance Appearance {" << endl;
      os << "  material Material { emissiveColor 0.2 0.2 0.2" << endl;
      os << "  diffuseColor " << color[0]/255 << " " << color[1]/255 << " " << color[2]/255 << endl;
      os << "  }" << endl;
      os << " }" << endl;

      os <<" geometry IndexedFaceSet {" << endl;
      os <<" solid FALSE " << endl;
      os <<" ccw TRUE " << endl;
      os <<" coord Coordinate {" << endl;
      os <<" point [" << endl;

      Vector3d const center = cam.cameraCenter();
      Vector3d const ll = cam.unprojectPixel(makeVector2(0.0, double(height)), scale);
      Vector3d const lr = cam.unprojectPixel(makeVector2(double(width), double(height)), scale);
      Vector3d const ul = cam.unprojectPixel(makeVector2(0.0, 0.0), scale);
      Vector3d const ur = cam.unprojectPixel(makeVector2(double(width), 0.0), scale);

      // write point coordinates
      os << ll[0] << " " << ll[1] << " " << ll[2] << "," << endl;
      os << lr[0] << " " << lr[1] << " " << lr[2] << "," << endl;
      os << ul[0] << " " << ul[1] << " " << ul[2] << "," << endl;
      os << ur[0] << " " << ur[1] << " " << ur[2] << "," << endl;
      os << center[0] << " " << center[1] << " " << center[2] << "," << endl;

      os << " ]\n } "<< endl;

      os << "  coordIndex [\n";
      os << "  0,1,2,3,-1" << endl;
      os << "  0,1,4,-1" << endl;
      os << "  1,3,4,-1" << endl;
      os << "  3,2,4,-1" << endl;
      os << "  2,0,4,-1\n  ]" << endl;
      os << " }\n}" << endl;
   } // end writeCameraFrustumToVRML()

} // end namespace <>

int
main(int argc, char** argv)
{
   try
   {
      bool const colorize = (argc == 7);
      bool const colorizeFrequencies = false;

      if (argc != 6 && argc != 7)
      {
         cerr << "Usage: " << argv[0] << " <poses file> <points file> <K file> <orig. width> <orig. height> [<image names file>]" << endl;
         return -1;
      }

      int const origWidth  = atoi(argv[4]);
      int const origHeight = atoi(argv[5]);

      vector<string> imageNames;
      if (argc == 7)
      {
         ifstream is(argv[6]);
         string name;
         while (is >> name)
         {
            imageNames.push_back(name);
         }
      } // end if

      vector<int>        viewIds;
      vector<Matrix3x4d> cameraPoses;
      {
         ifstream is(argv[1]);
         int nViews;
         is >> nViews;
         cout << "Going to read " << nViews << " poses." << endl;

         char buf[1024];
         Matrix3x4d P;

         for (int i = 0; i < nViews; ++i)
         {
            int viewId;
            is >> viewId;
            viewIds.push_back(viewId);

            is >> P[0][0] >> P[0][1] >> P[0][2] >> P[0][3];
            is >> P[1][0] >> P[1][1] >> P[1][2] >> P[1][3];
            is >> P[2][0] >> P[2][1] >> P[2][2] >> P[2][3];
            cameraPoses.push_back(P);
         }
      } // end scope

      int const nCams = viewIds.size();

      vector<TriangulatedPoint> pointModel;
      vector<Vector3d>          points3d;
      {
         ifstream is(argv[2]);
         int nPoints;
         is >> nPoints;

         for (int j = 0; j < nPoints; ++j)
         {
            TriangulatedPoint X;

            is >> X.pos[0] >> X.pos[1] >> X.pos[2];
            int nMeasurements = 0;
            is >> nMeasurements;
            for (int k = 0; k < nMeasurements; ++k)
            {
               PointMeasurement m;
               is >> m.view >> m.id >> m.pos[0] >> m.pos[1];
               X.measurements.push_back(m);
            }
            pointModel.push_back(X);
            points3d.push_back(X.pos);
         }
      } // end scope
      int const nPoints = pointModel.size();

      Matrix3x3d K; makeIdentityMatrix(K);

      {
         ifstream is(argv[3]);
         is >> K[0][0] >> K[0][1] >> K[0][2] >> K[1][1] >> K[1][2];
      }

      vector<Vector3f> colors(nPoints, makeVector3(0.0f, 0.0f, 0.0f));
      vector<float>    nMeasurements(nPoints, 0.0f);

      char const * wrlName = "result.wrl";

      if (colorize)
      {
         map<int, int> viewIdPosMap;
         for (int i = 0; i < nCams; ++i) viewIdPosMap.insert(make_pair(viewIds[i], i));

         for (int i = 0; i < nCams; ++i)
         {
            string const& imageName = imageNames[viewIds[i]];

            Image<unsigned char> srcIm;
            loadImageFile(imageName.c_str(), srcIm);
            int const w = srcIm.width();
            int const h = srcIm.height();
            int const nChan = srcIm.numChannels();

            double const s_x = double(w) / origWidth;
            double const s_y = double(h) / origHeight;
            cout << "s_x = " << s_x << ", s_y = " << s_y << endl;

            Matrix3x3d Kim, S;
            makeZeroMatrix(S);
            S[0][0] = s_x; S[1][1] = s_y; S[2][2] = 1.0;
            Kim = S * K;
            CameraMatrix cam;
            cam.setIntrinsic(Kim);
            cam.setOrientation(cameraPoses[i]);

            for (int j = 0; j < nPoints; ++j)
            {
               TriangulatedPoint const& X = pointModel[j];
               for (int k = 0; k < X.measurements.size(); ++k)
               {
                  PointMeasurement m = X.measurements[k];
                  //map<int, int>::const_iterator p;
                  //p = viewIdPosMap.find(m.view);
                  //if (p == viewIdPosMap.end() || p->second != i) continue;
                  if (m.view != i) continue;
                  m.pos[0] *= s_x;
                  m.pos[1] *= s_y;
                  if (m.pos[0] >= 0.0f && m.pos[0] < float(w) &&
                      m.pos[1] >= 0.0f && m.pos[1] < float(h))
                  {
                     if (nChan == 1)
                     {
                        float c = bilinearSample(srcIm, m.pos[0], m.pos[1], 0);
                        colors[j][0] += c;
                        colors[j][1] += c;
                        colors[j][2] += c;
                        nMeasurements[j] += 1.0f;
                     }
                     else if (nChan == 3)
                     {
                        colors[j][0] += bilinearSample(srcIm, m.pos[0], m.pos[1], 0);
                        colors[j][1] += bilinearSample(srcIm, m.pos[0], m.pos[1], 1);
                        colors[j][2] += bilinearSample(srcIm, m.pos[0], m.pos[1], 2);
                        nMeasurements[j] += 1.0f;
                     }
                  }
                  else
                  {
                     colors[j][0] += 255.0f;
                     nMeasurements[j] += 1.0f;
                  }
               } // end for (k)
            } // end for (j)
         } // end for (i)

         for (int j = 0; j < nPoints; ++j)
            scaleVectorIP(1.0f / nMeasurements[j], colors[j]);
      }
      else if (colorizeFrequencies)
      {
         float maxFrequency = 0.0f;
         double avgFrequency = 0.0;
         for (int j = 0; j < nPoints; ++j)
         {
            TriangulatedPoint const& X = pointModel[j];
            maxFrequency = std::max(maxFrequency, float(X.measurements.size()));
            avgFrequency += double(X.measurements.size());
         }
         cout << "maxFrequency = " << maxFrequency << ", avgFrequency = " << avgFrequency/nPoints << endl;

         for (int j = 0; j < nPoints; ++j)
         {
            TriangulatedPoint const& X = pointModel[j];
            float const frequency = float(X.measurements.size());
            float const alpha = frequency / maxFrequency;
#if 0
            colors[j][0] = (1.0f-alpha)*255.0f;
            colors[j][1] = alpha*255.0f;
            colors[j][2] = 0.0f;
#else
            colors[j][0] = alpha*255.0f;
            colors[j][1] = alpha*255.0f;
            colors[j][2] = alpha*255.0f;
#endif
         }
      } // end if (colorizeFrequencies)

      if (colorize || colorizeFrequencies)
         writeColoredPointsToVRML(points3d, colors, wrlName, false);
      else
         writePointsToVRML(points3d, wrlName, false);

      Vector3d bboxMin(points3d[0]);
      Vector3d bboxMax(points3d[0]);

      for (int j = 1; j < nPoints; ++j)
      {
         bboxMin[0] = std::min(points3d[j][0], bboxMin[0]);
         bboxMin[1] = std::min(points3d[j][1], bboxMin[1]);
         bboxMin[2] = std::min(points3d[j][2], bboxMin[2]);
         bboxMax[0] = std::max(points3d[j][0], bboxMax[0]);
         bboxMax[1] = std::max(points3d[j][1], bboxMax[1]);
         bboxMax[2] = std::max(points3d[j][2], bboxMax[2]);
      }
      double const diameter = distance_L2(bboxMin, bboxMax);
      cout << "diameter = " << diameter << endl;

      //double scale = 0.1 * diameter;
      double scale = 0.001 * diameter;
      Vector3f camColor = makeVector3(100.0f, 50.0f, 50.0f);
      for (int i = 0; i < nCams; ++i)
      {
         CameraMatrix cam;
         cam.setIntrinsic(K);
         cam.setOrientation(cameraPoses[i]);
         writeCameraFrustumToVRML(cam, origWidth, origHeight, scale, camColor, wrlName, true);
      }
   }
   catch (std::string s)
   {
      cerr << "Exception caught: " << s << endl;
   }
   catch (std::exception exn)
   {
      cerr << "Exception caught: " << exn.what() << endl;
   }
   catch (...)
   {
      cerr << "Unknown exception caught." << endl;
   }

   return 0;
}
