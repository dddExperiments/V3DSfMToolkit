#include "Base/v3d_image.h"
#include "GL/v3d_gpuundistort.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <GL/glew.h>
#include <GL/glut.h>

using namespace std;
using namespace V3D;
using namespace V3D_GPU;

int
main(int argc, char * argv[])
{
   unsigned int win;

   glutInitWindowPosition(0, 0);
   glutInitWindowSize(640, 480);
   glutInit(&argc, argv);

   if (argc != 4)
   {
      cerr << "Usage: " << argv[0] << " <Kmatrix file> <source image list> <path prefix (e.g. ``..'')>" << endl;
      return -1;
   }

   double fx, fy, s, px, py, k1, k2, p1, p2;

   {
      ifstream is(argv[1]);
      is >> fx >> s >> px >> fy >> py >> k1 >> k2 >> p1 >> p2;
   }

   std::vector<std::string> entries;
   {
      ifstream is(argv[2]);
      string name;
      while (is >> name) entries.push_back(name);
   }

   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

   if (!(win = glutCreateWindow("GPU KLT Test")))
   {
      cerr << "Error, couldn't open window" << endl;
      return -1;
   }

   glewInit();
   Cg_ProgramBase::initializeCg();

   Image<unsigned char> srcImage;
   for (size_t i = 0; i < entries.size(); ++i)
   {
      ostringstream oss;
      oss << argv[3] << "/" << entries[i];
      cout << "Resampling " << entries[i] << " to " << oss.str() << endl;

      loadImageFile(entries[i].c_str(), srcImage);

      int const w = srcImage.width();
      int const h = srcImage.height();

      Image<unsigned char> dstImage(w, h, 3);

      ParametricUndistortionFilter filter;
      filter.setDistortionParameters(fx, fy, k1, k2, p1, p2, px, py);
      filter.allocate(w, h);

      filter.undistortIntensityImage(&srcImage(0, 0, 0), &dstImage(0, 0, 0));
      filter.undistortIntensityImage(&srcImage(0, 0, 1), &dstImage(0, 0, 1));
      filter.undistortIntensityImage(&srcImage(0, 0, 2), &dstImage(0, 0, 2));

      filter.deallocate();

      saveImageFile(dstImage, oss.str().c_str());
   } // end for (i)

   return 0;
}
