#include <iostream>
#include <sstream>

#include "Base/v3d_image.h"
#include "Base/v3d_exifreader.h"
#include "Base/v3d_cfgfile.h"

using namespace std;
using namespace V3D;

int main(int argc, char * argv[])
{
   if (argc != 3)
   {
      cerr << "Usage: " << argv[0] << " <image list file> <config file>" << endl;
      return -1;
   }

   try
   {
      ConfigurationFile cf(argv[2]);

      double const focalLength = cf.get("FOCAL_LENGTH", -1.0);
      double const focalLengthY = cf.get("FOCAL_LENGTH_Y", focalLength);
      double const ppx = cf.get("PPX", -1.0);
      double const ppy = cf.get("PPY", -1.0);
      int const imageWidth  = cf.get("IMAGE_WIDTH", 2*int(ppx));
      int const imageHeight = cf.get("IMAGE_HEIGHT", 2*int(ppy));

      bool const enableEXIF = cf.get("ENABLE_EXIF_READING", true);

      std::vector<std::string> entries;

      {
         ifstream is(argv[1]);
         string name;
         while (is >> name)
         {
            entries.push_back(name);
         }
      }
      cout << "Checking calibration data for " << entries.size() << " images." << endl;

      char name[1024];

      int nFoundCalibFiles = 0, nFoundEXIFs = 0;

      ofstream os("calibration_db.txt");
      os.precision(10);
      os << entries.size() << endl;

      for (size_t i = 0; i < entries.size(); ++i)
      {
         sprintf(name, "%s.calib", entries[i].c_str());

         ifstream is(name);
         if (is)
         {
            ++nFoundCalibFiles;

            int w, h;
            double fx, skew, cx, fy, cy, k1, k2, p1, p2;

            is >> fx >> skew >> cx >> fy >> cy;
            is >> k1 >> k2 >> p1 >> p2;
            is >> w >> h;

            os << fx << " " << skew << " " << cx << " " << fy << " " << cy << " ";
            os << k1 << " " << k2 << " " << p1 << " " << p2 << " ";
            os << w << " " << h << endl;
         }
         else
         {
            bool readEXIF = false;

            if (enableEXIF)
            {
               int w, h;
               double fx, fy;
               readEXIF = getCalibrationFromEXIF(entries[i].c_str(), w, h, fx, fy, imageWidth, imageHeight);
               if (readEXIF)
               {
                  os << fx << " 0 " << w/2 << " " << fy << " " << h/2 << " ";
                  os << "0 0 0 0 " << w << " " << h << endl;
                  ++nFoundEXIFs;
               }
            }

            if (!readEXIF)
            {
               os << focalLength << " 0 " << ppx << " " << focalLengthY << " " << ppy << " ";
               os << "0 0 0 0 " << imageWidth << " " << imageHeight << endl;
            }
         }
      } // end for (i)

      cout << "Found " << nFoundCalibFiles << " image-specific calibration files." << endl;
      if (enableEXIF)
         cout << "Found " << nFoundEXIFs << " image-specific calibration data from EXIF." << endl;
   }
   catch (std::exception exn)
   {
      cerr << "Exception caught: " << exn.what() << endl;
   }
   catch (std::string s)
   {
      cerr << "Exception caught: " << s << endl;
   }
   catch (...)
   {
      cerr << "Unhandled exception." << endl;
   }

   return 0;
}
