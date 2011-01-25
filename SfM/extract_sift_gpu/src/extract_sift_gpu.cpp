#include <iostream>
#include <sstream>

#include "SiftGPU.h"

#include "Base/v3d_image.h"
#include "Base/v3d_feature.h"

using namespace std;
using namespace V3D;

int main(int argc, char * argv[])
{
   if (argc != 4 && argc != 5)
   {
      cerr << "Usage: " << argv[0] << " <image list file> <min. octave> <DoG threshold> [<silhouette images list file>]" << endl;
      return -1;
   }

   try
   {
      std::vector<std::string> entries;
      std::vector<std::string> silEntries;

      {
         ifstream is(argv[1]);
         string name;
         while (is >> name)
         {
            entries.push_back(name);
         }
      }

      if (argc == 5)
      {
         ifstream is(argv[4]);
         string name;
         while (is >> name)
         {
            silEntries.push_back(name);
         }
         if (entries.size() != silEntries.size())
         {
            cerr << "The number of images and the number of silhouettes do not match." << endl;
            return -2;
         }
      }

      SiftGPU sift;

      vector<float> descriptors;
      vector<SiftGPU::SiftKeypoint> keys;

      vector<char *> siftArgs;
      siftArgs.push_back("-fo"); siftArgs.push_back(argv[2]);
      siftArgs.push_back("-t"); siftArgs.push_back(argv[3]);
      siftArgs.push_back("-m"); siftArgs.push_back("-s");
      siftArgs.push_back("-v"); siftArgs.push_back("-1");
      siftArgs.push_back("-cg");
      //siftArgs.push_back("-lm"); siftArgs.push_back("8192");
      //siftArgs.push_back("-maxd"); siftArgs.push_back("3072");
      //-fo -1	use -1 octave 
      //-m,		up to 2 orientations for each feature
      //-s		enable subpixel subscale
      //-v 1		will invoke calling SiftGPU::SetVerbose(1),(only print out # feature and overal time)
      //-loweo	add a (.5, .5) offset


      sift.ParseParam(siftArgs.size(), &siftArgs[0]);
      if (sift.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
      {
         cerr << "Could not create GL context for SiftGPU." << endl;
         return 0;
      }

      char imgName[1024];

      for (size_t i = 0; i < entries.size(); ++i)
      {
         strncpy(imgName, entries[i].c_str(), 1024);

         if (sift.RunSIFT(imgName))
         {
            int const nFeatures = sift.GetFeatureNum();

            //allocate memory
            keys.resize(nFeatures);
            descriptors.resize(128*nFeatures);

            //read back a feature vector
            //faster than reading and writing files
            //if you dont need keys/descriptors, just put a NULL here
            sift.GetFeatureVector(&keys[0], &descriptors[0]);

            SerializableVector<SIFT_Feature> extractedFeatures;
            extractedFeatures.resize(nFeatures);

            for (int k = 0; k < nFeatures; ++k)
            {
               float const x           = keys[k].x;
               float const y           = keys[k].y;
               float const scale       = keys[k].s;
               float const orientation = keys[k].o;

               extractedFeatures[k].id        = k;
               extractedFeatures[k].position  = makeVector2<float>(x, y);
               extractedFeatures[k].scale     = scale;
               extractedFeatures[k].direction = makeVector2<float>(cos(orientation), sin(orientation));
               for (int l = 0; l < 128; ++l) extractedFeatures[k].descriptor[l] = descriptors[128*k + l];

               extractedFeatures[k].normalizeDescriptor();
            } // end for (k)

            if (!silEntries.empty())
            {
               Image<unsigned char> silImage;
               loadImageFile(silEntries[i].c_str(), silImage);

               SerializableVector<SIFT_Feature> filteredFeatures;
               for (int k = 0; k < extractedFeatures.size(); ++k)
               {
                  SIFT_Feature const& feature = extractedFeatures[k];
                  int const X = int(feature.position[0] + 0.5f);
                  int const Y = int(feature.position[1] + 0.5f);

                  if (silImage(X, Y) > 128)
                     filteredFeatures.push_back(feature);
               }
               extractedFeatures = filteredFeatures;
            } // end if

            ostringstream oss;
            oss << entries[i] << ".features";
            cout << "writing " << oss.str() << endl;

            serializeDataToFile(oss.str().c_str(), extractedFeatures, true);
         }
         else
         {
            cerr << "Cannot open file." << endl;
         }
      } // end for (i)
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
