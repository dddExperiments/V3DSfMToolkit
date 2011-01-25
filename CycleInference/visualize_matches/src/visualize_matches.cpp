#include "reconstruction_common.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Base/v3d_cfgfile.h"
#include "GL/v3d_gpubase.h"

#include <iostream>

using namespace V3D;
using namespace V3D_GPU;
using namespace std;

namespace
{

   SQLite3_Database * matchesDB;

   MatchDataTable * matchDataTable = 0;
   MatchDataTable::const_iterator * matchDataView;

   std::vector<std::string> imageNames;

   int scrwidth, scrheight;

   ImageTexture2D leftTex, rightTex;
   Matrix3x3d Kcombined;

   void
   reshape(int width, int height)
   {
      cout << "reshape" << endl;

      scrwidth = width;
      scrheight = height;
      glViewport(0, 0, (GLint) width, (GLint) height);
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0, 2010, 0, 1000);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
   }

   bool drawMatches = true;

   void
   drawscene()
   {
      static bool initialized = false;

      if (!initialized)
      {
         glewInit();
         //Cg_ProgramBase::initializeCg();

         leftTex.allocateID();
         rightTex.allocateID();
         initialized = true;
      }

      PairwiseMatch const& matchData = (*(*matchDataView)).second;
      int const v0 = matchData.views.view0;
      int const v1 = matchData.views.view1;

      cout << "read matches between " << v0 << " (" << imageNames[v0] << ") and "
           << v1 << " (" << imageNames[v1] << ") with " << matchData.corrs.size() << " matches." << endl;

      Image<unsigned char> im;
      loadImageFile(imageNames[v0].c_str(), im);
      int const wL = im.width(), hL = im.height();
      leftTex.reserve(im.width(), im.height(), TextureSpecification("rgb=8"));
      if (im.numChannels() == 3)
         leftTex.overwriteWith(im.begin(0), im.begin(1), im.begin(2));
      else
         leftTex.overwriteWith(im.begin(0), im.begin(0), im.begin(0));

      loadImageFile(imageNames[v1].c_str(), im);
      int const wR = im.width(), hR = im.height();
      rightTex.reserve(im.width(), im.height(), TextureSpecification("rgb=8"));
      if (im.numChannels() == 3)
         rightTex.overwriteWith(im.begin(0), im.begin(1), im.begin(2));
      else
         rightTex.overwriteWith(im.begin(0), im.begin(0), im.begin(0));

      bool const rotateRight = 1;

      leftTex.enable(GL_TEXTURE0);
      glBegin(GL_QUADS);
      if (!rotateRight)
      {
         glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 0.0);
         glTexCoord2f(1.0, 1.0); glVertex2f(1000.0, 0.0);
         glTexCoord2f(1.0, 0.0); glVertex2f(1000.0, 1000.0);
         glTexCoord2f(0.0, 0.0); glVertex2f(0.0, 1000.0);
      }
      else
      {
         glTexCoord2f(1.0, 1.0); glVertex2f(0.0, 0.0);
         glTexCoord2f(1.0, 0.0); glVertex2f(1000.0, 0.0);
         glTexCoord2f(0.0, 0.0); glVertex2f(1000.0, 1000.0);
         glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 1000.0);
      }
      glEnd();
      leftTex.disable(GL_TEXTURE0);

      rightTex.enable(GL_TEXTURE0);
      glBegin(GL_QUADS);
      if (!rotateRight)
      {
         glTexCoord2f(0.0, 1.0); glVertex2f(1010.0, 0.0);
         glTexCoord2f(1.0, 1.0); glVertex2f(2010.0, 0.0);
         glTexCoord2f(1.0, 0.0); glVertex2f(2010.0, 1000.0);
         glTexCoord2f(0.0, 0.0); glVertex2f(1010.0, 1000.0);
      }
      else
      {
         glTexCoord2f(1.0, 1.0); glVertex2f(1010.0, 0.0);
         glTexCoord2f(1.0, 0.0); glVertex2f(2010.0, 0.0);
         glTexCoord2f(0.0, 0.0); glVertex2f(2010.0, 1000.0);
         glTexCoord2f(0.0, 1.0); glVertex2f(1010.0, 1000.0);
      }
      glEnd();
      rightTex.disable(GL_TEXTURE0);

      Vector3f p, q;

      if (drawMatches)
      {
         srand(0);
         vector<int> indices(matchData.corrs.size());
         for (int i = 0; i < matchData.corrs.size(); ++i) indices[i] = i;
         random_shuffle(indices.begin(), indices.end());

         glLineWidth(2);
         glBegin(GL_LINES);
         //for (int i = 0; i < matchData.corrs.size(); ++i)
         for (int i = 0; i < std::min(size_t(250), matchData.corrs.size()); ++i)
         {
            //if ((i & 7) != 0) continue;

            PointCorrespondence const& corr = matchData.corrs[indices[i]];

            multiply_A_v_affine(Kcombined, corr.left.pos, p);
            multiply_A_v_affine(Kcombined, corr.right.pos, q);

            if (!rotateRight)
            {
               glVertex2f(p[0], 1000 - p[1]);
               glVertex2f(q[0] + 1010, 1000 - q[1]);
            }
            else
            {
               glVertex2f(1000-p[1], 1000 - p[0]);
               glVertex2f(1000-q[1] + 1010, 1000 - q[0]);
            }
         } // end for (i)
         glEnd();
      } // end if

      glutSwapBuffers();
   } // end drawscene()

   void
   keyFunc(unsigned char key, int x, int y)
   {
      if (key == 27) exit(0);
      switch (key)
      {
         case ' ':
            ++*matchDataView;
            break;
         case 'm':
            drawMatches = !drawMatches;
            break;
      }
      glutPostRedisplay();
   }

} // end namespace <>

int
main( int argc, char** argv) 
{
   unsigned int win;

   int const W = 1536; int const H = 768;

   glutInitWindowPosition(0, 0);
   glutInitWindowSize(W, H);
   glutInit(&argc, argv);

   if (argc != 4 && argc != 6)
   {
      cerr << "Usage: " << argv[0] << " <image list file> <match db file> <config file> [<left view> <right view>]" << endl;
      return -1;
   }

   {
      ifstream is(argv[1]);
      string name;
      while (is >> name) imageNames.push_back(name);
   }

   {
      ConfigurationFile cf(argv[3]);
      double const focalLength = cf.get("FOCAL_LENGTH", 1.0);
      double const ppx = cf.get("PPX", 0.0);
      double const ppy = cf.get("PPY", 0.0);
      int const imageWidth  = cf.get("IMAGE_WIDTH", 2*int(ppx));
      int const imageHeight = cf.get("IMAGE_HEIGHT", 2*int(ppy));

      Matrix3x3d Kintrinsic, Kviewport;

      makeIdentityMatrix(Kintrinsic);
      makeIdentityMatrix(Kviewport);
      Kintrinsic[0][0] = focalLength;
      Kintrinsic[1][1] = focalLength;
      Kintrinsic[0][2] = ppx;
      Kintrinsic[1][2] = ppy;

      Kviewport[0][0] = 1000.0 / imageWidth;
      Kviewport[1][1] = 1000.0 / imageHeight;
      Kviewport[0][2] = 0.0;
      Kviewport[1][2] = 0.0;

      multiply_A_B(Kviewport, Kintrinsic, Kcombined);
      displayMatrix(Kcombined);
   }

   matchesDB = new SQLite3_Database(argv[2]);

   map<ViewPair, int> viewPairOIDMap;

   matchDataTable = new MatchDataTable(matchesDB->getTable<PairwiseMatch>("matches_data"));
   matchDataView  = new MatchDataTable::const_iterator(matchDataTable->begin());

   {
      typedef SQLite3_Database::Table<ViewPair> Table;
      Table table = matchesDB->getTable<ViewPair>("matches_list");
      for (Table::const_iterator p = table.begin(); bool(p); ++p)
      {
         int const oid = (*p).first;
         ViewPair pair = (*p).second;
         int const view1 = pair.view0;
         int const view2 = pair.view1;
         
         viewPairOIDMap.insert(make_pair(pair, oid));
      }
   } // end scope
   cout << "Considering = " << viewPairOIDMap.size() << " view pairs." << endl;

   if (argc == 6)
   {
      int const v0 = atoi(argv[4]);
      int const v1 = atoi(argv[5]);
      if (viewPairOIDMap.find(ViewPair(v0, v1)) != viewPairOIDMap.end())
      {
         while (*matchDataView)
         {
            PairwiseMatch const& matchData = (*(*matchDataView)).second;
            int const vv0 = matchData.views.view0;
            int const vv1 = matchData.views.view1;
            if (v0 == vv0 && v1 == vv1) break;
            ++*matchDataView;
         } // end while
      }
      else
      {
         cout << "View pair not found in the database." << endl;
      }
   } // end if

   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

   if (!(win = glutCreateWindow("Matching Visualization")))
   {
      cerr << "Error, couldn't open window" << endl;
      return -1;
   }

   glutReshapeFunc(reshape);
   glutDisplayFunc(drawscene);
   //glutIdleFunc(drawscene);
   glutKeyboardFunc(keyFunc);
   glutMainLoop();

   return 0;
}
