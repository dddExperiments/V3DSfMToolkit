#include "v3d_database.h"
#include <fstream>

namespace V3D {

//----- Urbanscape Float Image -------

void statFLTImageFile( const char *filename, int &w, int &h )
{
    std::ifstream in(filename,std::ios_base::in|std::ios_base::binary);
    if(!in.is_open()) {
        w = h = 0;
        return;
    }
    in >> w >> h;
}

void readFLTImageFile( const char *filename, float *im, int w, int h )
{
    int ww,hh;
    std::ifstream in(filename,std::ios_base::in|std::ios_base::binary);
    if(!in.is_open())
        return;
    in >> ww >> hh;
    if(ww!=w && hh!=h)
        return;
    in.read((char*)im,1); // Read 1 byte to skip new-line.
    in.read((char*)im,w*h*sizeof(float));
}

void writeFLTImageFile( const char *filename, float *im, int w, int h )
{
    std::ofstream out(filename,std::ios_base::out|std::ios_base::binary);
    if(!out.is_open())
        return;
    out << w << " " << h << "\n";
    out.write((char*)im,w*h*sizeof(float));
}

//-------- Urbanscape Image Record ----------

bool ImageRecord::exists( const char *folder, int frame )
{
    char filename[1024];
    sprintf(filename,"%s/undistorted%08d.jpg",folder,frame);
    std::ifstream in(filename);
    return in.is_open();
}

void ImageRecord::create( const CreateParams &params )
{
    image.resize(params.w,params.h,3);
    gain = 1.0;
}

void ImageRecord::read( const char *folder, int frame )
{
    char filename[1024];
    sprintf(filename,"%s/undistorted%08d.jpg",folder,frame);
    loadImageFile(filename,image);
    // Look for gain in gains.txt
    sprintf(filename,"%s/gains.txt",folder);
    std::ifstream in(filename);
    if(!in.is_open()) {
        gain = 1.0;
        return;
    }
    while(in) {
        int f;
        double g;
        in >> f >> g;
        if(f == frame) {
            gain = g;
            return;
        }
    }
    gain = 1.0;
}

void ImageRecord::write( const char *folder, int frame )
{
}

// ------- Urbanscape Camera Record ---------

bool CameraRecord::exists( const char *folder, int frame )
{
    char filename[1024];
    sprintf(filename,"%s/camera%08d.txt",folder,frame);
    std::ifstream in(filename);
    return in.is_open();
}

void CameraRecord::create( const CreateParams &params )
{
}

void CameraRecord::read( const char *folder, int frame )
{
    char filename[1024];
    sprintf(filename,"%s/camera%08d.txt",folder,frame);
    std::ifstream in(filename);
    for(int r=0; r<3; r++)
        for(int c=0; c<4; c++)
            in >> P[r][c];
}

void CameraRecord::write( const char *folder, int frame )
{
    char filename[1024];
    sprintf(filename,"%s/camera%08d.txt",folder,frame);
    std::ofstream out(filename);
    out.precision(20);
    for(int r=0; r<3; r++) {
        for(int c=0; c<3; c++)
            out << P[r][c] << " ";
        out << P[r][3] << "\n";
    }
}
// ------- Urbanscape Camera Record ---------

bool Tracker2DRecord::exists( const char *folder, int frame )
{
    char filename[1024];
    sprintf(filename,"%s/tracks%08d.txt",folder,frame);
    std::ifstream in(filename);
    return in.is_open();
}

void Tracker2DRecord::create( const CreateParams &params )
{
}

void Tracker2DRecord::read( const char *folder, int frame )
{
    char filename[1024];
    sprintf(filename,"%s/camera%08d.txt",folder,frame);
    std::ifstream in(filename);

    int numFeatures;
    bool hasIds = false;
    in >> numFeatures;
    if(numFeatures==-1) {
        hasIds = true;
        in >> numFeatures;
    }

    tracks.resize(numFeatures);
    for(int i=0; i<numFeatures; i++) {
        Track &track = *(tracks.end()-1);
        in >> track.x >> track.y;
        int s;
        in >> s;
        switch(s) {
            case (int)Track::Invalid:
                track.status = Track::Invalid;
            case (int)Track::New:
                track.status = Track::New;
            case (int)Track::Continued:
                track.status = Track::Continued;
            default:
                verify(false,"Read invalid track status.");
        }
        if(hasIds)
            in >> track.id;
    }
}


void Tracker2DRecord::write( const char *folder, int frame )
{
    char filename[1024];
    sprintf(filename,"%s/tracks%08d.txt",folder,frame);
    std::ofstream out(filename);
    out.precision(17);

    out << -1 << "\n" << tracks.size() << "\n";
    for(int i=0; i < tracks.size(); i++) {
        out << tracks[i].x << " " << tracks[i].y << " "
            << tracks[i].status << " " << tracks[i].id << "\n";
    }
}

//-------- Urbanscape Depthmap Record -----------

bool DepthmapRecord::exists( const char *folder, int frame )
{
    char filename[1024];
    sprintf(filename,"%s/depth%08d.flt",folder,frame);
    std::ifstream in(filename);
    return in.is_open();
}

void DepthmapRecord::create( const DepthmapRecord::CreateParams &params )
{
    depthmap.resize(params.w,params.h);
    confidence.resize(params.w,params.h);
}

void DepthmapRecord::read( const char *folder, int frame )
{
    int w,h;
    char filename[1024];

    sprintf(filename,"%s/depth%08d.flt",folder,frame);
    statFLTImageFile(filename,w,h);
    depthmap.resize(w,h);
    readFLTImageFile(filename,depthmap.begin(),w,h);

    sprintf(filename,"%s/confidence%08d.flt",folder,frame);
    statFLTImageFile(filename,w,h);
    confidence.resize(w,h);
    readFLTImageFile(filename,confidence.begin(),w,h);
}

void DepthmapRecord::write( const char *folder, int frame )
{
    int w,h;
    char filename[1024];

    sprintf(filename,"%s/depth%08d.flt",folder,frame);
    writeFLTImageFile(filename,depthmap.begin(),depthmap.width(),depthmap.height());

    sprintf(filename,"%s/confidence%08d.flt",folder,frame);
    readFLTImageFile(filename,confidence.begin(),confidence.width(),confidence.height());
}

} // namespace V3D
