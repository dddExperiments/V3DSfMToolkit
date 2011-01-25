// -*- C++ -*-
#ifndef V3D_EXIF_READER_H
#define V3D_EXIF_READER_H

namespace V3D
{

   bool getCalibrationFromEXIF(char const * fileName, int& width, int& height, double& focalLengthX, double& focalLengthY,
                               double const defaultWidth = -1, double const defaultHeight = -1);

} // end namespace V3D

#endif // defined(V3D_EXIF_READER_H)
