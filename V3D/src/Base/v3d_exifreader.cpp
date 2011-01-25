#include "Base/v3d_exifreader.h"

#if defined(V3DLIB_ENABLE_LIBEXIF)

#include <libexif/exif-data.h>

#include <iostream>

using namespace std;

namespace
{

   inline double
   getEXIF_DoubleVal(ExifEntry * e, ExifByteOrder byteOrder)
   {
      double res = -1.0;

      switch (e->format)
      {
         case EXIF_FORMAT_SHORT:
            res = exif_get_short(e->data, byteOrder);
            break;
         case EXIF_FORMAT_LONG:
            res = exif_get_long(e->data, byteOrder);
            break;
         case EXIF_FORMAT_RATIONAL:
         {
            ExifRational rat = exif_get_rational(e->data, byteOrder);
            res = double(rat.numerator) / rat.denominator;
            break;
         }
         case EXIF_FORMAT_SRATIONAL:
         {
            ExifSRational rat = exif_get_srational(e->data, byteOrder);
            res = double(rat.numerator) / rat.denominator;
            break;
         }
         default:
            cerr << "Unknown format " << int(e->format) << endl;
      }
      return res;
   }

} // end namespace <>

namespace V3D
{

   bool
   getCalibrationFromEXIF(char const * fileName, int& width, int& height, double& focalLengthX, double& focalLengthY,
                          double const defaultWidth, double const defaultHeight)
   {
      width = defaultWidth;
      height = defaultHeight;

      focalLengthX = focalLengthY = -1;
      double fNative = -1, f35mm = -1;
      double fResX = -1, fResY = -1, fResUnits = -1;

      ExifData * ed = exif_data_new_from_file(fileName);
      if (!ed) return false;

      ExifByteOrder const byteOrder = exif_data_get_byte_order(ed);

      ExifEntry * entry;

      entry = exif_data_get_entry(ed, EXIF_TAG_PIXEL_X_DIMENSION);
      if (entry) width = getEXIF_DoubleVal(entry, byteOrder);

      entry = exif_data_get_entry(ed, EXIF_TAG_PIXEL_Y_DIMENSION);
      if (entry) height = getEXIF_DoubleVal(entry, byteOrder);

      entry = exif_data_get_entry(ed, EXIF_TAG_FOCAL_LENGTH);
      if (entry) fNative = getEXIF_DoubleVal(entry, byteOrder);

      entry = exif_data_get_entry(ed, EXIF_TAG_FOCAL_PLANE_X_RESOLUTION);
      if (entry) fResX = getEXIF_DoubleVal(entry, byteOrder);

      entry = exif_data_get_entry(ed, EXIF_TAG_FOCAL_PLANE_Y_RESOLUTION);
      if (entry) fResY = getEXIF_DoubleVal(entry, byteOrder);

      entry = exif_data_get_entry(ed, EXIF_TAG_FOCAL_PLANE_RESOLUTION_UNIT);
      if (entry) fResUnits = getEXIF_DoubleVal(entry, byteOrder);

      if (fResUnits == 1 || fResUnits == 2)
         fResUnits = 25.4;
      else if (fResUnits == 3)
         fResUnits = 10.0;
      else if (fResUnits == 4)
         fResUnits = 1.0;
      else if (fResUnits == 5)
         fResUnits = 0.001;
      else
         fResUnits = 25.4;

      if (fNative > 0 && fResX > 0 && fResY > 0 && fResUnits > 0)
      {
         focalLengthX = fNative * fResX / fResUnits;
         focalLengthY = fNative * fResY / fResUnits;
      }
      else
      {
         entry = exif_data_get_entry(ed, EXIF_TAG_FOCAL_LENGTH_IN_35MM_FILM);
         if (entry) f35mm = getEXIF_DoubleVal(entry, byteOrder);

         if (f35mm > 0) focalLengthX = focalLengthY = f35mm*width / 35.0;
      } // end if

      return focalLengthX > 0;
   } // getCalibrationFromEXIF()

} // end namespace V3D

#else

namespace V3D
{

   bool
   getCalibrationFromEXIF(char const * fileName, int& width, int& height, double& focalLengthX, double& focalLengthY,
                          double const defaultWidth, double const defaultHeight)
   {
      width = defaultWidth;
      height = defaultHeight;
      focalLengthX = focalLengthY = -1;
      return false;
   }

} // end namespace V3D

#endif
