// -*- C++ -*-

#ifndef V3D_DATABASE_H
#define V3D_DATABASE_H

#include <map>
#include <vector>
#include <fstream>
#include "Base/v3d_image.h"
#include "Math/v3d_linear.h"

namespace V3D {

    const unsigned int DATABASE_READ   = 0x0001;
    const unsigned int DATABASE_WRITE  = 0x0002;

    template<typename Record>
    class DatabaseFolder
    {
    public:
        DatabaseFolder() : _options(0) {}
        ~DatabaseFolder() { clean(); }

        void init( const char *folder = NULL,
                   unsigned int options = 0 )
        {
            clean();
            strncpy(_folder,folder,sizeof(_folder));
            _options = options;
        }

        void clean()
        {
            // Release all records to allow them to be written to disk.
            while(_records.size()>0)
                release(_records.begin()->first);
        }

        bool exists( int frame )
        {
            // Check if record exists in memory.
            if(_records.find(frame) != _records.end())
                return true;
            // Then look for it on disk.
            return Record::exists(_folder,frame);
        }

        Record *create( int frame,
                        const typename Record::CreateParams & params = Record::CreateParams() )
        {
            // TODO:  Reference-counting.
            // TODO-FOR-THREADING:  Write-lock record before returning.
            // Create new record.  If another record already existed for this
            // frame, it will be replaced.
            _records[frame].create(params);
            return &_records[frame];
        }

        Record *access( int frame )
        {
            // TODO:  Reference-counting.
            // TODO-FOR-THREADING:  Write-lock record before returning.
            // Check if record is in memory.
            if(_records.find(frame)==_records.end()) {
                // Not in memory, so read from disk, if enabled.
                if(_options & DATABASE_READ)
                    _records[frame].read(_folder,frame);
                else
                    return NULL;
            }
            return &_records[frame];
        }

        const Record *accessReadOnly( int frame )
        {
            // TODO:  Reference-counting.
            // TODO-FOR-THREADING:  Read-lock record before returning.
            // Not implemented.  Just access for now.
            return access(frame);
        }

        void release( int frame )
        {
            // TODO:  Reference-counting.
            // TODO-FOR-THREADING:  Release read/write lock.
            // Check if record is actually in memory.
            if(_records.find(frame)==_records.end())
                return;
            // Write record to disk, if enabled.
            if(_options & DATABASE_WRITE)
                _records[frame].write(_folder,frame);
            _records.erase(_records.find(frame));
        }

    private:
        char _folder[1024];
        unsigned int _options;
        std::map<int,Record> _records;
    };


    //------ Urbanscape Database Records ----------

    class ImageRecord
    {
    public:
        Image<unsigned char> image;
        double gain;

        struct CreateParams {
            int w;
            int h;
        };

        static bool exists( const char *folder, int frame );
        void create( const CreateParams &params );
        void read( const char *folder, int frame );
        void write( const char *folder, int frame );
    };

    class CameraRecord
    {
    public:
        Matrix3x4d P;

        struct CreateParams {};
        static bool exists( const char *folder, int frame );
        void create( const CreateParams &params );
        void read( const char *folder, int frame );
        void write( const char *folder, int frame );
    };

    class Tracker2DRecord
    {
    public:
        struct Track {
            float x, y;
            enum Status {
                New = 0,
                Continued = 1,
                Invalid = 2,
            } status;
            int id;
        };
        vector<Track> tracks;

        struct CreateParams {};

        static bool exists( const char *folder, int frame );
        void create( const CreateParams &params );
        void read( const char *folder, int frame );
        void write( const char *folder, int frame );
    };

    class DepthmapRecord
    {
    public:
        Image<float> depthmap;
        Image<float> confidence;

        struct CreateParams {
            int w;
            int h;
        };

        static bool exists( const char *folder, int frame );
        void create( const CreateParams &params );
        void read( const char *folder, int frame );
        void write( const char *folder, int frame );
    };

}

#endif
