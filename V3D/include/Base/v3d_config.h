// -*- C++ -*-
#ifndef V3D_CONFIG_H
#define V3D_CONFIG_H

namespace V3D {

   class Config
   {
     public:
      Config( const char *appName );
      ~Config();

      // File I/O
      void read( const char *filename );
      void write( const char *filename ) const;
      void use( const char *filename );
      void update( const char *filename );
      // read:   fails if file doesn't exist
      // write:  will overwrite existing file
      // use:    read, write if file doesn't exist
      // update: read followed by write, new parameters inserted

      // Defaults
      void defaultBool( const char *name,
                        bool defval,
                        const char *desc = 0x0 );
      void defaultInt( const char *name,
                       int defval,
                       const char *desc = 0x0 );
      void defaultDouble( const char *name,
                          double defval,
                          const char *desc = 0x0 );
      void defaultString( const char *name,
                          const char *defval,
                          const char *desc = 0x0 );
      void defaultEnum( const char *name,
                        const int defvals[],
                        const char *const defstrs[],
                        int num,
                        int defval,
                        const char *desc = 0x0 );

      // Get
      bool        getBool( const char *name ) const;
      int         getInt( const char *name ) const;
      double      getDouble( const char *name ) const;
      const char *getString( const char *name ) const;
      int         getEnum( const char *name ) const;

      // Set
      void set( const char *name, bool val );
      void set( const char *name, int val );
      void set( const char *name, double val );
      void set( const char *name, const char *val );
      void setEnum( const char *name, int val );

     private:
      struct ConfigPrivate *m_private;
   };

}

#endif
