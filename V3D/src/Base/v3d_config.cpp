#include "v3d_config.h"
#include "v3d_exception.h"
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <string.h>

#if defined(WIN32)
# define STRTOK strtok_s
#else
# define STRTOK strtok_r
#endif

namespace V3D {

   struct ConfigPrivate {
         std::string appName;
         std::vector<std::string> varNames;
         std::vector<std::string> varDescs;
         std::map<std::string,bool> boolVars;
         std::map<std::string,int> intVars;
         std::map<std::string,double> doubleVars;
         std::map<std::string,std::string> stringVars;
         std::map<std::string,std::map<std::string,int> > enumTypes;
         std::map<std::string,int> enumVars;
   };

   Config::Config( const char *appName )
   {
      m_private = new ConfigPrivate;
      m_private->appName = std::string(appName);
   }

   Config::~Config()
   {
      delete m_private;
   }

// File I/O

   void Config::read( const char *filename )
   {
      using namespace std;

      std::ifstream in;
      char line[256];
      std::string type, name;
      char *context;
      char *value,*p,*peq;

      // Open file.
      in.open(filename);
      verify(in.is_open(),"failed to open config file.");
      // Read one line at a time, ignoring comments.
      while(!in.eof()) {
         in.getline(line,sizeof(line));
         // Comments must have // as first character.
         if(line[0]=='#')
            continue;
         // White lines are ignored.
         bool ignore = true;
         for(p = line; *p!='\0' && ignore; p++)
            ignore = ignore && isspace(*p);
         if(ignore)
            continue;
         // Look for appname.
         if(strncmp(line,"appname",7)==0) {
            p = STRTOK(line," \t",&context);
            p = STRTOK(NULL," \t",&context);
            if(m_private->appName.compare(p)!=0)
               printf("appname %s does not match %s\n",p,m_private->appName.c_str());
            continue;
         }
         // Tokenize line and record type, name, =, value
         peq = strchr(line,'=');
         if(peq==NULL) {
            printf("Config::read failed to find \"=\"\n");
            continue;
         }
         p = STRTOK(line," \t",&context);
         if(p==NULL)
            continue;
         type = std::string(p);
         p = STRTOK(NULL," \t",&context);
         name = std::string(p);
         value = peq + 1;
         while(*value==' ' || *value=='\t')
            value++;
         // Type.
         if(type.compare("bool")==0) {
            //m_private->varNames.push_back(name);
            if(strcmp(value,"true")==0)
               m_private->boolVars[name] = true;
            else if(strcmp(value,"false")==0)
               m_private->boolVars[name] = false;
            else {
               printf("Config::read failed to recognize %s as bool\n",value);
               m_private->boolVars[name] = false;
            }
         } else if(type.compare("int")==0) {
            //m_private->varNames.push_back(name);
            m_private->intVars[name] = atoi(value);
         } else if(type.compare("double")==0) {
            //m_private->varNames.push_back(name);
            m_private->doubleVars[name] = atof(value);
         } else if(type.compare("string")==0) {
            //m_private->varNames.push_back(name);
            m_private->stringVars[name] = std::string(value);
         } else if(type.compare("enum")==0) {
            //m_private->varNames.push_back(name);
            if(m_private->enumTypes.find(name)==m_private->enumTypes.end()) {
               printf("Config::read failed to find enum %s\n",name.c_str());
               continue;
            }
            if(m_private->enumTypes[name].find(value)==m_private->enumTypes[name].end()) {
               printf("Config::read failed to find enum value %s\n",value);
               continue;
            }
            m_private->enumVars[name] = m_private->enumTypes[name][value];
         } else {
            printf("Config::read failed to recognize type \"%s\"\n",type.c_str());
         }
      }
   }

   void Config::write( const char *filename ) const
   {
      using namespace std;

      std::ofstream out;
      int varlen;
      int i,j;

      // Open file.
      out.open(filename);
      if(!out.is_open()) {
         printf("Config::write failed to open %s\n",filename);
         exit(0);
      }
      // Write appname.
      out << "appname " << m_private->appName << "\n";
      // Find the longest variable name.
      varlen = 0;
      for(i=0; i<(int)m_private->varNames.size(); i++) {
         if((int)m_private->varNames[i].length()>varlen)
            varlen = (int)m_private->varNames[i].length();
      }
      // Write variables.
      for(i=0; i<(int)m_private->varNames.size(); i++) {
         std::string name = m_private->varNames[i];
         std::string type;
         std::ostringstream value;
         if(m_private->boolVars.find(name)!=m_private->boolVars.end()) {
            type = "bool    ";
            value << std::string(m_private->boolVars.find(name)->second?"true":"false");
         } else if(m_private->intVars.find(name)!=m_private->intVars.end()) {
            type = "int     ";
            value << m_private->intVars.find(name)->second;
         } else if(m_private->doubleVars.find(name)!=m_private->doubleVars.end()) {
            type = "double  ";
            value << m_private->doubleVars.find(name)->second;
         } else if(m_private->stringVars.find(name)!=m_private->stringVars.end()) {
            type = "string  ";
            value << m_private->stringVars.find(name)->second;
         } else if(m_private->enumVars.find(name)!=m_private->enumVars.end()) {
            type = "enum    ";
            std::map<std::string,int>::const_iterator iter;
            for(iter=m_private->enumTypes.find(name)->second.begin();
                iter!=m_private->enumTypes.find(name)->second.end(); iter++)
            {
               if(iter->second==m_private->enumVars.find(name)->second) {
                  value << iter->first;
                  break;
               }
            }
         }
         if(m_private->varDescs[i].length()>0)
            out << "# " << m_private->varDescs[i] << "\n";
         if(m_private->enumVars.find(name)!=m_private->enumVars.end()) {
            out << "# values: ";
            std::map<std::string,int> e = m_private->enumTypes.find(name)->second;
            std::vector<std::string> names;
            for(std::map<std::string,int>::const_iterator iter=e.begin(); iter!=e.end(); iter++)
               names.push_back(iter->first);
            std::sort(names.begin(),names.end());
            for(int s=0; s<(int)names.size(); s++)
               out << names[s] << " ";
            out << "\n";
         }
         out << type << name;
         for(j=0; j<varlen-(int)name.length(); j++)
            out << " ";
         out << "  =  " << value.str() << "\n";
      }
   }

   void Config::use( const char *filename )
   {
      bool exists;
      std::ifstream in(filename);
      exists = in.is_open();
      in.close();
      if(exists)
         read(filename);
      else
         write(filename);        
   }

   void Config::update( const char *filename )
   {
      read(filename);
      write(filename);
      // TODO:  Old parameters that no longer have
      // defaults could be marked as stale.
   }

// Defaults

   void Config::defaultBool( const char *name,
                             bool defval,
                             const char *desc )
   {
      if(m_private->boolVars.find(name)==m_private->boolVars.end())
         m_private->varNames.push_back(std::string(name));
      m_private->boolVars[name] = defval;
      m_private->varDescs.push_back(desc?std::string(desc):std::string(""));
   }

   void Config::defaultInt( const char *name,
                            int defval,
                            const char *desc )
   {
      if(m_private->intVars.find(name)==m_private->intVars.end())
         m_private->varNames.push_back(std::string(name));
      m_private->intVars[name] = defval;
      m_private->varDescs.push_back(desc?std::string(desc):std::string(""));
   }

   void Config::defaultDouble( const char *name,
                               double defval,
                               const char *desc )
   {
      if(m_private->doubleVars.find(name)==m_private->doubleVars.end())
         m_private->varNames.push_back(std::string(name));
      m_private->doubleVars[name] = defval;
      m_private->varDescs.push_back(desc?std::string(desc):std::string(""));
   }

   void Config::defaultString( const char *name,
                               const char *defval,
                               const char *desc )
   {
      if(m_private->stringVars.find(name)==m_private->stringVars.end())
         m_private->varNames.push_back(std::string(name));
      m_private->stringVars[name] = std::string(defval);
      m_private->varDescs.push_back(desc?std::string(desc):std::string(""));
   }

   void Config::defaultEnum( const char *name,
                             const int defvals[],
                             const char *const defstrs[],
                             int num,
                             int defval,
                             const char *desc )
   {
      if(m_private->enumVars.find(name)==m_private->enumVars.end()) {
         m_private->varNames.push_back(std::string(name));
         m_private->enumTypes[name].clear();
      }
      for(int i=0; i<num; i++)
         m_private->enumTypes[name][defstrs[i]] = defvals[i];
      m_private->enumVars[name] = defval;
      m_private->varDescs.push_back(desc?std::string(desc):std::string(""));
   }

// Get

   bool Config::getBool( const char *name ) const
   {
      if(m_private->boolVars.find(name)==m_private->boolVars.end()) {
         printf("Config::getBool could not find %s\n",name);
         exit(0);
      }
      return m_private->boolVars.find(name)->second;
   }

   int Config::getInt( const char *name ) const
   {
      if(m_private->intVars.find(name)==m_private->intVars.end()) {
         printf("Config::getInt could not find %s\n",name);
         exit(0);
      }
      return m_private->intVars.find(name)->second;
   }

   double Config::getDouble( const char *name ) const
   {
      if(m_private->doubleVars.find(name)==m_private->doubleVars.end()) {
         printf("Config::getDouble could not find %s\n",name);
         exit(0);
      }
      return m_private->doubleVars.find(name)->second;
   }

   const char *Config::getString( const char *name ) const
   {
      if(m_private->stringVars.find(name)==m_private->stringVars.end()) {
         printf("Config::getString could not find %s\n",name);
         exit(0);
      }
      return m_private->stringVars.find(name)->second.c_str();
   }

   int Config::getEnum( const char *name ) const
   {
      if(m_private->enumVars.find(name)==m_private->enumVars.end()) {
         printf("Config::getEnum could not find %s\n",name);
         exit(0);
      }
      return m_private->enumVars.find(name)->second;
   }

// Set

   void Config::set( const char *name,
                     bool val )
   {
      m_private->boolVars[name] = val;
   }

   void Config::set( const char *name,
                     int val )
   {
      m_private->intVars[name] = val;
   }

   void Config::set( const char *name,
                     double val )
   {
      m_private->doubleVars[name] = val;
   }

   void Config::set( const char *name,
                     const char *val )
   {
      m_private->stringVars[name] = std::string(val);
   }

   void Config::setEnum( const char *name,
                         int val )
   {
      m_private->enumVars[name] = val;
   }

}
