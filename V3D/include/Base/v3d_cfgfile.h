// -*- C++ -*-

#ifndef V3D_CFG_FILE_H
#define V3D_CFG_FILE_H

#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <cctype>

#include "Base/v3d_exception.h"

namespace V3D
{

   // This is a very trivial config file class to read in configuration data based on key-value pairs.
   struct ConfigurationFile
   {
      private:
         static void toLowerCase(std::string& s)
         {
            size_t const len = s.size();
            for (size_t i = 0; i < len; ++i)
               s[i] = tolower(s[i]);
         }
      public:
         ConfigurationFile(char const * filename)
            : _assocs()
         {
            using namespace std;
            ifstream is(filename);
            if (!is) return;

            std::string line, key, value;

            while (std::getline(is, line))
            {
               if (line.empty() || line[0] == '#') continue;
               std::istringstream ss(line);
               ss >> key;
               if (key.empty() || key[0] == '#') continue;
               ss >> value;
               _assocs.insert(std::make_pair(key, value));
            } // end while
         }

         bool get(char const * key, bool defaultVal) const
         {
            std::string const& val = this->getString(key);
            if (val.empty()) return defaultVal;
            if (val == std::string("yes") || val == std::string("true") || val == std::string("1")) return true;
            return false;
         }

         float get(char const * key, float defaultVal) const
         {
            return this->getGeneric<float>(key, defaultVal);
         }

         double get(char const * key, double defaultVal) const
         {
            return this->getGeneric<double>(key, defaultVal);
         }

         int get(char const * key, int defaultVal) const
         {
            return this->getGeneric<int>(key, defaultVal);
         }

         std::string get(char const * key, std::string const& defaultVal) const
         {
            std::string const& val = this->getString(key);
            if (val.empty()) return defaultVal;
            return val;
         }

      protected:
         template <typename T> T getGeneric(char const * key, T defaultVal) const
         {
            std::string const& val = this->getString(key);
            if (val.empty()) return defaultVal;
            std::istringstream ss(val);
            T res;
            ss >> res;
            return res;
         }


         std::string const& getString(char const * key_) const
         {
            std::string key(key_);
            std::map<std::string, std::string>::const_iterator p = _assocs.find(key);
            if (p == _assocs.end())
               return _emptyString;
            return (*p).second;
         }

         std::map<std::string, std::string> _assocs;
         std::string _emptyString;
   }; // end struct ConfigurationFile

} // end namespace V3D

#endif
