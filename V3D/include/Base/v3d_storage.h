// -*- C++ -*-

#ifndef V3D_STORAGE_H
#define V3D_STORAGE_H

#include "Base/v3d_exception.h"
#include "Base/v3d_serialization.h"

#include <list>
#include <map>

#if defined(V3DLIB_ENABLE_SQLITE3)
struct sqlite3_stmt;
struct sqlite3;
#endif

namespace V3D
{

   template <typename Item>
   struct LRU_Cache
   {
      public:
         typedef int Key;
         typedef std::pair<Key, Item *> Pair;
         typedef std::list<Pair>        List;

         typedef std::map<Key, typename List::iterator> Map;

      public:
         LRU_Cache(int maxSize)
            : _maxSize(maxSize), _size(0), _cache()
         { }

         ~LRU_Cache()
         {
            for (typename List::iterator p = _cache.begin(); p != _cache.end(); ++p)
               delete (*p).second;
         }

         void insert(Key key, Item * item)
         {
            if (_rankMap.find(key) != _rankMap.end()) return;
            insertNew(key, item);
         }

         Item * operator[](Key key)
         {
            this->touch(key);
            return _cache.front().second;
         }

         bool has(Key const& key) const
         {
            return _rankMap.find(key) != _rankMap.end();
         }

         int size() const { return _size; }

      private:
         void insertNew(Key const& key, Item * item)
         {
            if (_size >= _maxSize) this->dropBack();

            _cache.push_front(make_pair(key, item));
            _rankMap.insert(make_pair(key, _cache.begin()));
            ++_size;
         }

         void dropBack()
         {
            Key key = _cache.back().first;
            delete _cache.back().second;
            _cache.pop_back();
            _rankMap.erase(key);
            --_size;
         }

         void touch(Key key)
         {
            typename Map::iterator p = _rankMap.find(key);
            if (p == _rankMap.end()) throwV3DErrorHere("Key not found in LRU_Cache");
            this->touch(p);
         }

         void touch(typename Map::iterator p)
         {
            typename List::iterator pos = (*p).second;
            if (pos != _cache.begin())
            {
               _cache.push_front(*pos);
               _cache.erase(pos);
               (*p).second = _cache.begin();
            }
         }

         int const _maxSize;
         int       _size;
         Map       _rankMap;
         List      _cache;
   }; // end struct LRU_Cache

//----------------------------------------------------------------------

#if defined(V3DLIB_ENABLE_SQLITE3)

   struct SQLite3_Database
   {
      protected:
         struct Stmts
         {
               sqlite3_stmt * overwriteStmt;
               sqlite3_stmt * deleteStmt;
               sqlite3_stmt * retrieveStmt;
         };

      public:
         typedef unsigned long oid_t;

         struct TraversalImpl
         {
               TraversalImpl(sqlite3 * db, char const * tableName);

               void getNextObject(oid_t& oid, unsigned char const *& blob);

               void retain() { ++_refCount; }
               void release();

            protected:
               int            _refCount;
               sqlite3_stmt * _stmt;
         }; // end struct TraversalImpl

         template <typename T>
         struct Table
         {
               typedef T value_type;

               Table(SQLite3_Database& db, char const * tableName)
                  : _db(db), _tableName(tableName), _oar(4096)
               { }

               bool updateObject(oid_t oid, T const& val)
               {
                  _oar.clear();
                  _oar & val;
                  return _db.updateBlob(_tableName, oid, _oar.getBlob(), _oar.blobSize());
               } // end updateObject()

               bool retrieveObject(oid_t oid, T& val)
               {
                  unsigned char * blobPtr = _db.retrieveBlob(_tableName, oid);
                  if (blobPtr)
                  {
                     BlobIArchive iar(blobPtr);
                     iar & val;
                     delete [] blobPtr;
                     return true;
                  }
                  return false;
               }

               bool deleteObject(oid_t oid)
               {
                  return _db.deleteObject(_tableName, oid);
               }

               unsigned int size()
               {
                  return _db.getRowCount(_tableName);
               }

               struct const_iterator
               {
                     typedef std::pair<oid_t, T> value_type;

                     const_iterator(sqlite3 * db, char const * tableName)
                        : _isGood(true)
                     {
                        _impl = new TraversalImpl(db, tableName);

                        unsigned char const * blob;
                        _impl->getNextObject(_val.first, blob);

                        if (blob)
                        {
                           BlobIArchive iar(blob);
                           iar & _val.second;
                        }
                        else
                           _isGood = false;
                     }

                     const_iterator(const_iterator const& b)
                        : _impl(b._impl), _val(b._val), _isGood(b._isGood)
                     {
                        _impl->retain();
                     }

                     ~const_iterator()
                     {
                        _impl->release();
                     }

                     const_iterator& operator=(const_iterator const& b)
                     {
                        _impl   = b._impl;
                        _val    = b._val;
                        _isGood = b._isGood;
                        _impl->retain();
                     }

                     operator bool() const { return _isGood; }

                     const_iterator& operator++()
                     {
                        unsigned char const * blob;
                        _impl->getNextObject(_val.first, blob);

                        if (blob)
                        {
                           BlobIArchive iar(blob);
                           iar & _val.second;
                        }
                        else
                           _isGood = false;

                        return *this;
                     }

                     value_type const& operator*() const { return _val; }

                  protected:
                     TraversalImpl * _impl;
                     value_type      _val;
                     bool            _isGood;
               }; // end struct const_iterator

               const_iterator begin()
               {
                  return const_iterator(_db._db, _tableName);
               }

            protected:
               SQLite3_Database&   _db;
               char const        * _tableName;
               BlobOArchive        _oar;
         }; // end struct Table

         SQLite3_Database(char const * dbName, bool synchronousMode = false);
         ~SQLite3_Database();

         bool createTable(char const * tableName, bool truncate = false);
         bool dropTable(char const * tableName);

         unsigned int getRowCount(char const * tableName);

         template <typename T>
         bool updateObject(char const * tableName, oid_t oid, T const& val)
         {
            _oar.clear();
            _oar & val;
            return this->updateBlob(tableName, oid, _oar.getBlob(), _oar.blobSize());
         } // end updateObject()

         template <typename T>
         bool retrieveObject(char const * tableName, oid_t oid, T& val)
         {
            unsigned char * blobPtr = this->retrieveBlob(tableName, oid);
            if (blobPtr)
            {
               BlobIArchive iar(blobPtr);
               iar & val;
               delete [] blobPtr;
               return true;
            }
            return false;
         }

         bool deleteObject(char const * tableName, oid_t oid);

         template <typename T>
         Table<T> getTable(char const * tableName)
         {
            return Table<T>(*this, tableName);
         }

      protected:
         typedef std::map<std::string, Stmts> StmtsMap;

         Stmts& lookupStatements(char const * tableName);

         bool            updateBlob(char const * tableName, oid_t oid, void const * blob, int blobSize);
         unsigned char * retrieveBlob(char const * tableName, oid_t oid);

         sqlite3    * _db;
         StmtsMap     _stmtsMap;
         BlobOArchive _oar;
   }; // end struct SQLite3_Database

#endif // defined (V3DLIB_ENABLE_SQLITE3)

//----------------------------------------------------------------------

   template <typename Storage>
   struct CachedStorage
   {
         typedef typename Storage::value_type value_type;

         CachedStorage(Storage& storage, int nMaxElements)
            : _storage(storage), _cache(nMaxElements)
         { }

         value_type * operator[](int oid)
         {
            if (_cache.has(oid)) return _cache[oid];

            value_type * val = new value_type;
            _storage.retrieveObject(oid, *val);
            _cache.insert(oid, val);
            return val;
         }

      protected:
         Storage&              _storage;
         LRU_Cache<value_type> _cache;
   }; // end struct CachedStorage

} // end namespace V3D

#endif
