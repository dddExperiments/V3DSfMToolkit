#include "Base/v3d_storage.h"

#if defined(V3DLIB_ENABLE_SQLITE3)

#include <sqlite3.h>
#include <cstdio>
#include <cstdlib>

using namespace std;

namespace
{

   int
   countCallback(void * dstVal_, int argc, char *argv[], char *azColName[])
   {
      int * dstVal = static_cast<int *>(dstVal_);
      if (argc > 0)
         *dstVal = atoi(argv[0]);
      else
         *dstVal = 0;
      return 0;
   }

} // end namespace <>

namespace V3D
{

   SQLite3_Database::TraversalImpl::TraversalImpl(sqlite3 * db, char const * tableName)
      : _refCount(1)
   {
      char buf[1024]; int res;

      sprintf(buf, "select OID, DATA from %s", tableName);
      res = sqlite3_prepare_v2(db, buf, -1, &_stmt, NULL);
   }

   void
   SQLite3_Database::TraversalImpl::getNextObject(oid_t& oid, unsigned char const *& blob)
   {
      int res = sqlite3_step(_stmt);
      //cout << "SQLite3_Database::TraversalImpl::getNextObject(): res = " << res << endl;
      if (res == SQLITE_ROW)
      {
         oid  = sqlite3_column_int(_stmt, 0);
         blob = (unsigned char const *)(sqlite3_column_blob(_stmt, 1));
         //cout << "OID = " << oid << " blob = " << (void *)blob << endl;
      }
      else
      {
         oid = -1;
         blob = 0;
      }
   }

   void
   SQLite3_Database::TraversalImpl::release()
   {
      --_refCount;
      if (_refCount <= 0)
      {
         sqlite3_finalize(_stmt);
         delete this;
      }
   }

//----------------------------------------------------------------------

   SQLite3_Database::SQLite3_Database(char const * dbName, bool synchronousMode)
      : _oar(4096)
   {
      int res = sqlite3_open(dbName, &_db);

      if (!synchronousMode)
      {
         char * errMsg = 0;
         res = sqlite3_exec(_db, "pragma synchronous = off", 0, 0, &errMsg);
         if (res != SQLITE_OK)
         {
            cerr << "SQL error: " << errMsg << endl;
            sqlite3_free(errMsg);
         }
      } // end if
   }

   SQLite3_Database::~SQLite3_Database()
   {
      sqlite3_close(_db);
   }

   bool
   SQLite3_Database::createTable(char const * tableName, bool truncate)
   {
      using namespace std;

      char buf[1024]; int res;
      char * errMsg = 0;

      if (truncate)
      {
         sprintf(buf, "drop table if exists %s", tableName);
         res = sqlite3_exec(_db, buf, 0, 0, &errMsg);
         if (res != SQLITE_OK)
         {
            cerr << "SQL error: " << errMsg << endl;
            sqlite3_free(errMsg);
            return false;
         }
      } // end if

      sprintf(buf, "create table if not exists %s (DATA BLOB)", tableName);
      res = sqlite3_exec(_db, buf, 0, 0, &errMsg);
      if (res != SQLITE_OK)
      {
         cerr << "SQL error: " << errMsg << endl;
         sqlite3_free(errMsg);
         return false;
      }
      return true;
   } // end SQLite3_Database::createTable()

   bool
   SQLite3_Database::dropTable(char const * tableName)
   {
      using namespace std;

      char buf[1024]; int res;
      char * errMsg = 0;

      sprintf(buf, "drop table if exists %s", tableName);
      res = sqlite3_exec(_db, buf, 0, 0, &errMsg);
      if (res != SQLITE_OK)
      {
         cerr << "SQL error: " << errMsg << endl;
         sqlite3_free(errMsg);
         return false;
      }
      return true;
   } // end SQLite3_Database::dropTable()

   unsigned int
   SQLite3_Database::getRowCount(char const * tableName)
   {
      using namespace std;

      char buf[1024]; int res, count;
      char * errMsg = 0;

      sprintf(buf, "select count(*) from %s", tableName);
      res = sqlite3_exec(_db, buf, countCallback, &count, &errMsg);
      if (res != SQLITE_OK)
      {
         cerr << "SQL error: " << errMsg << endl;
         sqlite3_free(errMsg);
         return 0;
      }
      return count;
   } // end SQLite3_Database::dropTable()

   bool
   SQLite3_Database::deleteObject(char const * tableName, oid_t oid)
   {
      Stmts& stmts = this->lookupStatements(tableName);
      sqlite3_stmt * stmt = stmts.deleteStmt;

      int res;
      res = sqlite3_bind_int(stmt, 1, oid);
      res = sqlite3_step(stmt);
      sqlite3_reset(stmt);

      if (res != SQLITE_DONE)
         return false;
      return true;
   } // end SQLite3_Database::deleteObject()

   SQLite3_Database::Stmts&
   SQLite3_Database::lookupStatements(char const * tableName)
   {
      StmtsMap::iterator p = _stmtsMap.find(tableName);
      if (p != _stmtsMap.end())
      {
         return (*p).second;
      }

      Stmts stmts;
      char buf[1024]; int res;

      sprintf(buf, "insert or replace into %s (OID, DATA) values (?, ?)", tableName);
      res = sqlite3_prepare_v2(_db, buf, -1, &stmts.overwriteStmt, NULL);

      sprintf(buf, "select DATA from %s where OID = ?", tableName);
      res = sqlite3_prepare_v2(_db, buf, -1, &stmts.retrieveStmt, NULL);

      sprintf(buf, "delete from %s where OID = ?", tableName);
      res = sqlite3_prepare_v2(_db, buf, -1, &stmts.deleteStmt, NULL);

      p = _stmtsMap.insert(make_pair(tableName, stmts)).first;
      return (*p).second;
   } // end SQLite3_Database::lookupStatements()

   bool
   SQLite3_Database::updateBlob(char const * tableName, oid_t oid, void const * blob, int blobSize)
   {
      Stmts& stmts = this->lookupStatements(tableName);
      sqlite3_stmt * stmt = stmts.overwriteStmt;

      int res;
      res = sqlite3_bind_int(stmt, 1, oid);
      res = sqlite3_bind_blob(stmt, 2, blob, blobSize, SQLITE_TRANSIENT);
      res = sqlite3_step(stmt);
      sqlite3_reset(stmt);

      return res == SQLITE_DONE;
   } // end SQLite3_Database::::updateBlob()

   unsigned char *
   SQLite3_Database::retrieveBlob(char const * tableName, oid_t oid)
   {
      Stmts& stmts = this->lookupStatements(tableName);
      sqlite3_stmt * stmt = stmts.retrieveStmt;

      unsigned char * blobPtr = 0;

      int res;
      res = sqlite3_bind_int(stmt, 1, oid);
      res = sqlite3_step(stmt);

      if (res == SQLITE_ROW)
      {
         unsigned char const * blob = (unsigned char const *)(sqlite3_column_blob(stmt, 0));
         int const blobLen = sqlite3_column_bytes(stmt, 0);

         // Make a transient copy, since sqlite3_reset() might destroy the column contents
         blobPtr = new unsigned char[blobLen];
         memcpy(blobPtr, blob, blobLen);
      }
      sqlite3_reset(stmt);
      return blobPtr;
   } // end SQLite3_Database::retrieveBlob()

} // end namespace V3D

#endif // defined (V3DLIB_ENABLE_SQLITE3)
