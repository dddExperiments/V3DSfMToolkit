#if defined(V3DLIB_ENABLE_ARPACK)

#include "Base/v3d_exception.h"
#include "Math/v3d_sparseeig.h"

using namespace std;

#define F77NAME(x) x ## _

typedef int F77_int;
typedef int F77_logical;

extern "C"
{

//    struct { 
//          F77_int logfil, ndigit, mgetv0;
//          F77_int msaupd, msaup2, msaitr, mseigt, msapps, msgets, mseupd;
//          F77_int mnaupd, mnaup2, mnaitr, mneigt, mnapps, mngets, mneupd;
//          F77_int mcaupd, mcaup2, mcaitr, mceigt, mcapps, mcgets, mceupd;
//    } F77NAME(debug);

   void F77NAME(dsaupd)(F77_int *ido, char *bmat, F77_int *n, char *which,
                        F77_int *nev, double *tol, double *resid,
                        F77_int *ncv, double *V, F77_int *ldv,
                        F77_int *iparam, F77_int *ipntr, double *workd,
                        double *workl, F77_int *lworkl, F77_int *info);

   void F77NAME(dseupd)(F77_logical *rvec, char *HowMny, F77_logical *select,
                        double *d, double *Z, F77_int *ldz,
                        double *sigma, char *bmat, F77_int *n,
                        char *which, F77_int *nev, double *tol,
                        double *resid, F77_int *ncv, double *V,
                        F77_int *ldv, F77_int *iparam, F77_int *ipntr,
                        double *workd, double *workl,
                        F77_int *lworkl, F77_int *info);

}

namespace
{

   inline char * getWhichFromMode(int mode)
   {
      using namespace V3D;

      switch (mode)
      {
         case V3D_ARPACK_LARGEST_EIGENVALUES:
            return "LA";
         case V3D_ARPACK_SMALLEST_EIGENVALUES:
            return "SA";
         case V3D_ARPACK_LARGEST_MAGNITUDE_EIGENVALUES:
            return "LM";
         case V3D_ARPACK_SMALLEST_MAGNITUDE_EIGENVALUES:
            return "SM";
         default:
            throwV3DErrorHere("Unknown mode specifier in computeSparseSymmetricEig().");
      }
   } // end getWhichFromMode()

}

namespace V3D
{

   bool
   computeSparseSymmetricEig(CCS_Matrix<double> const& A, int mode, int nWanted,
                             VectorBase<double>& lambda, MatrixBase<double>& U,
                             SparseSymmetricEigConfig cfg)
   {
      assert(A.num_cols() == A.num_rows());

      F77_int ido = 0;
      char bmat = 'I'; // Std eigenvalue problem Ax=lambda*x
      F77_int nev = nWanted;

      char * which = getWhichFromMode(mode);

      int N = A.num_cols();
      int ncv = cfg.nColumnsV;
      if (ncv <= 0) ncv = std::min(N-1, std::max(25, 2*nWanted));

      double tol = cfg.tolerance;
      vector<double> resid(N+1);
      vector<double> V(ncv*N+1);

      int ldv = N;
      int iparam[12];
      iparam[0] = 1;
      iparam[2] = cfg.maxArnoldiIterations;
      iparam[6] = 1;
      int ipntr[12];

      int lworkl = ncv*(ncv+8);
      vector<double> workd(3*N+1), workl(lworkl+1);
      int info = 0;

      VectorBase<double> X(N), Y(N);

      while (1)
      {
         F77NAME(dsaupd)(&ido, &bmat, &N, which, &nev, &tol, &resid[0],
                         &ncv, &V[0], &ldv, iparam, ipntr, &workd[0], &workl[0],
                         &lworkl, &info);

         //cout << "ido = " << ido << ", info = " << info << endl;

         if (ido == -1 || ido == 1)
         {
            // Compute Y=A*X
            double * Xptr = &workd[ipntr[0]-1];
            double * Yptr = &workd[ipntr[1]-1];

            for (int i = 0; i < N; ++i) X[i] = Xptr[i];

            multiply_At_v_Sparse(A, X, Y); // recall that A is symmetric

            for (int i = 0; i < N; ++i) Yptr[i] = Y[i];
         }
         else
            break;
      } // end while

      if (info == 1)
      {
         cerr << "computeSparseSymmetricEig(): Maximum number of Arnoldi iterations reached, please specify more iterations." << endl;
         return false;

      }
      if (ido != 99) return false;

      int rvec = 1; // generate eigenvectors, too
      char howMany = 'A';
      vector<F77_logical> select(ncv);
      vector<double> d(nev), Z(N*nev);

      int ldz = N;
      double sigma;

      F77NAME(dseupd)(&rvec, &howMany, &select[0], &d[0], &Z[0], &ldz, &sigma,
                      &bmat, &N, which, &nev, &tol, &resid[0], &ncv, &V[0], &ldv,
                      iparam, ipntr, &workd[0], &workl[0], &lworkl, &info);

      //cout << "info = " << info << endl;
      if (info < 0) return false;

      U.newsize(N, nev);
      lambda.newsize(nev);

      for (int i = 0; i < nev; ++i) lambda[i] = d[i];

      for (int i = 0; i < N; ++i)
         for (int j = 0; j < nev; ++j)
            U[i][j] = Z[N*j + i]; // recall that fortran matrices are column major

      return true;
   } // end computeSparseSymmetricEig()

   bool
   computeSparseSVD(CCS_Matrix<double> const& A, int mode, int nWanted,
                    VectorBase<double>& sigma, MatrixBase<double>& V,
                    SparseSymmetricEigConfig cfg)
   {
      int M = A.num_rows();
      int N = A.num_cols();

      //cout << "cfg.maxArnoldiIterations = " << cfg.maxArnoldiIterations << endl;

      // Create sparse matrix representation for A^T
      int const nnz = A.getNonzeroCount();
      vector<pair<int, int> > nzAt;
      vector<double>          valsAt;
      nzAt.reserve(nnz);
      valsAt.reserve(nnz);
      vector<int> rows(M);
      vector<double> values(M);
      for (int j = 0; j < N; ++j)
      {
         A.getSparseColumn(j, rows, values);
         int const nnzCol = A.getColumnNonzeroCount(j);

         for (int k = 0; k < nnzCol; ++k)
         {
            nzAt.push_back(make_pair(j, rows[k]));
            valsAt.push_back(values[k]);
         }
      } // end for (j)
      CCS_Matrix<double> At(N, M, nzAt, valsAt);

      F77_int ido = 0;
      char bmat = 'I'; // Std eigenvalue problem Ax=lambda*x
      F77_int nev = nWanted;

      char * which = getWhichFromMode(mode);

      int ncv = cfg.nColumnsV;
      if (ncv <= 0) ncv = std::min(N-1, std::max(25, 2*nWanted));

      double tol = cfg.tolerance;
      vector<double> resid(N+1);
      vector<double> VV(ncv*N+1);

      int ldv = N;
      int iparam[12];
      iparam[0] = 1;
      iparam[2] = cfg.maxArnoldiIterations;
      iparam[6] = 1;
      int ipntr[12];

      int lworkl = ncv*(ncv+8);
      vector<double> workd(3*N+1), workl(lworkl+1);
      int info = 0;

      VectorBase<double> X(N), Ax(M), Y(N);

      while (1)
      {
         F77NAME(dsaupd)(&ido, &bmat, &N, which, &nev, &tol, &resid[0],
                         &ncv, &VV[0], &ldv, iparam, ipntr, &workd[0], &workl[0],
                         &lworkl, &info);

         //cout << "ido = " << ido << ", info = " << info << endl;

         if (ido == -1 || ido == 1)
         {
            // Compute Y=A*X
            double * Xptr = &workd[ipntr[0]-1]; // Fortran has 1-based arrays
            double * Yptr = &workd[ipntr[1]-1];

            for (int i = 0; i < N; ++i) X[i] = Xptr[i];

            multiply_At_v_Sparse(At, X, Ax);
            multiply_At_v_Sparse(A, Ax, Y);

            for (int i = 0; i < N; ++i) Yptr[i] = Y[i];
         }
         else
            break;
      } // end while

      //cout << "Used iterations: " << iparam[2] << endl;
      //cout << "final ido = " << ido << ", info = " << info << endl;

      if (info == 1)
      {
         cerr << "computeSparseSVD(): Maximum number of Arnoldi iterations reached, please specify more iterations." << endl;
         return false;
      }

      if (ido != 99) return false;

      int rvec = 1; // generate eigenvectors, too
      char howMany = 'A';
      vector<F77_logical> select(ncv);
      vector<double> d(nev), Z(N*nev);

      int ldz = N;
      double shift = 0;

      F77NAME(dseupd)(&rvec, &howMany, &select[0], &d[0], &Z[0], &ldz, &shift,
                      &bmat, &N, which, &nev, &tol, &resid[0], &ncv, &VV[0], &ldv,
                      iparam, ipntr, &workd[0], &workl[0], &lworkl, &info);

      //cout << "info = " << info << endl;
      if (info < 0) return false;

      V.newsize(N, nev);
      sigma.newsize(nev);

      for (int i = 0; i < nev; ++i) sigma[i] = sqrt(d[i]);

      for (int i = 0; i < N; ++i)
         for (int j = 0; j < nev; ++j)
            V[i][j] = Z[N*j + i]; // recall that fortran matrices are column major

      return true;
   } // end computeSparseSVD()

} // end namespace V3D

#endif
