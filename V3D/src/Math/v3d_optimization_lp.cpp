#include "v3d_optimization.h"

using namespace std;
using namespace V3D;

#if defined(V3DLIB_ENABLE_LPSOLVE)
extern "C"
{
# include "lpsolve/lp_lib.h"
}
#endif

#if defined(V3DLIB_ENABLE_DSDP)
# include "dsdp5.h"
#endif

namespace
{

   bool solveLinearProgram_LPSOLVE(std::vector<double> const& costs,
                                   CCS_Matrix<double> const& A,
                                   std::vector<ConstraintType> const& constraintTypes,
                                   std::vector<double> const& b,
                                   std::vector<std::pair<int, double> > const& lowerBounds,
                                   std::vector<std::pair<int, double> > const& upperBounds,
                                   std::vector<int> const& nonNegativeVariables,
                                   std::vector<double>& result,
                                   bool verbose, bool maximize)
   {
#if defined(V3DLIB_ENABLE_LPSOLVE)
      // ATTENTION: lp_solve's arrays are 1-based, not 0-based.

      // We have n constraints and m unknowns
      int const n = A.num_rows();
      int const m = A.num_cols();

      assert(constraintTypes.size() == n);
      assert(b.size() == n);
      assert(costs.size() == m);

      result.resize(m);

      lprec *lp;
      lp = make_lp(n, 0);

      if (verbose)
         set_verbose(lp, 4);
      else
         set_verbose(lp, 0);

      if (maximize)
         set_maxim(lp);
      else
         set_minim(lp);

      //resize_lp(lp, n, get_Ncolumns(lp));

      // Add constraints column-wise
      vector<double> values(n);
      vector<int> rowno(n);

      for (int j = 0; j < m; ++j)
      {
         int const count = A.getColumnNonzeroCount(j);
         A.getSparseColumn(j, rowno, values);
         for (int k = 0; k < count; ++k) rowno[k] += 1;
         add_columnex(lp, count, &values[0], &rowno[0]);
      }

      vector<double> rhs(n+1);
      rhs[0] = 0;
      std::copy(b.begin(), b.end(), &rhs[1]);
      set_rh_vec(lp, &rhs[0]);

      int constrType;
      for (int i = 0; i < n; ++i)
      {
         switch (constraintTypes[i])
         {
            case LP_EQUAL:
               constrType = EQ;
               break;
            case LP_LESS_EQUAL:
               constrType = LE;
               break;
            case LP_GREATER_EQUAL:
               constrType = GE;
               break;
            default:
               constrType = LE;
               cerr << "solveLinearProgram_LPSOLVE55(): Unknown constraint type, assuming <=." << endl;
         }
         set_constr_type(lp, i+1, constrType);
      } // end for (i)

      // Set free variables
      // Default: set free var
      for (int j = 0; j < m; ++j) set_unbounded(lp, j + 1);
      for (int k = 0; k < nonNegativeVariables.size(); ++k)
         set_lowbo(lp, nonNegativeVariables[k] + 1, 0.0);

      // Add lower & upper bounds on the variables
      for (int k = 0; k < lowerBounds.size(); ++k)
         set_lowbo(lp, lowerBounds[k].first + 1, lowerBounds[k].second);
      for (int k = 0; k < upperBounds.size(); ++k)
         set_upbo(lp, upperBounds[k].first + 1, upperBounds[k].second);

      // Set objective
      vector<double> obj(m+1);
      obj[0] = 0;
      copy(costs.begin(), costs.end(), &obj[1]);
      set_obj_fn(lp, &obj[0]);

      // Solve
      int status = solve(lp);
      if (verbose) cout << "lp_solve status = " << status << endl;

      // Copy result back
      double * X;
      get_ptr_primal_solution(lp, &X);
      int const N = get_Nrows(lp);
      copy(X+1+N, X+1+N+m, &result[0]);

      delete_lp(lp);

      return true;
#else
      throwV3DErrorHere("solveLinearProgram(): lp_solve55 not enabled.");
      return false;
#endif
   } // end solveLinearProgram_LPSOLVE()

   bool solveLinearProgram_DSDP(std::vector<double> const& costs,
                                CCS_Matrix<double> const& A,
                                std::vector<ConstraintType> const& constraintTypes,
                                std::vector<double> const& b,
                                std::vector<std::pair<int, double> > const& lowerBounds,
                                std::vector<std::pair<int, double> > const& upperBounds,
                                std::vector<int> const& nonNegativeVariables,
                                std::vector<double>& result,
                                bool verbose, bool maximize, bool useInitialValue)
   {
#if defined(V3DLIB_ENABLE_DSDP)
      DSDP solver;
      size_t const n = A.num_cols();
      size_t const m = A.num_rows();
 
      assert(constraintTypes.size() == m);
      assert(b.size() == m);
      assert(costs.size() == n);

      if (!useInitialValue) result.resize(n);

      DSDPCreate(n, &solver);
  
      double const invertCost = maximize ? 1.0 : -1.0;

      for (size_t i = 0; i < costs.size(); ++i)
         DSDPSetDualObjective(solver, i + 1, costs[i] * invertCost);

      LPCone lpCone;
      DSDPCreateLPCone(solver, &lpCone);

      vector<int> nnzin;
      vector<int> row;
      vector<double> aval;

      //look for equality and double this row later
      map<int, int> row2double; 
  
      for (size_t i = 0; i < b.size(); ++i)
      {
         if (constraintTypes[i] == LP_EQUAL)
         {
            row2double[i] = row2double.size() + b.size() - 1; //index of additional next row
         }
      }

      nnzin.push_back(0);
      for (size_t j = 0; j < A.num_cols(); ++j)
      {
         int const count = A.getColumnNonzeroCount(j);

         vector<int> ids(count);
         vector<double> vals(count);

         vector<int> idsDouble; //add at the end, implemention could assume ordering
         vector<double> valsDouble;
      
         A.getSparseColumn(j, ids, vals);

         nnzin.push_back(ids.size() + nnzin.back());
         for (size_t i = 0; i < ids.size(); ++i)
         {
            row.push_back(ids[i]);
            double val = vals[i];

            if (constraintTypes[ids[i]] == LP_GREATER_EQUAL)
	    {
               val *= -1;
	    }
            aval.push_back(val);

            if (constraintTypes[ids[i]] == LP_EQUAL)
	    {
               idsDouble.push_back(ids[i]);
               valsDouble.push_back(val);
	    }
         }

         //add LP_EQUAL rows
         nnzin.back() += idsDouble.size();
         for (size_t i = 0; i < idsDouble.size(); ++i)
         {
            int id = row2double[idsDouble[i]];
            row.push_back(id);
            aval.push_back(valsDouble[i] * -1);
         }
      }

      nnzin.push_back(b.size() + nnzin.back());     

      vector<double> valsDouble;
      vector<int> idsDouble;
      for (size_t i = 0; i < b.size(); ++i)
      {
         double val = b[i];     

         row.push_back(i);
         if (constraintTypes[i] == LP_GREATER_EQUAL)
         {
            val *= -1;
         }
         aval.push_back(val);
      
         if (constraintTypes[i] == LP_EQUAL)
         {
            valsDouble.push_back(val);
            idsDouble.push_back(i);
         }
      }

      //add LP_EQUAL rows
      nnzin.back() += idsDouble.size();
      for (size_t i = 0; i < idsDouble.size(); ++i)
      {
         int id = row2double[idsDouble[i]];
         row.push_back(id);
         aval.push_back(valsDouble[i] * -1);
      }
  
      assert(nnzin.size() == n + 2);
      assert(row.size() == aval.size());

      int* nnzin_a = new int[nnzin.size()];
      int* row_a = new int[row.size()];
      double* aval_a = new double[aval.size()];
  
      copy(nnzin.begin(), nnzin.end(), &nnzin_a[0]);
      copy(row.begin(), row.end(), &row_a[0]);
      copy(aval.begin(), aval.end(), &aval_a[0]);

      LPConeSetData2(lpCone, m + row2double.size(), nnzin_a, row_a, aval_a);

      //add variable bounds
      BCone lower;
      BCone upper;
      DSDPCreateBCone(solver, &lower);
      DSDPCreateBCone(solver, &upper);
  
      for (size_t i = 0; i < lowerBounds.size(); ++i)
      {
         int varId = lowerBounds[i].first + 1; //vars begin with 1...
         double bound = lowerBounds[i].second;
         BConeSetLowerBound(lower, varId, bound);
      }
      for (size_t i = 0; i < upperBounds.size(); ++i)
      {
         int varId = upperBounds[i].first + 1; //vars begin with 1...
         double bound = upperBounds[i].second;
         BConeSetUpperBound(upper, varId, bound);
      }

      //limit R+ vars
      for (size_t i= 0; i < nonNegativeVariables.size(); ++i)
      {
         BConeSetPSurplusVariable(lower, nonNegativeVariables[i] + 1); //vars begin with 1...
      }

      if (verbose)
         DSDPSetStandardMonitor(solver, 1);

      if (useInitialValue)
      {
         for (int i = 0; i < result.size(); ++i)
            DSDPSetY0(solver, i+1, result[i]);
         DSDPSetR0(solver, 1000.0);
      }

      DSDPSetup(solver);
      DSDPSolve(solver);

      DSDPTerminationReason reason;
      DSDPStopReason(solver, &reason);

      bool success = true;
  
      if (reason == DSDP_CONVERGED)
      {
         if (verbose)
            cout << "converged" << endl;
      }
      else
      {
         if (verbose)
            cout << "not converged!1!" << endl;

         success = false;
      }

      double* yout = new double[n];

      DSDPGetY(solver, &yout[0], n);
      result.resize(n);
      for (size_t i = 0; i < n; ++i)
      {
         result[i] = yout[i];
      }

      delete[] yout;

      DSDPDestroy(solver); 
      delete[] nnzin_a;
      delete[] row_a;
      delete[] aval_a;

      return success;
#else
      throwV3DErrorHere("solveLinearProgram(): DSDP not enabled.");
      return false;
#endif
   } // end solveLinearProgram_DSDP()

} // end namespace <>

namespace V3D
{

   bool solveLinearProgram(std::vector<double> const& costs,
                           CCS_Matrix<double> const& A,
                           std::vector<ConstraintType> const& constraintTypes,
                           std::vector<double> const& b,
                           std::vector<std::pair<int, double> > const& lowerBounds,
                           std::vector<std::pair<int, double> > const& upperBounds,
                           std::vector<int> const& nonNegativeVariables,
                           std::vector<double>& result, LP_Configuration const& conf)
   {
      if (conf.solverType == LP_LPSOLVE55)
         return solveLinearProgram_LPSOLVE(costs, A, constraintTypes, b, lowerBounds, upperBounds, nonNegativeVariables, result,
                                           conf.verbose, conf.maximize);
      else if (conf.solverType == LP_DSDP)
         return solveLinearProgram_DSDP(costs, A, constraintTypes, b, lowerBounds, upperBounds, nonNegativeVariables, result,
                                        conf.verbose, conf.maximize, conf.useInitialValue);
      else
      {
         cerr << "solveLinearProgram(): Unknown solver type " << conf.solverType << endl;
         return false;
      }
   } // end solveLinearProgram()

} // end namespace V3D
