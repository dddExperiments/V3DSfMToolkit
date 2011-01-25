// -*- C++ -*-

#ifndef V3D_POLYNOMIAL_CODEGEN_H
#define V3D_POLYNOMIAL_CODEGEN_H

#include "Math/v3d_polynomial.h"

#include <set>
#include <list>

namespace V3D
{

   template <int N, typename Order>
   inline std::map<Monomial<N>, int, Order>
   makeMonomialIndexMap(std::set<Monomial<N>, Order> const& monomials)
   {
      std::map<Monomial<N>, int, Order> res;

      int pos = 0;
      for (typename std::set<Monomial<N>, Order>::const_iterator p = monomials.begin(); p != monomials.end(); ++p)
      {
         res.insert(make_pair(*p, pos));
         ++pos;
      }

      return res;
   } // end makeMonomialIndexMap()

   template <typename Field, int N, typename Order>
   struct PolynomialDivisionOp
   {
         bool divisionOccured;
         int j;
         Monomial<N> quotient, leadMonomialP, leadMonomialF;
         Polynomial<Field, N, Order> gj;
         Polynomial<Field, N, Order> src, dst;
   }; // end struct PolynomialDivisionOp

   template <typename Field, int N, typename Order>
   struct PolynomialDivisionTrace : public std::vector<PolynomialDivisionOp<Field, N, Order> >
   {
         std::set<Monomial<N>, Order> collectMonomials() const
         {
            std::set<Monomial<N>, Order> res;
            for (size_t i = 0; i < this->size(); ++i)
            {
               PolynomialDivisionOp<Field, N, Order> op = (*this)[i];
               op.src.collectMonomials(res);
               op.dst.collectMonomials(res);
            } // end for (i)
            return res;
         } // end 

         void display() const
         {
            using namespace std;

            set<Monomial<N>, Order> const allMonomials = this->collectMonomials();

            for (size_t i = 0; i < this->size(); ++i)
            {
               PolynomialDivisionOp<Field, N, Order> op = (*this)[i];
               if (op.divisionOccured)
               {
                  cout << " lead term divisible by g" << op.j << " with quotient " << op.quotient << endl;
//                   cout << "  src = " << op.src << endl;
//                   cout << "  gj = " << op.gj << endl;
//                   cout << "  result = " << op.dst << endl;
               }
               else
               {
                  cout << " lead term added to remainder." << endl;
               }
            } // end for (i)

            cout << "  allMonomials (" << allMonomials.size() << "): ";
            for (typename set<Monomial<N>, Order>::const_iterator p = allMonomials.begin(); p != allMonomials.end(); ++p)
               cout << *p << " ";
            cout << endl;
         } // end display()

         void generateCode(char const * fieldType,
                           std::map<Monomial<N>, int, Order> const globalMap,
                           std::ostream& os = std::cout) const
         {
            using namespace std;

            typedef Polynomial<Field, N, Order> Poly;
            typedef map<Monomial<N>, int, Order> MonoMap;
            typedef set<Monomial<N>, Order> MonoSet;

            MonoSet const localMonomials = this->collectMonomials();
            MonoMap const localMap = makeMonomialIndexMap(localMonomials);

            os << "  {" << endl;
            os << "   " << fieldType << " c;" << endl;

            for (size_t i = 0; i < this->size(); ++i)
            {
               PolynomialDivisionOp<Field, N, Order> op = (*this)[i];
               if (op.divisionOccured)
               {
                  assert(localMap.find(op.leadMonomialP) != localMap.end());
                  int const localLeadIx  = (*localMap.find(op.leadMonomialP)).second;
                  //int const globalLeadIx = (*globalMap.find(op.leadMonomialF)).second;

                  os << "   c = s[" << localLeadIx << "];" << endl;
                  os << "   s[" << localLeadIx << "] = 0;" << endl;
                  MonoSet const gMonomials = op.gj.collectMonomials();
                  for (typename MonoSet::const_iterator p = gMonomials.begin(); p != gMonomials.end(); ++p)
                  {
                     Monomial<N> const gMonomial = *p;
                     Monomial<N> const dstMonomial = gMonomial * op.quotient;
                     if (dstMonomial == op.leadMonomialP) continue; // We have set that explicitly to 0.

                     assert(globalMap.find(gMonomial) != globalMap.end());
                     assert(localMap.find(dstMonomial) != localMap.end());
                     int const globalIx = (*globalMap.find(gMonomial)).second;
                     int const dstIx = (*localMap.find(dstMonomial)).second;
                     os << "   s[" << dstIx << "] -= c*G[" << op.j << "][" << globalIx << "];" << endl;
                  }
               }
               else
               {
                  assert(localMap.find(op.leadMonomialP) != localMap.end());
                  int const ix = (*localMap.find(op.leadMonomialP)).second;
                  os << "   r[" << ix << "] += s[" << ix << "];" << endl;
                  os << "   s[" << ix << "] = 0;" << endl;
               }
            } // end for (i)
            os << "  }" << endl;
         } // end generateCode()

   }; // end struct PolynomialDivisionTrace

   template <typename Field, int N, typename Order>
   struct GroebnerSpairOp
   {
         int _what;
         int _index1, _index2, _degree;
         int _dstIndex;
         Polynomial<Field, N, Order> gj1, gj2, spair, result;

         PolynomialDivisionTrace<Field, N, Order> _divTrace;

         GroebnerSpairOp(int i1, int i2, int degree)
            : _index1(i1), _index2(i2), _degree(degree)
         { }
   }; // end struct GroebnerSpairOp

   template <typename Field, int N, typename Order>
   struct GroebnerSpairTrace : public std::vector<GroebnerSpairOp<Field, N, Order> >
   {
         std::set<Monomial<N> , Order> encounteredMonomials;

         std::set<int> collectDependencies(int index) const
         {
            std::set<int> res;
            this->collectDependencies(index, res);
            return res;
         } // end collectDependencies()

         void collectDependencies(int index, std::set<int>& deps) const
         {
            std::list<int> queue;

            queue.push_back(index);

            while (!queue.empty())
            {
               int targetIndex = queue.front();
               queue.pop_front();
               for (int j = 0; j < this->size(); ++j)
               {
                  if ((*this)[j]._dstIndex == targetIndex)
                  {
                     int const i1 = (*this)[j]._index1;
                     int const i2 = (*this)[j]._index2;
                     deps.insert(i1);
                     deps.insert(i2);
                     queue.push_back(i1);
                     queue.push_back(i2);
                  } // end if
               } // end for (j)
            } // end while()
         } // end collectDependencies()

         void display()
         {
            using namespace std;

            for (size_t i = 0; i < this->size(); ++i)
            {
               GroebnerSpairOp<Field, N, Order> const& op = (*this)[i];

               cout << "Adding spair of " << op._index1 << " and " << op._index2
                    << " with degree " << op._degree << " yielding g" << op._dstIndex << endl;
               op._divTrace.display();
            } // end for (i)
         } // end display()

         void generateCode(char const * fieldType, std::map<Monomial<N>, int, Order> const globalMap,
                           std::ostream& os = std::cout) const
         {
            using namespace std;

            typedef Polynomial<Field, N, Order> Poly;
            typedef map<Monomial<N>, int, Order> MonoMap;
            typedef set<Monomial<N>, Order> MonoSet;

            os << "{" << endl;

            for (size_t i = 0; i < this->size(); ++i)
            {
               GroebnerSpairOp<Field, N, Order> const& op = (*this)[i];

               MonoSet       localMonomials = op._divTrace.collectMonomials();

               // Also add terms in the generated spair
               op.spair.collectMonomials(localMonomials);

               MonoMap const localMap = makeMonomialIndexMap(localMonomials);

               os << " {" << endl;

               // s[] holds the result of the spair operation, r holds the remainder
               os << "  " << fieldType << " s[" << localMap.size() << "];" << endl;
               os << "  " << fieldType << " r[" << localMap.size() << "];" << endl;
               os << "  std::fill(s, s + " << localMap.size() << ", 0);" << endl;
               os << "  std::fill(r, r + " << localMap.size() << ", 0);" << endl;

               Monomial<N> const& ltg1 = op.gj1.leadTerm().x;
               Monomial<N> const& ltg2 = op.gj2.leadTerm().x;
               Monomial<N> const lcm = Monomial<N>::lcm(ltg1, ltg2);

               Monomial<N> const q1 = lcm / ltg1;
               Monomial<N> const q2 = lcm / ltg2;

               {
                  MonoSet gMonomials = op.gj1.collectMonomials();
                  Monomial<N> const& quotient = q1;
                  int const j = op._index1;

                  gMonomials.erase(gMonomials.begin()); // remove lead term, since it cancels out
                  for (typename MonoSet::const_iterator p = gMonomials.begin(); p != gMonomials.end(); ++p)
                  {
                     Monomial<N> const srcMonomial = *p;
                     Monomial<N> const dstMonomial = srcMonomial * quotient;

                     if (localMap.find(dstMonomial) == localMap.end())
                     {
                        // If we cannot find that monomial, it cancels out, so ignore that one.
                        continue;
                     }

                     assert(globalMap.find(srcMonomial) != globalMap.end());
                     assert(localMap.find(dstMonomial) != localMap.end());

                     int const srcIx = (*globalMap.find(srcMonomial)).second;
                     int const dstIx = (*localMap.find(dstMonomial)).second;

                     os << "  s[" << dstIx << "] = G[" << j << "][" << srcIx << "];" << endl;
                  } // end for (p)
               } // end scope

               {
                  MonoSet gMonomials = op.gj2.collectMonomials();
                  Monomial<N> const& quotient = q2;
                  int const j = op._index2;

                  gMonomials.erase(gMonomials.begin()); // remove lead term, since it cancels out
                  for (typename MonoSet::const_iterator p = gMonomials.begin(); p != gMonomials.end(); ++p)
                  {
                     Monomial<N> const srcMonomial = *p;
                     Monomial<N> const dstMonomial = srcMonomial * quotient;

                     if (localMap.find(dstMonomial) == localMap.end())
                     {
                        // If we cannot find that monomial, it cancels out, so ignore that one.
                        continue;
                     }

                     assert(globalMap.find(srcMonomial) != globalMap.end());
                     assert(localMap.find(dstMonomial) != localMap.end());

                     int const srcIx = (*globalMap.find(srcMonomial)).second;
                     int const dstIx = (*localMap.find(dstMonomial)).second;

                     os << "  s[" << dstIx << "] -= G[" << j << "][" << srcIx << "];" << endl;
                  } // end for (p)
               } // end scope

               op._divTrace.generateCode(fieldType, globalMap, os);

               {
                  MonoSet const dstMonomials = op.result.collectMonomials();

                  // Normalize the lead term
                  assert(localMap.find(op.result.leadTerm().x) != localMap.end());
                  int const leadIx = (*localMap.find(op.result.leadTerm().x)).second;
                  os << "  " << fieldType << " c = (" << fieldType << ")(1)/r[" << leadIx << "];";
                  os << "  r[" << leadIx << "] = 1;" << endl;
                  for (typename MonoSet::const_iterator p = dstMonomials.begin(); p != dstMonomials.end(); ++p)
                  {
                     if (p == dstMonomials.begin()) continue;
                     assert(localMap.find(*p) != localMap.end());
                     int const ix = (*localMap.find(*p)).second;
                     os << "  r[" << ix << "] *= c;" << endl;
                  } // end for (p)

                  // Copy back the result into the basis G
                  for (typename MonoSet::const_iterator p = dstMonomials.begin(); p != dstMonomials.end(); ++p)
                  {
                     assert(localMap.find(*p) != localMap.end());
                     assert(globalMap.find(*p) != globalMap.end());
                     int const srcIx = (*localMap.find(*p)).second;
                     int const dstIx = (*globalMap.find(*p)).second;
                     os << "  G[" << op._dstIndex << "][" << dstIx << "] = r[" << srcIx << "];" << endl;
                  }
               } // end scope

            os << " }" << endl;
            } // end for (i)

            os << "}" << endl;
         } // end generateCode()
   }; // end struct GroebnerSpairTrace

   template <typename Field, int N, typename Order>
   struct GroebnerBasisCodeGenerator
   {
         typedef Monomial<N>                 monomial_type;
         typedef MonomialTerm<Field, N>      term_type;
         typedef Polynomial<Field, N, Order> polynomial_type;

         GroebnerBasisCodeGenerator(std::vector<polynomial_type> const& F)
            : _F(F)
         {
            this->compute_StdBasis_Buchberger();
         }

         static polynomial_type polynomialRemainder(polynomial_type const& f, std::vector<polynomial_type > const& F,
                                                    PolynomialDivisionTrace<Field, N, Order>& trace)
         {
            using namespace std;

            trace.clear();

            polynomial_type p(f), r;

            while (!p.isZero())
            {
               size_t j = 0;
               while (j < F.size())
               {
                  term_type const ltf = F[j].leadTerm();
                  term_type const ltp = p.leadTerm();
                  if (ltf.divides(ltp))
                  {
                     term_type t(ltp.c/ltf.c, ltp.x/ltf.x);

                     polynomial_type p_saved(p);
                     polynomial_type g = -F[j] * t;
                     g.removeLeadTerm();
                     p.removeLeadTerm();

                     p = p + g;

                     PolynomialDivisionOp<Field, N, Order> op;
                     op.divisionOccured = true;
                     op.leadMonomialP = p_saved.leadTerm().x;
                     op.leadMonomialF = F[j].leadTerm().x;
                     op.j = j;
                     op.quotient = t.x;
                     op.gj = F[j];
                     op.src = p_saved;
                     op.dst = p;
                     trace.push_back(op);

                     break;
                  }
                  ++j;
               } // end for (j)

               if (j == F.size())
               {
                  // No division occured
                  PolynomialDivisionOp<Field, N, Order> op;
                  op.divisionOccured = false;
                  op.src = r;
                  op.leadMonomialP = p.leadTerm().x;
                  r = r + p.leadTerm();
                  op.dst = r;
                  trace.push_back(op);
                  p.removeLeadTerm();
               }
            } // end while
            return r;
         } // end polynomialRemainder()

         void compute_StdBasis_Buchberger()
         {
            _trace.clear();

            _G = _F;
            while (1)
            {
               std::vector<polynomial_type> const G2 = _G;
               for (size_t j1 = 0; j1 < G2.size(); ++j1)
                  for (size_t j2 = j1+1; j2 < G2.size(); ++j2)
                  {
                     polynomial_type s = compute_Spair(G2[j1], G2[j2]);

                     PolynomialDivisionTrace<Field, N, Order> divTrace;
                     polynomial_type rem = polynomialRemainder(s, _G, divTrace);

                     if (!rem.isZero())
                     {
                        s.collectMonomials(_trace.encounteredMonomials);
                        rem.collectMonomials(_trace.encounteredMonomials);

                        //std::cout << "Adding " << rem << " to the std basis." << std::endl;
                        //if (rem.leadTerm().x != s.leadTerm().x) cout << " Needed division." << endl;

                        _trace.push_back(GroebnerSpairOp<Field, N, Order>(j1, j2, rem.degree()));
                        _trace.back()._dstIndex = _G.size();
                        _trace.back()._divTrace = divTrace;
                        _trace.back().gj1 = G2[j1];
                        _trace.back().gj2 = G2[j2];
                        _trace.back().result = rem;
                        _trace.back().spair = s;

                        Field c = Field(1) / rem.leadTerm().c;
                        _G.push_back(rem * c);
                     } // end for (j2)
                  } // end for (j1)
               if (_G.size() == G2.size()) break;
            } // end while
            _allMonomials.clear();
            for (size_t j = 0; j < _G.size(); ++j) _G[j].collectMonomials(_allMonomials);
            _monomialsIndexMap = makeMonomialIndexMap(_allMonomials);
         } // end compute_StdBasis_Buchberger()

         void generateCode(char const * fieldType, std::ostream& os = std::cout) const
         {
            using namespace std;

            cout << "all required monomials (" << _allMonomials.size() << "): ";
            for (typename set<monomial_type, Order>::const_iterator p = _allMonomials.begin();
                 p != _allMonomials.end(); ++p)
               cout << (*p) << " ";
            cout << endl;

            os << "/* This code is automatically generated from a Groebner basis trace - do not edit! */" << endl;

            _trace.generateCode(fieldType, _monomialsIndexMap, os);
         } // end generateCode()

         void displayMonomialIndexMap() const
         {
            cout << "Monomial to index mapping: " << endl;
            for (typename std::map<monomial_type, int, Order>::const_iterator p = _monomialsIndexMap.begin();
                 p != _monomialsIndexMap.end(); ++p)
               cout << " " << (*p).first << " -> " << (*p).second << endl;
         }

         void displayBasisPolynomialInC(int j, std::ostream& os = std::cout) const
         {
            polynomial_type const& p = _G[j];
            for (int k = 0; k < p._nTerms; ++k)
            {
               int const ix = (*_monomialsIndexMap.find(p._terms[k].x)).second;
               os << "+G[" << j << "][" << ix << "]*";
               p._terms[k].x.displayInC(os);
            }
            cout << endl;
         } // end displayBasisPolynomialInC()

         std::vector<polynomial_type> const& _F;
         std::vector<polynomial_type>        _G;
         GroebnerSpairTrace<Field, N, Order> _trace;
         std::set<monomial_type, Order>      _allMonomials; // in G
         std::map<monomial_type, int, Order> _monomialsIndexMap; // for all monomials in G
   }; // end struct GroebnerBasisCodeGenerator

} // end namespace V3D

#endif

