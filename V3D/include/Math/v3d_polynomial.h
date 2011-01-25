// -*- C++ -*-

#ifndef V3D_POLYNOMIAL_H
#define V3D_POLYNOMIAL_H

#include "Math/v3d_linear.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <set>
#include <map>

namespace V3D
{

   template <int N>
   struct Monomial
   {
         Monomial()
         {
            std::fill(exponent, exponent+N, 0);
         }

         Monomial(int var, int degree = 1)
         {
            std::fill(exponent, exponent+N, 0);
            exponent[var] = degree;
         }

         // Parses strings like "xy2z3" for "xy^2z^3" etc.
         Monomial(char const * str)
         {
            std::fill(exponent, exponent+N, 0);

            int curVarIx = -1;
            int curExponent = -1;
            for (char const * p = str; ; ++p)
            {
               char const c = *p;
               if (c >= '0' && c <= '9')
               {
                  if (curExponent < 0)
                     curExponent = (c - '0'); // exponent not specified so far
                  else
                     curExponent = 10*curExponent + (c - '0');
               }
               else
               {
                  if (curVarIx >= 0)
                  {
                     if (curExponent < 0)
                        ++this->exponent[curVarIx]; // exponent not specified
                     else
                        this->exponent[curVarIx] += curExponent;
                  }
                  curExponent = -1;
                  curVarIx = varIndexFromChar(c);
                  if (curVarIx < 0 || curVarIx >= N)
                     curVarIx = -1;
               }
               if (c == 0) break;
            } // end for (p)
         }

         bool operator==(Monomial const& b) const
         {
            Monomial const& a = *this;

            for (int i = 0; i < N; ++i)
            {
               if (a.exponent[i] != b.exponent[i]) return false;
            }
            return true;
         }

         bool operator!=(Monomial const& b) const
         {
            Monomial const& a = *this;
            return !(a == b);
         }

         Monomial operator*(Monomial const& b) const
         {
            Monomial const& a = *this;

            Monomial res;
            for (int i = 0; i < N; ++i)
            {
               res.exponent[i] = a.exponent[i] + b.exponent[i];
            }
            return res;
         }

         Monomial operator/(Monomial const& b) const
         {
            Monomial const& a = *this;

            Monomial res;
            for (int i = 0; i < N; ++i)
            {
               res.exponent[i] = a.exponent[i] - b.exponent[i];
            }
            return res;
         }

         bool divides(Monomial const& b) const
         {
            Monomial const& a = *this;

            for (int i = 0; i < N; ++i)
            {
               if (a.exponent[i] > b.exponent[i]) return false;
            }
            return true;
         }

         int multiDegree() const
         {
            int res = 0;
            for (int i = 0; i < N; ++i) res += exponent[i];
            return res;
         }

         bool isConstant() const { return this->multiDegree() == 0; }

         static Monomial lcm(Monomial const& a, Monomial const& b)
         {
            Monomial res;
            for (int i = 0; i < N; ++i)
            {
               res.exponent[i] = std::max(a.exponent[i], b.exponent[i]);
            }
            return res;
         }

         void displayInC(std::ostream& os = std::cout) const
         {
            static char const vars[] = "xyzwabcdefgh";

            Monomial<N> const& x = *this;

            if (x.isConstant())
               os << "1";
            else
            {
               bool multRequired = false;
               for (int i = 0; i < N; ++i)
               {
                  if (multRequired && exponent[i] > 0) os << "*";
                  if (x.exponent[i] > 1)
                  {
                     os << vars[i] << x.exponent[i];
                     multRequired = true;
                  }
                  else if (x.exponent[i] == 1)
                  {
                     os << vars[i];
                     multRequired = true;
                  }
               } // end for (i)
            } // end if 
         } // end displayInC()

         // x.substituteVars("xyz", "yzx") changes e.g. x^2 + yz to y^2 + zx
         Monomial substituteVars(char const * srcVars, char const * dstVars) const
         {
            Monomial res;
            while (*srcVars)
            {
               int const srcId = varIndexFromChar(*srcVars);
               int const dstId = varIndexFromChar(*dstVars);
               res.exponent[dstId] += exponent[srcId];
               ++srcVars; ++dstVars;
            } // end while
            return res;
         } // end substituteVars()

         int exponent[N];

      private:
         static int varIndexFromChar(char c)
         {
            switch (c)
            {
               case 'x': return 0;
               case 'y': return 1;
               case 'z': return 2;
               case 'w': return 3;
            }
            return c - 'a' + 4;
         }
   }; // end struct Monomial

   template <typename Field, int N>
   struct MonomialTerm
   {
         MonomialTerm() : x(), c(0) { }

         MonomialTerm(Field c_, Monomial<N> const& x_)
            : x(x_), c(c_)
         { }

         MonomialTerm(Field c_)
            : x(), c(c_)
         { }

         MonomialTerm(Field c_, char const * x_)
            : x(x_), c(c_)
         { }

         bool divides(MonomialTerm const& b) const
         {
            return x.divides(b.x);
         }

         Monomial<N> x;
         Field       c;
   }; // end struct MonomialTerm

   template <int N>
   struct MonomialOrderLex
   {
         bool operator()(Monomial<N> const& a, Monomial<N> const& b) const
         {
            for (int i = 0; i < N; ++i)
            {
               if (a.exponent[i] > b.exponent[i]) return true;
               if (a.exponent[i] < b.exponent[i]) return false;
            }
            return false;
         }
   }; // end struct MonomialOrderLex

   template <int N>
   struct MonomialOrderGrevLex
   {
         bool operator()(Monomial<N> const& a, Monomial<N> const& b) const
         {
            int const deg_a = a.multiDegree();
            int const deg_b = b.multiDegree();

            if (deg_a > deg_b) return true;
            if (deg_a < deg_b) return false;

            for (int i = N-1; i >= 0; --i)
            {
               if (a.exponent[i] > b.exponent[i]) return false;
               if (a.exponent[i] < b.exponent[i]) return true;
            }
            return false;
         }
   }; // end struct MonomialOrderGrevLex

   template <typename Field, int N, typename Order>
   struct TermOrder
   {
         bool operator()(MonomialTerm<Field, N> const& a, MonomialTerm<Field, N> const& b) const
         {
            Order order;
            return order(a.x, b.x);
         }
   }; // end struct TermOrder

   template <typename Field, int N, typename Order> struct Polynomial;

   template <typename Field, int N, typename Order>
   Polynomial<Field, N, Order>
   compute_Spair(Polynomial<Field, N, Order> const& a, Polynomial<Field, N, Order> const & b);

   template <typename Field, int N, typename Order>
   struct Polynomial
   {
         typedef Field                  field_type;
         typedef Monomial<N>            monomial_type;
         typedef MonomialTerm<Field, N> term_type;

         Polynomial()
            : _nTerms(0), _terms(0)
         { }

         Polynomial(Field c, Monomial<N> const& x)
            : _nTerms(1)
         {
            _terms = new MonomialTerm<Field, N>[1];
            _terms[0].c = c;
            _terms[0].x = x;
         }

         Polynomial(MonomialTerm<Field, N> const& t)
            : _nTerms(1)
         {
            _terms = new MonomialTerm<Field, N>[1];
            _terms[0] = t;
         }

         Polynomial(Monomial<N> const& x)
            : _nTerms(1)
         {
            _terms = new MonomialTerm<Field, N>[1];
            _terms[0].c = 1;
            _terms[0].x = x;
         }

         // We assume that the provided monomials are unique.
         Polynomial(int nTerms, Field const * coeffs, Monomial<N> const * xs)
            : _nTerms(0)
         {
            for (int k = 0; k < nTerms; ++k)
            {
               if (coeffs[k] != Field(0)) ++_nTerms;
            }

            _terms = new MonomialTerm<Field, N>[_nTerms];

            int pos = 0;
            for (int k = 0; k < nTerms; ++k)
            {
               if (coeffs[k] != Field(0))
               {
                  _terms[pos].c = coeffs[k];
                  _terms[pos].x = xs[k];
                  ++pos;
               }
            } // end for (k)

            TermOrder<Field, N, Order> order;
            std::sort(_terms, _terms + _nTerms, order);
         }

         Polynomial(Polynomial const& p)
            : _nTerms(p._nTerms)
         {
            _terms = new MonomialTerm<Field, N>[_nTerms];
            for (int k = 0; k < _nTerms; ++k)
               _terms[k] = p._terms[k];
         }

         ~Polynomial()
         {
            if (_terms) delete [] _terms;
         }

         Polynomial& operator=(Polynomial const& p)
         {
            if (this == &p) return *this;
            if (_terms) delete [] _terms;
            _nTerms = p._nTerms;
            _terms = new MonomialTerm<Field, N>[_nTerms];
            for (int k = 0; k < _nTerms; ++k)
               _terms[k] = p._terms[k];
            return *this;
         }

         int degree() const
         {
            int res = 0;
            for (int k = 0; k < _nTerms; ++k)
               res = std::max(res, _terms[k].x.multiDegree());
            return res;
         }

         MonomialTerm<Field, N> leadTerm() const
         {
            if (_nTerms == 0)
               return MonomialTerm<Field, N>();

            return _terms[0];
         }

         Polynomial operator+(Monomial<N> const& b) const
         {
            return (*this) + MonomialTerm<Field, N>(1, b);
         }

         Polynomial operator+(MonomialTerm<Field, N> const& b) const
         {
            Order order;

            Polynomial const& a = *this;
            Polynomial res(a._nTerms + 1);

            int k0 = 0, k = 0;
            while (k0 < a._nTerms && order(a._terms[k0].x, b.x))
            {
               res._terms[k] = a._terms[k0];
               ++k0; ++k;
            }
            if (k0 < a._nTerms && a._terms[k0].x == b.x)
            {
               Field coeff = a._terms[k0].c + b.c;
               if (coeff != Field(0))
               {
                  res._terms[k].x = a._terms[k0].x;
                  res._terms[k].c = coeff;
                  ++k;
               }
               ++k0;
            }
            else
            {
               res._terms[k] = b;
               ++k;
            }
            while (k0 < a._nTerms)
            {
               res._terms[k] = a._terms[k0];
               ++k0; ++k;
            }
            res._nTerms = k;
            return res;
         } // end operator+()

         Polynomial operator+(Polynomial const& b) const
         {
            Order order;

            Polynomial const& a = *this;
            Polynomial res(a._nTerms + b._nTerms);
            int k0 = 0, k1 = 0;
            int k = 0;

            while (k0 < a._nTerms || k1 < b._nTerms)
            {
               if (k0 < a._nTerms && k1 < b._nTerms)
               {
                  if (a._terms[k0].x == b._terms[k1].x)
                  {
                     Field coeff = a._terms[k0].c + b._terms[k1].c;
                     if (coeff != Field(0))
                     {
                        res._terms[k].x = a._terms[k0].x;
                        res._terms[k].c = a._terms[k0].c + b._terms[k1].c;
                        ++k;
                     }
                     ++k0; ++k1;
                  }
                  else if (order(a._terms[k0].x, b._terms[k1].x))
                  {
                     res._terms[k].x = a._terms[k0].x;
                     res._terms[k].c = a._terms[k0].c;
                     ++k0; ++k;
                  }
                  else
                  {
                     res._terms[k].x = b._terms[k1].x;
                     res._terms[k].c = b._terms[k1].c;
                     ++k1; ++k;
                  }
               }
               else if (k0 < a._nTerms)
               {
                  res._terms[k].x = a._terms[k0].x;
                  res._terms[k].c = a._terms[k0].c;
                  ++k0; ++k;
               }
               else
               {
                  res._terms[k].x = b._terms[k1].x;
                  res._terms[k].c = b._terms[k1].c;
                  ++k1; ++k;
               } // end if
            } // end while
            res._nTerms = k;
            return res;
         } // end operator+()

         Polynomial operator-() const
         {
            Polynomial const& a = *this;
            Polynomial res(a._nTerms);

            for (int k = 0; k < a._nTerms; ++k)
            {
               res._terms[k].c = -a._terms[k].c;
               res._terms[k].x = a._terms[k].x;
            } // end for (k)
            res._nTerms = a._nTerms;
            return res;
         }

         Polynomial operator*(Field b) const
         {
            Polynomial const& a = *this;
            Polynomial res(a._nTerms);

            for (int k = 0; k < a._nTerms; ++k)
            {
               res._terms[k].c = b * a._terms[k].c;
               res._terms[k].x = a._terms[k].x;
            } // end for (k)
            res._nTerms = a._nTerms;
            return res;
         }

         Polynomial operator*(Monomial<N> const& b) const
         {
            Polynomial const& a = *this;
            Polynomial res(a._nTerms);

            for (int k = 0; k < a._nTerms; ++k)
            {
               res._terms[k].c = a._terms[k].c;
               res._terms[k].x = a._terms[k].x * b;
            } // end for (k)
            res._nTerms = a._nTerms;
            return res;
         } // end operator*()

         Polynomial operator*(MonomialTerm<Field, N> const& b) const
         {
            Polynomial const& a = *this;
            Polynomial res(a._nTerms);

            for (int k = 0; k < a._nTerms; ++k)
            {
               res._terms[k].c = b.c * a._terms[k].c;
               res._terms[k].x = b.x * a._terms[k].x;
            } // end for (k)
            res._nTerms = a._nTerms;
            return res;
         } // end operator*()

         void removeLeadTerm()
         {
            for (int k = 0; k < _nTerms-1; ++k)
               _terms[k] = _terms[k+1];
            --_nTerms;
         }

         bool isZero() const
         {
            if (_nTerms == 0) return true;

            for (int k = 0; k < _nTerms; ++k)
            {
               if (_terms[0].c != Field(0)) return false;
            }
            return true;
         }

         Polynomial remainder(std::vector<Polynomial> const& F) const
         {
            using namespace std;

            Polynomial p(*this);
            Polynomial r;

            while (!p.isZero())
            {
               int j = 0;
               while (j < F.size())
               {
                  MonomialTerm<Field, N> const ltf = F[j].leadTerm();
                  MonomialTerm<Field, N> const ltp = p.leadTerm();
                  if (ltf.divides(ltp))
                  {
                     MonomialTerm<Field, N> t(ltp.c/ltf.c, ltp.x/ltf.x);
                     Polynomial g = -F[j] * t;
                     g.removeLeadTerm();
                     p.removeLeadTerm();
                     //cout << "(" << p << ") + (" << g << ") = " << p+g << endl;
                     p = p + g;
                     break;
                  }
                  ++j;
               } // end for (j)

               if (j == F.size())
               {
                  // No division occured
                  r = r + Polynomial(p.leadTerm());
                  p.removeLeadTerm();
               }
            } // end while
            return r;
         } // end remainder()

         Polynomial gaussianRemainder(std::vector<Polynomial> const& F) const
         {
            using namespace std;

            Polynomial p(*this);
            Polynomial r;

            while (!p.isZero())
            {
               int j = 0;
               while (j < F.size())
               {
                  MonomialTerm<Field, N> const ltf = F[j].leadTerm();
                  MonomialTerm<Field, N> const ltp = p.leadTerm();
                  if (ltf.x == ltp.x)
                  {
                     MonomialTerm<Field, N> t(ltp.c/ltf.c, ltp.x/ltf.x);
                     Polynomial g = -F[j] * t;
                     g.removeLeadTerm();
                     p.removeLeadTerm();
                     //cout << "(" << p << ") + (" << g << ") = " << p+g << endl;
                     p = p + g;
                     break;
                  }
                  ++j;
               } // end for (j)

               if (j == F.size())
               {
                  // No division occured
                  r = r + Polynomial(p.leadTerm());
                  p.removeLeadTerm();
               }
            } // end while
            return r;
         } // end gaussianRemainder()

         void collectMonomials(std::set<Monomial<N>, Order>& res) const
         {
            for (int k = 0; k < _nTerms; ++k)
               res.insert(_terms[k].x);
         } // end collectMonomials()

         std::set<Monomial<N>, Order> collectMonomials() const
         {
            std::set<Monomial<N>, Order> res;
            this->collectMonomials(res);
            return res;
         } // end collectMonomials()

         void collectNonLeadMonomials(std::set<Monomial<N>, Order>& res) const
         {
            for (int k = 1; k < _nTerms; ++k)
               res.insert(_terms[k].x);
         } // end collectNonLeadMonomials()

         Polynomial substituteVars(char const * srcVars, char const * dstVars) const
         {
            Polynomial res(*this);
            for (int k = 0; k < _nTerms; ++k)
               res._terms[k].x = _terms[k].x.substituteVars(srcVars, dstVars);

            TermOrder<Field, N, Order> order;
            std::sort(res._terms, res._terms + res._nTerms, order);
            return res;
         } // end substituteVars()


         int                      _nTerms;
         MonomialTerm<Field, N> * _terms;

      private:
         friend Polynomial<Field, N, Order> compute_Spair<>(Polynomial<Field, N, Order> const& a, Polynomial<Field, N, Order> const & b);

         Polynomial(int nAllocated)
            : _nTerms(0)
         {
            _terms = new MonomialTerm<Field, N>[nAllocated];
         }
   }; // end struct Polynomial

   template <int N>
   inline std::ostream&
   operator<<(std::ostream& os, Monomial<N> const& x)
   {
      static char const vars[] = "xyzwabcdefgh";

      if (x.isConstant())
      {
         os << "1";
         return os;
      }

      for (int i = 0; i < N; ++i)
      {
         if (x.exponent[i] > 1)
         {
            os << vars[i] << "^" << x.exponent[i];
         }
         else if (x.exponent[i] == 1)
         {
            os << vars[i];
         }
      } // end for (i)
      return os;
   } // end operator<<()

   template <typename Field, int N>
   inline std::ostream&
   operator<<(std::ostream& os, MonomialTerm<Field, N> const& t)
   {
      static char const vars[] = "xyzwabcdefgh";

      if (t.x.isConstant())
      {
         os << t.c;
         return os;
      }

      if (t.c != Field(1)) os << t.c;

      for (int i = 0; i < N; ++i)
      {
         if (t.x.exponent[i] > 1)
         {
            os << vars[i] << "^" << t.x.exponent[i];
         }
         else if (t.x.exponent[i] == 1)
         {
            os << vars[i];
         }
      } // end for (i)
      return os;
   } // end operator<<()

   template <typename Field, int N, typename Order>
   inline std::ostream&
   operator<<(std::ostream& os, Polynomial<Field, N, Order> const& p)
   {
      if (p._nTerms == 0)
      {
         os << "0";
         return os;
      }

      for (int k = 0; k < p._nTerms; ++k)
      {
         if (k > 0) os << " + ";

         MonomialTerm<Field, N> const& t = p._terms[k];
         os << t;
      } // end for (k)
      return os;
   } // end operator<<()

   inline float  multInverse(float a)  { return 1.0f / a; }
   inline double multInverse(double a) { return 1.0 / a; }
   inline long double multInverse(long double a) { return 1.0 / a; }

   template <typename Field, int N, typename Order>
   inline Polynomial<Field, N, Order>
   compute_Spair(Polynomial<Field, N, Order> const& a, Polynomial<Field, N, Order> const& b)
   {
      Order order;

      MonomialTerm<Field, N> const& lta = a.leadTerm();
      MonomialTerm<Field, N> const& ltb = b.leadTerm();

      Monomial<N> lcm = Monomial<N>::lcm(lta.x, ltb.x);

//       std::cout << "lta = " << lta << std::endl;
//       std::cout << "ltb = " << ltb << std::endl;
//       std::cout << "multInverse(lta.c) = " << multInverse(lta.c) << std::endl;
//       std::cout << "multInverse(ltb.c) = " << multInverse(ltb.c) << std::endl;

      MonomialTerm<Field, N> fa, fb;
      fa.c = multInverse(lta.c);
      fa.x = lcm / lta.x;

      fb.c = -multInverse(ltb.c);
      fb.x = lcm / ltb.x;

//       std::cout << "fa = " << fa << std::endl;
//       std::cout << "fb = " << fb << std::endl;

      Polynomial<Field, N, Order> const ma = a * fa;
      Polynomial<Field, N, Order> const mb = b * fb;

      Polynomial<Field, N, Order> res(a._nTerms + b._nTerms - 2); // The leading terms cancel

      int k0 = 1, k1 = 1; // We start at 1, since the leading terms cancel exactly
      int k = 0;

      while (k0 < ma._nTerms || k1 < mb._nTerms)
      {
         if (k0 < ma._nTerms && k1 < mb._nTerms)
         {
            if (ma._terms[k0].x == mb._terms[k1].x)
            {
               Field coeff = ma._terms[k0].c + mb._terms[k1].c;
               if (coeff != Field(0))
               {
                  res._terms[k].x = ma._terms[k0].x;
                  res._terms[k].c = ma._terms[k0].c + mb._terms[k1].c;
                  ++k;
               }
               ++k0; ++k1;
            }
            else if (order(ma._terms[k0].x, mb._terms[k1].x))
            {
               res._terms[k].x = ma._terms[k0].x;
               res._terms[k].c = ma._terms[k0].c;
               ++k0; ++k;
            }
            else
            {
               res._terms[k].x = mb._terms[k1].x;
               res._terms[k].c = mb._terms[k1].c;
               ++k1; ++k;
            }
         }
         else if (k0 < ma._nTerms)
         {
            res._terms[k].x = ma._terms[k0].x;
            res._terms[k].c = ma._terms[k0].c;
            ++k0; ++k;
         }
         else
         {
            res._terms[k].x = mb._terms[k1].x;
            res._terms[k].c = mb._terms[k1].c;
            ++k1; ++k;
         } // end if
      } // end while
      res._nTerms = k;

      TermOrder<Field, N, Order> torder;

      std::sort(res._terms, res._terms+k, torder);

      return res;
   } // end compute_Spair()

   template <typename Field, int N, typename Order>
   inline void
   compute_StdBasis_Buchberger(std::vector<Polynomial<Field, N, Order> > const& F, std::vector<Polynomial<Field, N, Order> >& G)
   {
      typedef Polynomial<Field, N, Order> Poly;
      G = F;
      while (1)
      {
         std::vector<Poly> const G2 = G;
         for (int j1 = 0; j1 < G2.size(); ++j1)
            for (int j2 = j1+1; j2 < G2.size(); ++j2)
            {
               Poly s = compute_Spair(G2[j1], G2[j2]);
               //std::cout << "spair = " << s << std::endl;
               //s = s.remainder(G2);
               s = s.remainder(G);
               //std::cout << "rem = " << s << std::endl;
               if (!s.isZero())
               {
                  //std::cout << "Adding " << s << " to the std basis." << std::endl;
                  Field c = Field(1) / s.leadTerm().c;
                  G.push_back(s * c);
               }
            }
         if (G.size() == G2.size()) break;
      } // end while
   } // end compute_StdBasis_Buchberger()

   template <typename Field, int N, typename Order>
   inline void
   reducePolynomialBasis(std::vector<Polynomial<Field, N, Order> >& F)
   {
      using namespace std;

      typedef Polynomial<Field, N, Order> Poly;

      typedef set<Monomial<N>, Order> MonoSet;
      typedef map<Monomial<N>, int, Order> MonoMap;

      MonoSet allMonomials;
      for (size_t j = 0; j < F.size(); ++j)
      {
         Poly const& f = F[j];
         for (int k = 0; k < f._nTerms; ++k)
            allMonomials.insert(f._terms[k].x);
      }

      MonoMap monomialPosMap;
      vector<Monomial<N> > posMonomialMap(allMonomials.size());
      int pos = 0;
      for (typename MonoSet::const_iterator p = allMonomials.begin(); p != allMonomials.end(); ++p, ++pos)
      {
         monomialPosMap.insert(make_pair(*p, pos));
         posMonomialMap[pos] = *p;
      }

      Matrix<Field> A(F.size(), allMonomials.size());
      makeZeroMatrix(A);

      for (size_t j = 0; j < F.size(); ++j)
      {
         Poly const& f = F[j];
         for (int k = 0; k < f._nTerms; ++k)
         {
            typename MonoMap::const_iterator p = monomialPosMap.find(f._terms[k].x);
            if (p != monomialPosMap.end())
               A[j][(*p).second] = f._terms[k].c;
         }
      } // end for (j)
      //cout << "A = "; displayMatrix(A); cout << endl;

      convertToRowEchelonMatrix(A);
      convertToReducedRowEchelonMatrix(A);
      //cout << "A = "; displayMatrix(A);

      int const nPolys = F.size();

      F.clear();
      for (int j = 0; j < nPolys; ++j)
      {
         Poly const f = Poly(A.num_cols(), A[j], &posMonomialMap[0]);
         if (!f.isZero()) F.push_back(f);
      }
   } // end reducePolynomialBasis()

//**********************************************************************

   template <typename Field, int N, typename Order>
   inline std::vector<Field>
   checkLinearDependency_FGLM(std::vector<Polynomial<Field, N, Order> > const& F)
   {
      using namespace std;

      typedef Polynomial<Field, N, Order> Poly;

      typedef set<Monomial<N>, Order> MonoSet;
      typedef map<Monomial<N>, int, Order> MonoMap;

      MonoSet allMonomials;
      for (size_t j = 0; j < F.size(); ++j)
      {
         Poly const& f = F[j];
         for (int k = 0; k < f._nTerms; ++k)
            allMonomials.insert(f._terms[k].x);
      }

      MonoMap monomialPosMap;
      vector<Monomial<N> > posMonomialMap(allMonomials.size());
      int pos = 0;
      for (typename MonoSet::const_iterator p = allMonomials.begin(); p != allMonomials.end(); ++p, ++pos)
      {
         monomialPosMap.insert(make_pair(*p, pos));
         posMonomialMap[pos] = *p;
      }

      Matrix<Field> A(F.size(), 2*allMonomials.size());
      makeZeroMatrix(A);

      for (size_t j = 0; j < F.size(); ++j)
      {
         Poly const& f = F[j];
         for (int k = 0; k < f._nTerms; ++k)
         {
            typename MonoMap::const_iterator p = monomialPosMap.find(f._terms[k].x);
            if (p != monomialPosMap.end())
               A[j][(*p).second] = f._terms[k].c;
         }
      } // end for (j)

      for (size_t j = 0; j < F.size(); ++j)
         A[j][j+allMonomials.size()] = 1;

      //cout << "A = "; displayMatrix(A); cout << endl;
      convertToRowEchelonMatrix(A);
      convertToReducedRowEchelonMatrix(A);
      //cout << "A = "; displayMatrix(A);

      vector<Field> coeffs;

      int const lastRow = A.num_rows() - 1;
      bool isDependent = true;
      for (size_t j = 0; j < allMonomials.size(); ++j)
      {
         if (A[lastRow][j] != Field(0))
            return coeffs;
      } // end for (j)

      for (size_t j = 0; j < F.size()-1; ++j)
         coeffs.push_back(A[lastRow][j + allMonomials.size()]);

      return coeffs;
   } // end checkLinearDependency_FGLM()

   template <typename Field, int N, typename Order>
   inline void
   convertBasisToLexOrder_FGLM(std::vector<Polynomial<Field, N, Order> > const& G,
                               std::vector<Polynomial<Field, N, MonomialOrderLex<N> > >& Glex, bool debug = false)
   {
      using namespace std;

      typedef Monomial<N>            M;
      typedef MonomialTerm<Field, N> MT;
      typedef Polynomial<Field, N, Order> Poly;
      typedef Polynomial<Field, N, MonomialOrderLex<N> > PolyLex;

      vector<M>    Blex;
      vector<Poly> remBlex;

      Monomial<N> curMonomial; // start with 1
      while (1)
      {
         if (debug) cout << "Checking " << curMonomial << endl;

         Poly rem = Poly(curMonomial).remainder(G);
         vector<Poly> remainders(remBlex);
         remainders.push_back(rem);

         if (debug) cout << "rem = " << rem << endl;

         vector<Field> const coeffs = checkLinearDependency_FGLM(remainders);

         if (!coeffs.empty())
         {
            if (debug) cout << "remainder is linearly dependent." << endl;
            PolyLex g(curMonomial);
            Poly grem(rem);
            for (size_t j = 0; j < coeffs.size(); ++j)
            {
               if (coeffs[j] != Field(0))
               {
                  g = g + MT(coeffs[j], Blex[j]);
                  grem = grem + remBlex[j]*coeffs[j];
               }
            }
            Glex.push_back(g);
            if (debug) cout << "Added " << g << endl;
            if (debug) cout << "grem = " << grem << endl;

            MT const ltg = g.leadTerm();
            bool stop = true;
            for (int i = 1; i < N; ++i)
            {
               if (ltg.x.exponent[i] > 0) stop = false;
            }
            if (stop) break;
         }
         else
         {
            if (debug) cout << "remainder is linearly independent." << endl;
            Blex.push_back(curMonomial);
            remBlex.push_back(rem);
         } // end if

         if (debug) 
         {
            cout << "current Glex:" << endl;
            for (size_t j = 0; j < Glex.size(); ++j)
               cout << "G[" << j << "] = " << Glex[j] << endl;
            cout << "current Blex:" << endl;
            for (size_t j = 0; j < Blex.size(); ++j)
               cout << "B[" << j << "] = " << Blex[j] << endl;
            cout << "current remBlex:" << endl;
            for (size_t j = 0; j < remBlex.size(); ++j)
               cout << "remB[" << j << "] = " << remBlex[j] << endl;
         }

         // Next monomial:
         {
            int i = N-1;
            while (i >= 0)
            {
               M nextM = curMonomial;
               ++nextM.exponent[i];
               if (debug) cout << "Testing next monomial " << nextM << endl;
               bool success = true;
               for (size_t j = 0; j < Glex.size(); ++j)
               {
                  if (Glex[j].leadTerm().x.divides(nextM))
                  {
                     success = false;
                     break;
                  }
               } // end for (j)
               if (success)
               {
                  if (debug) cout << "Next monomial is " << nextM << endl;
                  curMonomial = nextM;
                  break;
               }
               else
               {
                  --i;
                  curMonomial = M(i);
               }
            } // end while
         } // end scope
      } // end while
   } // end convertBasisToLexOrder_FGLM()

} // end namespace V3D

#endif
