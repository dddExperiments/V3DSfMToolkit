// -*- C++ -*-

#ifndef V3D_PRIME_FIELD_H
#define V3D_PRIME_FIELD_H

#include <iostream>

namespace V3D
{

   template <typename Num>
   inline void
   extended_GCD(Num a, Num b, Num& s, Num& t)
   {
      s = 0; t = 1;
      Num lastS = 1, lastT = 0;

      Num tmp;

      while (b != 0)
      {
         tmp = b;
         Num quot = a / b;
         b = a % b; a = tmp;

         tmp = s; s = lastS - quot*s; lastS = tmp;

         tmp = t; t = lastT - quot*t; lastT = tmp;
      } // end while

      s = lastS; t = lastT;
   } // end extended_GCD()

   // p must be a prime number to make the ring of residual classes Z/pZ a field.
   // Because of the trivial implementation p should be small to avoid overflow in the multiplication, i.e. p < 32768.
   template <int p>
   struct PrimeField
   {
         PrimeField()
            : _x(0)
         { }

         PrimeField(int x)
         {
            _x = x % p;
            if (_x < 0) _x += p;
         }

         operator int() const
         {
            return _x;
         }

         bool operator==(PrimeField b) const
         {
            return _x == b._x;
         }

         bool operator!=(PrimeField b) const
         {
            return _x != b._x;
         }

         bool operator<(PrimeField b) const
         {
            return _x < b._x;
         }

         bool operator>(PrimeField b) const
         {
            return _x > b._x;
         }

         PrimeField inverse() const
         {
            int s, t;
            extended_GCD(_x, p, s, t);
            return PrimeField(s);
         } // end inverse()

         PrimeField operator+() const
         {
            return *this;
         }

         PrimeField operator-() const
         {
            return PrimeField(-_x);
         }

         PrimeField operator+(PrimeField const b) const
         {
            return PrimeField(_x + b._x);
         }

         PrimeField operator-(PrimeField const b) const
         {
            return PrimeField(_x - b._x);
         }

         PrimeField operator*(PrimeField const b) const
         {
            return PrimeField(_x * b._x);
         }

         PrimeField operator/(PrimeField const b) const
         {
            return PrimeField(_x * b.inverse());
         }

      private:
         int _x;
   }; // end struct PrimeField

   template <int p>
   inline PrimeField<p> multInverse(PrimeField<p> x)
   {
      int a = x;
      int b = p;

      int s = 0, t = 1;
      int lastS = 1, lastT = 0;
      int tmp;

      while (b != 0)
      {
         tmp = b;
         int quot = a / b;
         b = a % b; a = tmp;

         tmp = s; s = lastS - quot*s; lastS = tmp;

         tmp = t; t = lastT - quot*t; lastT = tmp;
      } // end while

      return PrimeField<p>(lastS);
   } // end multInverse()

   template <int p>
   inline std::ostream&
   operator<<(std::ostream& os, PrimeField<p> const& x)
   {
#if 0
      os << int(x);
#else
      // This code path outputs numbers in Z/pZ compatible with
      // Macaulay's and Singular's output format.
      int v = int(x);
      os << ((v < p/2) ? v : (v-p));
#endif
      return os;
   }

} // end namespace V3D

namespace std
{

   template <int p> inline V3D::PrimeField<p> abs(V3D::PrimeField<p> const x) { return x; }

} // end namespace std

#endif
