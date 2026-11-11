// fdlibm cbrt/hypot/exp/log/log1p/expm1/sinh/cosh/tanh (see bolt_fdlibm_base.h).
#pragma once

#include <cassert>

#include "bolt/kernels/bolt_fdlibm_base.h"

// Java does not fuse a*b+c; neither may these routines (a fused result differs in the last bit).
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace bolt::kernels::fdlibm {

namespace Cbrt {
        inline constexpr int32_t B1 = 715094163; 
        inline constexpr int32_t B2 = 696219795; 
        inline constexpr double C =  0x1.15f15f15f15f1p-1;
        inline constexpr double D = -0x1.691de2532c834p-1;
        inline constexpr double E =  0x1.6a0ea0ea0ea0fp0;
        inline constexpr double F =  0x1.9b6db6db6db6ep0;
        inline constexpr double G =  0x1.6db6db6db6db7p-2;
        inline double compute(double x) {
            double  t = 0.0;
            double sign;
            if (x == 0.0 || !std::isfinite(x))
                return x;
            sign = (x < 0.0) ? -1.0:  1.0;
            x = std::fabs(x);
            if (x < 0x1.0p-1022) {
                t = 0x1.0p54;
                t *= x;
                t = hi_w(t, hi_w(t)/3 + B2);
            } else {
                int32_t hx = hi_w(x);
                t = hi_w(t, hx/3 + B1);
            }
            double  r, s, w;
            r = t * t/x;
            s = C + r*t;
            t *= G + F/(s + E + D/s);
            t = lo_w(t, 0);
            t = hi_w(t, hi_w(t) + 0x00000001);
            s = t * t;
            r = x / s;
            w = t + t;
            r = (r - t)/(w + r);
            t = t + t*r;
            return sign * t;
        }
}  // namespace
namespace Hypot {
        inline constexpr double TWO_MINUS_600 = 0x1.0p-600;
        inline constexpr double TWO_PLUS_600 = 0x1.0p+600;
        inline double compute(double x, double y) {
            double a = std::fabs(x);
            double b = std::fabs(y);
            if (!std::isfinite(a) || !std::isfinite(b)) {
                if (a == INFINITY || b == INFINITY)
                    return INFINITY;
                else
                    return a + b;
            }
            if (b > a) {
                double tmp = a;
                a = b;
                b = tmp;
            }
            assert(a >= b);
            int32_t ha = hi_w(a);
            int32_t hb = hi_w(b);
            if ((ha - hb) > 0x3c00000) {
                return a + b;
            }
            int32_t k = 0;
            if (a > 0x1.00000ffffffffp500) {
                ha -= 0x25800000;
                hb -= 0x25800000;
                a = a * TWO_MINUS_600;
                b = b * TWO_MINUS_600;
                k += 600;
            }
            double t1, t2;
            if (b < 0x1.0p-500) {
                if (b < std::numeric_limits<double>::min()) {
                    if (b == 0.0)
                        return a;
                    t1 = 0x1.0p1022;
                    b *= t1;
                    a *= t1;
                    k -= 1022;
                } else {
                    ha += 0x25800000;
                    hb += 0x25800000;
                    a = a * TWO_PLUS_600;
                    b = b * TWO_PLUS_600;
                    k -= 600;
                }
            }
            double w = a - b;
            if (w > b) {
                t1 = 0;
                t1 = hi_w(t1, ha);
                t2 = a - t1;
                w  = std::sqrt(t1*t1 - (b*(-b) - t2 * (a + t1)));
            } else {
                double y1, y2;
                a  = a + a;
                y1 = 0;
                y1 = hi_w(y1, hb);
                y2 = b - y1;
                t1 = 0;
                t1 = hi_w(t1, ha + 0x00100000);
                t2 = a - t1;
                w  = std::sqrt(t1*y1 - (w*(-w) - (t1*y2 + t2*b)));
            }
            if (k != 0) {
                return pow2d(k) * w;
            } else
                return w;
        }
}  // namespace
namespace Exp {
        inline constexpr double half[] = {0.5, -0.5,};
        inline constexpr double huge = 1.0e+300;
        inline constexpr double twom1000 =     0x1.0p-1000;
        inline constexpr double o_threshold =  0x1.62e42fefa39efp9;
        inline constexpr double u_threshold = -0x1.74910d52d3051p9;
        inline constexpr double ln2HI[] ={  0x1.62e42feep-1,
                                                 -0x1.62e42feep-1};
        inline constexpr double ln2LO[] ={  0x1.a39ef35793c76p-33,
                                                 -0x1.a39ef35793c76p-33};
        inline constexpr double invln2 =      0x1.71547652b82fep0;
        inline constexpr double P1 =  0x1.555555555553ep-3;
        inline constexpr double P2 = -0x1.6c16c16bebd93p-9;
        inline constexpr double P3 =  0x1.1566aaf25de2cp-14;
        inline constexpr double P4 = -0x1.bbd41c5d26bf1p-20;
        inline constexpr double P5 =  0x1.6376972bea4d0p-25;
        inline double compute(double x) {
            double y;
            double hi = 0.0;
            double lo = 0.0;
            double c;
            double t;
            int32_t k = 0;
            int32_t xsb;
             int32_t hx;
            hx  = hi_w(x);  
            xsb = (hx >> 31) & 1;               
            hx &= EXP_SIGNIF_BITS;              
            if (hx >= 0x40862E42) {                  
                if (hx >= 0x7ff00000) {
                    if (((hx & 0xfffff) | lo_w(x)) != 0)
                        return x + x;                
                    else
                        return (xsb == 0) ? x : 0.0;    
                }
                if (x > o_threshold)
                    return huge * huge; 
                if (x < u_threshold)
                    return twom1000 * twom1000; 
            }
            if (hx > 0x3fd62e42) {           
                if(hx < 0x3FF0A2B2) {       
                    hi = x - ln2HI[xsb];
                    lo=ln2LO[xsb];
                    k = 1 - xsb - xsb;
                } else {
                    k  = (int32_t)(invln2 * x + half[xsb]);
                    t  = k;
                    hi = x - t*ln2HI[0];    
                    lo = t*ln2LO[0];
                }
                x  = hi - lo;
            } else if (hx < 0x3e300000)  {     
                if (huge + x > 1.0)
                    return 1.0 + x; 
            } else {
                k = 0;
            }
            t  = x * x;
            c  = x - t*(P1 + t*(P2 + t*(P3 + t*(P4 + t*P5))));
            if (k == 0)
                return 1.0 - ((x*c)/(c - 2.0) - x);
            else
                y = 1.0 - ((lo - (x*c)/(2.0 - c)) - hi);
            if(k >= -1021) {
                y = hi_w(y, hi_w(y) + (k << 20)); 
                return y;
            } else {
                y = hi_w(y, hi_w(y) + ((k + 1000) << 20)); 
                return y * twom1000;
            }
        }
}  // namespace
namespace Log {
        inline constexpr double             ln2_hi = 0x1.62e42feep-1,
            ln2_lo = 0x1.a39ef35793c76p-33,
            Lg1    = 0x1.5555555555593p-1,
            Lg2    = 0x1.999999997fa04p-2,
            Lg3    = 0x1.2492494229359p-2,
            Lg4    = 0x1.c71c51d8e78afp-3,
            Lg5    = 0x1.7466496cb03dep-3,
            Lg6    = 0x1.39a09d078c69fp-3,
            Lg7    = 0x1.2f112df3e5244p-3;
        inline double compute(double x) {
            double hfsq, f, s, z, R, w, t1, t2, dk;
            int32_t k, hx, i, j;
             int32_t lx;
            hx = hi_w(x);
            lx = lo_w(x);
            k=0;
            if (hx < 0x00100000) {
                if (((hx & EXP_SIGNIF_BITS) | lx) == 0) {
                    return -TWO54/0.0;
                }
                if (hx < 0) {
                    return (x - x)/0.0;
                }
                k -= 54;
                x *= TWO54;
                hx = hi_w(x);
            }
            if (hx >= EXP_BITS) {
                return x + x;
            }
            k += (hx >> 20) - 1023;
            hx &= 0x000fffff;
            i = (hx + 0x95f64) & 0x100000;
            x =hi_w(x, hx | (i ^ 0x3ff00000));
            k += (i >> 20);
            f = x - 1.0;
            if ((0x000fffff & (2 + hx)) < 3) {
                if (f == 0.0) {
                    if (k == 0) {
                        return 0.0;
                    } else {
                        dk = (double)k;
                        return dk*ln2_hi + dk*ln2_lo;
                    }
                }
                R = f*f*(0.5 - 0.33333333333333333*f);
                if (k == 0) {
                    return f - R;
                } else {
                    dk = (double)k;
                    return dk*ln2_hi - ((R - dk*ln2_lo) - f);
                }
            }
            s = f/(2.0 + f);
            dk = (double)k;
            z = s*s;
            i = hx - 0x6147a;
            w = z*z;
            j = 0x6b851 - hx;
            t1= w*(Lg2 + w*(Lg4 + w*Lg6));
            t2= z*(Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7)));
            i |= j;
            R = t2 + t1;
            if (i > 0) {
                hfsq = 0.5*f*f;
                if (k == 0) {
                    return f-(hfsq - s*(hfsq + R));
                } else {
                    return dk*ln2_hi - ((hfsq - (s*(hfsq + R) + dk*ln2_lo)) - f);
                }
            } else {
                if (k == 0) {
                    return f - s*(f - R);
                } else {
                    return dk*ln2_hi - ((s*(f - R) - dk*ln2_lo) - f);
                }
            }
        }
}  // namespace
namespace Log1p {
        inline constexpr double ln2_hi = 0x1.62e42feep-1;
        inline constexpr double ln2_lo = 0x1.a39ef35793c76p-33;
        inline constexpr double Lp1 = 0x1.5555555555593p-1;
        inline constexpr double Lp2 = 0x1.999999997fa04p-2;
        inline constexpr double Lp3 = 0x1.2492494229359p-2;
        inline constexpr double Lp4 = 0x1.c71c51d8e78afp-3;
        inline constexpr double Lp5 = 0x1.7466496cb03dep-3;
        inline constexpr double Lp6 = 0x1.39a09d078c69fp-3;
        inline constexpr double Lp7 = 0x1.2f112df3e5244p-3;
        inline double compute(double x) {
            double hfsq, f=0, c=0, s, z, R, u;
            int32_t k, hx, hu=0, ax;
            hx = hi_w(x);           
            ax = hx & EXP_SIGNIF_BITS;
            k = 1;
            if (hx < 0x3FDA827A) {                  
                if (ax >= 0x3ff00000) {             
                    if (x == -1.0) 
                        return -INFINITY;
                    else
                        return std::numeric_limits<double>::quiet_NaN();           
                }
                if (ax < 0x3e200000) {                
                    if (TWO54 + x > 0.0                
                       && ax < 0x3c900000)            
                        return x;
                    else
                        return x - x*x*0.5;
                }
                if (hx > 0 || hx <= static_cast<int32_t>(0xbfd2bec3u)) { 
                    k=0;
                    f=x;
                    hu=1;
                }
            }
            if (hx >= EXP_BITS) {
                return x + x;
            }
            if (k != 0) {
                if (hx < 0x43400000) {
                    u  = 1.0 + x;
                    hu = hi_w(u);           
                    k  = (hu >> 20) - 1023;
                    c  = (k > 0)? 1.0 - (u-x) : x-(u-1.0); 
                    c /= u;
                } else {
                    u  = x;
                    hu = hi_w(u);           
                    k  = (hu >> 20) - 1023;
                    c  = 0;
                }
                hu &= 0x000fffff;
                if (hu < 0x6a09e) {
                    u = hi_w(u, hu | 0x3ff00000);       
                } else {
                    k += 1;
                    u = hi_w(u, hu | 0x3fe00000);       
                    hu = (0x00100000 - hu) >> 2;
                }
                f = u - 1.0;
            }
            hfsq = 0.5*f*f;
            if (hu == 0) {     
                if (f == 0.0) {
                    if (k == 0) {
                        return 0.0;
                    } else {
                        c += k * ln2_lo;
                        return k * ln2_hi + c;
                    }
                }
                R = hfsq * (1.0 - 0.66666666666666666*f);
                if (k == 0) {
                    return f - R;
                } else {
                    return k * ln2_hi - ((R-(k * ln2_lo+c)) - f);
                }
            }
            s = f/(2.0 + f);
            z = s * s;
            R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z*Lp7))))));
            if (k == 0) {
                return f - (hfsq - s*(hfsq + R));
            } else {
                return k * ln2_hi - ((hfsq - (s*(hfsq + R) + (k * ln2_lo+c))) - f);
            }
        }
}  // namespace
namespace Expm1 {
        inline constexpr double huge =  1.0e+300;
        inline constexpr double tiny =  1.0e-300;
        inline constexpr double o_threshold =  0x1.62e42fefa39efp9;
        inline constexpr double ln2_hi =  0x1.62e42feep-1;
        inline constexpr double ln2_lo =  0x1.a39ef35793c76p-33;
        inline constexpr double invln2 =  0x1.71547652b82fep0;
        inline constexpr double Q1 = -0x1.11111111110f4p-5;
        inline constexpr double Q2 =  0x1.a01a019fe5585p-10;
        inline constexpr double Q3 = -0x1.4ce199eaadbb7p-14;
        inline constexpr double Q4 =  0x1.0cfca86e65239p-18;
        inline constexpr double Q5 = -0x1.afdb76e09c32dp-23;
        inline double compute(double x) {
            double y, hi, lo, c=0, t, e, hxs, hfx, r1;
            int32_t k, xsb;
             int32_t hx;
            hx  = hi_w(x);
            xsb = hx & SIGN_BIT;
            y = std::fabs(x);
            hx &= EXP_SIGNIF_BITS;
            if (hx >= 0x4043687A) {
                if (hx >= 0x40862E42) {
                    if (hx >= 0x7ff00000) {
                        if (((hx & 0xfffff) | lo_w(x)) != 0) {
                            return x + x;
                        } else {
                            return (xsb == 0)? x : -1.0;
                        }
                    }
                    if (x > o_threshold) {
                        return huge*huge;
                    }
                }
                if (xsb != 0) {
                    if (x + tiny < 0.0) {
                        return tiny - 1.0;
                    }
                }
            }
            if (hx > 0x3fd62e42) {
                if (hx < 0x3FF0A2B2) {
                    if (xsb == 0) {
                        hi = x - ln2_hi;
                        lo = ln2_lo;
                        k =  1;
                    } else {
                        hi = x + ln2_hi;
                        lo = -ln2_lo;
                        k = -1;
                    }
                } else {
                    k  = (int32_t)(invln2*x + ((xsb == 0) ? 0.5 : -0.5));
                    t  = k;
                    hi = x - t*ln2_hi;
                    lo = t*ln2_lo;
                }
                x  = hi - lo;
                c  = (hi - x) - lo;
            } else if (hx < 0x3c900000) {
                t = huge + x;
                return x - (t - (huge + x));
            } else {
                k = 0;
            }
            hfx = 0.5*x;
            hxs = x*hfx;
            r1 = 1.0 + hxs*(Q1 + hxs*(Q2 + hxs*(Q3 + hxs*(Q4 + hxs*Q5))));
            t  = 3.0 - r1*hfx;
            e  = hxs *((r1 - t)/(6.0 - x*t));
            if (k == 0) {
                return x - (x*e - hxs);
            } else {
                e  = (x*(e - c) - c);
                e -= hxs;
                if (k == -1) {
                    return 0.5*(x - e) - 0.5;
                }
                if (k == 1) {
                    if (x < -0.25) {
                        return -2.0*(e - (x + 0.5));
                    } else {
                        return 1.0 + 2.0*(x - e);
                    }
                }
                if (k <= -2 || k > 56) {
                    y = 1.0 - (e - x);
                    y = hi_w(y, hi_w(y) + (k << 20));
                    return y - 1.0;
                }
                t = 1.0;
                if (k < 20) {
                    t = hi_w(t, 0x3ff00000 - (0x200000 >> k));
                    y = t - ( e - x);
                    y = hi_w(y, hi_w(y) + (k << 20));
                } else {
                    t = hi_w(t, ((0x3ff - k) << 20));
                    y = x - (e + t);
                    y += 1.0;
                    y = hi_w(y, hi_w(y) + (k << 20));
                }
            }
            return y;
        }
}  // namespace
namespace Sinh {
        inline constexpr double shuge = 1.0e307;
        inline double compute(double x) {
            double t, w, h;
            int32_t ix, jx;
             int32_t lx;
            jx = hi_w(x);
            ix = jx & EXP_SIGNIF_BITS;
            if (ix >= EXP_BITS) {
                return x + x;
            }
            h = 0.5;
            if (jx < 0) {
                h = -h;
            }
            if (ix < 0x40360000) {
                if (ix < 0x3e300000)
                    if (shuge + x > 1.0) {
                        return x;
                    }
                t = Expm1::compute(std::fabs(x));
                if (ix < 0x3ff00000) {
                    return h*(2.0 * t - t*t/(t + 1.0));
                }
                return h*(t + t/(t + 1.0));
            }
            if (ix < 0x40862E42) {
                return h*Exp::compute(std::fabs(x));
            }
            lx = lo_w(x);
            if (ix < 0x408633CE ||
                ((ix == 0x408633ce) &&
                 ((static_cast<uint32_t>(lx) <= static_cast<uint32_t>(0x8fb9f87d))))) {
                w = Exp::compute(0.5 * std::fabs(x));
                t = h * w;
                return t * w;
            }
            return x * shuge;
        }
}  // namespace
namespace Cosh {
        inline constexpr double huge = 1.0e300;
        inline double compute(double x) {
            double t, w;
            int32_t ix;
             int32_t lx;
            ix = hi_w(x);
            ix &= EXP_SIGNIF_BITS;
            if (ix >= EXP_BITS) {
                return x*x;
            }
            if (ix < 0x3fd62e43) {
                t = Expm1::compute(std::fabs(x));
                w = 1.0 + t;
                if (ix < 0x3c800000) {
                    return w;
                }
                return 1.0 + (t * t)/(w + w);
            }
            if (ix < 0x40360000) {
                t = Exp::compute(std::fabs(x));
                return 0.5*t + 0.5/t;
            }
            if (ix < 0x40862E42) {
                return 0.5*Exp::compute(std::fabs(x));
            }
            lx = lo_w(x);
            if (ix<0x408633CE ||
                ((ix == 0x408633ce) &&
                 ((static_cast<uint32_t>(lx) <= static_cast<uint32_t>(0x8fb9f87d))))) {
                w = Exp::compute(0.5*std::fabs(x));
                t = 0.5*w;
                return t*w;
            }
            return huge*huge;
        }
}  // namespace
namespace Tanh {
        inline constexpr double tiny = 1.0e-300;
        inline double compute(double x) {
            double t, z;
            int32_t jx, ix;
            jx = hi_w(x);
            ix = jx & EXP_SIGNIF_BITS;
            if (ix >= EXP_BITS) {
                if (jx >= 0) {
                    return 1.0/x + 1.0;
                } else {
                    return 1.0/x - 1.0;
                }
            }
            if (ix < 0x40360000) {
                if (ix<0x3c800000)
                    return x*(1.0 + x);
                if (ix>=0x3ff00000) {
                    t = Expm1::compute(2.0*std::fabs(x));
                    z = 1.0 - 2.0/(t + 2.0);
                } else {
                    t = Expm1::compute(-2.0*std::fabs(x));
                    z= -t/(t + 2.0);
                }
            } else {
                z = 1.0 - tiny;
            }
            return (jx >= 0)? z: -z;
        }
}  // namespace

}  // namespace bolt::kernels::fdlibm

#if defined(__clang__)
#pragma clang fp contract(on)
#endif
