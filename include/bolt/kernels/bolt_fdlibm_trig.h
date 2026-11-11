// fdlibm sin/cos/tan/asin/acos/atan/atan2 (see bolt_fdlibm_base.h).
#pragma once

#include "bolt/kernels/bolt_fdlibm_base.h"

// Java does not fuse a*b+c; neither may these routines (a fused result differs in the last bit).
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace bolt::kernels::fdlibm {

namespace RemPio2 { inline int32_t ieee754_rem_pio2(double x, double* y); }
namespace Sin { inline double kernel_sin(double x, double y, int32_t iy); }
namespace Cos { inline double kernel_cos(double x, double y); }
namespace Tan { inline double kernel_tan(double x, double y, int32_t iy); }
namespace KernelRemPio2 {
inline int32_t kernel_rem_pio2(double* x, double* y, int32_t e0, int32_t nx, int32_t prec, const int32_t* ipio2);
}

namespace Sin {
        inline double compute(double x) {
            double y[2] = {};
            double z = 0.0;
            int32_t n, ix;
            ix = hi_w(x);
            ix &= EXP_SIGNIF_BITS;
            if (ix <= 0x3fe921fb) {
                return kernel_sin(x, z, 0);
            } else if (ix >= EXP_BITS) {
                return x - x;
            } else {
                n = RemPio2::ieee754_rem_pio2(x, y);
                switch(n & 3) {
                case 0: return  Sin::kernel_sin(y[0], y[1], 1);
                case 1: return  Cos::kernel_cos(y[0], y[1]);
                case 2: return -Sin::kernel_sin(y[0], y[1], 1);
                default:
                    return -Cos::kernel_cos(y[0], y[1]);
                }
            }
        }
        inline constexpr double             S1  = -0x1.5555555555549p-3,
            S2  =  0x1.111111110f8a6p-7,
            S3  = -0x1.a01a019c161d5p-13,
            S4  =  0x1.71de357b1fe7dp-19,
            S5  = -0x1.ae5e68a2b9cebp-26,
            S6  =  0x1.5d93a5acfd57cp-33;
        inline double kernel_sin(double x, double y, int32_t iy) {
            double z, r, v;
            int32_t ix;
            ix = hi_w(x) & EXP_SIGNIF_BITS;
            if (ix < 0x3e400000) {
                if ((int32_t)x == 0)
                    return x;
            }
            z       =  x*x;
            v       =  z*x;
            r       =  S2 + z*(S3 + z*(S4 + z*(S5 + z*S6)));
            if (iy == 0) {
                return x + v*(S1 + z*r);
            } else {
                return x - ((z*(0.5*y - v*r) - y) - v*S1);
            }
        }
}  // namespace
namespace Cos {
        inline double compute(double x) {
            double y[2] = {};
            double z = 0.0;
            int32_t n, ix;
            ix = hi_w(x);
            ix &= EXP_SIGNIF_BITS;
            if (ix <= 0x3fe921fb) {
                return kernel_cos(x, z);
            } else if (ix >= EXP_BITS) {
                return x - x;
            } else {
                n = RemPio2::ieee754_rem_pio2(x,y);
                switch (n & 3) {
                case 0: return  Cos::kernel_cos(y[0], y[1]);
                case 1: return -Sin::kernel_sin(y[0], y[1],1);
                case 2: return -Cos::kernel_cos(y[0], y[1]);
                default:
                    return  Sin::kernel_sin(y[0], y[1], 1);
                }
            }
        }
        inline constexpr double             C1  =  0x1.555555555554cp-5,
            C2  = -0x1.6c16c16c15177p-10,
            C3  =  0x1.a01a019cb159p-16,
            C4  = -0x1.27e4f809c52adp-22,
            C5  =  0x1.1ee9ebdb4b1c4p-29,
            C6  = -0x1.8fae9be8838d4p-37;
        inline double kernel_cos(double x, double y) {
            double a, hz, z, r, qx = 0.0;
            int32_t ix;
            ix = hi_w(x) & EXP_SIGNIF_BITS;
            if (ix < 0x3e400000) {
                if (((int32_t)x) == 0) {
                    return 1.0;
                }
            }
            z  = x*x;
            r  = z*(C1 + z*(C2 + z*(C3 + z*(C4 + z*(C5 + z*C6)))));
            if (ix < 0x3FD33333) {
                return 1.0 - (0.5*z - (z*r - x*y));
            } else {
                if (ix > 0x3fe90000) {
                    qx = 0.28125;
                } else {
                    qx = hi_lo(ix - 0x00200000, 0);
                }
                hz = 0.5*z - qx;
                a  = 1.0 - qx;
                return a - (hz - (z*r - x*y));
            }
        }
}  // namespace
namespace Tan {
        inline double compute(double x) {
            double y[2] = {};
            double z = 0.0;
            int32_t n, ix;
            ix = hi_w(x);
            ix &= EXP_SIGNIF_BITS;
            if (ix <= 0x3fe921fb) {
                return kernel_tan(x, z, 1);
            } else if (ix >= EXP_BITS) {
                return x - x;
            } else {
                n = RemPio2::ieee754_rem_pio2(x, y);
                return kernel_tan(y[0], y[1], 1 - ((n & 1) << 1));
            }
        }
        inline constexpr double             pio4  =  0x1.921fb54442d18p-1,
            pio4lo=  0x1.1a62633145c07p-55,
            T[] = {
             0x1.5555555555563p-2,
             0x1.111111110fe7ap-3,
             0x1.ba1ba1bb341fep-5,
             0x1.664f48406d637p-6,
             0x1.226e3e96e8493p-7,
             0x1.d6d22c9560328p-9,
             0x1.7dbc8fee08315p-10,
             0x1.344d8f2f26501p-11,
             0x1.026f71a8d1068p-12,
             0x1.47e88a03792a6p-14,
             0x1.2b80f32f0a7e9p-14,
            -0x1.375cbdb605373p-16,
             0x1.b2a7074bf7ad4p-16,
        };
        inline double kernel_tan(double x, double y, int32_t iy) {
            double z, r, v, w, s;
            int32_t ix, hx;
            hx = hi_w(x);
            ix = hx & EXP_SIGNIF_BITS;
            if (ix < 0x3e300000) {
                if ((int32_t)x == 0) {
                    if (((ix | lo_w(x)) | (iy + 1)) == 0) {
                        return 1.0 / std::fabs(x);
                    } else {
                        if (iy == 1) {
                            return x;
                        } else {
                            double a, t;
                            z = w = x + y;
                            z= lo_w(z, 0);
                            v = y - (z - x);
                            t = a = -1.0 / w;
                            t = lo_w(t, 0);
                            s = 1.0 + t * z;
                            return t + a * (s + t * v);
                        }
                    }
                }
            }
            if (ix >= 0x3FE59428) {
                if ( hx < 0) {
                    x = -x;
                    y = -y;
                }
                z = pio4 - x;
                w = pio4lo - y;
                x = z + w;
                y = 0.0;
            }
            z       =  x*x;
            w       =  z*z;
            r =    T[1] + w*(T[3] + w*(T[5] + w*(T[7] + w*(T[9]  + w*T[11]))));
            v = z*(T[2] + w*(T[4] + w*(T[6] + w*(T[8] + w*(T[10] + w*T[12])))));
            s = z*x;
            r = y + z*(s*(r + v) + y);
            r += T[0]*s;
            w = x + r;
            if (ix >= 0x3FE59428) {
                v = (double)iy;
                return (double)(1-((hx >> 30) & 2))*(v - 2.0*(x - (w*w/(w + v) - r)));
            }
            if (iy == 1) {
                return w;
            } else { 
                double a,t;
                z = w;
                z = lo_w(z, 0);
                v = r - (z - x);
                t = a  = -1.0/w;
                t = lo_w(t, 0);
                s = 1.0 + t*z;
                return t + a*(s + t*v);
            }
        }
}  // namespace
namespace RemPio2 {
        inline constexpr int32_t two_over_pi[] = {
            0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
            0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
            0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
            0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
            0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
            0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
            0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
            0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
            0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
            0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
            0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
        };
        inline constexpr int32_t npio2_hw[] = {
            0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C,
            0x4025FDBB, 0x402921FB, 0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C,
            0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A,
            0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C,
            0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB, 0x4046C6CB, 0x40478FDB,
            0x404858EB, 0x404921FB,
        };
        inline constexpr double             invpio2 =  0x1.45f306dc9c883p-1,
            pio21  =  0x1.921fb544p0,
            pio21t =  0x1.0b4611a626331p-34,
            pio22  =  0x1.0b4611a6p-34,
            pio22t =  0x1.3198a2e037073p-69,
            pio23  =  0x1.3198a2ep-69,
            pio23t =  0x1.b839a252049c1p-104;
        inline int32_t ieee754_rem_pio2(double x, double* y) {
            double z = 0.0, w, t, r, fn;
            double tx[3] = {};
            int32_t e0, i, j, nx, n, ix, hx;
            hx = hi_w(x);
            ix = hx & EXP_SIGNIF_BITS;
            if (ix <= 0x3fe921fb) {
                y[0] = x;
                y[1] = 0;
                return 0;
            }
            if (ix < 0x4002d97c) {
                if (hx > 0) {
                    z = x - pio21;
                    if (ix != 0x3ff921fb) {
                        y[0] = z - pio21t;
                        y[1] = (z - y[0]) - pio21t;
                    } else {
                        z -= pio22;
                        y[0] = z - pio22t;
                        y[1] = (z-y[0])-pio22t;
                    }
                    return 1;
                } else {
                    z = x + pio21;
                    if (ix != 0x3ff921fb) {
                        y[0] = z + pio21t;
                        y[1] = (z - y[0]) + pio21t;
                    } else {
                        z += pio22;
                        y[0] = z + pio22t;
                        y[1] = (z - y[0]) + pio22t;
                    }
                    return -1;
                }
            }
            if (ix <= 0x413921fb) {
                t  = std::fabs(x);
                n  = (int32_t) (t*invpio2 + 0.5);
                fn = (double)n;
                r  = t - fn*pio21;
                w  = fn*pio21t;
                if (n < 32 && ix != npio2_hw[n - 1]) {
                    y[0] = r - w;
                } else {
                    j = ix >> 20;
                    y[0] = r - w;
                    i = j - (((hi_w(y[0])) >> 20) & 0x7ff);
                    if (i > 16) {
                        t  = r;
                        w  = fn*pio22;
                        r  = t - w;
                        w  = fn*pio22t - ((t - r) - w);
                        y[0] = r - w;
                        i = j - (((hi_w(y[0])) >> 20) & 0x7ff);
                        if (i > 49)  {
                            t  = r;
                            w  = fn*pio23;
                            r  = t - w;
                            w  = fn*pio23t - ((t - r) - w);
                            y[0] = r - w;
                        }
                    }
                }
                y[1] = (r - y[0]) - w;
                if (hx < 0) {
                    y[0] = -y[0];
                    y[1] = -y[1];
                    return -n;
                } else {
                    return n;
                }
            }
            if (ix >= EXP_BITS) {
                y[0] = y[1] = x - x;
                return 0;
            }
            z = lo_w(z, lo_w(x));
            e0 = (ix >> 20) - 1046;
            z = hi_w(z, ix - (e0 << 20));
            for (i=0; i < 2; i++) {
                tx[i] = (double)((int32_t)(z));
                z     = (z - tx[i])*TWO24;
            }
            tx[2] = z;
            nx = 3;
            while (tx[nx - 1] == 0.0) {
                nx--;
            }
            n = KernelRemPio2::kernel_rem_pio2(tx, y, e0, nx, 2, two_over_pi);
            if (hx < 0) {
                y[0] = -y[0];
                y[1] = -y[1];
                return -n;
            }
            return n;
        }
}  // namespace
namespace KernelRemPio2 {
        inline constexpr int32_t init_jk[] = {2, 3, 4, 6};
        inline constexpr double PIo2[] = {
            0x1.921fb4p0,
            0x1.4442dp-24,
            0x1.846988p-48,
            0x1.8cc516p-72,
            0x1.01b838p-96,
            0x1.a25204p-120,
            0x1.382228p-145,
            0x1.9f31dp-169,
        };
        inline constexpr double             twon24  = 0x1.0p-24;
        inline int32_t kernel_rem_pio2(double* x, double* y, int32_t e0, int32_t nx, int32_t prec, const int32_t* ipio2) {
            int32_t jz, jx, jv, jp, jk, carry, n, i, j, k, m, q0, ih;
            int32_t iq[20] = {};
            double z,fw;
            double f[20] = {};
            double fq[20] = {};
            double q[20] = {};
            jk = init_jk[prec];
            jp = jk;
            jx = nx - 1;
            jv = (e0 - 3)/24;
            if (jv < 0) {
                jv = 0;
            }
            q0 =  e0 - 24*(jv + 1);
            j = jv - jx;
            m = jx + jk;
            for (i = 0; i <= m; i++, j++) {
                f[i] = (j < 0) ? 0.0 : (double) ipio2[j];
            }
            for (i=0; i <= jk; i++) {
                for(j = 0, fw = 0.0; j <= jx; j++) {
                    fw += x[j]*f[jx + i - j];
                }
                q[i] = fw;
            }
            jz = jk;
            while(true) {
                for(i=0, j=jz, z=q[jz]; j > 0; i++, j--) {
                    fw    =  (double)((int32_t)(twon24* z));
                    iq[i] =  (int32_t)(z - TWO24*fw);
                    z     =  q[j - 1] + fw;
                }
                z  = std::scalbn(z, q0);
                z -= 8.0*std::floor(z*0.125);
                n  = (int32_t) z;
                z -= (double)n;
                ih = 0;
                if (q0 > 0) {
                    i  = (iq[jz - 1] >> (24 - q0));
                    n += i;
                    iq[jz - 1] -= i << (24 - q0);
                    ih = iq[jz - 1] >> (23 - q0);
                } else if (q0 == 0) {
                    ih = iq[jz-1]>>23;
                } else if (z >= 0.5) {
                    ih=2;
                }
                if (ih > 0) {
                    n += 1;
                    carry = 0;
                    for (i=0; i < jz; i++) {
                        j = iq[i];
                        if (carry == 0) {
                            if (j != 0) {
                                carry = 1;
                                iq[i] = 0x1000000 - j;
                            }
                        } else {
                            iq[i] = 0xffffff - j;
                        }
                    }
                    if (q0 > 0) {
                        switch(q0) {
                        case 1:
                            iq[jz-1] &= 0x7fffff;
                            break;
                        case 2:
                            iq[jz-1] &= 0x3fffff;
                            break;
                        }
                    }
                    if (ih == 2) {
                        z = 1.0 - z;
                        if (carry != 0) {
                            z -= std::scalbn(1.0, q0);
                        }
                    }
                }
                if (z == 0.0) {
                    j = 0;
                    for (i = jz - 1; i >= jk; i--) {
                        j |= iq[i];
                    }
                    if (j == 0) {
                        for(k=1; iq[jk - k] == 0; k++);
                        for(i = jz + 1; i <= jz + k; i++) {
                            f[jx + i] = (double) ipio2[jv + i];
                            for (j=0, fw = 0.0; j <= jx; j++) {
                                fw += x[j]*f[jx + i - j];
                            }
                            q[i] = fw;
                        }
                        jz += k;
                        continue;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            if (z == 0.0) {
                jz -= 1;
                q0 -= 24;
                while (iq[jz] == 0) {
                    jz--;
                    q0-=24;
                }
            } else {
                z = std::scalbn(z, -q0);
                if (z >= TWO24) {
                    fw = (double)((int32_t)(twon24*z));
                    iq[jz] = (int32_t)(z - TWO24*fw);
                    jz += 1;
                    q0 += 24;
                    iq[jz] = (int32_t) fw;
                } else {
                    iq[jz] = (int32_t) z;
                }
            }
            fw = std::scalbn(1.0, q0);
            for(i = jz; i>=0; i--) {
                q[i] = fw*(double)iq[i];
                fw *= twon24;
            }
            for(i = jz; i>=0; i--) {
                for (fw = 0.0, k = 0; k <= jp && k <= jz-i; k++) {
                    fw += PIo2[k]*q[i + k];
                }
                fq[jz - i] = fw;
            }
            switch(prec) {
            case 0:
                fw = 0.0;
                for (i = jz; i >=0; i--) {
                    fw += fq[i];
                }
                y[0] = (ih == 0)? fw: -fw;
                break;
            case 1:
            case 2:
                fw = 0.0;
                for (i = jz; i>=0; i--) {
                    fw += fq[i];
                }
                y[0] = (ih == 0) ? fw: -fw;
                fw = fq[0] - fw;
                for (i = 1; i <= jz; i++) {
                    fw += fq[i];
                }
                y[1] = (ih == 0)? fw: -fw;
                break;
            case 3:
                for (i = jz; i > 0; i--) {
                    fw      = fq[i - 1] + fq[i];
                    fq[i]  += fq[i - 1] - fw;
                    fq[i - 1] = fw;
                }
                for (i = jz; i>1; i--) {
                    fw      = fq[i - 1] + fq[i];
                    fq[i]  += fq[i - 1] - fw;
                    fq[i-1] = fw;
                }
                for (fw = 0.0, i = jz; i >= 2; i--) {
                    fw += fq[i];
                }
                if (ih == 0) {
                    y[0] =  fq[0];
                    y[1] =  fq[1];
                    y[2] =  fw;
                } else {
                    y[0] = -fq[0];
                    y[1] = -fq[1];
                    y[2] = -fw;
                }
            }
            return n & 7;
        }
}  // namespace
namespace Asin {
        inline constexpr double             pio2_hi = 0x1.921fb54442d18p0,
            pio2_lo = 0x1.1a62633145c07p-54,
            pio4_hi = 0x1.921fb54442d18p-1,
            pS0 =  0x1.5555555555555p-3,
            pS1 = -0x1.4d61203eb6f7dp-2,
            pS2 =  0x1.9c1550e884455p-3,
            pS3 = -0x1.48228b5688f3bp-5,
            pS4 =  0x1.9efe07501b288p-11,
            pS5 =  0x1.23de10dfdf709p-15,
            qS1 = -0x1.33a271c8a2d4bp1,
            qS2 =  0x1.02ae59c598ac8p1,
            qS3 = -0x1.6066c1b8d0159p-1,
            qS4 =  0x1.3b8c5b12e9282p-4;
        inline double compute(double x) {
            double t = 0, w, p, q, c, r, s;
            int32_t hx, ix;
            hx = hi_w(x);
            ix = hx & EXP_SIGNIF_BITS;
            if (ix >= 0x3ff00000) {
                if(((ix - 0x3ff00000) | lo_w(x)) == 0) {
                    return x*pio2_hi + x*pio2_lo;
                }
                return (x - x)/(x - x);
            } else if (ix < 0x3fe00000) {
                if (ix < 0x3e400000) {
                    if (HUGE_D + x > 1.0) {
                        return x;
                    }
                } else {
                    t = x*x;
                }
                p = t*(pS0 + t*(pS1 + t*(pS2 + t*(pS3 + t*(pS4 + t*pS5)))));
                q = 1.0 + t*(qS1 + t*(qS2 + t*(qS3 + t*qS4)));
                w = p/q;
                return x + x*w;
            }
            w = 1.0 - std::fabs(x);
            t = w*0.5;
            p = t*(pS0 + t*(pS1 + t*(pS2 + t*(pS3 + t*(pS4 + t*pS5)))));
            q = 1.0 + t*(qS1 + t*(qS2 + t*(qS3 + t*qS4)));
            s = std::sqrt(t);
            if (ix >= 0x3FEF3333) {
                w = p/q;
                t = pio2_hi - (2.0*(s + s*w) - pio2_lo);
            } else {
                w  = s;
                w  = lo_w(w, 0);
                c  = (t - w*w)/(s + w);
                r  = p/q;
                p  = 2.0*s*r - (pio2_lo - 2.0*c);
                q  = pio4_hi - 2.0*w;
                t  = pio4_hi - (p - q);
            }
            return (hx > 0) ? t : -t;
        }
}  // namespace
namespace Acos {
        inline constexpr double             pio2_hi =  0x1.921fb54442d18p0,
            pio2_lo =  0x1.1a62633145c07p-54,
            pS0     =  0x1.5555555555555p-3,
            pS1     = -0x1.4d61203eb6f7dp-2,
            pS2     =  0x1.9c1550e884455p-3,
            pS3     = -0x1.48228b5688f3bp-5,
            pS4     =  0x1.9efe07501b288p-11,
            pS5     =  0x1.23de10dfdf709p-15,
            qS1     = -0x1.33a271c8a2d4bp1,
            qS2     =  0x1.02ae59c598ac8p1,
            qS3     = -0x1.6066c1b8d0159p-1,
            qS4     =  0x1.3b8c5b12e9282p-4;
        inline double compute(double x) {
            double z, p, q, r, w, s, c, df;
            int32_t hx, ix;
            hx = hi_w(x);
            ix = hx & EXP_SIGNIF_BITS;
            if (ix >= 0x3ff00000) {
                if (((ix - 0x3ff00000) | lo_w(x)) == 0) {
                    if (hx > 0) {
                        return 0.0;
                    }else {
                        return k_pi + 2.0*pio2_lo;
                    }
                }
                return (x - x)/(x - x);
            }
            if (ix < 0x3fe00000) {
                if (ix <= 0x3c600000) {
                    return pio2_hi + pio2_lo;
                }
                z = x*x;
                p = z*(pS0 + z*(pS1 + z*(pS2 + z*(pS3 + z*(pS4 + z*pS5)))));
                q = 1.0 + z*(qS1 + z*(qS2 + z*(qS3 + z*qS4)));
                r = p/q;
                return pio2_hi - (x - (pio2_lo - x*r));
            } else if (hx < 0) {
                z = (1.0 + x)*0.5;
                p = z*(pS0 + z*(pS1 + z*(pS2 + z*(pS3 + z*(pS4 + z*pS5)))));
                q = 1.0 + z*(qS1 + z*(qS2 + z*(qS3 + z*qS4)));
                s = std::sqrt(z);
                r = p/q;
                w = r*s - pio2_lo;
                return k_pi - 2.0*(s+w);
            } else {
                z = (1.0 - x)*0.5;
                s = std::sqrt(z);
                df = s;
                df = lo_w(df, 0);
                c  = (z - df*df)/(s + df);
                p = z*(pS0 + z*(pS1 + z*(pS2 + z*(pS3 + z*(pS4 + z*pS5)))));
                q = 1.0 + z*(qS1 + z*(qS2 + z*(qS3 + z*qS4)));
                r = p/q;
                w = r*s + c;
                return 2.0*(df + w);
            }
        }
}  // namespace
namespace Atan {
        inline constexpr double atanhi[] = {
            0x1.dac670561bb4fp-2,
            0x1.921fb54442d18p-1,
            0x1.f730bd281f69bp-1,
            0x1.921fb54442d18p0,
        };
        inline constexpr double atanlo[] = {
            0x1.a2b7f222f65e2p-56,
            0x1.1a62633145c07p-55,
            0x1.007887af0cbbdp-56,
            0x1.1a62633145c07p-54,
        };
        inline constexpr double aT[] = {
             0x1.555555555550dp-2,
            -0x1.999999998ebc4p-3,
             0x1.24924920083ffp-3,
            -0x1.c71c6fe231671p-4,
             0x1.745cdc54c206ep-4,
            -0x1.3b0f2af749a6dp-4,
             0x1.10d66a0d03d51p-4,
            -0x1.dde2d52defd9ap-5,
             0x1.97b4b24760debp-5,
            -0x1.2b4442c6a6c2fp-5,
             0x1.0ad3ae322da11p-6,
        };
        inline double compute(double x) {
            double w, s1, s2, z;
            int32_t ix, hx, id;
            hx = hi_w(x);
            ix = hx & EXP_SIGNIF_BITS;
            if (ix >= 0x44100000) {
                if (ix > EXP_BITS ||
                    (ix == EXP_BITS && (lo_w(x) != 0))) {
                    return x+x;
                }
                if (hx > 0) {
                    return atanhi[3] + atanlo[3];
                } else {
                    return -atanhi[3] - atanlo[3];
                }
            } if (ix < 0x3fdc0000) {
                if (ix < 0x3e200000) {
                    if (HUGE_D + x > 1.0) {
                        return x;
                    }
                }
                id = -1;
            } else {
                x = std::fabs(x);
                if (ix < 0x3ff30000) {
                    if (ix < 0x3fe60000) {
                        id = 0;
                        x = (2.0*x - 1.0)/(2.0 + x);
                    } else {
                        id = 1;
                        x  = (x - 1.0)/(x + 1.0);
                    }
                } else {
                    if (ix < 0x40038000) {
                        id = 2;
                        x  = (x - 1.5)/(1.0 + 1.5*x);
                    } else {
                        id = 3;
                        x  = -1.0/x;
                    }
                }
            }
            z = x*x;
            w = z*z;
            s1 = z*(aT[0] + w*(aT[2] + w*(aT[4] + w*(aT[6] + w*(aT[8] + w*aT[10])))));
            s2 = w*(aT[1] + w*(aT[3] + w*(aT[5] + w*(aT[7] + w*aT[9]))));
            if (id < 0) {
                return x - x*(s1 + s2);
            } else {
                z = atanhi[id] - ((x*(s1+s2) - atanlo[id]) - x);
                return (hx < 0) ? -z: z;
            }
        }
}  // namespace
namespace Atan2 {
        inline constexpr double             tiny    = 1.0e-300,
            pi_o_4  = 0x1.921fb54442d18p-1,
            pi_o_2  = 0x1.921fb54442d18p0,
            pi_lo   = 0x1.1a62633145c07p-53;
        inline double compute(double y, double x) {
            double z;
            int32_t k, m, hx, hy, ix, iy;
             int32_t lx, ly;
            hx = hi_w(x);
            ix = hx & EXP_SIGNIF_BITS;
            lx = lo_w(x);
            hy = hi_w(y);
            iy = hy & EXP_SIGNIF_BITS;
            ly = lo_w(y);
            if (std::isnan(x) || std::isnan(y))
                return x + y;
            if (((hx - 0x3ff00000) | lx) == 0)
                return Atan::compute(y);
            m = ((hy >> 31) & 1)|((hx >> 30) & 2);
            if ((iy | ly) == 0) {
                switch(m) {
                case 0:
                case 1: return y;
                case 2: return  k_pi + tiny;
                case 3: return -k_pi - tiny;
                }
            }
            if ((ix | lx) == 0) {
                return (hy < 0)?  -pi_o_2 - tiny : pi_o_2 + tiny;
            }
            if (ix == EXP_BITS) {
                if (iy == EXP_BITS) {
                    switch(m) {
                    case 0: return  pi_o_4 + tiny;
                    case 1: return -pi_o_4 - tiny;
                    case 2: return  3.0*pi_o_4 + tiny;
                    case 3: return -3.0*pi_o_4 - tiny;
                    }
                } else {
                    switch(m) {
                    case 0: return  0.0;
                    case 1: return -0.0;
                    case 2: return  k_pi + tiny;
                    case 3: return -k_pi - tiny;
                    }
                }
            }
            if (iy == EXP_BITS) {
                return (hy < 0)? -pi_o_2 - tiny : pi_o_2 + tiny;
            }
            k = (iy - ix) >> 20;
            if (k > 60) {
                z = pi_o_2+0.5*pi_lo;
            } else if (hx < 0 && k < -60) {
                z = 0.0;
            } else {
                z = Atan::compute(std::fabs(y/x));
            }
            switch (m) {
            case 0:  return  z;
            case 1:  return -z;
            case 2:  return  k_pi - (z - pi_lo);
            default: return (z - pi_lo) - k_pi;
            }
        }
}  // namespace

}  // namespace bolt::kernels::fdlibm

#if defined(__clang__)
#pragma clang fp contract(on)
#endif
