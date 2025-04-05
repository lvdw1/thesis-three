/* This file was automatically generated by CasADi 3.6.7.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) bicycle_model_expl_ode_hess_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_sq CASADI_PREFIX(sq)
#define casadi_trans CASADI_PREFIX(trans)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

void casadi_mtimes(const casadi_real* x, const casadi_int* sp_x, const casadi_real* y, const casadi_int* sp_y, casadi_real* z, const casadi_int* sp_z, casadi_real* w, casadi_int tr) {
  casadi_int ncol_x, ncol_y, ncol_z, cc;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y, *colind_z, *row_z;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  ncol_z = sp_z[1];
  colind_z = sp_z+2; row_z = sp_z + 2 + ncol_z+1;
  if (tr) {
    for (cc=0; cc<ncol_z; ++cc) {
      casadi_int kk;
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_z[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    for (cc=0; cc<ncol_y; ++cc) {
      casadi_int kk;
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_y[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}

static const casadi_int casadi_s0[33] = {3, 9, 0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
static const casadi_int casadi_s1[27] = {9, 3, 0, 7, 14, 21, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 7, 0, 1, 2, 3, 4, 5, 8};
static const casadi_int casadi_s2[87] = {9, 9, 0, 9, 18, 27, 36, 45, 54, 61, 68, 75, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 7, 0, 1, 2, 3, 4, 5, 8};
static const casadi_int casadi_s3[39] = {9, 9, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4};
static const casadi_int casadi_s4[93] = {9, 9, 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s5[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s6[45] = {6, 6, 0, 6, 12, 18, 24, 30, 36, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s7[24] = {6, 3, 0, 6, 12, 18, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s8[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s9[222] = {218, 1, 0, 218, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217};
static const casadi_int casadi_s10[10] = {9, 1, 0, 6, 2, 3, 4, 6, 7, 8};
static const casadi_int casadi_s11[49] = {45, 1, 0, 45, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44};

static const casadi_real casadi_c0[21] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.};

/* bicycle_model_expl_ode_hess:(i0[6],i1[6x6],i2[6x3],i3[6],i4[3],i5[218])->(o0[9x1,6nz],o1[45]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j;
  casadi_real *rr, *ss;
  const casadi_real *cs;
  casadi_real w0, w1, w2, *w3=w+12, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, *w17=w+31, *w18=w+112, *w19=w+148, *w20=w+166, *w21=w+220, *w22=w+274, *w23=w+295, *w24=w+316, *w26=w+391, *w27=w+466, *w28=w+475, *w29=w+484, *w30=w+493, *w31=w+502, *w32=w+511, *w33=w+520, *w34=w+527, *w35=w+534, w36, w37, *w38=w+543, *w39=w+546, *w40=w+549, *w41=w+552, *w42=w+555, *w43=w+558, *w44=w+561, *w45=w+564, *w46=w+567, *w47=w+570;
  /* #0: @0 = input[0][2] */
  w0 = arg[0] ? arg[0][2] : 0;
  /* #1: @1 = cos(@0) */
  w1 = cos( w0 );
  /* #2: @2 = input[0][4] */
  w2 = arg[0] ? arg[0][4] : 0;
  /* #3: @3 = input[3][0] */
  casadi_copy(arg[3], 6, w3);
  /* #4: {@4, @5, @6, @7, @8, @9} = vertsplit(@3) */
  w4 = w3[0];
  w5 = w3[1];
  w6 = w3[2];
  w7 = w3[3];
  w8 = w3[4];
  w9 = w3[5];
  /* #5: @10 = (@2*@5) */
  w10  = (w2*w5);
  /* #6: @11 = (@1*@10) */
  w11  = (w1*w10);
  /* #7: @0 = sin(@0) */
  w0 = sin( w0 );
  /* #8: @12 = (@2*@4) */
  w12  = (w2*w4);
  /* #9: @13 = (@0*@12) */
  w13  = (w0*w12);
  /* #10: @11 = (@11-@13) */
  w11 -= w13;
  /* #11: output[0][0] = @11 */
  if (res[0]) res[0][0] = w11;
  /* #12: @11 = 0.666667 */
  w11 = 6.6666666666666663e-01;
  /* #13: @11 = (@11*@6) */
  w11 *= w6;
  /* #14: @2 = (@2*@11) */
  w2 *= w11;
  /* #15: @6 = input[0][3] */
  w6 = arg[0] ? arg[0][3] : 0;
  /* #16: @13 = cos(@6) */
  w13 = cos( w6 );
  /* #17: @14 = sq(@13) */
  w14 = casadi_sq( w13 );
  /* #18: @2 = (@2/@14) */
  w2 /= w14;
  /* #19: output[0][1] = @2 */
  if (res[0]) res[0][1] = w2;
  /* #20: @15 = tan(@6) */
  w15 = tan( w6 );
  /* #21: @15 = (@15*@11) */
  w15 *= w11;
  /* #22: @16 = (@0*@5) */
  w16  = (w0*w5);
  /* #23: @15 = (@15+@16) */
  w15 += w16;
  /* #24: @16 = (@1*@4) */
  w16  = (w1*w4);
  /* #25: @15 = (@15+@16) */
  w15 += w16;
  /* #26: output[0][2] = @15 */
  if (res[0]) res[0][2] = w15;
  /* #27: @15 = 0.2032 */
  w15 = 2.0319999999999999e-01;
  /* #28: @15 = (@15*@8) */
  w15 *= w8;
  /* #29: output[0][3] = @15 */
  if (res[0]) res[0][3] = w15;
  /* #30: output[0][4] = @7 */
  if (res[0]) res[0][4] = w7;
  /* #31: output[0][5] = @9 */
  if (res[0]) res[0][5] = w9;
  /* #32: @17 = zeros(9x9) */
  casadi_clear(w17, 81);
  /* #33: @18 = input[1][0] */
  casadi_copy(arg[1], 36, w18);
  /* #34: @19 = input[2][0] */
  casadi_copy(arg[2], 18, w19);
  /* #35: @20 = horzcat(@18, @19) */
  rr=w20;
  for (i=0, cs=w18; i<36; ++i) *rr++ = *cs++;
  for (i=0, cs=w19; i<18; ++i) *rr++ = *cs++;
  /* #36: @21 = @20' */
  for (i=0, rr=w21, cs=w20; i<9; ++i) for (j=0; j<6; ++j) rr[i+j*9] = *cs++;
  /* #37: @22 = 
  [[0, 0, 0, 0, 0, 0, 1, 00, 00], 
   [0, 0, 0, 0, 0, 0, 00, 1, 00], 
   [0, 0, 0, 0, 0, 0, 00, 00, 1]] */
  casadi_copy(casadi_c0, 21, w22);
  /* #38: @23 = @22' */
  casadi_trans(w22,casadi_s0, w23, casadi_s1, iw);
  /* #39: @24 = horzcat(@21, @23) */
  rr=w24;
  for (i=0, cs=w21; i<54; ++i) *rr++ = *cs++;
  for (i=0, cs=w23; i<21; ++i) *rr++ = *cs++;
  /* #40: @25 = 00 */
  /* #41: @26 = @24' */
  casadi_trans(w24,casadi_s2, w26, casadi_s2, iw);
  /* #42: {@27, @28, @29, @30, @31, @32, @33, @34, @35} = horzsplit(@26) */
  casadi_copy(w26, 9, w27);
  casadi_copy(w26+9, 9, w28);
  casadi_copy(w26+18, 9, w29);
  casadi_copy(w26+27, 9, w30);
  casadi_copy(w26+36, 9, w31);
  casadi_copy(w26+45, 9, w32);
  casadi_copy(w26+54, 7, w33);
  casadi_copy(w26+61, 7, w34);
  casadi_copy(w26+68, 7, w35);
  /* #43: {NULL, NULL, @9, @7, @15, NULL, NULL, NULL, NULL} = vertsplit(@27) */
  w9 = w27[2];
  w7 = w27[3];
  w15 = w27[4];
  /* #44: @8 = (@5*@15) */
  w8  = (w5*w15);
  /* #45: @8 = (@1*@8) */
  w8  = (w1*w8);
  /* #46: @16 = (@0*@9) */
  w16  = (w0*w9);
  /* #47: @36 = (@10*@16) */
  w36  = (w10*w16);
  /* #48: @8 = (@8-@36) */
  w8 -= w36;
  /* #49: @9 = (@1*@9) */
  w9  = (w1*w9);
  /* #50: @36 = (@12*@9) */
  w36  = (w12*w9);
  /* #51: @37 = (@4*@15) */
  w37  = (w4*w15);
  /* #52: @37 = (@0*@37) */
  w37  = (w0*w37);
  /* #53: @36 = (@36+@37) */
  w36 += w37;
  /* #54: @8 = (@8-@36) */
  w8 -= w36;
  /* #55: @15 = (@11*@15) */
  w15  = (w11*w15);
  /* #56: @15 = (@15/@14) */
  w15 /= w14;
  /* #57: @2 = (@2/@14) */
  w2 /= w14;
  /* #58: @13 = (2.*@13) */
  w13 = (2.* w13 );
  /* #59: @6 = sin(@6) */
  w6 = sin( w6 );
  /* #60: @36 = (@6*@7) */
  w36  = (w6*w7);
  /* #61: @36 = (@13*@36) */
  w36  = (w13*w36);
  /* #62: @36 = (@2*@36) */
  w36  = (w2*w36);
  /* #63: @15 = (@15+@36) */
  w15 += w36;
  /* #64: @7 = (@7/@14) */
  w7 /= w14;
  /* #65: @7 = (@11*@7) */
  w7  = (w11*w7);
  /* #66: @9 = (@5*@9) */
  w9  = (w5*w9);
  /* #67: @7 = (@7+@9) */
  w7 += w9;
  /* #68: @16 = (@4*@16) */
  w16  = (w4*w16);
  /* #69: @7 = (@7-@16) */
  w7 -= w16;
  /* #70: @38 = vertcat(@25, @25, @8, @15, @7, @25, @25, @25, @25) */
  rr=w38;
  *rr++ = w8;
  *rr++ = w15;
  *rr++ = w7;
  /* #71: {NULL, NULL, @8, @15, @7, NULL, NULL, NULL, NULL} = vertsplit(@28) */
  w8 = w28[2];
  w15 = w28[3];
  w7 = w28[4];
  /* #72: @16 = (@5*@7) */
  w16  = (w5*w7);
  /* #73: @16 = (@1*@16) */
  w16  = (w1*w16);
  /* #74: @9 = (@0*@8) */
  w9  = (w0*w8);
  /* #75: @36 = (@10*@9) */
  w36  = (w10*w9);
  /* #76: @16 = (@16-@36) */
  w16 -= w36;
  /* #77: @8 = (@1*@8) */
  w8  = (w1*w8);
  /* #78: @36 = (@12*@8) */
  w36  = (w12*w8);
  /* #79: @37 = (@4*@7) */
  w37  = (w4*w7);
  /* #80: @37 = (@0*@37) */
  w37  = (w0*w37);
  /* #81: @36 = (@36+@37) */
  w36 += w37;
  /* #82: @16 = (@16-@36) */
  w16 -= w36;
  /* #83: @7 = (@11*@7) */
  w7  = (w11*w7);
  /* #84: @7 = (@7/@14) */
  w7 /= w14;
  /* #85: @36 = (@6*@15) */
  w36  = (w6*w15);
  /* #86: @36 = (@13*@36) */
  w36  = (w13*w36);
  /* #87: @36 = (@2*@36) */
  w36  = (w2*w36);
  /* #88: @7 = (@7+@36) */
  w7 += w36;
  /* #89: @15 = (@15/@14) */
  w15 /= w14;
  /* #90: @15 = (@11*@15) */
  w15  = (w11*w15);
  /* #91: @8 = (@5*@8) */
  w8  = (w5*w8);
  /* #92: @15 = (@15+@8) */
  w15 += w8;
  /* #93: @9 = (@4*@9) */
  w9  = (w4*w9);
  /* #94: @15 = (@15-@9) */
  w15 -= w9;
  /* #95: @39 = vertcat(@25, @25, @16, @7, @15, @25, @25, @25, @25) */
  rr=w39;
  *rr++ = w16;
  *rr++ = w7;
  *rr++ = w15;
  /* #96: {NULL, NULL, @16, @7, @15, NULL, NULL, NULL, NULL} = vertsplit(@29) */
  w16 = w29[2];
  w7 = w29[3];
  w15 = w29[4];
  /* #97: @9 = (@5*@15) */
  w9  = (w5*w15);
  /* #98: @9 = (@1*@9) */
  w9  = (w1*w9);
  /* #99: @8 = (@0*@16) */
  w8  = (w0*w16);
  /* #100: @36 = (@10*@8) */
  w36  = (w10*w8);
  /* #101: @9 = (@9-@36) */
  w9 -= w36;
  /* #102: @16 = (@1*@16) */
  w16  = (w1*w16);
  /* #103: @36 = (@12*@16) */
  w36  = (w12*w16);
  /* #104: @37 = (@4*@15) */
  w37  = (w4*w15);
  /* #105: @37 = (@0*@37) */
  w37  = (w0*w37);
  /* #106: @36 = (@36+@37) */
  w36 += w37;
  /* #107: @9 = (@9-@36) */
  w9 -= w36;
  /* #108: @15 = (@11*@15) */
  w15  = (w11*w15);
  /* #109: @15 = (@15/@14) */
  w15 /= w14;
  /* #110: @36 = (@6*@7) */
  w36  = (w6*w7);
  /* #111: @36 = (@13*@36) */
  w36  = (w13*w36);
  /* #112: @36 = (@2*@36) */
  w36  = (w2*w36);
  /* #113: @15 = (@15+@36) */
  w15 += w36;
  /* #114: @7 = (@7/@14) */
  w7 /= w14;
  /* #115: @7 = (@11*@7) */
  w7  = (w11*w7);
  /* #116: @16 = (@5*@16) */
  w16  = (w5*w16);
  /* #117: @7 = (@7+@16) */
  w7 += w16;
  /* #118: @8 = (@4*@8) */
  w8  = (w4*w8);
  /* #119: @7 = (@7-@8) */
  w7 -= w8;
  /* #120: @40 = vertcat(@25, @25, @9, @15, @7, @25, @25, @25, @25) */
  rr=w40;
  *rr++ = w9;
  *rr++ = w15;
  *rr++ = w7;
  /* #121: {NULL, NULL, @9, @15, @7, NULL, NULL, NULL, NULL} = vertsplit(@30) */
  w9 = w30[2];
  w15 = w30[3];
  w7 = w30[4];
  /* #122: @8 = (@5*@7) */
  w8  = (w5*w7);
  /* #123: @8 = (@1*@8) */
  w8  = (w1*w8);
  /* #124: @16 = (@0*@9) */
  w16  = (w0*w9);
  /* #125: @36 = (@10*@16) */
  w36  = (w10*w16);
  /* #126: @8 = (@8-@36) */
  w8 -= w36;
  /* #127: @9 = (@1*@9) */
  w9  = (w1*w9);
  /* #128: @36 = (@12*@9) */
  w36  = (w12*w9);
  /* #129: @37 = (@4*@7) */
  w37  = (w4*w7);
  /* #130: @37 = (@0*@37) */
  w37  = (w0*w37);
  /* #131: @36 = (@36+@37) */
  w36 += w37;
  /* #132: @8 = (@8-@36) */
  w8 -= w36;
  /* #133: @7 = (@11*@7) */
  w7  = (w11*w7);
  /* #134: @7 = (@7/@14) */
  w7 /= w14;
  /* #135: @36 = (@6*@15) */
  w36  = (w6*w15);
  /* #136: @36 = (@13*@36) */
  w36  = (w13*w36);
  /* #137: @36 = (@2*@36) */
  w36  = (w2*w36);
  /* #138: @7 = (@7+@36) */
  w7 += w36;
  /* #139: @15 = (@15/@14) */
  w15 /= w14;
  /* #140: @15 = (@11*@15) */
  w15  = (w11*w15);
  /* #141: @9 = (@5*@9) */
  w9  = (w5*w9);
  /* #142: @15 = (@15+@9) */
  w15 += w9;
  /* #143: @16 = (@4*@16) */
  w16  = (w4*w16);
  /* #144: @15 = (@15-@16) */
  w15 -= w16;
  /* #145: @41 = vertcat(@25, @25, @8, @7, @15, @25, @25, @25, @25) */
  rr=w41;
  *rr++ = w8;
  *rr++ = w7;
  *rr++ = w15;
  /* #146: {NULL, NULL, @8, @7, @15, NULL, NULL, NULL, NULL} = vertsplit(@31) */
  w8 = w31[2];
  w7 = w31[3];
  w15 = w31[4];
  /* #147: @16 = (@5*@15) */
  w16  = (w5*w15);
  /* #148: @16 = (@1*@16) */
  w16  = (w1*w16);
  /* #149: @9 = (@0*@8) */
  w9  = (w0*w8);
  /* #150: @36 = (@10*@9) */
  w36  = (w10*w9);
  /* #151: @16 = (@16-@36) */
  w16 -= w36;
  /* #152: @8 = (@1*@8) */
  w8  = (w1*w8);
  /* #153: @36 = (@12*@8) */
  w36  = (w12*w8);
  /* #154: @37 = (@4*@15) */
  w37  = (w4*w15);
  /* #155: @37 = (@0*@37) */
  w37  = (w0*w37);
  /* #156: @36 = (@36+@37) */
  w36 += w37;
  /* #157: @16 = (@16-@36) */
  w16 -= w36;
  /* #158: @15 = (@11*@15) */
  w15  = (w11*w15);
  /* #159: @15 = (@15/@14) */
  w15 /= w14;
  /* #160: @36 = (@6*@7) */
  w36  = (w6*w7);
  /* #161: @36 = (@13*@36) */
  w36  = (w13*w36);
  /* #162: @36 = (@2*@36) */
  w36  = (w2*w36);
  /* #163: @15 = (@15+@36) */
  w15 += w36;
  /* #164: @7 = (@7/@14) */
  w7 /= w14;
  /* #165: @7 = (@11*@7) */
  w7  = (w11*w7);
  /* #166: @8 = (@5*@8) */
  w8  = (w5*w8);
  /* #167: @7 = (@7+@8) */
  w7 += w8;
  /* #168: @9 = (@4*@9) */
  w9  = (w4*w9);
  /* #169: @7 = (@7-@9) */
  w7 -= w9;
  /* #170: @42 = vertcat(@25, @25, @16, @15, @7, @25, @25, @25, @25) */
  rr=w42;
  *rr++ = w16;
  *rr++ = w15;
  *rr++ = w7;
  /* #171: {NULL, NULL, @16, @15, @7, NULL, NULL, NULL, NULL} = vertsplit(@32) */
  w16 = w32[2];
  w15 = w32[3];
  w7 = w32[4];
  /* #172: @9 = (@5*@7) */
  w9  = (w5*w7);
  /* #173: @9 = (@1*@9) */
  w9  = (w1*w9);
  /* #174: @8 = (@0*@16) */
  w8  = (w0*w16);
  /* #175: @36 = (@10*@8) */
  w36  = (w10*w8);
  /* #176: @9 = (@9-@36) */
  w9 -= w36;
  /* #177: @16 = (@1*@16) */
  w16  = (w1*w16);
  /* #178: @36 = (@12*@16) */
  w36  = (w12*w16);
  /* #179: @37 = (@4*@7) */
  w37  = (w4*w7);
  /* #180: @37 = (@0*@37) */
  w37  = (w0*w37);
  /* #181: @36 = (@36+@37) */
  w36 += w37;
  /* #182: @9 = (@9-@36) */
  w9 -= w36;
  /* #183: @7 = (@11*@7) */
  w7  = (w11*w7);
  /* #184: @7 = (@7/@14) */
  w7 /= w14;
  /* #185: @36 = (@6*@15) */
  w36  = (w6*w15);
  /* #186: @36 = (@13*@36) */
  w36  = (w13*w36);
  /* #187: @36 = (@2*@36) */
  w36  = (w2*w36);
  /* #188: @7 = (@7+@36) */
  w7 += w36;
  /* #189: @15 = (@15/@14) */
  w15 /= w14;
  /* #190: @15 = (@11*@15) */
  w15  = (w11*w15);
  /* #191: @16 = (@5*@16) */
  w16  = (w5*w16);
  /* #192: @15 = (@15+@16) */
  w15 += w16;
  /* #193: @8 = (@4*@8) */
  w8  = (w4*w8);
  /* #194: @15 = (@15-@8) */
  w15 -= w8;
  /* #195: @43 = vertcat(@25, @25, @9, @7, @15, @25, @25, @25, @25) */
  rr=w43;
  *rr++ = w9;
  *rr++ = w7;
  *rr++ = w15;
  /* #196: {NULL, NULL, @9, @7, @15, NULL, NULL, NULL, NULL} = vertsplit(@33) */
  w9 = w33[2];
  w7 = w33[3];
  w15 = w33[4];
  /* #197: @8 = (@5*@15) */
  w8  = (w5*w15);
  /* #198: @8 = (@1*@8) */
  w8  = (w1*w8);
  /* #199: @16 = (@0*@9) */
  w16  = (w0*w9);
  /* #200: @36 = (@10*@16) */
  w36  = (w10*w16);
  /* #201: @8 = (@8-@36) */
  w8 -= w36;
  /* #202: @9 = (@1*@9) */
  w9  = (w1*w9);
  /* #203: @36 = (@12*@9) */
  w36  = (w12*w9);
  /* #204: @37 = (@4*@15) */
  w37  = (w4*w15);
  /* #205: @37 = (@0*@37) */
  w37  = (w0*w37);
  /* #206: @36 = (@36+@37) */
  w36 += w37;
  /* #207: @8 = (@8-@36) */
  w8 -= w36;
  /* #208: @15 = (@11*@15) */
  w15  = (w11*w15);
  /* #209: @15 = (@15/@14) */
  w15 /= w14;
  /* #210: @36 = (@6*@7) */
  w36  = (w6*w7);
  /* #211: @36 = (@13*@36) */
  w36  = (w13*w36);
  /* #212: @36 = (@2*@36) */
  w36  = (w2*w36);
  /* #213: @15 = (@15+@36) */
  w15 += w36;
  /* #214: @7 = (@7/@14) */
  w7 /= w14;
  /* #215: @7 = (@11*@7) */
  w7  = (w11*w7);
  /* #216: @9 = (@5*@9) */
  w9  = (w5*w9);
  /* #217: @7 = (@7+@9) */
  w7 += w9;
  /* #218: @16 = (@4*@16) */
  w16  = (w4*w16);
  /* #219: @7 = (@7-@16) */
  w7 -= w16;
  /* #220: @44 = vertcat(@25, @25, @8, @15, @7, @25, @25, @25, @25) */
  rr=w44;
  *rr++ = w8;
  *rr++ = w15;
  *rr++ = w7;
  /* #221: {NULL, NULL, @8, @15, @7, NULL, NULL, NULL, NULL} = vertsplit(@34) */
  w8 = w34[2];
  w15 = w34[3];
  w7 = w34[4];
  /* #222: @16 = (@5*@7) */
  w16  = (w5*w7);
  /* #223: @16 = (@1*@16) */
  w16  = (w1*w16);
  /* #224: @9 = (@0*@8) */
  w9  = (w0*w8);
  /* #225: @36 = (@10*@9) */
  w36  = (w10*w9);
  /* #226: @16 = (@16-@36) */
  w16 -= w36;
  /* #227: @8 = (@1*@8) */
  w8  = (w1*w8);
  /* #228: @36 = (@12*@8) */
  w36  = (w12*w8);
  /* #229: @37 = (@4*@7) */
  w37  = (w4*w7);
  /* #230: @37 = (@0*@37) */
  w37  = (w0*w37);
  /* #231: @36 = (@36+@37) */
  w36 += w37;
  /* #232: @16 = (@16-@36) */
  w16 -= w36;
  /* #233: @7 = (@11*@7) */
  w7  = (w11*w7);
  /* #234: @7 = (@7/@14) */
  w7 /= w14;
  /* #235: @36 = (@6*@15) */
  w36  = (w6*w15);
  /* #236: @36 = (@13*@36) */
  w36  = (w13*w36);
  /* #237: @36 = (@2*@36) */
  w36  = (w2*w36);
  /* #238: @7 = (@7+@36) */
  w7 += w36;
  /* #239: @15 = (@15/@14) */
  w15 /= w14;
  /* #240: @15 = (@11*@15) */
  w15  = (w11*w15);
  /* #241: @8 = (@5*@8) */
  w8  = (w5*w8);
  /* #242: @15 = (@15+@8) */
  w15 += w8;
  /* #243: @9 = (@4*@9) */
  w9  = (w4*w9);
  /* #244: @15 = (@15-@9) */
  w15 -= w9;
  /* #245: @45 = vertcat(@25, @25, @16, @7, @15, @25, @25, @25, @25) */
  rr=w45;
  *rr++ = w16;
  *rr++ = w7;
  *rr++ = w15;
  /* #246: {NULL, NULL, @16, @7, @15, NULL, NULL, NULL, NULL} = vertsplit(@35) */
  w16 = w35[2];
  w7 = w35[3];
  w15 = w35[4];
  /* #247: @9 = (@5*@15) */
  w9  = (w5*w15);
  /* #248: @9 = (@1*@9) */
  w9  = (w1*w9);
  /* #249: @8 = (@0*@16) */
  w8  = (w0*w16);
  /* #250: @10 = (@10*@8) */
  w10 *= w8;
  /* #251: @9 = (@9-@10) */
  w9 -= w10;
  /* #252: @1 = (@1*@16) */
  w1 *= w16;
  /* #253: @12 = (@12*@1) */
  w12 *= w1;
  /* #254: @16 = (@4*@15) */
  w16  = (w4*w15);
  /* #255: @0 = (@0*@16) */
  w0 *= w16;
  /* #256: @12 = (@12+@0) */
  w12 += w0;
  /* #257: @9 = (@9-@12) */
  w9 -= w12;
  /* #258: @15 = (@11*@15) */
  w15  = (w11*w15);
  /* #259: @15 = (@15/@14) */
  w15 /= w14;
  /* #260: @6 = (@6*@7) */
  w6 *= w7;
  /* #261: @13 = (@13*@6) */
  w13 *= w6;
  /* #262: @2 = (@2*@13) */
  w2 *= w13;
  /* #263: @15 = (@15+@2) */
  w15 += w2;
  /* #264: @7 = (@7/@14) */
  w7 /= w14;
  /* #265: @11 = (@11*@7) */
  w11 *= w7;
  /* #266: @5 = (@5*@1) */
  w5 *= w1;
  /* #267: @11 = (@11+@5) */
  w11 += w5;
  /* #268: @4 = (@4*@8) */
  w4 *= w8;
  /* #269: @11 = (@11-@4) */
  w11 -= w4;
  /* #270: @46 = vertcat(@25, @25, @9, @15, @11, @25, @25, @25, @25) */
  rr=w46;
  *rr++ = w9;
  *rr++ = w15;
  *rr++ = w11;
  /* #271: @47 = horzcat(@38, @39, @40, @41, @42, @43, @44, @45, @46) */
  rr=w47;
  for (i=0, cs=w38; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w39; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w40; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w41; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w42; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w43; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w44; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w45; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w46; i<3; ++i) *rr++ = *cs++;
  /* #272: @17 = mac(@24,@47,@17) */
  casadi_mtimes(w24, casadi_s2, w47, casadi_s3, w17, casadi_s4, w, 0);
  /* #273: @9 = @17[0] */
  for (rr=(&w9), ss=w17+0; ss!=w17+1; ss+=1) *rr++ = *ss;
  /* #274: output[1][0] = @9 */
  if (res[1]) res[1][0] = w9;
  /* #275: @9 = @17[1] */
  for (rr=(&w9), ss=w17+1; ss!=w17+2; ss+=1) *rr++ = *ss;
  /* #276: output[1][1] = @9 */
  if (res[1]) res[1][1] = w9;
  /* #277: @9 = @17[2] */
  for (rr=(&w9), ss=w17+2; ss!=w17+3; ss+=1) *rr++ = *ss;
  /* #278: output[1][2] = @9 */
  if (res[1]) res[1][2] = w9;
  /* #279: @9 = @17[3] */
  for (rr=(&w9), ss=w17+3; ss!=w17+4; ss+=1) *rr++ = *ss;
  /* #280: output[1][3] = @9 */
  if (res[1]) res[1][3] = w9;
  /* #281: @9 = @17[4] */
  for (rr=(&w9), ss=w17+4; ss!=w17+5; ss+=1) *rr++ = *ss;
  /* #282: output[1][4] = @9 */
  if (res[1]) res[1][4] = w9;
  /* #283: @9 = @17[5] */
  for (rr=(&w9), ss=w17+5; ss!=w17+6; ss+=1) *rr++ = *ss;
  /* #284: output[1][5] = @9 */
  if (res[1]) res[1][5] = w9;
  /* #285: @9 = @17[6] */
  for (rr=(&w9), ss=w17+6; ss!=w17+7; ss+=1) *rr++ = *ss;
  /* #286: output[1][6] = @9 */
  if (res[1]) res[1][6] = w9;
  /* #287: @9 = @17[7] */
  for (rr=(&w9), ss=w17+7; ss!=w17+8; ss+=1) *rr++ = *ss;
  /* #288: output[1][7] = @9 */
  if (res[1]) res[1][7] = w9;
  /* #289: @9 = @17[8] */
  for (rr=(&w9), ss=w17+8; ss!=w17+9; ss+=1) *rr++ = *ss;
  /* #290: output[1][8] = @9 */
  if (res[1]) res[1][8] = w9;
  /* #291: @9 = @17[10] */
  for (rr=(&w9), ss=w17+10; ss!=w17+11; ss+=1) *rr++ = *ss;
  /* #292: output[1][9] = @9 */
  if (res[1]) res[1][9] = w9;
  /* #293: @9 = @17[11] */
  for (rr=(&w9), ss=w17+11; ss!=w17+12; ss+=1) *rr++ = *ss;
  /* #294: output[1][10] = @9 */
  if (res[1]) res[1][10] = w9;
  /* #295: @9 = @17[12] */
  for (rr=(&w9), ss=w17+12; ss!=w17+13; ss+=1) *rr++ = *ss;
  /* #296: output[1][11] = @9 */
  if (res[1]) res[1][11] = w9;
  /* #297: @9 = @17[13] */
  for (rr=(&w9), ss=w17+13; ss!=w17+14; ss+=1) *rr++ = *ss;
  /* #298: output[1][12] = @9 */
  if (res[1]) res[1][12] = w9;
  /* #299: @9 = @17[14] */
  for (rr=(&w9), ss=w17+14; ss!=w17+15; ss+=1) *rr++ = *ss;
  /* #300: output[1][13] = @9 */
  if (res[1]) res[1][13] = w9;
  /* #301: @9 = @17[15] */
  for (rr=(&w9), ss=w17+15; ss!=w17+16; ss+=1) *rr++ = *ss;
  /* #302: output[1][14] = @9 */
  if (res[1]) res[1][14] = w9;
  /* #303: @9 = @17[16] */
  for (rr=(&w9), ss=w17+16; ss!=w17+17; ss+=1) *rr++ = *ss;
  /* #304: output[1][15] = @9 */
  if (res[1]) res[1][15] = w9;
  /* #305: @9 = @17[17] */
  for (rr=(&w9), ss=w17+17; ss!=w17+18; ss+=1) *rr++ = *ss;
  /* #306: output[1][16] = @9 */
  if (res[1]) res[1][16] = w9;
  /* #307: @9 = @17[20] */
  for (rr=(&w9), ss=w17+20; ss!=w17+21; ss+=1) *rr++ = *ss;
  /* #308: output[1][17] = @9 */
  if (res[1]) res[1][17] = w9;
  /* #309: @9 = @17[21] */
  for (rr=(&w9), ss=w17+21; ss!=w17+22; ss+=1) *rr++ = *ss;
  /* #310: output[1][18] = @9 */
  if (res[1]) res[1][18] = w9;
  /* #311: @9 = @17[22] */
  for (rr=(&w9), ss=w17+22; ss!=w17+23; ss+=1) *rr++ = *ss;
  /* #312: output[1][19] = @9 */
  if (res[1]) res[1][19] = w9;
  /* #313: @9 = @17[23] */
  for (rr=(&w9), ss=w17+23; ss!=w17+24; ss+=1) *rr++ = *ss;
  /* #314: output[1][20] = @9 */
  if (res[1]) res[1][20] = w9;
  /* #315: @9 = @17[24] */
  for (rr=(&w9), ss=w17+24; ss!=w17+25; ss+=1) *rr++ = *ss;
  /* #316: output[1][21] = @9 */
  if (res[1]) res[1][21] = w9;
  /* #317: @9 = @17[25] */
  for (rr=(&w9), ss=w17+25; ss!=w17+26; ss+=1) *rr++ = *ss;
  /* #318: output[1][22] = @9 */
  if (res[1]) res[1][22] = w9;
  /* #319: @9 = @17[26] */
  for (rr=(&w9), ss=w17+26; ss!=w17+27; ss+=1) *rr++ = *ss;
  /* #320: output[1][23] = @9 */
  if (res[1]) res[1][23] = w9;
  /* #321: @9 = @17[30] */
  for (rr=(&w9), ss=w17+30; ss!=w17+31; ss+=1) *rr++ = *ss;
  /* #322: output[1][24] = @9 */
  if (res[1]) res[1][24] = w9;
  /* #323: @9 = @17[31] */
  for (rr=(&w9), ss=w17+31; ss!=w17+32; ss+=1) *rr++ = *ss;
  /* #324: output[1][25] = @9 */
  if (res[1]) res[1][25] = w9;
  /* #325: @9 = @17[32] */
  for (rr=(&w9), ss=w17+32; ss!=w17+33; ss+=1) *rr++ = *ss;
  /* #326: output[1][26] = @9 */
  if (res[1]) res[1][26] = w9;
  /* #327: @9 = @17[33] */
  for (rr=(&w9), ss=w17+33; ss!=w17+34; ss+=1) *rr++ = *ss;
  /* #328: output[1][27] = @9 */
  if (res[1]) res[1][27] = w9;
  /* #329: @9 = @17[34] */
  for (rr=(&w9), ss=w17+34; ss!=w17+35; ss+=1) *rr++ = *ss;
  /* #330: output[1][28] = @9 */
  if (res[1]) res[1][28] = w9;
  /* #331: @9 = @17[35] */
  for (rr=(&w9), ss=w17+35; ss!=w17+36; ss+=1) *rr++ = *ss;
  /* #332: output[1][29] = @9 */
  if (res[1]) res[1][29] = w9;
  /* #333: @9 = @17[40] */
  for (rr=(&w9), ss=w17+40; ss!=w17+41; ss+=1) *rr++ = *ss;
  /* #334: output[1][30] = @9 */
  if (res[1]) res[1][30] = w9;
  /* #335: @9 = @17[41] */
  for (rr=(&w9), ss=w17+41; ss!=w17+42; ss+=1) *rr++ = *ss;
  /* #336: output[1][31] = @9 */
  if (res[1]) res[1][31] = w9;
  /* #337: @9 = @17[42] */
  for (rr=(&w9), ss=w17+42; ss!=w17+43; ss+=1) *rr++ = *ss;
  /* #338: output[1][32] = @9 */
  if (res[1]) res[1][32] = w9;
  /* #339: @9 = @17[43] */
  for (rr=(&w9), ss=w17+43; ss!=w17+44; ss+=1) *rr++ = *ss;
  /* #340: output[1][33] = @9 */
  if (res[1]) res[1][33] = w9;
  /* #341: @9 = @17[44] */
  for (rr=(&w9), ss=w17+44; ss!=w17+45; ss+=1) *rr++ = *ss;
  /* #342: output[1][34] = @9 */
  if (res[1]) res[1][34] = w9;
  /* #343: @9 = @17[50] */
  for (rr=(&w9), ss=w17+50; ss!=w17+51; ss+=1) *rr++ = *ss;
  /* #344: output[1][35] = @9 */
  if (res[1]) res[1][35] = w9;
  /* #345: @9 = @17[51] */
  for (rr=(&w9), ss=w17+51; ss!=w17+52; ss+=1) *rr++ = *ss;
  /* #346: output[1][36] = @9 */
  if (res[1]) res[1][36] = w9;
  /* #347: @9 = @17[52] */
  for (rr=(&w9), ss=w17+52; ss!=w17+53; ss+=1) *rr++ = *ss;
  /* #348: output[1][37] = @9 */
  if (res[1]) res[1][37] = w9;
  /* #349: @9 = @17[53] */
  for (rr=(&w9), ss=w17+53; ss!=w17+54; ss+=1) *rr++ = *ss;
  /* #350: output[1][38] = @9 */
  if (res[1]) res[1][38] = w9;
  /* #351: @9 = @17[60] */
  for (rr=(&w9), ss=w17+60; ss!=w17+61; ss+=1) *rr++ = *ss;
  /* #352: output[1][39] = @9 */
  if (res[1]) res[1][39] = w9;
  /* #353: @9 = @17[61] */
  for (rr=(&w9), ss=w17+61; ss!=w17+62; ss+=1) *rr++ = *ss;
  /* #354: output[1][40] = @9 */
  if (res[1]) res[1][40] = w9;
  /* #355: @9 = @17[62] */
  for (rr=(&w9), ss=w17+62; ss!=w17+63; ss+=1) *rr++ = *ss;
  /* #356: output[1][41] = @9 */
  if (res[1]) res[1][41] = w9;
  /* #357: @9 = @17[70] */
  for (rr=(&w9), ss=w17+70; ss!=w17+71; ss+=1) *rr++ = *ss;
  /* #358: output[1][42] = @9 */
  if (res[1]) res[1][42] = w9;
  /* #359: @9 = @17[71] */
  for (rr=(&w9), ss=w17+71; ss!=w17+72; ss+=1) *rr++ = *ss;
  /* #360: output[1][43] = @9 */
  if (res[1]) res[1][43] = w9;
  /* #361: @9 = @17[80] */
  for (rr=(&w9), ss=w17+80; ss!=w17+81; ss+=1) *rr++ = *ss;
  /* #362: output[1][44] = @9 */
  if (res[1]) res[1][44] = w9;
  return 0;
}

CASADI_SYMBOL_EXPORT int bicycle_model_expl_ode_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int bicycle_model_expl_ode_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int bicycle_model_expl_ode_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void bicycle_model_expl_ode_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int bicycle_model_expl_ode_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void bicycle_model_expl_ode_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void bicycle_model_expl_ode_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void bicycle_model_expl_ode_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int bicycle_model_expl_ode_hess_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int bicycle_model_expl_ode_hess_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real bicycle_model_expl_ode_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* bicycle_model_expl_ode_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    case 5: return "i5";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* bicycle_model_expl_ode_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* bicycle_model_expl_ode_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    case 1: return casadi_s6;
    case 2: return casadi_s7;
    case 3: return casadi_s5;
    case 4: return casadi_s8;
    case 5: return casadi_s9;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* bicycle_model_expl_ode_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s10;
    case 1: return casadi_s11;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int bicycle_model_expl_ode_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 15;
  if (sz_res) *sz_res = 11;
  if (sz_iw) *sz_iw = 10;
  if (sz_w) *sz_w = 597;
  return 0;
}

CASADI_SYMBOL_EXPORT int bicycle_model_expl_ode_hess_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 15*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 11*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 10*sizeof(casadi_int);
  if (sz_w) *sz_w = 597*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
