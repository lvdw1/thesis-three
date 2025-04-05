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
  #define CASADI_PREFIX(ID) bicycle_model_cost_ext_cost_fun_ ## ID
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
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_c2 CASADI_PREFIX(c2)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_clear_casadi_int CASADI_PREFIX(clear_casadi_int)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_de_boor CASADI_PREFIX(de_boor)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_fill_casadi_int CASADI_PREFIX(fill_casadi_int)
#define casadi_low CASADI_PREFIX(low)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_nd_boor_eval CASADI_PREFIX(nd_boor_eval)
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

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

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

void casadi_de_boor(casadi_real x, const casadi_real* knots, casadi_int n_knots, casadi_int degree, casadi_real* boor) {
  casadi_int d, i;
  for (d=1;d<degree+1;++d) {
    for (i=0;i<n_knots-d-1;++i) {
      casadi_real b, bottom;
      b = 0;
      bottom = knots[i + d] - knots[i];
      if (bottom) b = (x - knots[i]) * boor[i] / bottom;
      bottom = knots[i + d + 1] - knots[i + 1];
      if (bottom) b += (knots[i + d + 1] - x) * boor[i + 1] / bottom;
      boor[i] = b;
    }
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_fill_casadi_int(casadi_int* x, casadi_int n, casadi_int alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_clear_casadi_int(casadi_int* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

casadi_int casadi_low(casadi_real x, const casadi_real* grid, casadi_int ng, casadi_int lookup_mode) {
  switch (lookup_mode) {
    case 1:
      {
        casadi_real g0, dg;
        casadi_int ret;
        g0 = grid[0];
        dg = grid[ng-1]-g0;
        ret = (casadi_int) ((x-g0)*(ng-1)/dg);
        if (ret<0) ret=0;
        if (ret>ng-2) ret=ng-2;
        return ret;
      }
    case 2:
      {
        casadi_int start, stop, pivot;
        if (ng<2 || x<grid[1]) return 0;
        if (x>grid[ng-1]) return ng-2;
        start = 0;
        stop  = ng-1;
        while (1) {
          pivot = (stop+start)/2;
          if (x < grid[pivot]) {
            if (pivot==stop) return pivot;
            stop = pivot;
          } else {
            if (pivot==start) return pivot;
            start = pivot;
          }
        }
      }
    default:
      {
        casadi_int i;
        for (i=0; i<ng-2; ++i) {
          if (x < grid[i+1]) break;
        }
        return i;
      }
  }
}

void casadi_nd_boor_eval(casadi_real* ret, casadi_int n_dims, const casadi_real* all_knots, const casadi_int* offset, const casadi_int* all_degree, const casadi_int* strides, const casadi_real* c, casadi_int m, const casadi_real* all_x, const casadi_int* lookup_mode, casadi_int* iw, casadi_real* w) {
  casadi_int n_iter, k, i, pivot;
  casadi_int *boor_offset, *starts, *index, *coeff_offset;
  casadi_real *cumprod, *all_boor;
  boor_offset = iw; iw+=n_dims+1;
  starts = iw; iw+=n_dims;
  index = iw; iw+=n_dims;
  coeff_offset = iw;
  cumprod = w; w+= n_dims+1;
  all_boor = w;
  boor_offset[0] = 0;
  cumprod[n_dims] = 1;
  coeff_offset[n_dims] = 0;
  n_iter = 1;
  for (k=0;k<n_dims;++k) {
    casadi_real *boor;
    const casadi_real* knots;
    casadi_real x;
    casadi_int degree, n_knots, n_b, L, start;
    boor = all_boor+boor_offset[k];
    degree = all_degree[k];
    knots = all_knots + offset[k];
    n_knots = offset[k+1]-offset[k];
    n_b = n_knots-degree-1;
    x = all_x[k];
    L = casadi_low(x, knots+degree, n_knots-2*degree, lookup_mode[k]);
    start = L;
    if (start>n_b-degree-1) start = n_b-degree-1;
    starts[k] = start;
    casadi_clear(boor, 2*degree+1);
    if (x>=knots[0] && x<=knots[n_knots-1]) {
      if (x==knots[1]) {
        casadi_fill(boor, degree+1, 1.0);
      } else if (x==knots[n_knots-1]) {
        boor[degree] = 1;
      } else if (knots[L+degree]==x) {
        boor[degree-1] = 1;
      } else {
        boor[degree] = 1;
      }
    }
    casadi_de_boor(x, knots+start, 2*degree+2, degree, boor);
    boor+= degree+1;
    n_iter*= degree+1;
    boor_offset[k+1] = boor_offset[k] + degree+1;
  }
  casadi_clear_casadi_int(index, n_dims);
  for (pivot=n_dims-1;pivot>=0;--pivot) {
    cumprod[pivot] = (*(all_boor+boor_offset[pivot]))*cumprod[pivot+1];
    coeff_offset[pivot] = starts[pivot]*strides[pivot]+coeff_offset[pivot+1];
  }
  for (k=0;k<n_iter;++k) {
    casadi_int pivot = 0;
    for (i=0;i<m;++i) ret[i] += c[coeff_offset[0]+i]*cumprod[0];
    index[0]++;
    {
      while (index[pivot]==boor_offset[pivot+1]-boor_offset[pivot]) {
        index[pivot] = 0;
        if (pivot==n_dims-1) break;
        index[++pivot]++;
      }
      while (pivot>0) {
        cumprod[pivot] = (*(all_boor+boor_offset[pivot]+index[pivot]))*cumprod[pivot+1];
        coeff_offset[pivot] = (starts[pivot]+index[pivot])*strides[pivot]+coeff_offset[pivot+1];
        pivot--;
      }
    }
    cumprod[0] = (*(all_boor+index[0]))*cumprod[1];
    coeff_offset[0] = (starts[0]+index[0])*m+coeff_offset[1];
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[330] = {2, 109, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
static const casadi_int casadi_s1[327] = {109, 108, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90, 90, 91, 91, 92, 92, 93, 93, 94, 94, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99, 100, 100, 101, 101, 102, 102, 103, 103, 104, 104, 105, 105, 106, 106, 107, 107, 108};
static const casadi_int casadi_s2[327] = {2, 108, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
static const casadi_int casadi_s3[2] = {0, 110};
static const casadi_int casadi_s4[1] = {1};
static const casadi_int casadi_s5[1] = {2};
static const casadi_int casadi_s6[2] = {0, 112};
static const casadi_int casadi_s7[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s8[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s9[4] = {0, 1, 0, 0};
static const casadi_int casadi_s10[222] = {218, 1, 0, 218, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217};
static const casadi_int casadi_s11[5] = {1, 1, 0, 1, 0};

static const casadi_real casadi_c0[216] = {-2.1400000000000003e+02, 2.1400000000000003e+02, -1.0700000000000001e+02, 1.0700000000000001e+02, -1.0700000000000003e+02, 1.0700000000000003e+02, -1.0700000000000001e+02, 1.0700000000000001e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0700000000000003e+02, 1.0700000000000003e+02, -1.0700000000000003e+02, 1.0700000000000003e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0700000000000007e+02, 1.0700000000000007e+02, -1.0700000000000007e+02, 1.0700000000000007e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0700000000000014e+02, 1.0700000000000014e+02, -1.0700000000000014e+02, 1.0700000000000014e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0700000000000031e+02, 1.0700000000000031e+02, -1.0700000000000031e+02, 1.0700000000000031e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0700000000000063e+02, 1.0700000000000063e+02, -1.0700000000000063e+02, 1.0700000000000063e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999999e+02, 1.0699999999999999e+02, -1.0699999999999935e+02, 1.0699999999999935e+02, -2.1399999999999744e+02, 2.1399999999999744e+02};
static const casadi_real casadi_c1[110] = {0., 0., 9.3457943925233638e-03, 1.8691588785046728e-02, 2.8037383177570090e-02, 3.7383177570093455e-02, 4.6728971962616821e-02, 5.6074766355140179e-02, 6.5420560747663545e-02, 7.4766355140186910e-02, 8.4112149532710276e-02, 9.3457943925233641e-02, 1.0280373831775701e-01, 1.1214953271028036e-01, 1.2149532710280372e-01, 1.3084112149532709e-01, 1.4018691588785046e-01, 1.4953271028037382e-01, 1.5887850467289719e-01, 1.6822429906542055e-01, 1.7757009345794392e-01, 1.8691588785046728e-01, 1.9626168224299065e-01, 2.0560747663551401e-01, 2.1495327102803738e-01, 2.2429906542056072e-01, 2.3364485981308408e-01, 2.4299065420560745e-01, 2.5233644859813081e-01, 2.6168224299065418e-01, 2.7102803738317754e-01, 2.8037383177570091e-01, 2.8971962616822428e-01, 2.9906542056074764e-01, 3.0841121495327101e-01, 3.1775700934579437e-01, 3.2710280373831774e-01, 3.3644859813084110e-01, 3.4579439252336447e-01, 3.5514018691588783e-01, 3.6448598130841120e-01, 3.7383177570093457e-01, 3.8317757009345793e-01, 3.9252336448598130e-01, 4.0186915887850466e-01, 4.1121495327102803e-01, 4.2056074766355139e-01, 4.2990654205607476e-01, 4.3925233644859812e-01, 4.4859813084112143e-01, 4.5794392523364480e-01, 4.6728971962616817e-01, 4.7663551401869153e-01, 4.8598130841121490e-01, 4.9532710280373826e-01, 5.0467289719626163e-01, 5.1401869158878499e-01, 5.2336448598130836e-01, 5.3271028037383172e-01, 5.4205607476635509e-01, 5.5140186915887845e-01, 5.6074766355140182e-01, 5.7009345794392519e-01, 5.7943925233644855e-01, 5.8878504672897192e-01, 5.9813084112149528e-01, 6.0747663551401865e-01, 6.1682242990654201e-01, 6.2616822429906538e-01, 6.3551401869158874e-01, 6.4485981308411211e-01, 6.5420560747663548e-01, 6.6355140186915884e-01, 6.7289719626168221e-01, 6.8224299065420557e-01, 6.9158878504672894e-01, 7.0093457943925230e-01, 7.1028037383177567e-01, 7.1962616822429903e-01, 7.2897196261682240e-01, 7.3831775700934577e-01, 7.4766355140186913e-01, 7.5700934579439250e-01, 7.6635514018691586e-01, 7.7570093457943923e-01, 7.8504672897196259e-01, 7.9439252336448596e-01, 8.0373831775700932e-01, 8.1308411214953269e-01, 8.2242990654205606e-01, 8.3177570093457942e-01, 8.4112149532710279e-01, 8.5046728971962615e-01, 8.5981308411214952e-01, 8.6915887850467288e-01, 8.7850467289719625e-01, 8.8785046728971961e-01, 8.9719626168224287e-01, 9.0654205607476623e-01, 9.1588785046728960e-01, 9.2523364485981296e-01, 9.3457943925233633e-01, 9.4392523364485970e-01, 9.5327102803738306e-01, 9.6261682242990643e-01, 9.7196261682242979e-01, 9.8130841121495316e-01, 9.9065420560747652e-01, 1., 1.};
static const casadi_real casadi_c2[112] = {0., 0., 0., 9.3457943925233638e-03, 1.8691588785046728e-02, 2.8037383177570090e-02, 3.7383177570093455e-02, 4.6728971962616821e-02, 5.6074766355140179e-02, 6.5420560747663545e-02, 7.4766355140186910e-02, 8.4112149532710276e-02, 9.3457943925233641e-02, 1.0280373831775701e-01, 1.1214953271028036e-01, 1.2149532710280372e-01, 1.3084112149532709e-01, 1.4018691588785046e-01, 1.4953271028037382e-01, 1.5887850467289719e-01, 1.6822429906542055e-01, 1.7757009345794392e-01, 1.8691588785046728e-01, 1.9626168224299065e-01, 2.0560747663551401e-01, 2.1495327102803738e-01, 2.2429906542056072e-01, 2.3364485981308408e-01, 2.4299065420560745e-01, 2.5233644859813081e-01, 2.6168224299065418e-01, 2.7102803738317754e-01, 2.8037383177570091e-01, 2.8971962616822428e-01, 2.9906542056074764e-01, 3.0841121495327101e-01, 3.1775700934579437e-01, 3.2710280373831774e-01, 3.3644859813084110e-01, 3.4579439252336447e-01, 3.5514018691588783e-01, 3.6448598130841120e-01, 3.7383177570093457e-01, 3.8317757009345793e-01, 3.9252336448598130e-01, 4.0186915887850466e-01, 4.1121495327102803e-01, 4.2056074766355139e-01, 4.2990654205607476e-01, 4.3925233644859812e-01, 4.4859813084112143e-01, 4.5794392523364480e-01, 4.6728971962616817e-01, 4.7663551401869153e-01, 4.8598130841121490e-01, 4.9532710280373826e-01, 5.0467289719626163e-01, 5.1401869158878499e-01, 5.2336448598130836e-01, 5.3271028037383172e-01, 5.4205607476635509e-01, 5.5140186915887845e-01, 5.6074766355140182e-01, 5.7009345794392519e-01, 5.7943925233644855e-01, 5.8878504672897192e-01, 5.9813084112149528e-01, 6.0747663551401865e-01, 6.1682242990654201e-01, 6.2616822429906538e-01, 6.3551401869158874e-01, 6.4485981308411211e-01, 6.5420560747663548e-01, 6.6355140186915884e-01, 6.7289719626168221e-01, 6.8224299065420557e-01, 6.9158878504672894e-01, 7.0093457943925230e-01, 7.1028037383177567e-01, 7.1962616822429903e-01, 7.2897196261682240e-01, 7.3831775700934577e-01, 7.4766355140186913e-01, 7.5700934579439250e-01, 7.6635514018691586e-01, 7.7570093457943923e-01, 7.8504672897196259e-01, 7.9439252336448596e-01, 8.0373831775700932e-01, 8.1308411214953269e-01, 8.2242990654205606e-01, 8.3177570093457942e-01, 8.4112149532710279e-01, 8.5046728971962615e-01, 8.5981308411214952e-01, 8.6915887850467288e-01, 8.7850467289719625e-01, 8.8785046728971961e-01, 8.9719626168224287e-01, 9.0654205607476623e-01, 9.1588785046728960e-01, 9.2523364485981296e-01, 9.3457943925233633e-01, 9.4392523364485970e-01, 9.5327102803738306e-01, 9.6261682242990643e-01, 9.7196261682242979e-01, 9.8130841121495316e-01, 9.9065420560747652e-01, 1., 1., 1.};

/* spline_derivative:(i0,i1[2x109])->(o0[2]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real *rr, *ss;
  casadi_real *w0=w+5, w1, *w2=w+8, *w3=w+224, *w4=w+442, *w5=w+658;
  /* #0: @0 = zeros(1x2) */
  casadi_clear(w0, 2);
  /* #1: @1 = input[0][0] */
  w1 = arg[0] ? arg[0][0] : 0;
  /* #2: @2 = zeros(2x108) */
  casadi_clear(w2, 216);
  /* #3: @3 = input[1][0] */
  casadi_copy(arg[1], 218, w3);
  /* #4: @4 = sparse: 109-by-108, 216 nnz
   (0, 0) -> -214
   (1, 0) -> 214
   (1, 1) -> -107
   (2, 1) -> 107
   (2, 2) -> -107
   (3, 2) -> 107
   (3, 3) -> -107
   (4, 3) -> 107
   (4, 4) -> -107
   (5, 4) -> 107
   (5, 5) -> -107
   (6, 5) -> 107
   (6, 6) -> -107
   (7, 6) -> 107
   (7, 7) -> -107
   (8, 7) -> 107
   (8, 8) -> -107
   (9, 8) -> 107
   (9, 9) -> -107
   (10, 9) -> 107
   (10, 10) -> -107
   (11, 10) -> 107
   (11, 11) -> -107
   (12, 11) -> 107
   (12, 12) -> -107
   (13, 12) -> 107
   (13, 13) -> -107
   (14, 13) -> 107
   (14, 14) -> -107
   (15, 14) -> 107
   (15, 15) -> -107
   (16, 15) -> 107
   (16, 16) -> -107
   (17, 16) -> 107
   (17, 17) -> -107
   (18, 17) -> 107
   (18, 18) -> -107
   (19, 18) -> 107
   (19, 19) -> -107
   (20, 19) -> 107
   (20, 20) -> -107
   (21, 20) -> 107
   (21, 21) -> -107
   (22, 21) -> 107
   (22, 22) -> -107
   (23, 22) -> 107
   (23, 23) -> -107
   (24, 23) -> 107
   (24, 24) -> -107
   (25, 24) -> 107
   (25, 25) -> -107
   (26, 25) -> 107
   (26, 26) -> -107
   (27, 26) -> 107
   (27, 27) -> -107
   (28, 27) -> 107
   (28, 28) -> -107
   (29, 28) -> 107
   (29, 29) -> -107
   (30, 29) -> 107
   (30, 30) -> -107
   (31, 30) -> 107
   (31, 31) -> -107
   (32, 31) -> 107
   (32, 32) -> -107
   (33, 32) -> 107
   (33, 33) -> -107
   (34, 33) -> 107
   (34, 34) -> -107
   (35, 34) -> 107
   (35, 35) -> -107
   (36, 35) -> 107
   (36, 36) -> -107
   (37, 36) -> 107
   (37, 37) -> -107
   (38, 37) -> 107
   (38, 38) -> -107
   (39, 38) -> 107
   (39, 39) -> -107
   (40, 39) -> 107
   (40, 40) -> -107
   (41, 40) -> 107
   (41, 41) -> -107
   (42, 41) -> 107
   (42, 42) -> -107
   (43, 42) -> 107
   (43, 43) -> -107
   (44, 43) -> 107
   (44, 44) -> -107
   (45, 44) -> 107
   (45, 45) -> -107
   (46, 45) -> 107
   (46, 46) -> -107
   (47, 46) -> 107
   (47, 47) -> -107
   (48, 47) -> 107
   (48, 48) -> -107
   (49, 48) -> 107
   (49, 49) -> -107
   (50, 49) -> 107
   (50, 50) -> -107
   (51, 50) -> 107
   (51, 51) -> -107
   (52, 51) -> 107
   (52, 52) -> -107
   (53, 52) -> 107
   (53, 53) -> -107
   (54, 53) -> 107
   (54, 54) -> -107
   (55, 54) -> 107
   (55, 55) -> -107
   (56, 55) -> 107
   (56, 56) -> -107
   (57, 56) -> 107
   (57, 57) -> -107
   (58, 57) -> 107
   (58, 58) -> -107
   (59, 58) -> 107
   (59, 59) -> -107
   (60, 59) -> 107
   (60, 60) -> -107
   (61, 60) -> 107
   (61, 61) -> -107
   (62, 61) -> 107
   (62, 62) -> -107
   (63, 62) -> 107
   (63, 63) -> -107
   (64, 63) -> 107
   (64, 64) -> -107
   (65, 64) -> 107
   (65, 65) -> -107
   (66, 65) -> 107
   (66, 66) -> -107
   (67, 66) -> 107
   (67, 67) -> -107
   (68, 67) -> 107
   (68, 68) -> -107
   (69, 68) -> 107
   (69, 69) -> -107
   (70, 69) -> 107
   (70, 70) -> -107
   (71, 70) -> 107
   (71, 71) -> -107
   (72, 71) -> 107
   (72, 72) -> -107
   (73, 72) -> 107
   (73, 73) -> -107
   (74, 73) -> 107
   (74, 74) -> -107
   (75, 74) -> 107
   (75, 75) -> -107
   (76, 75) -> 107
   (76, 76) -> -107
   (77, 76) -> 107
   (77, 77) -> -107
   (78, 77) -> 107
   (78, 78) -> -107
   (79, 78) -> 107
   (79, 79) -> -107
   (80, 79) -> 107
   (80, 80) -> -107
   (81, 80) -> 107
   (81, 81) -> -107
   (82, 81) -> 107
   (82, 82) -> -107
   (83, 82) -> 107
   (83, 83) -> -107
   (84, 83) -> 107
   (84, 84) -> -107
   (85, 84) -> 107
   (85, 85) -> -107
   (86, 85) -> 107
   (86, 86) -> -107
   (87, 86) -> 107
   (87, 87) -> -107
   (88, 87) -> 107
   (88, 88) -> -107
   (89, 88) -> 107
   (89, 89) -> -107
   (90, 89) -> 107
   (90, 90) -> -107
   (91, 90) -> 107
   (91, 91) -> -107
   (92, 91) -> 107
   (92, 92) -> -107
   (93, 92) -> 107
   (93, 93) -> -107
   (94, 93) -> 107
   (94, 94) -> -107
   (95, 94) -> 107
   (95, 95) -> -107
   (96, 95) -> 107
   (96, 96) -> -107
   (97, 96) -> 107
   (97, 97) -> -107
   (98, 97) -> 107
   (98, 98) -> -107
   (99, 98) -> 107
   (99, 99) -> -107
   (100, 99) -> 107
   (100, 100) -> -107
   (101, 100) -> 107
   (101, 101) -> -107
   (102, 101) -> 107
   (102, 102) -> -107
   (103, 102) -> 107
   (103, 103) -> -107
   (104, 103) -> 107
   (104, 104) -> -107
   (105, 104) -> 107
   (105, 105) -> -107
   (106, 105) -> 107
   (106, 106) -> -107
   (107, 106) -> 107
   (107, 107) -> -214
   (108, 107) -> 214 */
  casadi_copy(casadi_c0, 216, w4);
  /* #5: @2 = mac(@3,@4,@2) */
  casadi_mtimes(w3, casadi_s0, w4, casadi_s1, w2, casadi_s2, w, 0);
  /* #6: @2 = nonzeros(@2) */
  /* #7: @5 = BSplineParametric(@1, @2) */
  casadi_clear(w5, 2);
  CASADI_PREFIX(nd_boor_eval)(w5,1,casadi_c1,casadi_s3,casadi_s4,casadi_s5,w2,2,(&w1),casadi_s5, iw, w);
  /* #8: (@0[:2] = @5) */
  for (rr=w0+0, ss=w5; rr!=w0+2; rr+=1) *rr = *ss++;
  /* #9: output[0][0] = @0 */
  casadi_copy(w0, 2, res[0]);
  return 0;
}

/* spline:(i0,i1[2x109])->(o0[2]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, *w1=w+8, *w2=w+226;
  /* #0: @0 = input[0][0] */
  w0 = arg[0] ? arg[0][0] : 0;
  /* #1: @1 = input[1][0] */
  casadi_copy(arg[1], 218, w1);
  /* #2: @2 = BSplineParametric(@0, @1) */
  casadi_clear(w2, 2);
  CASADI_PREFIX(nd_boor_eval)(w2,1,casadi_c2,casadi_s6,casadi_s5,casadi_s5,w1,2,(&w0),casadi_s5, iw, w);
  /* #3: output[0][0] = @2 */
  casadi_copy(w2, 2, res[0]);
  return 0;
}

/* bicycle_model_cost_ext_cost_fun:(i0[6],i1[3],i2[0],i3[218])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real **res1=res+1, *rr, *ss;
  const casadi_real **arg1=arg+4;
  casadi_real w0, w1, w2, w3, *w4=w+664, *w5=w+882, w6, w7, w8, w9;
  /* #0: @0 = 0.02 */
  w0 = 2.0000000000000000e-02;
  /* #1: @1 = 1e-06 */
  w1 = 9.9999999999999995e-07;
  /* #2: @2 = input[0][5] */
  w2 = arg[0] ? arg[0][5] : 0;
  /* #3: @3 = 1 */
  w3 = 1.;
  /* #4: @2 = fmod(@2,@3) */
  w2  = fmod(w2,w3);
  /* #5: @4 = input[3][0] */
  casadi_copy(arg[3], 218, w4);
  /* #6: @4 = reshape(@4) */
  /* #7: @5 = spline_derivative(@2, @4) */
  arg1[0]=(&w2);
  arg1[1]=w4;
  res1[0]=w5;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #8: @3 = @5[1] */
  for (rr=(&w3), ss=w5+1; ss!=w5+2; ss+=1) *rr++ = *ss;
  /* #9: @3 = (@1+@3) */
  w3  = (w1+w3);
  /* #10: @6 = @5[0] */
  for (rr=(&w6), ss=w5+0; ss!=w5+1; ss+=1) *rr++ = *ss;
  /* #11: @1 = (@1+@6) */
  w1 += w6;
  /* #12: @3 = atan2(@3,@1) */
  w3  = atan2(w3,w1);
  /* #13: @1 = sin(@3) */
  w1 = sin( w3 );
  /* #14: @6 = input[0][0] */
  w6 = arg[0] ? arg[0][0] : 0;
  /* #15: @5 = spline(@2, @4) */
  arg1[0]=(&w2);
  arg1[1]=w4;
  res1[0]=w5;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #16: @2 = @5[0] */
  for (rr=(&w2), ss=w5+0; ss!=w5+1; ss+=1) *rr++ = *ss;
  /* #17: @6 = (@6-@2) */
  w6 -= w2;
  /* #18: @2 = (@1*@6) */
  w2  = (w1*w6);
  /* #19: @7 = cos(@3) */
  w7 = cos( w3 );
  /* #20: @8 = input[0][1] */
  w8 = arg[0] ? arg[0][1] : 0;
  /* #21: @9 = @5[1] */
  for (rr=(&w9), ss=w5+1; ss!=w5+2; ss+=1) *rr++ = *ss;
  /* #22: @8 = (@8-@9) */
  w8 -= w9;
  /* #23: @9 = (@7*@8) */
  w9  = (w7*w8);
  /* #24: @2 = (@2-@9) */
  w2 -= w9;
  /* #25: @2 = sq(@2) */
  w2 = casadi_sq( w2 );
  /* #26: @0 = (@0*@2) */
  w0 *= w2;
  /* #27: @7 = (@7*@6) */
  w7 *= w6;
  /* #28: @7 = (-@7) */
  w7 = (- w7 );
  /* #29: @1 = (@1*@8) */
  w1 *= w8;
  /* #30: @7 = (@7-@1) */
  w7 -= w1;
  /* #31: @7 = sq(@7) */
  w7 = casadi_sq( w7 );
  /* #32: @0 = (@0+@7) */
  w0 += w7;
  /* #33: @7 = 1e-05 */
  w7 = 1.0000000000000001e-05;
  /* #34: @1 = input[1][0] */
  w1 = arg[1] ? arg[1][0] : 0;
  /* #35: @1 = sq(@1) */
  w1 = casadi_sq( w1 );
  /* #36: @7 = (@7*@1) */
  w7 *= w1;
  /* #37: @0 = (@0+@7) */
  w0 += w7;
  /* #38: @7 = 1e-09 */
  w7 = 1.0000000000000001e-09;
  /* #39: @3 = sq(@3) */
  w3 = casadi_sq( w3 );
  /* #40: @7 = (@7*@3) */
  w7 *= w3;
  /* #41: @0 = (@0+@7) */
  w0 += w7;
  /* #42: @7 = 5 */
  w7 = 5.;
  /* #43: @3 = input[1][2] */
  w3 = arg[1] ? arg[1][2] : 0;
  /* #44: @7 = (@7*@3) */
  w7 *= w3;
  /* #45: @0 = (@0-@7) */
  w0 -= w7;
  /* #46: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int bicycle_model_cost_ext_cost_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int bicycle_model_cost_ext_cost_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int bicycle_model_cost_ext_cost_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void bicycle_model_cost_ext_cost_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int bicycle_model_cost_ext_cost_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void bicycle_model_cost_ext_cost_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void bicycle_model_cost_ext_cost_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void bicycle_model_cost_ext_cost_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int bicycle_model_cost_ext_cost_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int bicycle_model_cost_ext_cost_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real bicycle_model_cost_ext_cost_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* bicycle_model_cost_ext_cost_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* bicycle_model_cost_ext_cost_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* bicycle_model_cost_ext_cost_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s7;
    case 1: return casadi_s8;
    case 2: return casadi_s9;
    case 3: return casadi_s10;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* bicycle_model_cost_ext_cost_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s11;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int bicycle_model_cost_ext_cost_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 6;
  if (sz_w) *sz_w = 888;
  return 0;
}

CASADI_SYMBOL_EXPORT int bicycle_model_cost_ext_cost_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 3*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 6*sizeof(casadi_int);
  if (sz_w) *sz_w = 888*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
