// TIME testing utility for policy inference
// Samurice: Hanoi 29 Oct 2025
// How to use:
// Ubuntu/Linux
// gcc -O3 -std=c99 -march=native -ffast-math -fno-trapping-math -fno-math-errno \
  policy_bench.c -lm -o policy_bench
  // Windows (clang):cl /O2 /std:c11 policy_bench.c
// MacOS (clang): clang -O3 -std=c99 -march=native -ffast-math policy_bench.c -lm -o policy_bench
#if !defined(_WIN32) && !defined(__APPLE__)
  #ifndef _POSIX_C_SOURCE
  #define _POSIX_C_SOURCE 199309L
  #endif
#endif

#define WEIGHTS_HEADER "policy_random_weights.h"
#include "src/policy_infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// -------- portable monotonic clock (ns) --------
#if defined(_WIN32)

  #include <windows.h>
  static inline uint64_t nsec_now(void){
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER ctr;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ctr);
    long double s = (long double)ctr.QuadPart / (long double)freq.QuadPart;
    return (uint64_t)(s * 1.0e9L);
  }

#elif defined(__APPLE__)

  #include <mach/mach_time.h>
  static inline uint64_t nsec_now(void){
    static mach_timebase_info_data_t ti = {0,0};
    if (ti.denom == 0) mach_timebase_info(&ti);
    uint64_t t = mach_absolute_time();
    return (t * (uint64_t)ti.numer) / (uint64_t)ti.denom;
  }

#else

  #include <time.h>     // _POSIX_C_SOURCE 
  static inline uint64_t nsec_now(void){
    struct timespec ts;
    #ifdef CLOCK_MONOTONIC_RAW
      clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    #else
      clock_gettime(CLOCK_MONOTONIC, &ts);
    #endif
    return (uint64_t)ts.tv_sec*1000000000ull + (uint64_t)ts.tv_nsec;
  }

#endif
// -----------------------------------------------

#ifndef WARMUP_STEPS
#define WARMUP_STEPS   2000
#endif
#ifndef BENCH_STEPS
#define BENCH_STEPS    20000
#endif

// Simple deterministic RNG (xorshift32 + Box–Muller)
static inline uint32_t rng_u32(uint32_t *s){ uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; return *s=x; }
static inline float rng_norm(uint32_t *s){
    float u1 = ((rng_u32(s)>>8) + 1) * (1.0f/16777217.0f);
    float u2 = ((rng_u32(s)>>8) + 1) * (1.0f/16777217.0f);
    float r = sqrtf(-2.0f * logf(u1));
    float th = 6.28318530718f * u2;
    return r * cosf(th);
}

int main(void){
    printf("POL_INPUT=%d  POL_HIDDEN=%d  POL_LAYERS=%d  OUT=%d  ACT_ID=%d\n",
           POL_INPUT, POL_HIDDEN, POL_LAYERS, POL_LIN_OUT[POL_LAYERS-1], POL_ACT_ID);

    float h[POL_HIDDEN], c[POL_HIDDEN];
    pol_reset_exported(h, c); // or pol_reset_zero(h,c)

    float x[POL_INPUT];
    float y[256]; // >= OUT
    int outdim = POL_LIN_OUT[POL_LAYERS-1];

    // Warmup
    uint32_t seed = 1u;
    for (int i=0;i<WARMUP_STEPS;++i){
        for (int j=0;j<POL_INPUT;++j) x[j] = rng_norm(&seed);
        pol_step(x, h, c, y);
    }

    volatile float checksum = 0.f;
    uint64_t t0 = nsec_now();
    for (int i=0;i<BENCH_STEPS;++i){
        for (int j=0;j<POL_INPUT;++j) x[j] = rng_norm(&seed);
        pol_step(x, h, c, y);
        for (int k=0;k<outdim;++k) checksum += y[k];
    }
    uint64_t t1 = nsec_now();

    double ns = (double)(t1 - t0);
    double us_per = ns / (double)BENCH_STEPS / 1000.0;
    double khz = 1000.0 / us_per;

    printf("checksum=%.6f\n", checksum);
    printf("bench: %d steps in %.3f ms  =>  %.3f us/step  (%.2f kHz)\n",
           BENCH_STEPS, ns/1e6, us_per, khz);
    return 0;
}
