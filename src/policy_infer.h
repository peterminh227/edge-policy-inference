// policy_infer.h
// LSTM-based policy inference (header-only)
// Samurice: Hanoi 29 Oct 2025
#ifndef POLICY_INFER_H
#define POLICY_INFER_H


#ifndef WEIGHTS_HEADER
#define WEIGHTS_HEADER "policy_random_weights.h"
#endif
#include WEIGHTS_HEADER


#ifdef __cplusplus
extern "C" {
#endif
#include <string.h>  // memcpy
#include <math.h>    // expf, tanhf
// --- START Helpers for state init / access -----------------------------------------

// Reset LSTM state to exported h0,c0
static inline void pol_reset(float* h, float* c);

// Initialize LSTM state to zeros (strict cold start).
static inline void pol_reset_zero(float* h, float* c){
    for (int i=0;i<POL_HIDDEN;++i){ h[i]=0.0f; c[i]=0.0f; }
}

// Initialize LSTM state to exported buffers (warm start matching training/export).
static inline void pol_reset_exported(float* h, float* c){
    memcpy(h, POL_H0, sizeof(float)*POL_HIDDEN);
    memcpy(c, POL_C0, sizeof(float)*POL_HIDDEN);
}

// Copy provided state into runtime state (deterministic restore / testing).
static inline void pol_set_state(const float* h_in, const float* c_in, float* h, float* c){
    memcpy(h, h_in, sizeof(float)*POL_HIDDEN);
    memcpy(c, c_in, sizeof(float)*POL_HIDDEN);
}
// Read current state (e.g., to checkpoint state or compare against Torch).
static inline void pol_get_state(const float* h, const float* c, float* h_out, float* c_out){
    memcpy(h_out, h, sizeof(float)*POL_HIDDEN);
    memcpy(c_out, c, sizeof(float)*POL_HIDDEN);
}
// Matrix-vector with bias: y = W*x + b  - SIMPLE HACK
static inline void __pol_matvec_bias(const float* __restrict W,
                                     const float* __restrict x,
                                     const float* __restrict b,
                                     float* __restrict y, int M, int N){
    for(int i=0;i<M;++i){
        const float* Wi = W + i*N;
        float acc = b[i];
        for(int j=0;j<N;++j) acc += Wi[j]*x[j];
        y[i]=acc;
    }
}

// ---END Helpers for state init / access -----------------------------------------

// One-step inference (no malloc).
// Inputs:
//   x[POL_INPUT], h[POL_HIDDEN], c[POL_HIDDEN]
// Output:
//   y_out[ POL_LIN_OUT[POL_LAYERS-1] ]
static inline void pol_step(const float x[], float h[], float c[], float* y_out);

#ifdef __cplusplus
}
#endif

// ======== Implementation (header-only) ========

#include <math.h>
#include <string.h>

static inline float __pol_sigmoid(float v){ return 1.0f/(1.0f+expf(-v)); }
static inline float __pol_relu(float v){ return v>0.0f? v:0.0f; }
static inline float __pol_elu(float v){ return v>=0.0f? v:(expf(v)-1.0f); }
static inline float __pol_selu(float v){
    const float l=1.0507009873554806f, a=1.6732632423543772f;
    return v>0? l*v: l*a*(expf(v)-1.0f);
}
static inline float __pol_lrelu(float v){ const float a=0.01f; return v>=0.0f? v: a*v; }
static inline float __pol_tanh(float v){ return tanhf(v); }
static inline float __pol_sigmoid_id(float v){ return __pol_sigmoid(v); }

static inline void __pol_act_vec(float* y, int n, int act_id){
    switch(act_id){
        case 0: for(int i=0;i<n;++i) y[i]=__pol_relu(y[i]); break;     // ReLU
        case 1: for(int i=0;i<n;++i) y[i]=__pol_elu(y[i]); break;      // ELU
        case 2: for(int i=0;i<n;++i) y[i]=__pol_selu(y[i]); break;     // SELU
        case 3: for(int i=0;i<n;++i) y[i]=__pol_lrelu(y[i]); break;    // LeakyReLU
        case 4: for(int i=0;i<n;++i) y[i]=__pol_tanh(y[i]); break;     // tanh
        case 5: for(int i=0;i<n;++i) y[i]=__pol_sigmoid_id(y[i]); break; // sigmoid
        default: for(int i=0;i<n;++i) y[i]=__pol_elu(y[i]); break;     // default ELU
    }
}

static inline void __pol_matvec(const float* W, const float* x, float* y, int M, int N){
    for(int i=0;i<M;++i){
        float acc=0.0f;
        const float* Wi=W + i*N;
        for(int j=0;j<N;++j) acc += Wi[j]*x[j];
        y[i]=acc;
    }
}
static inline void __pol_addbias(float* y, const float* b, int M){ for(int i=0;i<M;++i) y[i]+=b[i]; }
static inline void __pol_matvec_acc(const float* W, const float* x, float* y, int M, int N){
    for(int i=0;i<M;++i){
        float acc=0.0f;
        const float* Wi=W + i*N;
        for(int j=0;j<N;++j) acc += Wi[j]*x[j];
        y[i]+=acc;
    }
}

static inline void __pol_lstm_step(const float* x, const float* h_in, const float* c_in,
                                   float* h_out, float* c_out){
    const int H = POL_HIDDEN;
    const int H4 = 4*H;
    float g[4*POL_HIDDEN];

    // gates = W_ih*x + b_ih + W_hh*h + b_hh
    __pol_matvec(POL_W_IH, x, g, H4, POL_INPUT);
    __pol_addbias(g, POL_B_IH, H4);
    __pol_matvec_acc(POL_W_HH, h_in, g, H4, H);
    __pol_addbias(g, POL_B_HH, H4);

    const float* gi = g + 0*H;
    const float* gf = g + 1*H;
    const float* gg = g + 2*H;
    const float* go = g + 3*H;
    for(int k=0;k<H;++k){
        float i = __pol_sigmoid(gi[k]);
        float f = __pol_sigmoid(gf[k]);
        float gtil = tanhf(gg[k]);
        float o = __pol_sigmoid(go[k]);
        float c = f*c_in[k] + i*gtil;
        c_out[k] = c;
        h_out[k] = o * tanhf(c);
    }
}

static inline void pol_reset(float* h, float* c){
    memcpy(h, POL_H0, sizeof(float)*POL_HIDDEN);
    memcpy(c, POL_C0, sizeof(float)*POL_HIDDEN);
}

// Macros to call per-layer arrays by index in a portable way:
#define __POL_DO_LAYER(L, SRC, DST, OUTL, INL) \
    case L: __pol_matvec(POL_AW_##L, SRC, DST, OUTL, INL); __pol_addbias(DST, POL_AB_##L, OUTL); break

static inline void pol_step(const float x[], float h[], float c[], float* y_out){
    float h2[POL_HIDDEN], c2[POL_HIDDEN];
    __pol_lstm_step(x, h, c, h2, c2);

    // First actor layer
    int out0 = POL_LIN_OUT[0];
    float bufA[512]; // scratch (ensure >= max hidden size in your policies)
    __pol_matvec(POL_AW_0, h2, bufA, out0, POL_LIN_IN[0]);
    __pol_addbias(bufA, POL_AB_0, out0);
    if(POL_LAYERS>1) __pol_act_vec(bufA, out0, POL_ACT_ID);

    // Middle layers
    for(int li=1; li<POL_LAYERS-1; ++li){
        int inL = POL_LIN_IN[li], outL = POL_LIN_OUT[li];
        float bufB[512];
        switch(li){
        #if POL_LAYERS > 1
            __POL_DO_LAYER(1, bufA, bufB, outL, inL);
        #endif
        #if POL_LAYERS > 2
            __POL_DO_LAYER(2, bufA, bufB, outL, inL);
        #endif
        #if POL_LAYERS > 3
            __POL_DO_LAYER(3, bufA, bufB, outL, inL);
        #endif
        #if POL_LAYERS > 4
            __POL_DO_LAYER(4, bufA, bufB, outL, inL);
        #endif
        #if POL_LAYERS > 5
            __POL_DO_LAYER(5, bufA, bufB, outL, inL);
        #endif
        default: break;
        }
        __pol_act_vec(bufB, outL, POL_ACT_ID);
        memcpy(bufA, bufB, sizeof(float)*outL);
    }

    // Final linear
    if(POL_LAYERS==1){
        memcpy(y_out, bufA, sizeof(float)*out0);
    }else{
        int last = POL_LAYERS-1;
        int inL = POL_LIN_IN[last], outL = POL_LIN_OUT[last];
        switch(last){
        #if POL_LAYERS > 1
            __POL_DO_LAYER(1, bufA, y_out, outL, inL);
        #endif
        #if POL_LAYERS > 2
            __POL_DO_LAYER(2, bufA, y_out, outL, inL);
        #endif
        #if POL_LAYERS > 3
            __POL_DO_LAYER(3, bufA, y_out, outL, inL);
        #endif
        #if POL_LAYERS > 4
            __POL_DO_LAYER(4, bufA, y_out, outL, inL);
        #endif
        #if POL_LAYERS > 5
            __POL_DO_LAYER(5, bufA, y_out, outL, inL);
        #endif
        default: break;
        }
    }

    // Commit recurrent state
    memcpy(h, h2, sizeof(float)*POL_HIDDEN);
    memcpy(c, c2, sizeof(float)*POL_HIDDEN);
}

#endif // POLICY_INFER_H
