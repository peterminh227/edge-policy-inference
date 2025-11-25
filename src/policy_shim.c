// policy_shim.c
// Build as a shared library to be called from Python via ctypes.
// How to use:
// Ubuntu/Linux to buyld shared lib:
// gcc -O3 -std=c99 -fPIC -shared     policy_shim.c -lm -o libpolicy_m2.so
#define WEIGHTS_HEADER "policy_M2_weights.h"
#include "policy_infer.h"

// Exported symbols must be visible from the shared lib
#if defined(_WIN32)
  #define API __declspec(dllexport)
#else
  #define API __attribute__((visibility("default")))
#endif

// Return dims so Python can assert shapes
API int pol_get_input_dim(void)  { return POL_INPUT; }
API int pol_get_hidden_dim(void) { return POL_HIDDEN; }
API int pol_get_output_dim(void) { return POL_LIN_OUT[POL_LAYERS-1]; }

// Initialize LSTM state (use exported warm-start; alternative is zero-init version)
API void pol_init(float *h, float *c) {
    pol_reset_exported(h, c);   // or pol_reset_zero(h, c);
}

// One inference step: obs + state_in -> action + state_out (updated in-place)
API void pol_step_api(const float *obs,
                      float *h, float *c,
                      float *action_out)
{
    pol_step(obs, h, c, action_out);
}
// End of file
