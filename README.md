# Model Inference Utilities (Quick and Dirty (QnD) version)

A lightweight C implementation for neural network policy inference with validation and benchmarking tools against PyTorch models.

---

**Version:** 1.0.0  
**Author:** Samurice  
**Date:** October 29, 2025  
**Location:** Hanoi, Vietnam

---
## Directory Structure

```
inference_opt/
├── policy_cpkt/
│   └── policy_random.pt
├── policy_test.c
├── policy_test.c
├── src/
│   ├── export_policy_weights.py
│   ├── policy_infer.h
│   ├── policy_random.json
│   └── policy_random_weights.h
├── validate_c_vs_torch.py
└── validate_QnD_vs_torch.py
```

## Quick Start

### 1. Export Policy Weights from PyTorch Checkpoint

First, generate the weights header file from the JIT checkpoint:

```bash
python export_policy_weights.py ../policy_cpkt/policy_random.pt \
    --out-prefix=policy_random \
    --activation=elu
```

This creates `policy_random_weights.h` and `policy_random.json` in the `src/` directory.

### 2. Quick Validation: Custom Implementation vs PyTorch

Test the mathematical formulation against the PyTorch library:

```bash
python validate_QnD_vs_torch.py ./policy_cpkt/policy_random.pt \
    --trials 10 \
    --steps 10 \
    --init zeros
```

**Options:**
- `--trials`: Number of random trials to run
- `--steps`: Number of inference steps per trial
- `--init`: Initialization method (`zeros`, `random`)

### 3. Test C Implementation

Compile and run the C implementation:

```bash
# Compile
gcc -O3 -std=c99 policy_test.c -lm -o policy_test

# Run
./policy_test
```

### 4. Validate C Implementation vs PyTorch

Ensure the C implementation matches PyTorch numerically:

```bash
python validate_c_vs_torch.py \
    --pt ./policy_cpkt/policy_random.pt \
    --weights_h ./src/policy_random_weights.h \
    --infer_h ./src/policy_infer.h \
    --trials 10 \
    --steps 20 \
    --init zeros
```

This validator will report:
- Maximum absolute error between C and PyTorch outputs
- Mean absolute error across all trials
- Whether the implementation passes numerical tolerance

### 5. Benchmark Performance

Measure computation speed and memory usage:

```bash
# Ubuntu/Linux
gcc -O3 -std=c99 -march=native -ffast-math -fno-trapping-math -fno-math-errno \
    policy_bench.c -lm -o policy_bench

# Windows (MSVC)
cl /O2 /std:c11 policy_bench.c

# macOS (Clang)
clang -O3 -std=c99 -march=native -ffast-math policy_bench.c -lm -o policy_bench

# Run benchmark
./policy_bench
```
### 6. Vinrobotics Policy Deployment Tutorial

#### For Python Deployment

**Step 1: Compile the policy shared library**

Compile `src/policy_shim.c` as a shared library:

```bash
gcc -O3 -std=c99 -fPIC -shared policy_shim.c -lm -o libpolicy_random.so
```

**Step 2: Test and deploy**

We can now test and deploy the policy inference in Python:

```bash
# Test the policy inference
python examples/test_policy_infer.py

# Deploy to real robot
python examples/deploy_real_policy_infer.py
```



#### For C++ Deployment

**Modified files for C++ integration:**

```
utils/inference_opt/examples/
├── vr_cali_locomotion_controller_infer.cpp
└── vr_cali_locomotion_controller_infer.hpp
```

**Build system notes (CMake):**

In the controller package `CMakeLists.txt`, link against the policy library:

```cmake
# Find or specify the policy library
find_library(POLICY_random_LIB 
    NAMES policy_random 
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# Or if you built a shared lib `libpolicy_random.so` installed somewhere:
set(POLICY_random_LIB "/path/to/libpolicy_random.so")

# Add your controller executable
add_executable(vr_cali_controller
    src/vr_cali_locomotion_controller_infer.cpp
)

# Link against the policy library
target_link_libraries(vr_cali_controller
    ${POLICY_random_LIB}
    m  # Math library
)

# Include directories for policy headers
target_include_directories(vr_cali_controller PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/inference_opt/src
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/inference_opt/examples
)
```

**Alternative: Build C source directly in CMake**

If you prefer to compile the C source directly instead of using a shared library:

```cmake
# Add the C policy implementation
add_library(policy_random_static STATIC
    utils/inference_opt/src/policy_infer.c
    utils/inference_opt/src/policy_shim.c
)

# Set C99 standard for the policy library
set_target_properties(policy_random_static PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)

# Optimization flags for the policy
target_compile_options(policy_random_static PRIVATE
    -O3
    -march=native
    -ffast-math
)

# Link your controller against the static library
target_link_libraries(vr_cali_controller
    policy_random_static
    m
)
```
### 7. Export Model to ONNX for NXP ARM Cortex-A53/A55

For deployment on NXP i.MX platforms with ARM Cortex-A53/A55 processors.

#### Step 1: Export PyTorch Model to ONNX

Convert your TorchScript checkpoint to ONNX format:

```bash
python export_policy_to_onnx.py \
    --ts ../policy_cpkt/policy_random.pt \
    --out policy_random.onnx
```

This generates `policy_random.onnx` that can be used with ONNX Runtime on embedded systems.

#### Step 2: Validation

Validate the ONNX export against both C implementation and PyTorch:

```bash
# Validate ONNX vs C implementation
python validate_onnx_vs_c.py

# Validate ONNX vs PyTorch (reference)
python validate_onnx_vs_torch.py
```

#### Deployment Options for NXP i.MX Platforms

When deploying on NXP ARM processors (i.MX 8M Plus, i.MX 93, etc.), you have three main options:

##### 1. Handwritten C (Current Implementation) ⭐ **Recommended for Hard Real-Time**

**Pros:**
- ✅ **Tiny footprint**: Minimal binary size, perfect for embedded systems
- ✅ **Deterministic timing**: No runtime surprises or garbage collection
- ✅ **Static memory**: All allocations known at compile time
- ✅ **No runtime dependencies**: Zero external libraries needed
- ✅ **Perfect for hard real-time control loops**: Sub-millisecond, jitter-free execution

**Cons:**
- ❌ No automatic NPU/GPU usage (but you don't need it for small policy networks)
- ❌ Requires manual updates if network architecture changes

**When to use:**
- Safety-critical control systems
- Low-jitter, high-frequency control loops (>100 Hz)
- Resource-constrained environments
- When determinism is paramount

##### 2. ONNX Runtime with NXP eIQ

**Pros:**
- ✅ **No custom math implementation**: Just export ONNX and run
- ✅ **Easier maintenance**: Policy changes don't require code rewrite
- ✅ **Cross-platform**: Same model runs on different targets

**Cons:**
- ⚠️ **Runtime overhead**: Additional latency from runtime + allocator + threading
- ⚠️ **Potential jitter**: Watch for timing variability in control loops
- ⚠️ **NPU acceleration not guaranteed**: Your small policy network may not benefit
- ⚠️ **Marginal gains**: For tiny models, NPU speedup may be negligible vs. optimized C

**Installation on i.MX:**
```bash
# Install ONNX Runtime for ARM
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-aarch64-1.16.0.tgz
tar -xzf onnxruntime-linux-aarch64-1.16.0.tgz

# Link against ONNX Runtime in your build
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
```

**Usage example:**
```cpp
#include <onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PolicyInference");
Ort::SessionOptions session_options;
Ort::Session session(env, "policy_random.onnx", session_options);

// Run inference
auto input_tensor = Ort::Value::CreateTensor<float>(/*...*/);
auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                   input_names, &input_tensor, 1,
                                   output_names, 1);
```

**When to use:**
- Policies change frequently during development
- Not latency-critical (>10ms acceptable)
- Want framework-agnostic deployment

##### 3. TensorFlow Lite (LiteRT) with NXP eIQ

**Pros:**
- ✅ **Best NXP integration**: Optimized delegates for Neutron NPU and GPU
- ✅ **Future-proof**: Great for adding vision/ML models later
- ✅ **Mature tooling**: Extensive optimization and quantization support

**Cons:**
- ⚠️ **Complex conversion**: PyTorch → ONNX → TFLite or PyTorch → TFLite
- ⚠️ **Questionable benefits**: For tiny recurrent controllers, gains over handwritten C are minimal
- ⚠️ **Runtime overhead**: Still has interpreter/delegate overhead

**Conversion pipeline:**
```bash
# PyTorch → ONNX → TensorFlow → TFLite
pip install onnx-tf tf2onnx tensorflow

# Convert ONNX to TensorFlow
onnx-tf convert -i policy_random.onnx -o policy_random_tf

# Convert TensorFlow to TFLite
python -c "
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('policy_random_tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('policy_random.tflite', 'wb').write(tflite_model)
"
```

**When to use:**
- Planning to add CNN/vision models that benefit from NPU
- Need INT8 quantization for extreme efficiency
- Building a larger ML pipeline on i.MX

#### Recommendation for Different Use Cases

**For safety-critical, low-jitter locomotion control:**

> **Stick with the handwritten C implementation.** It's ideal and already validated against PyTorch.

The current C engine provides:
- ✅ Deterministic sub-millisecond inference
- ✅ Zero runtime dependencies
- ✅ Proven numerical accuracy vs. PyTorch
- ✅ No jitter in control loops
- ✅ Minimal power consumption

**Consider ONNX/TFLite only if:**
- You need to rapidly iterate on policy architectures
- You're adding vision processing that genuinely needs NPU acceleration
- Control loop timing requirements are relaxed (>10ms acceptable)

#### Performance Comparison on NXP i.MX 8M Plus

Typical inference times for a small policy network (256-256 MLP):

| Method | Latency | Jitter | Memory | Power |
|--------|---------|--------|--------|-------|
| **Handwritten C** | **50-200 µs** | **<5 µs** | **~50 KB** | **Minimal** |
| ONNX Runtime | 500-2000 µs | ~50 µs | ~5 MB | Low |
| TFLite (CPU) | 300-1000 µs | ~30 µs | ~2 MB | Low |
| TFLite (NPU) | 200-800 µs* | ~100 µs | ~3 MB | Medium |

*NPU gains depend heavily on model architecture and may not materialize for small MLPs.

#### Testing on NXP Hardware

```bash
# Cross-compile for ARM (on x86 host)
aarch64-linux-gnu-gcc -O3 -march=armv8-a+crc+simd \
    -ffast-math -funroll-loops \
    policy_infer.c -lm -o policy_infer_arm

# Copy to i.MX board and run
scp policy_infer_arm root@imx-board:/home/root/
ssh root@imx-board "./policy_infer_arm"
```
## CPU Optimization Playbook (Intel & ARM)

### 1. Compiler Flags (Safe → Aggressive)

**Baseline (safe, reproducible):**
```bash
-O3 -march=native -fno-math-errno -fno-signaling-nans
```

**Faster math (allows slight numeric drift):**
```bash
-ffast-math -fno-trapping-math -funsafe-math-optimizations
```

**Note:** Clang/LLVM often vectorizes better than GCC on some kernels—try both compilers.

**Tip:** Maintain two builds:
- **Reference build:** No fast-math for numerical accuracy verification
- **Fast build:** All optimizations enabled
- Use this validator to ensure max error vs TorchScript stays within tolerance

### 2. Vectorization / SIMD

**For Intel/AMD:**
```bash
# Haswell and newer (AVX2 + FMA)
-mavx2 -mfma

# Skylake-X/Ice Lake/Sapphire Rapids (AVX-512)
-mavx512f
```

**Note:** If using `-march=native`, these are auto-detected.

**Optimization checklist:**
- ✅ Ensure inner loops use unit-stride memory access
- ✅ Keep loops simple for compiler auto-vectorization
- ✅ Our matvec uses contiguous rows (`Wi[j]`) which compilers optimize well
- ✅ Consider unrolling inner loops: `-funroll-loops`
- ✅ Add `restrict` qualifiers to pointer arguments to hint no aliasing

**Example optimization:**
```c
// Add restrict to help compiler vectorize
void matvec(const float* restrict W, const float* restrict x, 
            float* restrict y, int rows, int cols) {
    #pragma omp simd  // OpenMP SIMD hint
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}
```

### 3. Architecture-Specific Builds

**Complete optimization examples:**

```bash
# Intel/AMD (AVX2)
gcc -O3 -march=haswell -mavx2 -mfma -ffast-math -funroll-loops \
    -fno-trapping-math policy_bench.c -lm -o policy_bench_avx2

# Intel (AVX-512)
gcc -O3 -march=skylake-avx512 -mavx512f -ffast-math -funroll-loops \
    policy_bench.c -lm -o policy_bench_avx512

# ARM NEON (Raspberry Pi 4, Apple Silicon)
gcc -O3 -march=native -ffast-math -funroll-loops \
    policy_bench.c -lm -o policy_bench_arm
```

## Validation Workflow

```
┌─────────────────────────┐
│  PyTorch Checkpoint     │
│  (policy_random.pt)         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Export Weights         │
│  export_policy_weights  │
└───────────┬─────────────┘
            │
            ├─────────────────┬─────────────────┐
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Quick Test   │  │   C Test     │  │  Validation  │
    │ QnD vs Torch │  │ policy_test  │  │ C vs Torch   │
    └──────────────┘  └──────────────┘  └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │  Benchmark   │
                                        │ policy_bench │
                                        └──────────────┘
```

## Expected Performance

With proper optimization flags, the C implementation should achieve:
- **10-100x faster** than Python/PyTorch for single inference
- **Sub-millisecond latency** for typical policy networks
- **Minimal memory footprint** (no framework overhead)

## Troubleshooting

**Numerical differences:**
- Small differences (< 1e-5) are normal due to floating-point operations
- Check if `-ffast-math` causes larger errors; use reference build to verify
- Ensure consistent activation functions (e.g., ELU parameters)

**Performance issues:**
- Verify compiler flags with: `gcc -Q --help=optimizers`
- Check vectorization reports: Add `-fopt-info-vec` to see what vectorized
- Profile with: `perf stat ./policy_bench` on Linux

**Compilation errors:**
- Ensure C99 or C11 standard support
- Link math library with `-lm` on Unix-like systems
- Check that generated headers match this network architecture

## Real-Time Integration

### Integration Patterns for Production Systems

#### A) Real-Time Control: One Run = One Episode

For real-time control systems where the application runs continuously

**Key principle:** Reset once at startup; no "steps" concept required.

```c
// main_rt_control.c
#include "src/policy_infer.h"
#include "src/policy_random_weights.h"

int main() {
    // Initialize policy network once at startup
    PolicyNetwork policy;
    init_policy(&policy, &policy_random_weights);
    
    // Optional: warm-up inference to ensure caches are loaded
    float dummy_obs[OBS_DIM] = {0};
    float dummy_action[ACT_DIM];
    policy_forward(&policy, dummy_obs, dummy_action);
    
    // Main control loop - runs continuously
    while (system_running) {
        // 1. Get current observation from sensors/state estimator
        float observation[OBS_DIM];
        read_sensors(observation);
        
        // 2. Run inference (sub-millisecond latency)
        float action[ACT_DIM];
        policy_forward(&policy, observation, action);
        
        // 3. Send commands to actuators
        send_commands(action);
        
        // 4. Wait for next control cycle (e.g., 1kHz = 1ms)
        wait_next_cycle();
    }
    
    // Cleanup (only when shutting down)
    cleanup_policy(&policy);
    
    return 0;
}
```

**Compilation for real-time systems:**
```bash
# Maximum performance with deterministic timing
gcc -O3 -march=native -ffast-math -fno-trapping-math \
    -fno-math-errno -funroll-loops -flto \
    main_rt_control.c policy_infer.c -lm -o rt_controller

# For hard real-time (PREEMPT_RT kernel)
gcc -O3 -march=native -ffast-math -static \
    main_rt_control.c policy_infer.c -lm -o rt_controller
```

**Real-time considerations:**
- 🎯 **No dynamic memory allocation** in the control loop
- ⚡ **Predictable execution time** (use `-fno-trapping-math`)
- 🔒 **Lock memory pages** to prevent page faults (`mlockall()`)
- 📊 **Monitor worst-case execution time** during validation
- 🏃 **Set real-time scheduling** (`SCHED_FIFO` on Linux)

**Example with RT scheduling:**
```c
#include <sched.h>
#include <sys/mman.h>

int main() {
    // Lock all memory to prevent page faults
    mlockall(MCL_CURRENT | MCL_FUTURE);
    
    // Set real-time priority
    struct sched_param param;
    param.sched_priority = 80; // High priority
    sched_setscheduler(0, SCHED_FIFO, &param);
    
    // ... rest of control loop ...
}
```

#### B) Episodic / Batch Processing

For applications that process episodes or batches:

```c
// Process multiple episodes
for (int episode = 0; episode < num_episodes; episode++) {
    reset_environment(&env);
    
    for (int step = 0; step < max_steps; step++) {
        policy_forward(&policy, env.obs, action);
        step_environment(&env, action);
        
        if (env.done) break;
    }
}
```

#### C) Multi-Threading for Parallel Inference

For systems running multiple policies or parallel environments:

```c
#include <pthread.h>

typedef struct {
    PolicyNetwork policy;
    int thread_id;
} ThreadData;

void* control_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    // Each thread has its own policy instance (thread-safe)
    while (running) {
        policy_forward(&data->policy, obs, action);
        // ... control logic ...
    }
    return NULL;
}

int main() {
    pthread_t threads[NUM_ROBOTS];
    ThreadData thread_data[NUM_ROBOTS];
    
    // Initialize separate policy for each robot
    for (int i = 0; i < NUM_ROBOTS; i++) {
        init_policy(&thread_data[i].policy, &policy_random_weights);
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], NULL, control_thread, &thread_data[i]);
    }
    
    // Wait for threads
    for (int i = 0; i < NUM_ROBOTS; i++) {
        pthread_join(threads[i], NULL);
    }
}
```

### Performance Monitoring

Add timing tools to verify real-time performance:

```c
#include <time.h>

void measure_inference_time(PolicyNetwork* policy, float* obs, float* action) {
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    policy_forward(policy, obs, action);
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed_us = (end.tv_sec - start.tv_sec) * 1e6 +
                        (end.tv_nsec - start.tv_nsec) / 1e3;
    
    printf("Inference time: %.2f µs\n", elapsed_us);
}
```

## License


## Citation
