#!/usr/bin/env python3
import numpy as np
import ctypes
import onnxruntime as ort

ONNX_PATH = "./src/policy_random.onnx"
SO_PATH   = "./src/libpolicy_random.so"

# Load C library
lib = ctypes.cdll.LoadLibrary(SO_PATH)
lib.pol_get_input_dim.restype  = ctypes.c_int
lib.pol_get_hidden_dim.restype = ctypes.c_int
lib.pol_get_output_dim.restype = ctypes.c_int

INPUT  = lib.pol_get_input_dim()
HIDDEN = lib.pol_get_hidden_dim()
ACTDim = lib.pol_get_output_dim()

lib.pol_step_api.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
if hasattr(lib, "pol_init"):
    lib.pol_init.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

# ONNX Runtime
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

def c_step(obs_np, h, c):
    act = np.zeros(ACTDim, dtype=np.float32)
    lib.pol_step_api(
        obs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return act, h, c  # h,c updated in place

def onnx_step(obs_np, h_prev, c_prev):
    out = sess.run(
        None,
        {
            "obs":  obs_np[None,:].astype(np.float32),
            "h_in": h_prev[None,:].astype(np.float32),
            "c_in": c_prev[None,:].astype(np.float32),
        },
    )
    act, h_out, c_out = out
    return (act.reshape(-1).astype(np.float32),
            h_out.reshape(-1).astype(np.float32),
            c_out.reshape(-1).astype(np.float32))

TRIALS = 5
STEPS  = 16
rng = np.random.default_rng(1)

max_abs = 0.0

for t in range(TRIALS):
    # init both sides
    h_c = np.zeros(HIDDEN, dtype=np.float32)
    c_c = np.zeros(HIDDEN, dtype=np.float32)
    if hasattr(lib, "pol_init"):
        lib.pol_init(
            h_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        h_o = h_c.copy()
        c_o = c_c.copy()
    else:
        h_o = np.zeros(HIDDEN, dtype=np.float32)
        c_o = np.zeros(HIDDEN, dtype=np.float32)

    for s in range(STEPS):
        obs = rng.standard_normal(INPUT).astype(np.float32)

        act_c, h_c, c_c = c_step(obs, h_c, c_c)
        act_o, h_o, c_o = onnx_step(obs, h_o, c_o)

        diff = float(np.max(np.abs(act_c - act_o)))
        if diff > max_abs:
            max_abs = diff

print(f"Max abs diff C vs ONNX: {max_abs:.3e}")
# Expected: very small, e.g. < 1e-6