#!/usr/bin/env python3
import numpy as np
import torch
import onnxruntime as ort

from src.export_policy_to_onnx import PolicyONNX  

PT_PATH   = "policy_cpkt/policy_random.pt"
ONNX_PATH = "src/policy_random.onnx"

# Load TS
policy_ts = torch.jit.load(PT_PATH, map_location="cpu").eval()
sd = policy_ts.state_dict()

# Build same math module in PyTorch for convenience
pol = PolicyONNX(sd).eval()
INPUT  = pol.input_dim
HIDDEN = pol.hidden_dim
ACTDim = pol.A4_b.shape[0]

# ONNX Runtime session
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

def torch_step(obs_np, h_prev, c_prev):
    obs = torch.from_numpy(obs_np[None, :])   # (1,INPUT)
    h   = torch.from_numpy(h_prev[None, :])
    c   = torch.from_numpy(c_prev[None, :])
    with torch.no_grad():
        act, h_out, c_out = pol(obs, h, c)
    return (act.numpy().reshape(-1).astype(np.float32),
            h_out.numpy().reshape(-1).astype(np.float32),
            c_out.numpy().reshape(-1).astype(np.float32))

def onnx_step(obs_np, h_prev, c_prev):
    inp = {
        "obs":  obs_np[None, :].astype(np.float32),
        "h_in": h_prev[None, :].astype(np.float32),
        "c_in": c_prev[None, :].astype(np.float32),
    }
    act, h_out, c_out = sess.run(None, inp)
    return (act.reshape(-1).astype(np.float32),
            h_out.reshape(-1).astype(np.float32),
            c_out.reshape(-1).astype(np.float32))

TRIALS = 5
STEPS  = 16
rng = np.random.default_rng(0)

max_abs = 0.0

for t in range(TRIALS):
    h_t = np.zeros(HIDDEN, dtype=np.float32)
    c_t = np.zeros(HIDDEN, dtype=np.float32)
    h_o = h_t.copy()
    c_o = c_t.copy()

    for s in range(STEPS):
        obs = rng.standard_normal(INPUT).astype(np.float32)

        act_ts, h_t, c_t = torch_step(obs, h_t, c_t)
        act_ox, h_o, c_o = onnx_step(obs, h_o, c_o)

        diff = float(np.max(np.abs(act_ts - act_ox)))
        if diff > max_abs:
            max_abs = diff

print(f"Max abs diff Torch(Python) vs ONNX: {max_abs:.3e}")
