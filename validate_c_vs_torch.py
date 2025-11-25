#!/usr/bin/env python3
# Samurice: Hanoi 29 Oct 2025
# How to run:
#   python validate_c_vs_torch.py   --pt ./policy_cpkt/policy_M2.pt   --weights_h ./src/policy_M2_weights.h   --infer_h ./src/policy_infer.h   --trials 10 --steps 20 --init zeros

import argparse, ctypes, numpy as np, torch, os, subprocess, tempfile, textwrap, sys, pathlib

def build_shared(weights_header, policy_infer_h):
    cc = os.environ.get("CC","gcc")
    tmpdir = tempfile.mkdtemp()
    shim_c = os.path.join(tmpdir, "shim.c")
    so_path = os.path.join(tmpdir, "libpol.so")

    # C shim to expose a minimal C API for ctypes
    code = f'''
    #define WEIGHTS_HEADER "{os.path.basename(weights_header)}"
    #include "{os.path.basename(policy_infer_h)}"

    // Expose thin wrappers for ctypes
    __attribute__((visibility("default")))
    void c_reset(float* h, float* c) {{ pol_reset(h, c); }}

    __attribute__((visibility("default")))
    void c_step(const float* x, float* h, float* c, float* y) {{ pol_step(x, h, c, y); }}

    __attribute__((visibility("default")))
    int c_input_dim() {{ return POL_INPUT; }}
    __attribute__((visibility("default")))
    int c_hidden_dim() {{ return POL_HIDDEN; }}
    __attribute__((visibility("default")))
    int c_out_dim() {{ return POL_LIN_OUT[POL_LAYERS-1]; }}
    '''
    open(shim_c,"w").write(code)

    # Copy headers to tmpdir
    for src in (weights_header, policy_infer_h):
        dst = os.path.join(tmpdir, os.path.basename(src))
        open(dst,"wb").write(open(src,"rb").read())

    # Build shared library
    cmd = [cc, "-O3", "-std=c99", "-shared", "-fPIC", shim_c, "-lm", "-o", so_path]
    subprocess.check_call(cmd, cwd=tmpdir)
    return so_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="TorchScript policy .pt")
    ap.add_argument("--weights_h", required=True, help="generated policy_RoboXXX_weights.h")
    ap.add_argument("--infer_h", required=True, help="policy_infer.h")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", choices=["zeros","exported"], default="zeros")
    args = ap.parse_args()

    # Build C shared lib
    so = build_shared(args.weights_h, args.infer_h)
    lib = ctypes.cdll.LoadLibrary(so)
    lib.c_reset.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.c_step.argtypes  = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.c_input_dim.restype = ctypes.c_int
    lib.c_hidden_dim.restype = ctypes.c_int
    lib.c_out_dim.restype    = ctypes.c_int

    # Load TorchScript
    ts = torch.jit.load(args.pt, map_location='cpu')
    sd = ts.state_dict()
    INPUT = sd["memory.weight_ih_l0"].shape[1]
    H     = sd["memory.weight_ih_l0"].shape[0]//4
    OUT   = lib.c_out_dim()

    # Sanity
    assert INPUT == lib.c_input_dim()
    assert H     == lib.c_hidden_dim()

    # Exported h0/c0 if requested
    if args.init == "exported" and "hidden_state" in sd and "cell_state" in sd:
        h0 = sd["hidden_state"].view(-1)[:H].cpu().numpy().astype(np.float32)
        c0 = sd["cell_state"].view(-1)[:H].cpu().numpy().astype(np.float32)
    else:
        h0 = np.zeros(H, dtype=np.float32)
        c0 = np.zeros(H, dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    max_abs = 0.0

    for t in range(args.trials):
        # Torch side state
        h_t = h0.copy(); c_t = c0.copy()
        # C side state
        h_c = (ctypes.c_float * H)(); c_c = (ctypes.c_float * H)()
        # initialize C to same initial state
        for i in range(H):
            h_c[i] = h0[i]; c_c[i] = c0[i]

        for s in range(args.steps):
            x = rng.standard_normal(INPUT).astype(np.float32)

            # Torch step (sync internal buffers)
            with torch.no_grad():
                ts.hidden_state = torch.from_numpy(h_t.reshape(1,-1)).clone()
                ts.cell_state   = torch.from_numpy(c_t.reshape(1,-1)).clone()
                y_ts = ts(torch.from_numpy(x)).cpu().numpy()

            # C step
            x_c = (ctypes.c_float * INPUT)(*x)
            y_c = (ctypes.c_float * OUT)()
            lib.c_step(x_c, h_c, c_c, y_c)

            # Update numpy copies from C state for next iteration’s Torch sync
            h_t = np.frombuffer(h_c, dtype=np.float32, count=H).copy()
            c_t = np.frombuffer(c_c, dtype=np.float32, count=H).copy()

            # Compare actions
            y_c_np = np.frombuffer(y_c, dtype=np.float32, count=OUT).copy()
            err = float(np.max(np.abs(y_c_np - y_ts)))
            if err > max_abs: max_abs = err

    print(f"OK: max_abs_diff TorchScript vs C = {max_abs:.3e}")

if __name__ == "__main__":
    main()
