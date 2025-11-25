# Samurice: Hanoi 29 Oct 2025
# QnD version of validator for VR-robot-control recurrent policys
#!/usr/bin/env python3
# How to use:  python validate_QnD_vs_torch.py ./policy_cpkt/policy_random.pt --trials 10 --steps 10 --init zeros

import argparse, numpy as np, torch

def load_policy(pt_path):
    ts = torch.jit.load(pt_path, map_location='cpu')
    sd = ts.state_dict()
    # LSTM tensors
    W_ih = sd["memory.weight_ih_l0"].cpu().numpy()
    W_hh = sd["memory.weight_hh_l0"].cpu().numpy()
    B_ih = sd["memory.bias_ih_l0"].cpu().numpy()
    B_hh = sd["memory.bias_hh_l0"].cpu().numpy()
    H4, INPUT = W_ih.shape
    H = H4 // 4
    # Actor layers 
    A0_W = sd["actor.0.weight"].cpu().numpy()
    A0_b = sd["actor.0.bias"].cpu().numpy()
    A2_W = sd["actor.2.weight"].cpu().numpy()
    A2_b = sd["actor.2.bias"].cpu().numpy()
    A4_W = sd["actor.4.weight"].cpu().numpy()
    A4_b = sd["actor.4.bias"].cpu().numpy()
    return ts, dict(INPUT=INPUT, H=H,
                    W_ih=W_ih, W_hh=W_hh, B_ih=B_ih, B_hh=B_hh,
                    A0_W=A0_W, A0_b=A0_b, A2_W=A2_W, A2_b=A2_b, A4_W=A4_W, A4_b=A4_b)

def elu(x):
    return np.where(x >= 0, x, np.exp(x) - 1).astype(np.float32)
# TODO add other activations if needed 

def step_numpy(x, h, c, p):
    """One step of LSTM + actor MLP in NumPy that matches PyTorch math."""
    gates = p["W_ih"] @ x + p["B_ih"] + p["W_hh"] @ h + p["B_hh"]  # (4H,)
    gi, gf, gg, go = np.split(gates, 4)
    i = 1.0 / (1.0 + np.exp(-gi))
    f = 1.0 / (1.0 + np.exp(-gf))
    g = np.tanh(gg)
    o = 1.0 / (1.0 + np.exp(-go))
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)

    a1 = elu(p["A0_W"] @ h_new + p["A0_b"])
    a2 = elu(p["A2_W"] @ a1    + p["A2_b"])
    y  =      p["A4_W"] @ a2    + p["A4_b"]
    return y.astype(np.float32), h_new.astype(np.float32), c_new.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt", help="TorchScript policy .pt (from RSL-RL repo)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", choices=["zeros","exported"], default="zeros",
                    help="initial LSTM state: zeros or exported buffers if present")
    args = ap.parse_args()

    ts, P = load_policy(args.pt)
    INPUT, H = P["INPUT"], P["H"]

    if args.init == "exported" and "hidden_state" in ts.state_dict() and "cell_state" in ts.state_dict():
        h0 = ts.state_dict()["hidden_state"].view(-1)[:H].cpu().numpy().astype(np.float32)
        c0 = ts.state_dict()["cell_state"].view(-1)[:H].cpu().numpy().astype(np.float32)
    else:
        h0 = np.zeros((H,), dtype=np.float32)
        c0 = np.zeros((H,), dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    max_abs_overall = 0.0
    per_trial = []

    for t in range(args.trials):
        h = h0.copy(); c = c0.copy()
        max_abs_trial = 0.0
        for s in range(args.steps):
            x = rng.standard_normal(INPUT).astype(np.float32)
            with torch.no_grad():
                ts.hidden_state = torch.from_numpy(h.reshape(1,-1)).clone()
                ts.cell_state   = torch.from_numpy(c.reshape(1,-1)).clone()
                y_ts = ts(torch.from_numpy(x)).cpu().numpy()

            y_np, h, c = step_numpy(x, h, c, P)
            err = float(np.max(np.abs(y_np - y_ts)))
            if err > max_abs_trial: max_abs_trial = err
            if err > max_abs_overall: max_abs_overall = err

        per_trial.append(max_abs_trial)

    print("Validator summary")
    print(f"  INPUT={INPUT}, HIDDEN={H}, ACTION={P['A4_b'].shape[0]}")
    print(f"  trials={args.trials}, steps/trial={args.steps}, seed={args.seed}")
    print(f"  max_abs_diff_overall = {max_abs_overall:.3e}")
    print("  max_abs_diff_per_trial =", " ".join(f"{e:.2e}" for e in per_trial))

if __name__ == "__main__":
    main()
