# For generating C header and JSON files from a TorchScript policy model
# Author: Samurice, Vinrobotics JSC
# Date: 30 October 2025
# How to export:
#   python export_policy_weights.py ../policy_cpkt/policy_random.pt --out-prefix=policy_random --activation=elu
#!/usr/bin/env python3
import argparse, json, re, os
import torch, numpy as np, sys

def flat_f32(t):
    return t.detach().cpu().numpy().astype(np.float32).ravel()

def guess_activation_id(act_name:str):
    name = (act_name or "").lower()
    if name in ("relu",): return 0
    if name in ("elu",): return 1
    if name in ("selu",): return 2
    if name in ("lrelu","leaky_relu","leaky-relu"): return 3
    if name in ("tanh",): return 4
    if name in ("sigmoid",): return 5
    return 1  # default ELU

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt_path", help="TorchScript policy .pt")
    ap.add_argument("--out-prefix", default="policy", help="output prefix (files: <prefix>_weights.h, <prefix>.json)")
    ap.add_argument("--activation", default="elu", help="actor hidden activation: relu|elu|selu|lrelu|tanh|sigmoid")
    args = ap.parse_args()

    m = torch.jit.load(args.pt_path, map_location="cpu")
    sd = m.state_dict()

    # Required LSTM parameters
    req = ["memory.weight_ih_l0","memory.weight_hh_l0","memory.bias_ih_l0","memory.bias_hh_l0"]
    for k in req:
        if k not in sd:
            sys.exit(f"ERROR: missing '{k}' in state_dict")

    W_ih = sd["memory.weight_ih_l0"]; W_hh = sd["memory.weight_hh_l0"]
    b_ih = sd["memory.bias_ih_l0"];   b_hh = sd["memory.bias_hh_l0"]
    H4, INPUT = W_ih.shape
    H = H4//4

    # Optional initial states
    H0 = sd.get("hidden_state", torch.zeros((1,1,H))).view(-1)[:H]
    C0 = sd.get("cell_state",  torch.zeros((1,1,H))).view(-1)[:H]

    # Actor: collect linear layers in ascending name order
    keys = list(sd.keys())
    pairs = sorted([(k, k.replace(".weight",".bias"))
                    for k in keys if k.startswith("actor.") and k.endswith(".weight")])
    if not pairs:
        sys.exit("ERROR: no actor.*.weight found (export expects an actor MLP head)")

    actor = []
    for wkey, bkey in pairs:
        W, b = sd[wkey], sd[bkey]
        M, N = W.shape
        if b.numel()!=M:
            sys.exit(f"ERROR: bias {bkey} length mismatch.")
        actor.append((wkey,bkey,M,N))

    act_id = guess_activation_id(args.activation)
    prefix = args.out_prefix
    guard = re.sub(r"[^A-Za-z0-9_]", "_", prefix.upper()) + "_WEIGHTS_H"
    out_h = f"{prefix}_weights.h"
    out_json = f"{prefix}.json"

    def emit_array(name, arr):
        values = ", ".join(f"{x:.8e}" for x in arr.astype(np.float32).ravel())
        return f"static const float {name}[{arr.size}] = {{ {values} }};\n"

    with open(out_h, "w") as f:
        f.write(f"// Auto-generated from {os.path.basename(args.pt_path)}\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"#define POL_INPUT {INPUT}\n#define POL_HIDDEN {H}\n")
        f.write(f"#define POL_LAYERS {len(actor)}\n#define POL_ACT_ID {act_id}\n\n")
        f.write("// LSTM (gate order i,f,g,o), row-major (out x in)\n")
        f.write(emit_array("POL_W_IH", W_ih.detach().cpu().numpy()))
        f.write(emit_array("POL_W_HH", W_hh.detach().cpu().numpy()))
        f.write(emit_array("POL_B_IH", b_ih.detach().cpu().numpy()))
        f.write(emit_array("POL_B_HH", b_hh.detach().cpu().numpy()))
        f.write("\n// Initial hidden/cell state\n")
        f.write(emit_array("POL_H0", H0.detach().cpu().numpy()))
        f.write(emit_array("POL_C0", C0.detach().cpu().numpy()))
        f.write("\n// Actor per-layer dims\n")
        f.write("static const int POL_LIN_IN[POL_LAYERS]  = { " + ", ".join(str(N) for (_,_,_,N) in actor) + " };\n")
        f.write("static const int POL_LIN_OUT[POL_LAYERS] = { " + ", ".join(str(M) for (_,_,M,_) in actor) + " };\n\n")
        for li,(wkey,bkey,M,N) in enumerate(actor):
            W = sd[wkey].detach().cpu().numpy()
            b = sd[bkey].detach().cpu().numpy()
            f.write(emit_array(f"POL_AW_{li}", W))
            f.write(emit_array(f"POL_AB_{li}", b))
            f.write("\n")
        f.write(f"#endif // {guard}\n")

    with open(out_json, "w") as jf:
        json.dump({
            "pt_file": os.path.basename(args.pt_path),
            "input_dim": int(INPUT),
            "hidden_dim": int(H),
            "layers": [{"in": int(N), "out": int(M)} for (_,_,M,N) in actor],
            "activation": args.activation
        }, jf, indent=2)

    print(f"Wrote {out_h} and {out_json}")

if __name__ == "__main__":
    main()