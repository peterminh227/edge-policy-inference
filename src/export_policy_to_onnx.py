#!/usr/bin/env python3
# Export a TorchScript policy to ONNX format for inference optimization
# Author: Samurice, Vinrobotics JSC
# Date: 9 Nov 2025
# How to export:
#   python export_policy_to_onnx.py --ts ../policy_cpkt/policy_M2.pt --out policy_M2.onnx
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class PolicyONNX(nn.Module):
    def __init__(self, sd):
        super().__init__()

        # LSTM weights (1-layer)
        W_ih = sd["memory.weight_ih_l0"]      # (4H, input_dim)
        W_hh = sd["memory.weight_hh_l0"]      # (4H, H)
        B_ih = sd["memory.bias_ih_l0"]        # (4H,)
        B_hh = sd["memory.bias_hh_l0"]        # (4H,)

        self.input_dim  = W_ih.shape[1]
        self.hidden_dim = W_ih.shape[0] // 4

        # Register as params so they’re embedded; requires_grad=False so no training.
        self.W_ih = nn.Parameter(W_ih.clone(), requires_grad=False)
        self.W_hh = nn.Parameter(W_hh.clone(), requires_grad=False)
        self.B_ih = nn.Parameter(B_ih.clone(), requires_grad=False)
        self.B_hh = nn.Parameter(B_hh.clone(), requires_grad=False)

        # Actor MLP (ELU hidden)
        self.A0_W = nn.Parameter(sd["actor.0.weight"].clone(), requires_grad=False)
        self.A0_b = nn.Parameter(sd["actor.0.bias"].clone(),  requires_grad=False)
        self.A2_W = nn.Parameter(sd["actor.2.weight"].clone(), requires_grad=False)
        self.A2_b = nn.Parameter(sd["actor.2.bias"].clone(),  requires_grad=False)
        self.A4_W = nn.Parameter(sd["actor.4.weight"].clone(), requires_grad=False)
        self.A4_b = nn.Parameter(sd["actor.4.bias"].clone(),  requires_grad=False)

    def forward(self, obs, h_in, c_in):
        """
        obs  : (B, input_dim)
        h_in : (B, hidden_dim)
        c_in : (B, hidden_dim)
        """
        H = self.hidden_dim

        # LSTM gates: (B, 4H)
        gates = F.linear(obs, self.W_ih, self.B_ih) + F.linear(h_in, self.W_hh, self.B_hh)

        # IMPORTANT: use slicing; this becomes Slice ops, NOT Split(num_outputs)
        gi = gates[:, 0:H]
        gf = gates[:, H:2*H]
        gg = gates[:, 2*H:3*H]
        go = gates[:, 3*H:4*H]

        i = torch.sigmoid(gi)
        f = torch.sigmoid(gf)
        g = torch.tanh(gg)
        o = torch.sigmoid(go)

        c_out = f * c_in + i * g
        h_out = o * torch.tanh(c_out)

        # Actor MLP with ELU activations on hidden layers
        x = F.elu(F.linear(h_out, self.A0_W, self.A0_b))
        x = F.elu(F.linear(x,    self.A2_W, self.A2_b))
        action = F.linear(x,     self.A4_W, self.A4_b)

        return action, h_out, c_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts",  default="policy_M2.pt")
    parser.add_argument("--out", default="policy_M2.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    # Load TorchScript
    ts = torch.jit.load(args.ts, map_location="cpu")
    ts.eval()
    sd = ts.state_dict()

    # Build exportable module
    pol = PolicyONNX(sd).eval()
    print(f"input_dim={pol.input_dim}, hidden_dim={pol.hidden_dim}")

    B = 1
    obs = torch.zeros((B, pol.input_dim),  dtype=torch.float32)
    h   = torch.zeros((B, pol.hidden_dim), dtype=torch.float32)
    c   = torch.zeros((B, pol.hidden_dim), dtype=torch.float32)

    input_names  = ["obs", "h_in", "c_in"]
    output_names = ["action", "h_out", "c_out"]
    dynamic_axes = {
        "obs":    {0: "batch"},
        "h_in":   {0: "batch"},
        "c_in":   {0: "batch"},
        "action": {0: "batch"},
        "h_out":  {0: "batch"},
        "c_out":  {0: "batch"},
    }

    torch.onnx.export(
        pol,
        (obs, h, c),
        args.out,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=args.opset,
    )

    print(f"Exported ONNX to {args.out}")

    # Quick structural check
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    print("ONNX model structure OK.")

if __name__ == "__main__":
    main()