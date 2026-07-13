import torch

ckpt_path = "results/checkpoints/FetchSlideDense-v4/20260711_232654/sacher_final.pt"

ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False,
)

print("Checkpoint keys:")
for k in ckpt.keys():
    print(" -", k)

print("\nAgent state keys:")
for k in ckpt["agent_state"].keys():
    print(" -", k)