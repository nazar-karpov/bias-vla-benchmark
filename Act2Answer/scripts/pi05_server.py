"""Minimal zmq policy server for the lerobot pi0.5 (pi05) policy.

Runs in the `lerobot_pi05` conda env (heavy lerobot/transformers deps). The
Act2Answer eval client (env `magma_act2answer`) talks to it over the same
REQ/REP + msgpack wire protocol as the RLDX server, so the client reuses the
same _encode/_decode helpers (see policies/pi05/pi05.py).

Wire protocol (REQ -> REP, msgpack, numpy as {"__ndarray_class__":True,"as_npy":...}):
  request  {"endpoint": "get_action", "data": {"images": (B,H,W,3) uint8,
                                                "state":  (B,7) f32,
                                                "tasks":  [str]*B,
                                                "reset":  [bool]*B}}
  response {"action": (B,7) f32}            # per env: [dx,dy,dz,droll,dpitch,dyaw,grip]
  request  {"endpoint": "reset"}  -> {"ok": True}
  request  {"endpoint": "ping"}   -> {"ok": True}

pi05 select_action manages its own 50-step action-chunk queue internally and pops
one action per call; we therefore query the model every env.step (receding
horizon) and let lerobot handle chunk buffering. The queue must be reset per
episode -> we hold ONE policy.reset() shared across the batch (the eval runner
resets all envs together at episode start via the client's reset()).

Checkpoint: qownscks/pi05_widowx (real pi05, WidowX/Bridge, 7-dim eef delta +
gripper, ships quantile norm stats). cam_wrist is absent in SimplerEnv -> we feed
a copy of the top image (the norm stats expect *some* wrist image; a duplicate is
closer than the -1 empty-camera fill for a single-view sim).
"""
import io
import os
import sys
import argparse

import numpy as np
import msgpack
import zmq
import torch

CKPT = os.environ.get("PI05_CKPT", "qownscks/pi05_widowx")
PORT = int(os.environ.get("PI05_PORT", "20005"))


# ---------------------------------------------------------------- wire codec ---
def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _pack(d):
    return msgpack.packb(d, default=_encode)


def _unpack(b):
    return msgpack.unpackb(b, object_hook=_decode)


# ------------------------------------------------------------------- policy ---
class Pi05Runner:
    def __init__(self, ckpt: str):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors

        print(f"[pi05] loading {ckpt} ...", flush=True)
        self.policy = PI05Policy.from_pretrained(ckpt).eval().to("cuda")
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=ckpt
        )
        # camera keys this checkpoint expects
        self.img_keys = [k for k in self.policy.config.input_features
                         if k.startswith("observation.images.")]
        self.state_dim = self.policy.config.input_features["observation.state"].shape[0]
        print(f"[pi05] ready. img_keys={self.img_keys} state_dim={self.state_dim}", flush=True)

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def _step_one(self, image_hw3_uint8, state_vec, task):
        # image -> batched CHW (1,3,H,W); model accepts uint8 or float CHW.
        chw = np.transpose(image_hw3_uint8, (2, 0, 1))[None].astype(np.uint8)  # (1,3,H,W)
        obs = {"observation.state": state_vec[None].astype(np.float32),        # (1,7)
               "task": [task]}
        for k in self.img_keys:
            obs[k] = chw  # top and (duplicated) cam_wrist both = current view
        batch = self.pre(obs)
        act = self.post(self.policy.select_action(batch))
        act = act.detach().cpu().numpy().reshape(-1)
        return act[: self.state_dim] if act.shape[0] >= self.state_dim else act

    @torch.no_grad()
    def get_action(self, images, states, tasks, resets):
        # Reset the shared chunk queue if any env signals a new episode. All envs
        # in this bench reset together, so a single reset per batch is correct.
        if resets is not None and bool(np.any(resets)):
            self.policy.reset()
        out = []
        B = images.shape[0]
        for i in range(B):
            out.append(self._step_one(images[i], states[i], tasks[i]))
        return np.stack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    runner = Pi05Runner(args.ckpt)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[pi05] serving on tcp://0.0.0.0:{args.port}", flush=True)

    while True:
        try:
            req = _unpack(sock.recv())
        except Exception as e:
            print(f"[pi05] recv error: {e}", flush=True)
            continue
        ep = req.get("endpoint")
        try:
            if ep == "ping":
                sock.send(_pack({"ok": True}))
            elif ep == "reset":
                runner.reset()
                sock.send(_pack({"ok": True}))
            elif ep == "get_action":
                d = req["data"]
                images = np.asarray(d["images"], dtype=np.uint8)
                states = np.asarray(d["state"], dtype=np.float32)
                tasks = list(d["tasks"])
                resets = d.get("reset")
                action = runner.get_action(images, states, tasks, resets)
                sock.send(_pack({"action": action}))
            else:
                sock.send(_pack({"error": f"unknown endpoint {ep}"}))
        except Exception as e:
            import traceback
            traceback.print_exc()
            sock.send(_pack({"error": str(e)}))


if __name__ == "__main__":
    main()
