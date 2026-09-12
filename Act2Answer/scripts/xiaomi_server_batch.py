#!/usr/bin/env python3
# Сервер Xiaomi-Robotics-0 с БАТЧЕВЫМИ запросами (12.09.2026).
# Оригинал deploy/server.py обрабатывает по одному наблюдению: forward ~1.1 с на 4090 и ~2.2 с на
# троттлящемся H100 — модель упирается не в GPU, а в запуск ядер/CPU (5 шагов flow × слои DiT + VLM).
# Здесь клиент может прислать {"batch": [obs, obs, ...]} — весь буфер симулятора одним forward'ом:
# processor(text=[...], images=[...], padding=True), state (B,1,32), action_mask (B,1,D).
# Одиночный формат (как у оригинала) тоже принимается. Протокол: 4 байта длины + pickle, как у deploy/server.py.
import argparse
import os
import pickle
import socket
import struct
import time
import traceback

import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

_BRIDGE_ONLY = ("bridge", "fractal")


class Server(mp.Process):
    def __init__(self, model_path, host, port):
        super().__init__()
        self.host, self.port, self.model_path = host, port, model_path

    def _load(self):
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True,
            attn_implementation=os.environ.get("XR0_ATTN", "flash_attention_2"),
            dtype=torch.bfloat16).cuda().to(torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)

    @staticmethod
    def _recv_all(conn, n):
        buf = b""
        while len(buf) < n:
            p = conn.recv(n - len(buf))
            if not p:
                return None
            buf += p
        return buf

    @torch.no_grad()
    def _infer(self, samples):
        """samples: список dict с ключами task_id, state (np, (1,1,32)), language, base (PIL), seed."""
        robot_type = samples[0]["task_id"]
        assert all(s["task_id"] == robot_type for s in samples)
        dev, dt = self.model.device, self.model.dtype
        if any(k in robot_type for k in _BRIDGE_ONLY):
            vl = self.processor(text=[s["language"] for s in samples], images=[s["base"] for s in samples],
                                videos=None, padding=True, return_tensors="pt")
        else:
            imgs = []
            for s in samples:
                imgs += [s["base"], s["wrist_left"]]
            vl = self.processor(text=[s["language"] for s in samples], images=imgs,
                                videos=None, padding=True, return_tensors="pt")
        data = dict(vl.to(dev))
        B = len(samples)
        data["action_mask"] = self.processor.get_action_mask(robot_type, batch_size=B).to(dev, dt)
        data["state"] = torch.cat([torch.as_tensor(s["state"]).to(dev, dt).view(1, 1, -1) for s in samples], 0)
        data["seed"] = int(samples[0]["seed"])
        # как в оригинале, лишние ключи уходят в vlm(**kwargs) и игнорируются; task_id нужен для формы
        data["task_id"] = robot_type
        out = self.model(**data)
        act = self.processor.decode_action(out.actions, robot_type=robot_type).cpu()  # (B, T, D)
        return act

    def run(self):
        self._load()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
            ss.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            ss.bind((self.host, self.port))
            ss.listen(4)
            print(f"Server running on {self.host}:{self.port}...", flush=True)
            while True:
                conn, _ = ss.accept()
                try:
                    with tqdm(desc="Processing Requests", unit=" req") as pbar:
                        while True:
                            hdr = self._recv_all(conn, 4)
                            if not hdr:
                                break
                            raw = self._recv_all(conn, struct.unpack(">I", hdr)[0])
                            if not raw:
                                break
                            tic = time.time()
                            req = pickle.loads(raw)
                            if "batch" in req:
                                act = self._infer(req["batch"])          # (B, T, D)
                                resp = [act[i] for i in range(act.shape[0])]
                            else:
                                resp = self._infer([req])[0:1][0]        # (T, D) — как action[0] у оригинала
                                resp = resp[None]                        # оригинал отдаёт (1, T, D)
                            payload = pickle.dumps(resp)
                            conn.sendall(struct.pack(">I", len(payload)) + payload)
                            pbar.update(1)
                            pbar.set_postfix({"avg_time": f"{(time.time() - tic) * 1000:.0f}ms",
                                              "B": len(req.get("batch", [1]))})
                except Exception as e:
                    print(f"Error handling connection: {e}", flush=True)
                    traceback.print_exc()
                finally:
                    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=10086)
    a = ap.parse_args()
    s = Server(a.model, a.host, a.port)
    s.start()
    s.join()
