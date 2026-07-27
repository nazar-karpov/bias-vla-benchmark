import torch, numpy as np
from PIL import Image
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
for tag, img in [("square512", Image.new("RGB",(512,512),(120,120,120))),
                 ("sim640x480", Image.new("RGB",(640,480),(120,120,120)))]:
    out = proc(images=img, texts="<image>\nhi", return_tensors="pt")
    pv = out["pixel_values"]; isz = out["image_sizes"]
    print(tag, "pixel_values", tuple(pv.shape), "image_sizes", tuple(isz.shape), "isz=", isz.tolist())
