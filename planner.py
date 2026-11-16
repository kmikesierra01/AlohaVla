"""High-level planner for the colour-command task: SmolVLM2 (SmolVLA's own base VLM) lists the cube colours left to
right from the top camera's first frame (the policy's own camera), which turns "Pick up the red cube ..." into
"Pick up the second cube ..." for the place-command SmolVLA policy (AlohaFourCubesSlot-v0).

Zero-shot SmolVLM2 gets the order right in 0/10 scenes (cubes ~18 px wide), so it is fine-tuned on
simulator-labelled images (layout seeds 100000+, disjoint from the policy's training seeds 0-50 and the eval seeds
1000+).

Usage: python planner.py train <out_dir> [n_images] [steps]
       python planner.py eval <model_dir|base> [from:to]      (default 1000:1200 = the 200 eval scenes)
"""
import re
import sys

import numpy as np
import torch
from PIL import Image

import two_cubes

BASE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
QUESTION = "There are four cubes in a row. List their colours from left to right, separated by commas."
COLORS = two_cubes.TASKS["AlohaFourCubes-v0"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def view(pixels):
    """Planner image from a top camera frame: the cubes' row (+y) runs bottom to top there, so turn it 90° clockwise
    to read it left to right."""
    return Image.fromarray(pixels).transpose(Image.Transpose.ROTATE_270)


def scenes(seeds):
    """(planner image, colours left to right) per layout."""
    env = two_cubes.TwoCubesEnv(COLORS, task="two_cubes", obs_type="pixels_agent_pos")
    for seed in seeds:
        poses, _ = two_cubes.sample(seed, COLORS)
        yield view(env.reset(seed=seed)[0]["pixels"]["top"]), sorted(poses, key=lambda c: poses[c][1])


def load(path, dtype=torch.bfloat16):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(BASE)
    proc.image_processor.do_image_splitting = False  # one 512 px view
    proc.tokenizer.padding_side = "right"
    return proc, AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to(DEVICE)


def prompt(proc, answer=None):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    if answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return proc.apply_chat_template(msgs, add_generation_prompt=answer is None)


def read_order(proc, model, image):
    """Colours left to right as the VLM reads them (yellow = our orange, which renders yellowish)."""
    inp = proc(text=[prompt(proc)], images=[[image]], return_tensors="pt").to(DEVICE)
    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=20, do_sample=False)
    text = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return [{"yellow": "orange"}.get(w, w) for w in re.findall(r"red|blue|orange|yellow|green", text.lower())]


def train(out, n=2000, steps=500, batch=8, lr=1e-5):
    data = list(scenes(range(100000, 100000 + n)))
    proc, model = load(BASE, torch.float32)
    for p in model.model.vision_model.parameters():  # SigLIP stays frozen, as in SmolVLA
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(0)
    model.train()
    for step in range(steps):
        items = [data[i] for i in rng.choice(n, batch, replace=False)]
        # our orange renders yellowish and the base VLM calls it yellow: teach that name, read_order maps it back
        full = proc(text=[prompt(proc, ", ".join(order).replace("orange", "yellow")) for _, order in items],
                    images=[[img] for img, _ in items],
                    return_tensors="pt", padding=True).to(DEVICE)
        start = proc(text=[prompt(proc)], images=[[items[0][0]]], return_tensors="pt")["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :start] = -100  # loss on the answer only (every prompt has the same length)
        labels[full["attention_mask"] == 0] = -100
        loss = model(**full, labels=labels).loss  # fp32 (autocast breaks SmolVLM's image-token scatter)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    model.save_pretrained(out)
    print(f"-> {out}")


def evaluate(path, seeds):
    proc, model = load(BASE if path == "base" else path)
    model.eval()
    right = place = 0
    for i, (img, truth) in enumerate(scenes(seeds)):
        said = read_order(proc, model, img)
        right += said == truth
        target = two_cubes.sample(seeds[i], COLORS)[1]  # the eval episode's commanded colour
        place += target in said and said.index(target) == truth.index(target)
        if said != truth:
            print(seeds[i], "truth", truth, "said", said, flush=True)
    print(f"{len(seeds)} scenes: order right {right / len(seeds):.1%}, commanded cube's place right "
          f"{place / len(seeds):.1%}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], *map(int, sys.argv[3:5]))
    else:
        evaluate(sys.argv[2], range(*map(int, (sys.argv[3] if len(sys.argv) > 3 else "1000:1200").split(":"))))
