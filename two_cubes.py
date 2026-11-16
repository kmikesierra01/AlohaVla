"""Coloured-cube ALOHA tasks: several cubes on the table, the language command says which one to hand over.

AlohaTwoCubes-v0: red + blue, mirrored about the right arm's line (y = 0.5). AlohaFourCubes-v0: red, blue, orange,
green on an arc around the right arm's base. Either way every cube is equally far from the right arm, the colour order
is random and the commanded colour is drawn independently, so only the command tells the policy which cube to take
(ignoring it gets 1/n right).
AlohaFourCubesSlot-v0: the four cubes, but the command names the cube's place ("the second cube"), not its colour.
AlohaFourCubesPlan-v0: colour commands for a place-command policy; planner.py turns them into place commands.
Importing this module registers them (gym-aloha's transfer-cube scene and stages).

Reward, as lerobot-eval's max_reward reports it:
  0-4    commanded cube: 1 right gripper touches, 2 lifted, 3 left gripper touches, 4 handed over = the left
         gripper alone has held it off the table for 0.5 s (gym-aloha counts any instant the left touches it)
  10+s   the right arm lifted ANOTHER cube first (s = that cube's stage), i.e. wrong colour

Usage (colours: red,blue = two cubes, default; red,blue,orange,green = four cubes):
       python two_cubes.py record <n_demos> <repo_id> [paired] [colours] [slot]  (scripted demos -> LeRobot dataset;
                paired: each layout once per colour; slot: commands by place)
       python two_cubes.py preview <seed> <out.mp4> [colours]   (one layout, a demo per colour, top | side camera)
       python two_cubes.py breakdown <eval_dir> [colours]       (right-colour rate by command and cube position)
       python two_cubes.py probe <checkpoint> <seed|from:to>  (does the command decide where the arm goes?
                                                               from:to = seeds from..to-1; task from the checkpoint)
       python two_cubes.py demo <checkpoint> <seed> <out.mp4>  (one scene, a policy episode per colour command)
"""
import collections
import os
import re
import sys

import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control

from gym_aloha.constants import ASSETS_DIR, DT, JOINTS, START_ARM_POSE, normalize_puppet_gripper_position
from gym_aloha.env import AlohaEnv
from gym_aloha.tasks.sim import BimanualViperXTask, TransferCubeTask
from gym_aloha.tasks.sim_end_effector import BimanualViperXEndEffectorTask

RGBA = {"red": "1 0 0 1", "blue": "0 0 1 1", "orange": "1 0.5 0 1", "green": "0 0.7 0 1"}
TASKS = {"AlohaTwoCubes-v0": ("red", "blue"), "AlohaFourCubes-v0": ("red", "blue", "orange", "green"),
         "AlohaFourCubesSlot-v0": ("red", "blue", "orange", "green"),
         "AlohaFourCubesPlan-v0": ("red", "blue", "orange", "green")}
SLOT = {"AlohaFourCubesSlot-v0"}  # tasks whose command names the cube's place ("the second cube"), not its colour
PLAN = {"AlohaFourCubesPlan-v0"}  # colour commands, turned into place commands by planner.py (PLANNER=<model dir>)
ORDINALS = ("first", "second", "third", "fourth")  # places counted along +y
COLORS = TASKS["AlohaTwoCubes-v0"]
RIGHT_BASE = np.array([0.469, 0.5])  # right arm's base on the table
COMMAND = "Pick up the {} cube and hand it to the left arm."
HOLD = 25  # steps (0.5 s) the left gripper must hold the cube alone before it counts as handed over


def make_physics(colors=COLORS, ee=False):
    """gym-aloha's transfer-cube scene plus a recoloured copy of its red cube per extra colour (keyframe dropped:
    qpos grew by 7 per cube)."""
    xml = (ASSETS_DIR / f"bimanual_viperx_{'end_effector_' if ee else ''}transfer_cube.xml").read_text()
    red = re.search(r'<body name="box".*?</body>', xml, re.S).group(0)
    copies = "".join(red.replace('name="box"', f'name="{c}_box_body"').replace("red_box", f"{c}_box").replace(
        f'rgba="{RGBA["red"]}"', f'rgba="{RGBA[c]}"') for c in colors[1:])
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml.replace(red, red + copies), flags=re.S)
    # ACT's soft mocap weld can't drag the arm (joint frictionloss 50-60) down to the cube on MuJoCo 3: 0/6 grasps.
    # MuJoCo's default stiffness: 6/6. Only the scripted end-effector env has the weld; the policy env is unchanged.
    xml = xml.replace('solimp=".25 .25 0.001"', 'solimp="0.9 0.95 0.001"')
    assets = {p.name: p.read_bytes() for p in ASSETS_DIR.iterdir() if p.is_file()}
    return mujoco.Physics.from_xml_string(xml, assets=assets)


def sample(seed, colors=COLORS):
    """Cube poses (all equally far from the right arm) in random colour order, and the commanded colour.
    Two cubes: same x, y = 0.5 -/+ d. More: on an arc around the right arm's base, 9-11 cm apart."""
    rng = np.random.RandomState(seed)
    q = [1, 0, 0, 0]
    if len(colors) == 2:  # unchanged, so the two-cube dataset and results stay reproducible
        x, d = rng.uniform(0.0, 0.2), rng.uniform(0.07, 0.1)
        near, far = colors if rng.rand() < 0.5 else colors[::-1]
        return {near: [x, 0.5 - d, 0.05, *q], far: [x, 0.5 + d, 0.05, *q]}, colors[rng.randint(2)]
    r, gap = rng.uniform(0.30, 0.40), rng.uniform(0.09, 0.11)  # arc radius, spacing along it
    angles = (np.arange(len(colors)) - (len(colors) - 1) / 2) * gap / r  # symmetric about the arms' line
    order = rng.permutation(len(colors))
    poses = {colors[i]: [*RIGHT_BASE + r * np.array([-np.cos(a), np.sin(a)]), 0.05, *q] for i, a in zip(order, angles)}
    return poses, colors[rng.randint(len(colors))]


def command(poses, target, slot=False):
    """The sentence asking for `target` in this layout: by colour, or (slot) by place along +y."""
    return COMMAND.format(ORDINALS[sorted(poses, key=lambda c: poses[c][1]).index(target)] if slot else target)


_planner = None


def plan(physics, text):
    """Colour command -> place command, read off the top camera by planner.py's VLM (unchanged if it can't)."""
    global _planner
    import planner

    if _planner is None:
        _planner = planner.load(os.environ["PLANNER"])
    order = planner.read_order(*_planner, planner.view(physics.render(height=480, width=640, camera_id="top")))
    color = next((c for c in RGBA if c in text.lower()), None)
    return COMMAND.format(ORDINALS[order.index(color)]) if color in order[:len(ORDINALS)] else text


class Cubes:
    """Cube placement + reward, shared by the joint-space task (policy, eval) and the end-effector one (demos)."""
    poses, target, first, held = None, "red", None, 0  # first: cube lifted first ("target"/"other")

    def __init__(self, colors=COLORS):
        super().__init__()
        self.colors = colors

    def place(self, physics):
        self.first, self.held = None, 0
        for color, pose in self.poses.items():
            physics.named.data.qpos[f"{color}_box_joint"] = pose

    def get_env_state(self, physics):
        return physics.data.qpos.copy()[16:]

    def get_reward(self, physics):
        def part(g):  # finger geoms by body: gym-aloha names only one geom per gripper, the rest hold the cube too
            body = physics.model.id2name(physics.model.geom_bodyid[g], "body") or ""
            arm = "right" if body.startswith("vx300s_right/") else "left" if body.startswith("vx300s_left/") else None
            return arm if arm and "finger" in body else physics.model.id2name(g, "geom")

        contacts = [physics.data.contact[i] for i in range(physics.data.ncon)]
        pairs = {frozenset((part(c.geom1), part(c.geom2))) for c in contacts}

        def stage(box):  # gym-aloha's transfer-cube stages for one cube
            touch = lambda what: frozenset((box, what)) in pairs
            if touch("left"):
                return 3 if touch("table") or touch("right") else 4
            if touch("right"):
                return 1 if touch("table") else 2
            return 0

        s = stage(f"{self.target}_box")
        o = max(stage(f"{c}_box") for c in self.colors if c != self.target)
        self.held = self.held + 1 if s == 4 else 0
        s = 3 if s == 4 and self.held < HOLD else s
        if self.first is None and max(s, o) >= 2:
            self.first = "target" if s >= 2 else "other"
        return 10 + o if self.first == "other" else s


class TwoCubesTask(Cubes, TransferCubeTask):
    def initialize_episode(self, physics):
        physics.named.data.qpos[:16] = START_ARM_POSE
        np.copyto(physics.data.ctrl, START_ARM_POSE)
        self.place(physics)
        BimanualViperXTask.initialize_episode(self, physics)  # skip TransferCubeTask's single-cube BOX_POSE

    def get_observation(self, physics):  # only the top camera: the policy sees nothing else (3x faster than gym-aloha)
        obs = collections.OrderedDict(qpos=self.get_qpos(physics), qvel=self.get_qvel(physics))
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {"top": physics.render(height=480, width=640, camera_id="top")}
        return obs


class TwoCubesEETask(Cubes, BimanualViperXEndEffectorTask):
    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        self.place(physics)
        BimanualViperXEndEffectorTask.initialize_episode(self, physics)

    def get_observation(self, physics):  # no cameras while scripting
        return {"qpos": self.get_qpos(physics), "gripper_ctrl": physics.data.ctrl.copy()}


def make_control_env(physics, task):
    return control.Environment(physics, task, float("inf"), control_timestep=DT, n_sub_steps=None,
                               flat_observation=False)


class TwoCubesEnv(AlohaEnv):
    """gym-aloha's AlohaEnv on a coloured-cube task. The seed picks the layout and the commanded colour."""
    command = None  # view.py / record() set one: it is fed to the policy and its colour becomes the target
    planned = None  # plan tasks: the place command the planner made of it at reset

    def __init__(self, colors=COLORS, slot=False, plan=False, **kwargs):
        self.colors, self.slot, self.plan = tuple(colors), slot, plan
        super().__init__(**kwargs)

    def _make_env_task(self, task_name):
        return make_control_env(make_physics(self.colors), TwoCubesTask(self.colors))

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        task = self._env.task
        task.poses, task.target = sample(seed, self.colors)
        if self.command:  # the colour it names, else the cube at the place it names
            text, by_y = self.command.lower(), sorted(task.poses, key=lambda c: task.poses[c][1])
            task.target = next((c for c in self.colors if c in text), None) or next(
                (by_y[i] for i, w in enumerate(ORDINALS[:len(by_y)]) if w in text), task.target)
        obs = self._format_raw_obs(self._env.reset().observation)
        if self.plan:  # the planner looks once, before the policy moves
            self.planned = plan(self._env.physics, self.command or COMMAND.format(task.target))
        return obs, {"is_success": False}

    @property
    def task_description(self):  # lerobot's rollout() hands this to the policy as the language command
        if self.plan:
            return self.planned
        return self.command or command(self._env.task.poses, self._env.task.target, self.slot)

    @task_description.setter
    def task_description(self, text):
        self.command = text


if os.environ.get("FOCUS"):  # FOCUS=50 run.sh train ...: draw each episode's first 50 frames 8x as often
    # Only the first action chunk (frames 0-49) needs the command: after it the arm already heads for a cube and the
    # state shows which. That is 1/8 of every 400-frame demo; repeating it 8x makes it about half of all samples.
    import torch
    import lerobot.scripts.lerobot_train as _train

    class FocusSampler(_train.EpisodeAwareSampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pos = np.arange(self._num_frames)
            in_episode = pos - np.repeat(np.r_[0, self._cum_lengths[:-1]], np.diff(np.r_[0, self._cum_lengths]))
            self._pool = np.concatenate([pos] + [pos[in_episode < int(os.environ["FOCUS"])]] * 7)

        def __len__(self):
            return len(self._pool)

        def __iter__(self):  # ponytail: no sample-exact resume (load_state_dict offset ignored); fine for fresh runs
            g = self._epoch_generator(self._epoch)
            self._epoch += 1
            return (self._frame_index(int(self._pool[k])) for k in torch.randperm(len(self._pool), generator=g))

    _train.EpisodeAwareSampler = FocusSampler
    print(f"FOCUS: first {os.environ['FOCUS']} frames of each episode drawn 8x as often", flush=True)

for _id, _colors in TASKS.items():
    gym.register(id=f"gym_aloha/{_id}", entry_point=TwoCubesEnv, max_episode_steps=400, nondeterministic=True,
                 kwargs={"obs_type": "pixels", "task": "two_cubes", "colors": _colors, "slot": _id in SLOT,
                         "plan": _id in PLAN})


PICK = [np.cos(np.pi / 4), 0, -np.sin(np.pi / 4), 0]  # right gripper pointing straight down: grasps from above
GIVE = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # right gripper level, rolled 90 deg about x: fingers vertical
# PICK -> GIVE turns the cube by whole 90s, so it stays square to the left gripper (a 60 deg pick left it skewed)
LEVEL = [1, 0, 0, 0]  # left gripper level, fingers horizontal: takes the cube by the two sides the right one left free
GRASP_Z = 0.03  # right fingertip height when closing on a cube resting on the table (cube centre at 0.02)
GIVE_AT = np.array([-0.05, 0.5, 0.22])  # right fingertips held out past the middle, toward the left arm
PLAN_LEFT = 270  # the right arm reached GIVE_AT at 245 and has settled: the left arm plans its grasp
TAKE_DX = 0.02  # left fingertips this far past the cube centre (grasp z 0.025-0.035 x take 0.02-0.03: 12/12 held)


def right_waypoints(right0, cube):
    """Rows: t, xyz, quat wxyz, gripper (1 = open). Pick from above, lift straight up (clear of the other cubes),
    then turn the tool level, roll it 90 deg and hold the cube out to the left arm, like the human demos."""
    x, y = cube[:2]
    return np.array([(0, *right0, 0), (90, x, y, GRASP_Z + 0.08, *PICK, 1), (130, x, y, GRASP_Z, *PICK, 1),
                     (170, x, y, GRASP_Z, *PICK, 0), (200, x, y, GRASP_Z + 0.1, *PICK, 0),
                     (245, *GIVE_AT, *GIVE, 0), (345, *GIVE_AT, *GIVE, 0), (360, *GIVE_AT, *GIVE, 1),
                     (390, *GIVE_AT + [0.1, 0, 0], *GIVE, 1), (400, *GIVE_AT + [0.1, 0, 0], *GIVE, 1)])


def left_waypoints(left0, cube):
    """Wait until the cube is held out, come in level from the left, close, hold while the right gripper lets go,
    then pull back with it. `cube` = where the right arm actually holds it at PLAN_LEFT."""
    take, back = cube + [TAKE_DX, 0, 0], cube + [-0.1, 0, 0]  # take: fingertips TAKE_DX past the cube centre
    return np.array([(0, *left0, 0), (PLAN_LEFT, *left0, 0), (300, *back, *LEVEL, 1), (320, *take, *LEVEL, 1),
                     (335, *take, *LEVEL, 0), (365, *take, *LEVEL, 0), (395, *back, *LEVEL, 0),
                     (400, *back, *LEVEL, 0)])


def scripted_joints(ee, seed, target=None):
    """Run the scripted demo for this seed's layout in the end-effector env; returns its joint trajectory (400 x 14).
    target: the colour to hand over (default: the seed's commanded colour)."""
    poses, seed_target = sample(seed, ee.task.colors)
    target = target or seed_target
    ee.task.poses, ee.task.target = poses, target
    ts = ee.reset()
    p = ee.physics.data
    left0, right0 = np.r_[p.mocap_pos[0], p.mocap_quat[0]], np.r_[p.mocap_pos[1], p.mocap_quat[1]]
    right = right_waypoints(right0, np.array(poses[target][:3]))
    left = left_waypoints(left0, np.zeros(3))  # only its waiting part is used before PLAN_LEFT
    joints = []
    for t in range(400):
        if t == PLAN_LEFT:
            left = left_waypoints(left0, ee.physics.named.data.qpos[f"{target}_box_joint"][:3].copy())
        q = ts.observation["qpos"].copy()  # gripper: commanded position, not the (blocked) measured one
        q[6], q[13] = (normalize_puppet_gripper_position(ts.observation["gripper_ctrl"][i]) for i in (0, 2))
        joints.append(q)
        a = np.concatenate([[np.interp(t, w[:, 0], c) for c in w[:, 1:].T] for w in (left, right)])
        for i in (3, 11):  # linearly blended quaternions -> unit length
            a[i:i + 4] /= np.linalg.norm(a[i:i + 4])
        ts = ee.step(a)
    return joints


def preview(seed, out, colors=COLORS):
    """Replay one layout's scripted demos, one per colour, back to back as video:
    top camera (what the policy sees) | side camera."""
    import imageio
    from PIL import Image, ImageDraw

    env = TwoCubesEnv(colors, task="two_cubes", obs_type="pixels_agent_pos")
    ee = make_control_env(make_physics(colors, ee=True), TwoCubesEETask(colors))
    frames = []
    for color in colors:
        joints = scripted_joints(ee, seed, color)
        env.command = COMMAND.format(color)
        obs, _ = env.reset(seed=seed)
        reward = 0
        for t, a in enumerate(joints):
            side = env._env.physics.render(height=480, width=640, camera_id="angle")
            img = Image.fromarray(np.concatenate([obs["pixels"]["top"], side], 1))
            ImageDraw.Draw(img).text((8, 8), f"seed {seed}  t={t}  reward {reward}  {env.command}", fill="white")
            frames.append(np.asarray(img))
            obs, reward, *_ = env.step(a)
        print(f"seed {seed} {color}: final reward {reward}")
    imageio.mimsave(out, frames, fps=50)
    print(f"-> {out}")


def record(n, repo_id, paired=False, colors=COLORS, slot=False):
    """ACT's sim recipe: end-effector waypoints -> joint trajectory -> replay it in the joint-space env (the one the
    policy is evaluated in) to record camera + state. Keeps a demo only if the left gripper still holds the
    commanded cube alone at the last step (no drop); n / len(colors) demos per colour.
    paired: every layout twice, once per colour (kept only if both work), so the image alone never tells which cube
    to take: without it each layout had one answer and the policy learned to pick by position, not by command.
    slot: commands name the cube's place, not its colour."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    motors = {"dtype": "float32", "shape": (14,), "names": {"motors": JOINTS}}
    ds = LeRobotDataset.create(repo_id, fps=50, robot_type="aloha", features={
        "observation.images.top": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
        "observation.state": motors, "action": motors})
    ee = make_control_env(make_physics(colors, ee=True), TwoCubesEETask(colors))
    env = TwoCubesEnv(colors, task="two_cubes", obs_type="pixels_agent_pos")
    seed, kept, count = 0, 0, collections.Counter()
    while kept < n:
        assert seed < 2 * n + 10, f"scripted demos keep failing ({kept}/{seed} kept)"
        targets = colors if paired else [sample(seed, colors)[1]]
        if not paired and count[targets[0]] >= n // len(colors):  # same count per colour: no colour prior
            seed += 1
            continue
        demos = []
        for target in targets:
            joints = scripted_joints(ee, seed, target)
            env.command = command(sample(seed, colors)[0], target, slot)  # replay env's target = this cube
            obs, _ = env.reset(seed=seed)
            frames = []
            for a in joints:
                frames.append(({"observation.images.top": obs["pixels"]["top"]},
                               obs["agent_pos"].astype(np.float32), a.astype(np.float32)))
                obs, reward, *_ = env.step(a)
            print(f"seed {seed} {target}: final reward {reward}", flush=True)
            demos.append((target, env.command, frames, reward))
        if all(reward == 4 for *_, reward in demos):
            for target, text, frames, _ in demos:
                for imgs, state, action in frames:
                    ds.add_frame({**imgs, "observation.state": state, "action": action, "task": text})
                ds.save_episode()
                kept, count[target] = kept + 1, count[target] + 1
        seed += 1
    ds.finalize()
    print(f"kept {kept} scripted demos from {seed} layouts {dict(count)} -> {ds.root}")


def breakdown(eval_dir, colors=COLORS):
    """Per-episode lerobot-eval results split by commanded colour and by where the commanded cube was (slot 0..n-1
    by y; two cubes: slot 0 = near side of the arms' line). Ignoring the command gets 1/n colours right."""
    import itertools
    import json
    import math

    m = json.load(open(f"{eval_dir}/eval_info.json"))["per_task"][0]["metrics"]
    rows = collections.defaultdict(collections.Counter)
    for seed, r, ok in zip(itertools.count(1000), m["max_rewards"], m["successes"]):  # episode i: seed 1000+i
        poses, target = sample(seed, colors)
        pick = "right" if 2 <= r < 10 else "wrong" if r >= 10 else "none"
        slot = sorted(poses, key=lambda c: poses[c][1]).index(target)
        for key in ("all", f"command {target}", f"commanded cube slot {slot}"):
            rows[key].update({pick: 1, "handed over": bool(ok), "n": 1})
    for key, c in rows.items():
        lifted = c["right"] + c["wrong"]
        k = 1 / len(colors)  # one-sided binomial test against guessing
        p = sum(math.comb(lifted, i) * k ** i * (1 - k) ** (lifted - i) for i in range(c["right"], lifted + 1))
        print(f"{key:22s} n={c['n']:3d}  right colour {c['right']}/{lifted} = {c['right'] / max(lifted, 1):.1%} "
              f"(p={p:.2g} vs chance)  nothing lifted {c['none']}  handed over {c['handed over']}/{c['n']}")


def probe(checkpoint, seeds, noises=8):
    """Does the command decide where the arm goes? Per scene and command: the first predicted chunk's right waist at
    t=49 (which way the right arm turns) for `noises` flow-matching noise draws (the same draws for every command).
    heading = draws whose waist is nearest the commanded cube's scripted-demo waist (ignoring the command: 1/n).
    correlation = per-command mean waist vs the demos', scene mean removed (a fixed per-word bias scores ~0)."""
    import torch
    from lerobot.envs.configs import AlohaEnv as AlohaEnvConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    import json
    from pathlib import Path

    policy = get_policy_class("smolvla").from_pretrained(checkpoint).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    cfg = AlohaEnvConfig(task=json.loads(Path(checkpoint, "train_config.json").read_text())["env"]["task"])
    colors = TASKS[cfg.task]
    env = make_env(cfg, n_envs=1)["aloha"][0]
    env_pre, _ = make_env_pre_post_processors(cfg, policy.config)
    ee = make_control_env(make_physics(colors, ee=True), TwoCubesEETask(colors))
    model, demo, hits = [], [], []
    for seed in seeds:
        d = {c: scripted_joints(ee, seed, c)[49][7] for c in colors}
        for c in colors:
            text = command(sample(seed, colors)[0], c, cfg.task in SLOT)
            env.envs[0].unwrapped.command = text
            obs, _ = env.reset(seed=[seed])
            batch = preprocess_observation(obs)
            batch["task"] = [text]
            batch = pre(env_pre(batch))
            w = []
            for n in range(noises):
                torch.manual_seed(n)
                with torch.no_grad():
                    w.append(float(post(policy.predict_action_chunk(batch))[0, 49, 7]))
            hits += [min(d, key=lambda k: abs(d[k] - x)) == c for x in w]
            model.append(np.mean(w))
            demo.append(d[c])
        print(f"seed {seed}: right waist at t=49, model mean per command "
              f"{({c: round(x, 2) for c, x in zip(colors, model[-len(colors):])})} vs demos "
              f"{({c: round(float(x), 2) for c, x in d.items()})}", flush=True)
    m, d = (np.reshape(x, (len(seeds), -1)) for x in (model, demo))
    r = np.corrcoef((m - m.mean(1, keepdims=True)).ravel(), (d - d.mean(1, keepdims=True)).ravel())[0, 1]
    print(f"{len(seeds)} scenes x {len(colors)} commands x {noises} noises: heading for the commanded cube "
          f"{np.mean(hits):.1%} (chance {1 / len(colors):.0%}), waist correlation with demos {r:.2f}")


def demo(checkpoint, seed, out):
    """One scene, one policy episode per colour command, back to back: the top camera (what the policy and the planner
    see), with the command and the planner's place command. Task: EVAL_TASK, else the checkpoint's
    (EVAL_TASK=AlohaFourCubesPlan-v0 PLANNER=<dir> for the two-stage system)."""
    import json
    from pathlib import Path

    import imageio
    from PIL import Image, ImageDraw
    from lerobot.envs.configs import AlohaEnv as AlohaEnvConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout

    task = os.environ.get("EVAL_TASK") or json.loads(Path(checkpoint, "train_config.json").read_text())["env"]["task"]
    cfg = AlohaEnvConfig(task=task, episode_length=500)
    env = make_env(cfg, n_envs=1)["aloha"][0]
    policy = get_policy_class("smolvla").from_pretrained(checkpoint).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    env_pre, env_post = make_env_pre_post_processors(cfg, policy.config)
    u = env.envs[0].unwrapped
    frames, plans = [], []

    def grab(_env):
        if len(frames) == start:
            plans.append(u.planned)  # the episode's; the vector env's auto-reset at the end plans a new scene
        p = u._env.physics
        img = Image.fromarray(p.render(height=480, width=640, camera_id="top"))
        text = f"seed {seed}  t={len(frames) - start}  command: {u.command}" + (f"\n-> planner: {u.planned}" if u.plan else "")
        ImageDraw.Draw(img).text((8, 8), text, fill="white")
        frames.append(np.asarray(img))

    for c in TASKS[task]:
        u.command, start = COMMAND.format(c), len(frames)
        r = rollout(env, policy, env_pre, env_post, pre, post, seeds=[seed], render_callback=grab)
        frames.pop()  # rendered after the auto-reset: a new random scene
        print(f"seed {seed} {c}: max reward {r['reward'].max().item():.0f}, success {bool(r['success'].any())}"
              + (f", planner -> {plans[-1]!r}" if u.plan else ""), flush=True)
    imageio.mimsave(out, frames, fps=50)
    print(f"-> {out}")


if __name__ == "__main__":
    cols = next((tuple(a.split(",")) for a in sys.argv[2:] if "," in a), COLORS)
    {"record": lambda: record(int(sys.argv[2]), sys.argv[3], "paired" in sys.argv[4:], cols, "slot" in sys.argv[4:]),
     "preview": lambda: preview(int(sys.argv[2]), sys.argv[3], cols),
     "breakdown": lambda: breakdown(sys.argv[2], cols),
     "probe": lambda: probe(sys.argv[2], range(*map(int, sys.argv[3].split(":"))) if ":" in sys.argv[3]
                            else [int(sys.argv[3])]),
     "demo": lambda: demo(sys.argv[2], int(sys.argv[3]), sys.argv[4])}[sys.argv[1]]()
