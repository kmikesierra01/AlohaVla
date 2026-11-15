"""Watch an ACT or SmolVLA policy live in a MuJoCo viewer window (opens on Windows via WSLg).

Runs lerobot's own eval rollout. Episode i uses seed 1000+i, the same seed lerobot-eval and the
end-of-training eval use, so episode i replays the scene of eval_episode_i.mp4. Close the window to quit.
Runs the task the checkpoint was trained on (TransferCube or the red/blue two_cubes task).
SmolVLA: type the language command in the terminal (two cubes: the colour named in it is the one to hand over).
After each episode, a new command replays the same scene (compare commands on one scene); Enter keeps the
command and moves on to the next scene.
Usage: python view.py <pretrained_model dir> [first_episode]
"""
import json
import os
import sys
import time
from pathlib import Path

import mujoco.viewer

from lerobot.envs.configs import AlohaEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout

import two_cubes  # registers gym_aloha/AlohaTwoCubes-v0

path = sys.argv[1]
ep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
kind = json.load(open(f"{path}/config.json"))["type"]  # act | smolvla
cfg = Path(path, "train_config.json")
task = json.loads(cfg.read_text())["env"]["task"] if cfg.exists() else "AlohaTransferCube-v0"
task = os.environ.get("EVAL_TASK", task)  # EVAL_TASK=AlohaFourCubesPlan-v0 PLANNER=<dir>: type colour commands
cmd = (two_cubes.COMMAND.format("first" if task in two_cubes.SLOT else "red") if task in two_cubes.TASKS
       else "Pick up the cube with the right arm and transfer it to the left arm.")  # TransferCube's only sentence

env_cfg = AlohaEnv(task=task)
env = make_env(env_cfg, n_envs=1)["aloha"][0]
policy = get_policy_class(kind).from_pretrained(path).eval()
pre, post = make_pre_post_processors(  # saved processors already hold SmolVLA's camera rename + tokenizer
    policy.config, pretrained_path=path,
    preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
)
env_pre, env_post = make_env_pre_post_processors(env_cfg, policy.config)
physics = env.envs[0].unwrapped._env.physics  # dm_control physics that gym-aloha steps

with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
    # start on the same fixed "top" camera the videos use ([ ] cycles cameras, Esc = free camera)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    last = time.time()

    def show(_env):
        global last
        if not viewer.is_running():
            raise KeyboardInterrupt
        viewer.sync()
        time.sleep(max(0.0, last + 1 / env_cfg.fps - time.time()))  # play at real time (50 Hz)
        last = time.time()

    try:
        if kind == "smolvla":
            cmd = input(f"command (Enter = training sentence '{cmd}'): ").strip() or cmd
        while viewer.is_running():
            # rollout() feeds env.task_description to the policy as the language command (ACT ignores it)
            env.envs[0].unwrapped.task_description = cmd
            if kind == "smolvla":
                print(f"> episode {ep} (seed {1000 + ep}), command: {cmd!r}", flush=True)
            out = rollout(env, policy, env_pre, env_post, pre, post, seeds=[1000 + ep], render_callback=show)
            planned = getattr(env.envs[0].unwrapped, "planned", None)  # plan tasks: what the planner told the policy
            print(f"episode {ep} (seed {1000 + ep}): max_reward {out['reward'].max().item():.0f}/4, "
                  f"success {bool(out['success'].any())}" + (f", planner -> {planned!r}" if planned else ""), flush=True)
            if kind == "smolvla":
                new = input("new command = replay this scene, Enter = next scene: ").strip()
                if new:
                    cmd = new
                    continue
            ep += 1
    except (KeyboardInterrupt, EOFError):  # window closed, Ctrl+C, or stdin ended
        pass
