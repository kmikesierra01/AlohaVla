#!/usr/bin/env bash
# Usage (Windows): wsl -d Ubuntu-22.04 -- bash /mnt/c/<repo path>/run.sh <eval|inspect|record|train|view> [args]
# Outputs land in ./outputs so videos can be opened from Windows.
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd "$(dirname "$0")"
export MUJOCO_GL=egl   # headless MuJoCo rendering on the WSL GPU
# eval.batch_size=1: several MuJoCo EGL contexts in one process crash on WSL Mesa (malloc corruption), and
# async envs (one per process) fail in lerobot 0.6.1 because forkserver workers never import gym_aloha.

TASK=${TASK:-AlohaTransferCube-v0}   # TASK=AlohaTwoCubes-v0 / AlohaFourCubes-v0: coloured-cube tasks (two_cubes.py)
DATASET=${DATASET:-lerobot/aloha_sim_transfer_cube_human}
POLICY=${POLICY:-act}   # POLICY=smolvla run.sh train ... fine-tunes lerobot/smolvla_base instead of training ACT from scratch
# smolvla_base was pretrained with 3 SO-100 cameras (camera1..3): ALOHA's one top camera goes in as camera1, the
# other two are left out (empty_cameras=0). Needed at eval too: lerobot-eval otherwise overrides the saved map with {}.
RENAME='{"observation.images.top":"observation.images.camera1","observation.images.zoom":"observation.images.camera2"}'

# lerobot-train / lerobot-eval, but the env hands the policy the dataset's language command. gym-aloha only has
# env.task ("transfer_cube"), which lerobot would feed SmolVLA instead of the sentence it was trained on. ACT ignores it.
# TASK_TEXT="..." run.sh eval ... feeds another command instead (language-sensitivity test; may be empty).
lr() {  # lr <train|eval> [args]
  python -c "import os, two_cubes, gym_aloha.env as e
e.AlohaEnv.task_description = property(lambda s: os.environ['TASK_TEXT'] if 'TASK_TEXT' in os.environ else {'transfer_cube': 'Pick up the cube with the right arm and transfer it to the left arm.', 'insertion': 'Insert the peg into the socket.'}[s.task])
from lerobot.scripts.lerobot_$1 import main; main()" "${@:2}"
}

tag() { [ $TASK = AlohaTransferCube-v0 ] || echo "${TASK%-v0}_"; }   # output-dir prefix for non-default tasks

case "$1" in
eval)  # eval [model_dir] [n_episodes]; default = pretrained ACT from the Hub -> success rate + rollout videos
  # Hub checkpoint predates the processor format (no policy_preprocessor.json) -> convert once, locally
  HUB=~/models/act_aloha_sim_transfer_cube_human_migrated
  [ -f $HUB/policy_preprocessor.json ] || python -m lerobot.processor.migrate_policy_normalization \
    --pretrained-path lerobot/act_aloha_sim_transfer_cube_human --output-dir $HUB
  MODEL=${2:-$HUB}
  NAME=$([ "$MODEL" = "$HUB" ] && echo pretrained || basename "$(dirname "$MODEL")")   # checkpoint step, e.g. 080000
  TYPE=$(grep -q '"type": "smolvla"' $MODEL/config.json && echo smolvla || echo act)
  # evaluate on the task the checkpoint was trained on (the Hub ACT has no train_config.json -> TransferCube).
  # Coloured cubes: 500-step limit (fixed before any final result): the demos themselves only reach success (0.5 s solo
  # hold) at ~step 380, so the default 400 would leave 20 steps of slack and count slightly slow hand-overs as failures.
  [ -f $MODEL/train_config.json ] && TASK=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['env']['task'])" $MODEL/train_config.json)
  # EVAL_TASK=AlohaFourCubesPlan-v0 PLANNER=<planner.py model> eval <place-command model>: colour commands via the planner
  TASK=${EVAL_TASK:-$TASK}
  lr eval \
    --policy.path=$MODEL \
    --policy.device=cuda \
    $([ $TYPE = smolvla ] && echo --rename_map=$RENAME) \
    --env.type=aloha --env.task=$TASK \
    $(case $TASK in *Cubes*-v0) echo --env.episode_length=500;; esac) \
    --eval.n_episodes=${3:-10} --eval.batch_size=1 \
    --output_dir=outputs/eval/$(tag)${TYPE}_${NAME}_$(date +%m%d_%H%M)
  ;;
inspect)  # dataset structure: episodes, fps, features, one frame
  python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("$DATASET")
print("episodes:", ds.num_episodes, "frames:", ds.num_frames, "fps:", ds.fps)
for k, v in ds.features.items():
    print(f"  {k:32s} {v['dtype']:8s} {v['shape']}")
f = ds[0]
for k, v in f.items():
    print(f"  frame[0].{k:24s}", getattr(v, "shape", v))
PY
  ;;
train)  # [POLICY=smolvla] train [steps] [every] [batch] [--lerobot.args ...]: checkpoint + 50-episode sim eval every N steps (ACT ~13 step/s on RTX 4070 SUPER)
  # 50 episodes because 5 was misleading: the 2k model scored 1/5 there but 30/66 in the viewer
  OUT=outputs/train/$(tag)${POLICY}_$(date +%m%d_%H%M)
  # SmolVLA recipe is batch 64, but on 12GB that spills into shared memory (34 s/step); 32 peaks at 9.7GB, ~1 s/step
  if [ $POLICY = smolvla ]; then P="--policy.path=lerobot/smolvla_base --rename_map=$RENAME"; BS=32
  else P=--policy.type=act; BS=8; fi
  mkdir -p outputs/train
  lr train \
    $P --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=$DATASET \
    --env.type=aloha --env.task=$TASK \
    --steps=${2:-2000} --batch_size=${4:-$BS} --log_freq=100 \
    --save_freq=${3:-${2:-2000}} --env_eval_freq=${3:-${2:-2000}} \
    --eval.n_episodes=50 --eval.batch_size=1 \
    --output_dir=$OUT "${@:5}" 2>&1 | tee $OUT.log   # log keeps the per-checkpoint pc_success after the terminal closes
  ;;
record)  # record [n] [repo_id] [paired] [red,blue,orange,green] [zoom]: scripted coloured-cube demos -> local LeRobot dataset (~10 s each)
  python two_cubes.py record ${2:-100} ${3:-local/aloha_sim_two_cubes} "${@:4}"
  ;;
view)  # live MuJoCo window; default = latest own ACT checkpoint, or pass a model dir (SmolVLA: type commands in the terminal)
  python view.py "${2:-$(ls -d outputs/1_transfercube_act_vs_smolvla/train/act_*/checkpoints/0*/pretrained_model | tail -1)}" $3
  ;;
*) echo "usage: $0 eval [model_dir] [n_episodes] | inspect | record [n] [repo_id] [paired] | [POLICY=smolvla] train [steps] [every] [batch] | view [model_dir] [first_episode]"; exit 1 ;;
esac
