# uv run $HOME/projects/hf-reinforcement-leanring/scripts/unit2/unit2.py \
#   --n_training_episodes=1000000 \
#   --learning_rate=0.5 \
#   --n_eval_episodes=100 \
#   --env_id="FrozenLake-v1" \
#   --map_name="4x4" \
#   --slippery \
#   --max_steps=500 \
#   --gamma=0.95 \
#   --max_epsilon=1.0 \
#   --min_epsilon=0.01 \
#   --decay_rate=5e-4 \
#   --username=$HF_UNAME \
#   --repo_name="q-FrozenLake-v1-4x4-Slippery" \
#   --save_config_path="$HOME/projects/hf-reinforcement-leanring/scripts/unit2/config-lake-slip.json"
#
# uv run $HOME/projects/hf-reinforcement-leanring/scripts/unit2/unit2.py \
#   --n_training_episodes=1000000 \
#   --learning_rate=0.5 \
#   --n_eval_episodes=100 \
#   --env_id="FrozenLake-v1" \
#   --map_name="4x4" \
#   --max_steps=500 \
#   --gamma=0.95 \
#   --max_epsilon=1.0 \
#   --min_epsilon=0.01 \
#   --decay_rate=5e-4 \
#   --username=$HF_UNAME \
#   --repo_name="q-FrozenLake-v1-4x4-noSlippery" \
#   --save_config_path="$HOME/projects/hf-reinforcement-leanring/scripts/unit2/config-lake-noslip.json"
#
uv run $HOME/projects/hf-reinforcement-leanring/scripts/unit2/unit2.py \
  --n_training_episodes=1000000 \
  --learning_rate=0.5 \
  --n_eval_episodes=100 \
  --env_id="Taxi-v3" \
  --max_steps=500 \
  --gamma=0.95 \
  --max_epsilon=1.0 \
  --min_epsilon=0.01 \
  --decay_rate=5e-4 \
  --username=$HF_UNAME \
  --repo_name="q-Taxi-v3-4x4" \
  --save_config_path="$HOME/projects/hf-reinforcement-leanring/scripts/unit2/config-taxi-v3.json"
