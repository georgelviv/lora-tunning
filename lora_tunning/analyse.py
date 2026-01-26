from .utils import get_results_dir
import logging
from .models import LoraBase, Algorithm
import pandas as pd
import json

high_reward = 0.3

def count_time(history_df: pd.DataFrame) -> int:
  from_time = history_df['timestamp'].iloc[0]
  to_time = history_df['timestamp'].iloc[-1]
  diff = to_time - from_time
  return int(diff.total_seconds())

def count_high_rewards(history_df: pd.DataFrame) -> int:
  high_rewards = (history_df['reward'] > high_reward).sum()
  return int(high_rewards)

def get_convergence_iteration(history_df: pd.DataFrame) -> int | None:
  mask = history_df['reward_ma'] > high_reward
  if not mask.any():
    return None
  return int(history_df.index[mask][0])

def get_rewards_sum(history_df: pd.DataFrame) -> int:
  rewards_sum = (history_df['reward']).sum()
  return rewards_sum

def get_highest_reward(history_df: pd.DataFrame) -> int:
  highest_reward = history_df['reward'].max()
  return highest_reward

def analyse(logger: logging.Logger, backend: LoraBase, algorithm: Algorithm) -> None:
  results_dir = get_results_dir(backend, algorithm)
  history_path = results_dir / "history.csv"
  history_df = pd.read_csv(history_path)

  history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
  history_df['reward_ma'] = history_df['reward'].rolling(window=50).mean()
  analysis_results_path = results_dir / 'analysis_results.json'

  results = {
    'algorithm': algorithm.name,
    'environment': backend.name,
    'time_diff': count_time(history_df),
    'high_rewards': count_high_rewards(history_df),
    'convergence_iteration': get_convergence_iteration(history_df),
    'rewards_sum': get_rewards_sum(history_df),
    'highest_reward': get_highest_reward(history_df)
  }

  with open(analysis_results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

  logger.info('Analysis results:')
  logger.info(f'Time diff: {results['time_diff']}')
  logger.info(f'High rewards: {results['high_rewards']}')
  logger.info(f'Convergence iteration: {results['convergence_iteration']}')
  logger.info(f'Rewards sum: {results['rewards_sum']}')
  logger.info(f'Highest reward: {results['highest_reward']}')
  logger.info('Analysis results saved')