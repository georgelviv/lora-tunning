from ..models import Action
import random
import itertools
import pandas as pd
from ..utils import current_limit_for_tp
import os
import math
import sys
from .models import PrimaryAction, SecondaryAction

class UCB:
  def __init__(self, history_file, ubf_file, results_file, epsilon=0.9,
               alpha=0.3, exploration_factor=0.1, decay=True, exponential_reward=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    self.alpha = alpha
    self.epsilon = epsilon
    self.decay = decay
    self.exploration_factor = exploration_factor
    self.exponential_reward = exponential_reward

    self.history_file = os.path.join(base_dir, history_file)
    self.ucb_file = os.path.join(base_dir, ubf_file)
    self.results_file = os.path.join(base_dir, results_file)

    self.history_df = pd.DataFrame(columns=[
      "iteration", "reward", "timestamp"
    ]).astype({
      "iteration": "int64",
      "reward": "float64",
      "timestamp": "datetime64[ns]"
    })

    self.ucb_df = pd.DataFrame(
      columns=["SF", "BW", "CR", "IH", "count", "reward", "ucb"]
    ).astype({
      "count": "int64",
      "reward": "float64",
      "ucb": "float64"
    })

    self.q_df = pd.DataFrame(
      columns=["SF", "FQ", "BW", "CR", "TP", "IH", "HS", "PL", "CL", "RT", 
        "count", "reward"]
    ).astype({
      "count": "int64",
      "reward": "float64"
    })

    self.load()

  def ucb(self, reward, count, total_count):
    if count == 0:
      return float('inf')
    return reward + self.exploration_factor * math.sqrt(
      2 * math.log(total_count + 1) / count
    )

  def choose_action(self) -> Action:
    if self.ucb_df.empty:
      self.setup_initial_ucb_table()
      self.save()

    total_count = self.ucb_df["count"].sum()
    self.ucb_df["ucb"] = self.ucb_df.apply(
      lambda row: self.ucb(row["reward"], row["count"], total_count),
      axis=1
    )

    best_idx = self.ucb_df["ucb"].idxmax()
    primary = self.ucb_df.loc[best_idx, ["SF", "BW", "CR", "IH"]].to_dict()

    if random.random() < self.epsilon or self.q_df.empty:
      secondary = self.random_secondary_action()
    else:
      best_idx = self.q_df["reward"].idxmax()
      secondary = self.q_df.loc[best_idx, ["FQ", "TP", "HS", "PL", "CL", "RT"]].to_dict()
    return {**primary, **secondary}

  
  def update(self, action: Action, reward):
    self.update_q_table(action, reward)
    self.update_ucb_table(action, reward)
    self.update_history(reward)
    self.update_epsilon()

  def update_ucb_table(self, action: Action, reward) -> None:
    active_keys = ["SF", "BW", "CR", "IH"]
    action_active = {k: action[k] for k in active_keys}

    mask = (self.ucb_df[active_keys] == pd.Series(action_active)).all(axis=1)
    if mask.any():
      idx = self.ucb_df[mask].index[0]
      n = self.ucb_df.at[idx, "count"] + 1
      old_reward = self.ucb_df.at[idx, "reward"]
      if self.exponential_reward:
        new_reward = self.compute_reward_exponential(old_reward, reward)
      else:
        new_reward = self.compute_reward(old_reward, reward, n)
      self.ucb_df.at[idx, "count"] = n
      self.ucb_df.at[idx, "reward"] = float(new_reward)
    else:
      new_row = {**action_active, "count": 1, "reward": float(reward)}
      self.ucb_df = pd.concat([self.ucb_df, pd.DataFrame([new_row])], ignore_index=True)

  def update_q_table(self, action: Action, reward) -> None:
    mask = (self.q_df[list(action)] == pd.Series(action)).all(axis=1)

    if mask.any():
      idx = self.q_df[mask].index[0]
      n = self.q_df.at[idx, "count"] + 1
      reward_val = self.q_df.at[idx, "reward"]
      if self.exponential_reward:
        reward_val = self.compute_reward_exponential(reward_val, reward)
      else:
        reward_val = self.compute_reward(reward_val, reward, n)
      self.q_df.at[idx, "count"] = n
      self.q_df.at[idx, "reward"] = float(reward_val) 
    else:
      action_with_stats = {**action, "count": 1, "reward": float(reward)}
      self.q_df = pd.concat([self.q_df, pd.DataFrame([action_with_stats])], ignore_index=True)

  def update_history(self, reward) -> None:
    new_row = {
      "iteration": len(self.history_df) + 1,
      "reward": float(reward),
      "timestamp": pd.Timestamp.now()
    }

    self.history_df = pd.concat([self.history_df, pd.DataFrame([new_row])], ignore_index=True)

  def compute_reward(self, old_value: float, new_value: float, n: int) -> float:
    return old_value + (new_value- old_value) / n
  
  def compute_reward_exponential(self, old_value: float, new_value: float) -> float:
    # exponential recency-weighted average
    return old_value * (1 - self.alpha) + new_value * self.alpha
  
  def save(self) -> None:
    if not self.q_df.empty:
      df_sorted = self.q_df.sort_values(by="reward", ascending=False)
      df_sorted.to_csv(self.results_file, index=False, float_format="%.4f")

    if not self.history_df.empty:
      self.history_df.to_csv(self.history_file, index=False,
                             float_format="%.4f", date_format="%Y-%m-%d %H:%M:%S")

    if not self.ucb_df.empty:
      df_ucb_sorted = self.ucb_df.sort_values(by="reward", ascending=False)
      df_ucb_sorted.to_csv(self.ucb_file, index=False, float_format="%.4f")

  def update_epsilon(self):
    if self.decay:
      self.epsilon = max(0.01, self.epsilon * 0.995)

  def load(self) -> None:
    try:
      self.ucb_df = pd.read_csv(self.ucb_file)
    except FileNotFoundError:
      pass
    try:
      self.q_df = pd.read_csv(self.results_file)
    except FileNotFoundError:
      pass
    try:
      self.history = pd.read_csv(self.history_file)
    except FileNotFoundError:
      pass

  def setup_initial_ucb_table(self) -> PrimaryAction:
    sf_values = list(range(6, 13))
    bw_values = [125, 250, 500]
    cr_values = list(range(5, 9))

    actions = []
    for sf, bw, cr in itertools.product(sf_values, bw_values, cr_values):
      ih = 1 if sf == 6 else 0
      actions.append((sf, ih, bw, cr))

    df = pd.DataFrame(actions, columns=["SF", "IH", "BW", "CR"])
    df["reward"] = 0.0
    df["count"] = 0

    self.ucb_df = df
  
  def random_secondary_action(self) -> SecondaryAction:
    tp = random.choice([x for x in range(2, 21) if x not in (18, 19)])
    cl = current_limit_for_tp(tp)

    return {
      "FQ": random.choice(range(860, 891)),
      "TP": tp,
      "HS": 200,
      "PL": random.choice(range(6, 101)),
      "CL": cl,
      "RT": 1
    } 