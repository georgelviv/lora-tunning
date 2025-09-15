from ..models import Action
import random
import pandas as pd
from ..utils import current_limit_for_tp
import os

class MultiArmedBanditDecay:
  def __init__(self, results_file, history_file, epsilon=0.9):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    self.epsilon = epsilon
    self.results_file = os.path.join(base_dir, results_file)
    self.history_file = os.path.join(base_dir, history_file)

    self.history_df = pd.DataFrame(columns=[
      "iteration", "reward", "timestamp", "epsilon"
    ]).astype({
      "iteration": "int64",
      "reward": "float64",
      "timestamp": "datetime64[ns]",
      "epsilon": "float64"
    })

    self.df = pd.DataFrame(
      columns=["SF", "FQ", "BW", "CR", "TP", "IH", "HS", "PL", "CL", "RT", "count", "reward"]
    ).astype({
      "count": "int64",
      "reward": "float64"
    })

    self.load()

  def choose_action(self) -> Action:
    if self.df.empty or random.random() < self.epsilon:
      return self.random_action()
    
    best_row = self.df.sort_values(by='reward', ascending=False).iloc[0]
    return best_row.drop(["count", "reward"]).to_dict()
  
  def update(self, action: Action, reward):
    mask = (self.df[list(action)] == pd.Series(action)).all(axis=1)

    if mask.any():
      idx = self.df[mask].index[0]
      n = self.df.at[idx, "count"] + 1
      reward_val = self.df.at[idx, "reward"]
      reward_val = reward_val + (reward- reward_val) / n
      self.df.at[idx, "count"] = n
      self.df.at[idx, "reward"] = float(reward_val) 
    else:
      action_with_stats = {**action, "count": 1, "reward": float(reward)}
      self.df = pd.concat([self.df, pd.DataFrame([action_with_stats])], ignore_index=True)

    new_row = {
      "iteration": len(self.history_df) + 1,
      "reward": float(reward),
      "timestamp": pd.Timestamp.now(),
      "epsilon": float(self.epsilon)
    }

    self.history_df = pd.concat([self.history_df, pd.DataFrame([new_row])], ignore_index=True)
    self.epsilon = max(0.01, self.epsilon * 0.995)
  
  def save(self) -> None:
    if not self.df.empty:
      df_sorted = self.df.sort_values(by="reward", ascending=False)
      df_sorted.to_csv(self.results_file, index=False)

    if not self.history_df.empty:
      self.history_df.to_csv(self.history_file, index=False)

  def load(self) -> None:
    try:
      self.df = pd.read_csv(self.results_file)
    except FileNotFoundError:
      pass
    try:
      self.history = pd.read_csv(self.history_file)
    except FileNotFoundError:
      pass

  def random_action(self) -> Action:
    sf = random.choice(range(6, 13))
    ih = 1 if sf == 6 else 0
    tp = random.choice([x for x in range(2, 21) if x not in (18, 19)])
    cl = current_limit_for_tp(tp)

    return {
      "SF": sf,
      "FQ": random.choice(range(860, 891)),
      "BW": random.choice([125, 250, 500]),
      "CR": random.choice(range(5, 9)),
      "TP": tp,
      "IH": ih,
      "HS": 200,
      "PL": random.choice(range(6, 101)),
      "CL": cl,
      "RT": 1
    }