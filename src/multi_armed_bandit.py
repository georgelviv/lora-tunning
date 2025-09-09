from .models import Action
from collections import defaultdict
import random
import json
import pandas as pd
from .utils import current_limit_for_tp

class MultiArmedBandit:
  def __init__(self, epsilon=0.3):
    self.epsilon = epsilon
    self.df = pd.DataFrame(
      columns=["SF", "FQ", "BW", "CR", "TP", "IH", "HS", "PL", "CL", "RT", "count", "reward"]
    )

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
      self.df.at[idx, "reward"] = reward
    else:
      action_with_stats = {**action, "count": 1, "reward": reward}
      self.df = pd.concat([self.df, pd.DataFrame([action_with_stats])], ignore_index=True)
  
  def save_results(self, file_path: str) -> None:
    if not self.df.empty:
      df_sorted = self.df.sort_values(by="reward", ascending=False)
      df_sorted.to_csv(file_path, index=False)

  def load_results(self, file_path: str) -> None:
      self.df = pd.read_csv(file_path)

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