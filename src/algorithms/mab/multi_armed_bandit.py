from typing import Dict
from ...models import Action
import random
import pandas as pd
from ..utils import current_limit_for_tp
from ..utils import prepare_results

from pathlib import Path

class MultiArmedBandit:
  def __init__(self, epsilon=0.9, extra_files: Dict[str, str] | None = None):
    self.epsilon = epsilon

    base_dir = Path(__file__).resolve().parent
    self.files = prepare_results(base_dir, extra_files)

    self.history_df = pd.DataFrame(columns=[
      "iteration", "reward", "timestamp"
    ]).astype({
      "iteration": "int64",
      "reward": "float64",
      "timestamp": "datetime64[ns]"
    })

    self.q_df = pd.DataFrame(
      columns=["SF", "FQ", "BW", "CR", "TP", "IH", "HS", "PL", "CL", "RT", 
        "count", "reward"]
    ).astype({
      "count": "int64",
      "reward": "float64"
    })

  def choose_action(self) -> Action:
    if self.q_df.empty:
      return self.random_action()
    
    if random.random() > self.epsilon:
      best_row = self.q_df.sort_values(by='reward', ascending=False).iloc[0]
      return best_row.drop(["count", "reward"]).to_dict()
    
    return self.random_action()
  
  def update(self, action: Action, reward):
    if action == 0:
      ## ignore unsuccess 
      return
    self.update_q_table(action, reward)
    self.update_history(reward)

  def update_q_table(self, action: Action, reward) -> None:
    mask = (self.q_df[list(action)] == pd.Series(action)).all(axis=1)

    if mask.any():
      idx = self.q_df[mask].index[0]
      n = self.q_df.at[idx, "count"] + 1
      reward_val = self.q_df.at[idx, "reward"]
      self.compute_reward(reward_val, reward, n)
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
  
  def save(self) -> None:
    if not self.q_df.empty:
      df_sorted = self.q_df.sort_values(by="reward", ascending=False)
      df_sorted.to_csv(self.files["results"], index=False, float_format="%.4f")

    if not self.history_df.empty:
      self.history_df.to_csv(self.files["history"], index=False,
                             float_format="%.4f", date_format="%Y-%m-%d %H:%M:%S")

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
  
  def get_iteration(self) -> int:
    return len(self.history_df)