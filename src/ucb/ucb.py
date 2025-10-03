from src.mab_reward_exponential import MultiArmedBanditRewardExponential
from ..models import Action
import random
import itertools
import pandas as pd
from ..utils import current_limit_for_tp
import math
from .models import PrimaryAction, SecondaryAction
import os

class UCB(MultiArmedBanditRewardExponential):
  def __init__(self, results_file, history_file, ubf_file, epsilon=0.9,
               alpha=0.3, exploration_factor=0.1):

    self.ucb_file = ubf_file
    self.exploration_factor = exploration_factor
    super().__init__(results_file, history_file, epsilon=epsilon, alpha=alpha)
  

    if os.path.exists(self.ucb_file):
      os.remove(self.ucb_file)

    self.ucb_df = pd.DataFrame(
      columns=["SF", "BW", "CR", "IH", "count", "reward", "ucb"]
    ).astype({
      "count": "int64",
      "reward": "float64",
      "ucb": "float64"
    })

    self.setup_initial_ucb_table()

  def ucb(self, reward, count, total_count):
    if count == 0:
      return float('inf')
    return reward + self.exploration_factor * math.sqrt(
      2 * math.log(total_count + 1) / count
    )

  def choose_action(self) -> Action:
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
    super().update(action, reward)
    self.update_ucb_table(action, reward)

  def update_ucb_table(self, action: Action, reward) -> None:
    active_keys = ["SF", "BW", "CR", "IH"]
    action_active = {k: action[k] for k in active_keys}

    mask = (self.ucb_df[active_keys] == pd.Series(action_active)).all(axis=1)
    if mask.any():
      idx = self.ucb_df[mask].index[0]
      n = self.ucb_df.at[idx, "count"] + 1
      old_reward = self.ucb_df.at[idx, "reward"]
      new_reward = self.compute_reward(old_reward, reward)
      self.ucb_df.at[idx, "count"] = n
      self.ucb_df.at[idx, "reward"] = float(new_reward)
    else:
      new_row = {**action_active, "count": 1, "reward": float(reward)}
      self.ucb_df = pd.concat([self.ucb_df, pd.DataFrame([new_row])], ignore_index=True)
  
  def save(self) -> None:
    super().save()

    if not self.ucb_df.empty:
      df_ucb_sorted = self.ucb_df.sort_values(by="reward", ascending=False)
      df_ucb_sorted.to_csv(self.ucb_file, index=False, float_format="%.4f")

  def load(self) -> None:
    super().load()
    try:
      self.ucb_df = pd.read_csv(self.ucb_file)
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