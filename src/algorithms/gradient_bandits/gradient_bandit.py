import pandas as pd
import numpy as np
from ..ucb import PrimaryAction, SecondaryAction
from ..models import Action
from ..mab_reward_exponential import MultiArmedBanditRewardExponential
import os
import random
import itertools
from ..utils import current_limit_for_tp

class GradientBandit(MultiArmedBanditRewardExponential):
  def __init__(self, results_file, history_file, gradients_file, alpha=0.1, epsilon=0.9):
    super().__init__(results_file, history_file, epsilon=epsilon, alpha=alpha)
  
    self.gradients_file = gradients_file
    self.alpha = alpha

    if os.path.exists(self.gradients_file):
      os.remove(self.gradients_file)

    self.gradients_df = pd.DataFrame(
      columns=["SF", "BW", "CR", "IH", "preference", "probability"]
    ).astype({"preference": "float64", "probability": "float64"})

    self.setup_initial_gradients_table()
    self.avg_reward = 0

  def choose_action(self) -> Action:
    if self.q_df.empty:
      return self.random_action()

    prefs = np.array(self.gradients_df["preference"])
    exp_prefs = np.exp(prefs - np.max(prefs))
    probs = exp_prefs / np.sum(exp_prefs)
    self.gradients_df["probability"] = probs

    idx = np.random.choice(len(self.gradients_df), p=probs)
    action_row = self.gradients_df.iloc[idx]
    primary: PrimaryAction = action_row.drop(["preference", "probability"]).to_dict()

    if random.random() < self.epsilon or self.q_df.empty:
      secondary = self.random_secondary_action()
    else:
      best_idx = self.q_df["reward"].idxmax()
      secondary: SecondaryAction = self.q_df.loc[best_idx, ["FQ", "TP", "HS", "PL", "CL", "RT"]].to_dict()
    return {**primary, **secondary}
  
  def update(self, action: Action, reward: float):
    super().update(action, reward)
    self.update_gradients_table(action, reward)

  
  def update_gradients_table(self, action: Action, reward: float) -> None:
    active_keys = ["SF", "BW", "CR", "IH"]
    action_active = {k: action[k] for k in active_keys}
  
    mask = (self.gradients_df[active_keys] == pd.Series(action_active)).all(axis=1)
  
    if not mask.any():
      action_with_pref = {**action_active, "preference": 0.0, "probability": 0.0}
      self.gradients_df = pd.concat([self.gradients_df, pd.DataFrame([action_with_pref])], ignore_index=True)
      return

    idx = self.gradients_df[mask].index[0]
    prefs = np.array(self.gradients_df["preference"])
    exp_prefs = np.exp(prefs - np.max(prefs))
    probs = exp_prefs / np.sum(exp_prefs)

    n = len(self.history_df) + 1
    self.avg_reward += (reward - self.avg_reward) / n

    for i in range(len(self.gradients_df)):
      if i == idx:
        self.gradients_df.at[i, "preference"] += self.alpha * (reward - self.avg_reward) * (1 - probs[i])
      else:
        self.gradients_df.at[i, "preference"] -= self.alpha * (reward - self.avg_reward) * probs[i]

    exp_prefs = np.exp(self.gradients_df["preference"] - np.max(self.gradients_df["preference"]))
    self.gradients_df["probability"] = exp_prefs / np.sum(exp_prefs)

  def save(self) -> None:
    super().save()

    if not self.gradients_df.empty:
      df_gradients_sorted = self.gradients_df.sort_values(by="preference", ascending=False)
      df_gradients_sorted.to_csv(self.gradients_file, index=False, float_format="%.4f")
      
  def setup_initial_gradients_table(self) -> None:
    sf_values = list(range(6, 13))
    bw_values = [125, 250, 500]
    cr_values = list(range(5, 9))

    actions = []
    for sf, bw, cr in itertools.product(sf_values, bw_values, cr_values):
      ih = 1 if sf == 6 else 0
      actions.append((sf, ih, bw, cr))

    df = pd.DataFrame(actions, columns=["SF", "IH", "BW", "CR"])
    df["preference"] = 0.0
    df["probability"] = 1.0 / len(df)

    self.gradients_df = df

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