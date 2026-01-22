import numpy as np
import pandas as pd
import os
from ..models import Action
from ..utils import current_limit_for_tp
import random

class GradientBandit:

  def __init__(self, results_file, history_file, alpha=0.1):
    self.alpha = alpha
    self.results_file = results_file
    self.history_file = history_file

    for f in [self.results_file, self.history_file]:
      if os.path.exists(f):
        os.remove(f)

    self.history_df = pd.DataFrame(columns=["iteration", "reward", "timestamp"])
    self.q_df = pd.DataFrame(columns=[
      "SF", "FQ", "BW", "CR", "TP", "IH", "HS", "PL", "CL", "RT",
      "preference", "probability"
    ])

    self.avg_reward = 0
    self.load()

  def choose_action(self) -> Action:
    if self.q_df.empty:
      return self.random_action()

    # Softmax по preference
    prefs = np.array(self.q_df["preference"])
    exp_prefs = np.exp(prefs - np.max(prefs))  # стабілізація
    probs = exp_prefs / np.sum(exp_prefs)
    self.q_df["probability"] = probs

    # Вибираємо дію за розподілом
    idx = np.random.choice(len(self.q_df), p=probs)
    action_row = self.q_df.iloc[idx]
    return action_row.drop(["preference", "probability"]).to_dict()

  def update(self, action: Action, reward: float):
    if action == 0:
      return

    # Додаємо дію, якщо її ще не було
    mask = (self.q_df[list(action)] == pd.Series(action)).all(axis=1)
    if not mask.any():
      action_with_pref = {**action, "preference": 0.0, "probability": 0.0}
      self.q_df = pd.concat([self.q_df, pd.DataFrame([action_with_pref])], ignore_index=True)
      return

    # Оновлюємо переваги
    idx = self.q_df[mask].index[0]
    prefs = np.array(self.q_df["preference"])
    exp_prefs = np.exp(prefs - np.max(prefs))
    probs = exp_prefs / np.sum(exp_prefs)
    chosen_prob = probs[idx]

    # Оновлюємо середню винагороду
    n = len(self.history_df) + 1
    self.avg_reward += (reward - self.avg_reward) / n

    # Оновлення переваг
    for i in range(len(self.q_df)):
      if i == idx:
        self.q_df.at[i, "preference"] += self.alpha * (reward - self.avg_reward) * (1 - probs[i])
      else:
        self.q_df.at[i, "preference"] -= self.alpha * (reward - self.avg_reward) * probs[i]

    # Історія
    new_row = {
      "iteration": n,
      "reward": float(reward),
      "timestamp": pd.Timestamp.now()
    }
    self.history_df = pd.concat([self.history_df, pd.DataFrame([new_row])], ignore_index=True)

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

  def save(self):
    if not self.q_df.empty:
      self.q_df.to_csv(self.results_file, index=False, float_format="%.4f")
    if not self.history_df.empty:
      self.history_df.to_csv(self.history_file, index=False,
                             float_format="%.4f", date_format="%Y-%m-%d %H:%M:%S")

  def load(self):
    try:
      self.q_df = pd.read_csv(self.results_file)
    except FileNotFoundError:
      pass
    try:
      self.history_df = pd.read_csv(self.history_file)
    except FileNotFoundError:
      pass
