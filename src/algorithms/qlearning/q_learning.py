import os
import pandas as pd
import random
from ..models import Action, State
from ..utils import current_limit_for_tp

class QLearning:
  def __init__(self, results_file: str, history_file: str, alpha=0.1, gamma=0.95, epsilon=0.9):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    self.results_file = os.path.join(base_dir, results_file)
    self.history_file = os.path.join(base_dir, history_file)

    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon

    self.history_df = pd.DataFrame(columns=[
      "iteration", "reward", "timestamp", "epsilon"
    ]).astype({
      "iteration": "int64",
      "reward": "float64",
      "timestamp": "datetime64[ns]",
      "epsilon": "float64"
    })

    self.q_df = pd.DataFrame(
      columns=["state", "SF", "FQ", "BW", "CR", "TP", "IH", "HS", "PL", "CL", "RT", "value"]
    ).astype({
      "value": "float64"
    })

    self.load()

  def load(self):
    try:
      self.q_df = pd.read_csv(self.results_file)
    except FileNotFoundError:
      pass
    try:
      self.history_df = pd.read_csv(self.history_file)
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
  
  def save(self):
    if not self.q_df.empty:
      self.q_df.to_csv(self.results_file, index=False)
    if not self.history_df.empty:
      self.history_df.to_csv(self.history_file, index=False)


  def update(self, state: State, action: Action, reward, next_state):
    mask = (self.q_df["state"] == state)
    for k, v in action.items():
      mask &= (self.q_df[k] == v)

    old_value = self.q_df.loc[mask, "value"].iloc[0] if mask.any() else 0.0
    next_actions = self.q_df[self.q_df["state"] == next_state]
    next_max = next_actions["value"].max() if not next_actions.empty else 0.0

    new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

    if mask.any():
      self.q_df.loc[mask, "value"] = new_value
    else:
      row = {"state": state, **action, "value": new_value}
      self.q_df = pd.concat([self.q_df, pd.DataFrame([row])], ignore_index=True)

    self.history_df.loc[len(self.history_df)] = {
      "iteration": len(self.history_df) + 1,
      "state": state,
      "reward": reward,
      "epsilon": self.epsilon
    }

    self.epsilon = max(0.01, self.epsilon * 0.995)

  def choose_action(self, state: State) -> Action:
    if random.random() < self.epsilon:
        return self.random_action()

    state_rows = self.q_df[self.q_df["state"] == state]
    if state_rows.empty:
      return self.random_action()

    best_row = state_rows.loc[state_rows["value"].idxmax()]
    action = {
      "SF": int(best_row["SF"]),
      "FQ": int(best_row["FQ"]),
      "BW": int(best_row["BW"]),
      "CR": int(best_row["CR"]),
      "TP": int(best_row["TP"]),
      "IH": int(best_row["IH"]),
      "HS": int(best_row["HS"]),
      "PL": int(best_row["PL"]),
      "CL": int(best_row["CL"]),
      "RT": int(best_row["RT"]),
    }
    return action