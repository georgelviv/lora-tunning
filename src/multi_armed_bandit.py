from .models import Action
from collections import defaultdict
import random
import json
import pandas as pd
from .utils import current_limit_for_tp

class MultiArmedBandit:
  def __init__(self, epsilon=0.3):
    self.epsilon = epsilon
    self.counts = defaultdict(int)
    self.values = defaultdict(float)

  def choose_action(self) -> Action:
    if random.random() < self.epsilon or not self.values:
      return self.random_action()
    
    best_action = max(self.values, key=lambda k: self.values[k])
    return json.loads(best_action)
  
  def update(self, action: Action, reward):
    key = self.get_action_key(action)
    self.counts[key] += 1
    n = self.counts[key]
    value = self.values[key]
    self.values[key] = value + (reward - value) / n

  def get_action_key(self, action: Action) -> str:
    return json.dumps(action, sort_keys=True)
  
  def save_results(self, file_path: str) -> None:
    rows = []
    for action_json, reward in self.values.items():
      action = json.loads(action_json)
      action["reward"] = reward
      rows.append(action)

    if rows:
      df = pd.DataFrame(rows)
      df = df.sort_values(by='reward', ascending=False)
      df.to_csv(file_path, index=False)

  def random_action(self) -> Action:
    sf = random.choice(range(6, 13))
    ih = 0

    if sf == 6:
      ih = 1


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