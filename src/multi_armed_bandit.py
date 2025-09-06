from .models import Action
from collections import defaultdict
import random
import json
import pandas as pd

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
      df.to_csv(file_path, index=False)

  def random_action(self) -> Action:
    return {
      "SF": random.choice([7,8,9,10,11,12]),
      "FQ": 869,
      "BW": 500,
      "CR": 8,
      "TP": 10,
      "IH": 0,
      "HS": 10,
      "PL": 10,
      "CL": 45,
      "RT": 1
    }