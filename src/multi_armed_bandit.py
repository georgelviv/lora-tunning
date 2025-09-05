from .models import Action
from collections import defaultdict
import random
import json

class MultiArmedBandit:
  def __init__(self, actions: list[Action], epsilon=0.1):
    self.actions = actions
    self.epsilon = epsilon
    self.counts = defaultdict(int)
    self.values = defaultdict(float)

  def choose_action(self) -> Action:
    if random.random() < self.epsilon:
      return random.choice(self.actions)
    
    return max(self.actions, key=lambda a: self.values[self.get_action_key(a)])
  
  def update(self, action: Action, reward):
    key = self.get_action_key(action)
    self.counts[key] += 1
    n = self.counts[key]
    value = self.values[key]
    self.values[key] = value + (reward - value) / n

  def get_action_key(self, action: Action) -> str:
    return json.dumps(action, sort_keys=True)