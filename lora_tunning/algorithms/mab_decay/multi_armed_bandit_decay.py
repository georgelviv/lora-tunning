import random
from typing import Dict
from ...models import Action
from ..mab import MultiArmedBandit

class MultiArmedBanditDecay(MultiArmedBandit):
  @property
  def name(self) -> str:
    return "mab_decay"
  
  @property
  def configs(self) -> Dict[str, float]:
    return {
      'epsilon': self.epsilon,
      'decay': self.decay
    }
  
  def __init__(self, epsilon=0.9, decay=0.995):
    super().__init__(epsilon=epsilon)
    self.decay = decay
    self.current_epsilon = epsilon

  def update(self, action: Action, reward):
    super().update(action, reward)
    self.current_epsilon = max(0.01, self.current_epsilon * self.decay)

  def choose_action(self) -> Action:
    if self.q_df.empty:
      return self.random_action()
    
    if random.random() > self.current_epsilon:
      best_row = self.q_df.sort_values(by='reward', ascending=False).iloc[0]
      return best_row.drop(["count", "reward"]).to_dict()
    
    return self.random_action()