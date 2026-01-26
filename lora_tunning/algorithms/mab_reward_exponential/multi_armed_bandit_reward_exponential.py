from typing import Dict
from ..mab_decay import MultiArmedBanditDecay

class MultiArmedBanditRewardExponential(MultiArmedBanditDecay):
  @property
  def name(self) -> str:
    return "mab_exponential"
  
  @property
  def configs(self) -> Dict[str, float]:
    return {
      'epsilon': self.epsilon,
      'alpha': self.alpha
    }

  def __init__(self, epsilon=0.9, alpha=0.3):
    super().__init__(epsilon=epsilon)
    self.alpha = alpha
  
  def compute_reward(self, old_value: float, new_value: float, *args, **kwargs) -> float:
    # exponential recency-weighted average
    return old_value * (1 - self.alpha) + new_value * self.alpha
