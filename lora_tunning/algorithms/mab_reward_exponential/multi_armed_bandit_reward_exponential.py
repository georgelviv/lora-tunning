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
      'alpha': self.alpha,
      'decay': self.decay
    }

  def __init__(self, epsilon=0.9, decay=0.995, alpha=0.3):
    super().__init__(epsilon=epsilon, decay=decay)
    self.alpha = alpha
  
  def compute_reward(self, old_value: float, new_value: float, *args, **kwargs) -> float:
    # exponential recency-weighted average
    return old_value * (1 - self.alpha) + new_value * self.alpha
