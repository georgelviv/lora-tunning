from ..mab_decay import MultiArmedBanditDecay

class MultiArmedBanditRewardExponential(MultiArmedBanditDecay):
  def __init__(self, results_file, history_file, epsilon=0.9, alpha=0.3):
    super().__init__(results_file, history_file, epsilon=epsilon)
    self.alpha = alpha
  
  def compute_reward(self, old_value: float, new_value: float) -> float:
    # exponential recency-weighted average
    return old_value * (1 - self.alpha) + new_value * self.alpha
