from ...models import Action
from ..mab import MultiArmedBandit

class MultiArmedBanditDecay(MultiArmedBandit):
  @property
  def name(self) -> str:
    return "mab_decay"

  def update(self, action: Action, reward):
    super().update(action, reward)
    self.epsilon = max(0.01, self.epsilon * 0.995)