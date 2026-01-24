from ...models import Action
from ..mab import MultiArmedBandit

class MultiArmedBanditDecay(MultiArmedBandit):
  def update(self, action: Action, reward):
    super().update(action, reward)
    self.epsilon = max(0.01, self.epsilon * 0.995)