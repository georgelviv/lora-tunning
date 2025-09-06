import logging
from .lora import Lora
from .models import Action, State
from .utils import estimate_reward
from .multi_armed_bandit import MultiArmedBandit
from .constants import actions

class LoraTunning:
  def __init__(self, port_filter) -> None:
    self.logger: logging.Logger = self.getLogger()
    self.lora: Lora = Lora(self.logger, port_filter)

  def getLogger(self) -> logging.Logger:
    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s [%(levelname)s] %(message)s'
    )

    return logging.getLogger(__name__)
  
  async def multi_armed_bandit(self):
    bandit = MultiArmedBandit(actions, epsilon=0.2)
    total = 100

    for t in range(total):
      self.logger.info(f'Running {t} of {total}')
      action: Action = bandit.choose_action()
      configs = list(action.items())
      await self.lora.config_sync(1, configs)
      action: Action = await self.lora.config_get()
      state: State = await self.lora.ping(id=1)
      reward = estimate_reward(state, action)
      bandit.update(action, reward)

    self.logger.info("Rewards:")
    for a, v in bandit.values.items():
      self.logger.info(f'{a},{v}')