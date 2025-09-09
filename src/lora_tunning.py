import logging
from .lora import Lora
from .models import Action, State
from .utils import estimate_reward
from .multi_armed_bandit import MultiArmedBandit
import sys

class LoraTunning:
  def __init__(self, port_filter) -> None:
    self.logger: logging.Logger = self.getLogger()
    self.lora: Lora = Lora(self.logger, port_filter)

  def getLogger(self) -> logging.Logger:
    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s [%(levelname)s] %(message)s',
      datefmt='%H:%M:%S'
    )

    return logging.getLogger(__name__)
  
  async def multi_armed_bandit(self):
    bandit = MultiArmedBandit()
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

    bandit.save_results('results.csv')
    self.logger.info("Results saved in results.csv")
    sys.exit(0)