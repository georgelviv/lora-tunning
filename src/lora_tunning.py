import logging
from .lora import Lora
from .models import Action, State
from .utils import estimate_reward
from .mab import MultiArmedBandit
from .mab_decay import MultiArmedBanditDecay
from .mab_reward_exponential import MultiArmedBanditRewardExponential
from .qlearning import QLearning
from .ucb import UCB

class LoraTunning:
  def __init__(self, port_filter) -> None:
    self.logger: logging.Logger = self.getLogger()
    self.lora: Lora = Lora(self.logger, port_filter)

  def getLogger(self) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
  
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler('app.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
  
  async def mab(self, bandit: MultiArmedBandit):
    while True:
      action: Action = bandit.choose_action()
      configs = list(action.items())
      await self.lora.config_sync(1, configs)
      action: Action = await self.lora.config_get()
      state: State = await self.lora.ping(id=1)
      reward = estimate_reward(state, action)
      bandit.update(action, reward)
      bandit.save()

      if bandit.get_iteration() > 1000:
        self.logger.info("Done!")
        break
  
  async def multi_armed_bandit(self):
    bandit = MultiArmedBandit('results.csv', 'history.csv')
    await self.mab(bandit)

  async def multi_armed_bandit_decay(self):
    bandit = MultiArmedBanditDecay('results.csv', 'history.csv')
    await self.mab(bandit)

  async def multi_armed_bandit_reward_exponential(self):
    bandit = MultiArmedBanditRewardExponential('results.csv', 'history.csv')
    await self.mab(bandit)   

  async def ucb(self):
    bandit = UCB(history_file='history.csv', ubf_file='ucb.csv', results_file='results.csv')

    await self.mab(bandit)

  async def q_learning(self):
    q_agent = QLearning('results.csv', 'history.csv')

    state: State = await self.lora.ping(id=1)

    while True:
      action: Action = q_agent.choose_action(state)
      configs = list(action.items())
      await self.lora.config_sync(1, configs)

      next_state: State = await self.lora.ping(id=1)
      reward = estimate_reward(next_state, action)

      q_agent.update(state, action, reward, next_state)
      q_agent.save()

      state = next_state