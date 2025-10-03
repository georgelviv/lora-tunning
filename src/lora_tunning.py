import logging
from .lora import Lora
from .models import Action, State
from .utils import estimate_reward
from .mab import MultiArmedBandit
from .mab_decay import MultiArmedBanditDecay
from .mab_reward_exponential import MultiArmedBanditRewardExponential
from .qlearning import QLearning
from .ucb import UCB
import os

class LoraTunning:
  def __init__(self, port_filter) -> None:
    self.logger: logging.Logger = self.getLogger()
    self.base_dir = os.path.dirname(os.path.abspath(__file__))
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
    results_file_path = os.path.join(self.base_dir, 'mab/results.csv')
    history_file_path = os.path.join(self.base_dir, 'mab/history.csv')
    bandit = MultiArmedBandit(results_file_path, history_file_path, epsilon=0.3)
    await self.mab(bandit)

  async def multi_armed_bandit_decay(self):
    results_file_path = os.path.join(self.base_dir, 'mab_decay/results.csv')
    history_file_path = os.path.join(self.base_dir, 'mab_decay/history.csv')
    bandit = MultiArmedBanditDecay(results_file_path, history_file_path)
    await self.mab(bandit)

  async def multi_armed_bandit_reward_exponential(self):
    results_file_path = os.path.join(self.base_dir, 'mab_reward_exponential/results.csv')
    history_file_path = os.path.join(self.base_dir, 'mab_reward_exponential/history.csv')
    bandit = MultiArmedBanditRewardExponential(results_file_path, history_file_path)
    await self.mab(bandit)   

  async def ucb(self):
    results_file_path = os.path.join(self.base_dir, 'ucb/results.csv')
    history_file_path = os.path.join(self.base_dir, 'ucb/history.csv')
    ucb_file_path = os.path.join(self.base_dir, 'ucb/ucb.csv')
    bandit = UCB(history_file=history_file_path, ubf_file=ucb_file_path, results_file=results_file_path)

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