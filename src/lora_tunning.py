import logging
from .lora import Lora, LoraBase
from .models import Action, State
from .utils import estimate_reward
from .mab import MultiArmedBandit
from .mab_decay import MultiArmedBanditDecay
from .mab_reward_exponential import MultiArmedBanditRewardExponential
from .qlearning import QLearning
from .ucb import UCB
from .gradient_bandits import GradientBandit
import os
import logging

class LoraTunning:
  def __init__(self, logger: logging.Logger, backend: LoraBase) -> None:
    self.logger: logging.Logger = logger
    self.base_dir = os.path.dirname(os.path.abspath(__file__))
    self.lora: Lora = Lora(self.logger, backend)
  
  async def mab(self, bandit: MultiArmedBandit):
    self.logger.info(f"Starting {bandit.__class__.__name__}")
    await self.lora.start()

    while True:
      action: Action = bandit.choose_action()
      await self.lora.config_sync(1, action)
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

  async def gradient_bandits(self):
    results_file_path = os.path.join(self.base_dir, 'gradient_bandits/results.csv')
    history_file_path = os.path.join(self.base_dir, 'gradient_bandits/history.csv')
    gradients_file_path = os.path.join(self.base_dir, 'gradient_bandits/gradients.csv')
    bandit = GradientBandit(results_file_path, history_file_path, gradients_file_path, alpha=0.01)
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