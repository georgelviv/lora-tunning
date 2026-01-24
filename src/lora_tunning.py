import logging
from .models import Action, State, LoraBase, Algorithm
from .utils import estimate_reward
import os
import logging

class LoraTunning:
  def __init__(self, logger: logging.Logger, backend: LoraBase, algorithm: Algorithm) -> None:
    self.logger: logging.Logger = logger
    self.base_dir = os.path.dirname(os.path.abspath(__file__))
    self.lora: LoraBase = backend
    self.algorithm = algorithm

  async def run(self):
    self.logger.info(f"Starting {self.algorithm.__class__.__name__}")
    await self.lora.start()

    while True:
      action: Action = self.algorithm.choose_action()
      await self.lora.config_sync(1, action)
      action: Action = await self.lora.config_get()
      state: State = await self.lora.ping(id=1)
      reward = estimate_reward(state, action)
      self.algorithm.update(action, reward)
      self.algorithm.save()

      if self.algorithm.get_iteration() > 1000:
        self.logger.info("Done!")
        break

  # async def gradient_bandits(self):
  #   results_file_path = os.path.join(self.base_dir, 'gradient_bandits/results.csv')
  #   history_file_path = os.path.join(self.base_dir, 'gradient_bandits/history.csv')
  #   gradients_file_path = os.path.join(self.base_dir, 'gradient_bandits/gradients.csv')
  #   bandit = GradientBandit(results_file_path, history_file_path, gradients_file_path, alpha=0.01)
  #   await self.mab(bandit)  


  # async def q_learning(self):
  #   q_agent = QLearning('results.csv', 'history.csv')

  #   state: State = await self.lora.ping(id=1)

  #   while True:
  #     action: Action = q_agent.choose_action(state)
  #     configs = list(action.items())
  #     await self.lora.config_sync(1, configs)

  #     next_state: State = await self.lora.ping(id=1)
  #     reward = estimate_reward(next_state, action)

  #     q_agent.update(state, action, reward, next_state)
  #     q_agent.save()

  #     state = next_state