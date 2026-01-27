import logging
from .models import Action, Args, State, LoraBase, Algorithm
from .reward import estimate_reward
import os
import logging
import time

class LoraTunning:
  def __init__(self, logger: logging.Logger, backend: LoraBase, algorithm: Algorithm, args: Args) -> None:
    self.logger: logging.Logger = logger
    self.base_dir = os.path.dirname(os.path.abspath(__file__))
    self.lora: LoraBase = backend
    self.algorithm = algorithm
    self.has_delays = args['has_delays']
    self.iterations = args["iterations"]

  async def run(self):
    self.logger.info(f"Starting {self.algorithm.__class__.__name__} ")
    await self.lora.start()

    for i in range(1, self.iterations + 1):
      self.logger.info(f"------ Iteration {i} / {self.iterations}  ----- ")
  
      action: Action = self.algorithm.choose_action()
      await self.config_sync_with_retries(action)
    
      action: Action = await self.lora.config_get()
      state: State = await self.lora.ping(id=1)
      reward = estimate_reward(state)
    
      self.logger.info(f'Reward {reward}')
      self.algorithm.update(action, reward)
      self.algorithm.save()
      

      if self.algorithm.get_iteration() >= self.iterations:
        self.logger.info("Running done")
        break

        
  async def config_sync_with_retries(self, action: Action):
    attempts = 10
    for attempt in range(1, attempts + 1):
      config_is_updated: bool = await self.lora.config_sync(1, action)
      if self.has_delays:
        time.sleep(2)

      if config_is_updated:
        break
      else:
        if self.has_delays:
          time.sleep(attempt ** 2)