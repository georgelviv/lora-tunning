import logging
from .models import Action, Args, State, LoraBase, Algorithm
from .reward import estimate_reward
import os
import logging

class LoraTunning:
  def __init__(self, logger: logging.Logger, backend: LoraBase, algorithm: Algorithm, args: Args) -> None:
    self.logger: logging.Logger = logger
    self.base_dir = os.path.dirname(os.path.abspath(__file__))
    self.lora: LoraBase = backend
    self.algorithm = algorithm
    self.iterations = args["iterations"]

  async def run(self):
    self.logger.info(f"Starting {self.algorithm.__class__.__name__} ")
    await self.lora.start()

    while True:
      action: Action = self.algorithm.choose_action()
      await self.lora.config_sync(1, action)
      action: Action = await self.lora.config_get()
      state: State = await self.lora.ping(id=1)
      reward = estimate_reward(state)
      self.algorithm.update(action, reward)
      self.algorithm.save()

      if self.algorithm.get_iteration() >= self.iterations:
        self.logger.info("Done!")
        break
      else:
        self.logger.info(f"------ Iteration {self.algorithm.get_iteration()} / {self.iterations}  ----- ")