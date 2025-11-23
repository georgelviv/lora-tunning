import logging
from .lora_base import LoraBase
from ..models import Action, State
from lora_simulation import LoraSimulation as Simulation

class LoraSimulation(LoraBase):
  def __init__(self, logger: logging.Logger):
    self.logger = logger
    self.simulation = Simulation()

  async def start(self):
    pass

  async def stop(self):
    pass

  async def config_get(self) -> Action:
    return self.simulation.get_config()

  async def ping(self, id: int) -> State:
    return self.simulation.ping()

  async def config_sync(self, id: int, params: Action) -> None:
    return self.simulation.set_config(params)