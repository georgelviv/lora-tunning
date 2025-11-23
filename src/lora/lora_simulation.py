import logging
from .lora_base import LoraBase
from ..models import Action, State

class LoraSimulation(LoraBase):
  def __init__(self, logger: logging.Logger):
    self.logger = logger

  @abstractmethod
  async def start(self):
    pass

  @abstractmethod
  async def stop(self):
    pass

  @abstractmethod
  async def config_get(self) -> Action:
    pass

  @abstractmethod
  async def ping(self, id: int) -> State:
    pass

  @abstractmethod
  async def config_sync(self, id: int, params: Action) -> None:
    pass