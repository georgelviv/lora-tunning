import logging
from ..models import Action, State
from .lora_base import LoraBase

class Lora:
  def __init__(self, logger: logging.Logger, backend: LoraBase) -> None:
    self.logger: logging.Logger = logger

    self.backend = backend
    
  async def start(self) -> None:
    await self.backend.start()

  def stop(self):
    self.backend.stop()

  async def config_get(self) -> Action:
    return await self.backend.config_get()
  
  async def ping(self, id: int) -> State:
    return await self.backend.ping(id)
  
  async def config_sync(self, id: int, params) -> None:
    return await self.backend.config_sync(id, params)
      