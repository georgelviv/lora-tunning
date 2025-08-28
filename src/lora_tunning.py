import logging
from .lora import Lora

class LoraTunning:
  def __init__(self, port_filter) -> None:
    self.logger: logging.Logger = self.getLogger()
    self.lora: Lora = Lora(self.logger, port_filter)

  def getLogger(self) -> logging.Logger:
    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s [%(levelname)s] %(message)s'
    )

    return logging.getLogger(__name__)

  async def get_config(self):
    return await self.lora.config_get()
  
  async def ping(self):
    return await self.lora.ping(id=1)