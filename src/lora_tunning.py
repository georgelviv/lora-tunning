import logging
from .lora import Lora
from .models import Action, State
from .utils import estimate_tx_energy

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
  
  async def sf_test(self):
    available_sf = list(range(7, 13))
    for sf in available_sf:
      await self.config_sync([('SF', sf)])
      action: Action = await self.get_config()
      state: State = await self.ping()
      energy = estimate_tx_energy(action['transmission_power'], state['timeOverAir'], action['current_limit'])
      self.logger.info(f"Energy for sf={sf} is {energy}")

  async def get_config(self) -> Action:
    return await self.lora.config_get()
  
  async def ping(self) -> State:
    return await self.lora.ping(id=1)
  
  async def config_sync(self, params):
    return await self.lora.config_sync(1, params)