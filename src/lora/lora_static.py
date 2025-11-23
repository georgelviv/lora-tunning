import logging
from .lora_base import LoraBase
from ..models import Action, State

class LoraStatic(LoraBase):
  def __init__(self, logger: logging.Logger):
    self.logger = logger
    self.config = {
        "SF": 8.0,
        "FQ": 878,
        "BW": 125.0,
        "CR": 8.0,
        "TP": 10.0,
        "IH": 0.0,
        "HS": 10.0,
        "PL": 10.0,
        "CL": 45.0,
        "RT": 1.0
    }

  async def start(self):
    pass

  async def stop(self):
    pass

  async def config_get(self) -> Action:
    return self.config

  async def ping(self, id: int) -> State:
    return {
      'bytes_per_second': 611.0,
      'chunks_count': 1.0,
      'delay': 151.0,
      'rssi': -32.0,
      'snr': 7.25,
      'time_over_air': 36.0,
      'attempt': 2
    }

  async def config_sync(self, id: int, params: Action) -> None:
    self.config = params