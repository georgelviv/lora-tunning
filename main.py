import asyncio
from dotenv import load_dotenv
import os
from src import (
  LoraTunning, getLogger, LoraBase, Algorithm, MultiArmedBandit, MultiArmedBanditDecay
)
from lora_hardware_model import LoraHardwareModel

load_dotenv(override=True)

PORT_FILTER = os.getenv('PORT_FILTER')

async def main():
  logger = getLogger()
  # algorithm: Algorithm = MultiArmedBandit()
  algorithm: Algorithm = MultiArmedBanditDecay()
  backend: LoraBase = LoraHardwareModel(logger, PORT_FILTER)
  # backend = LoraStatic(logger)
  # backend = LoraSimulation(logger, LORA_SIMULATION_ENVIRONMENTS['dense_urban'])
  loraTunning = LoraTunning(logger, backend, algorithm)
  await loraTunning.run()

if __name__ == "__main__":
  asyncio.run(main())
