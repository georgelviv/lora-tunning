import asyncio
from dotenv import load_dotenv
import os
from src import (
  LoraTunning, LoraHardware, getLogger, LoraStatic, 
  LoraSimulation, LORA_SIMULATION_ENVIRONMENTS
)

load_dotenv(override=True)

PORT_FILTER = os.getenv('PORT_FILTER')

async def main():
  logger = getLogger()
  # backend = LoraHardware(logger, PORT_FILTER)
  backend = LoraSimulation(logger, LORA_SIMULATION_ENVIRONMENTS['open_field'])
  loraTunning = LoraTunning(logger, backend)
  await loraTunning.gradient_bandits()

  logger.info("2222")

if __name__ == "__main__":
  asyncio.run(main())
