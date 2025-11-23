import asyncio
from dotenv import load_dotenv
import os
from src import LoraTunning, LoraHardware, getLogger, LoraStatic

load_dotenv(override=True)

PORT_FILTER = os.getenv('PORT_FILTER')

async def main():
  logger = getLogger()
  # backend = LoraHardware(logger, PORT_FILTER)
  backend = LoraStatic(logger)
  loraTunning = LoraTunning(logger, backend)
  await loraTunning.gradient_bandits()
  try:
    while True:
      await asyncio.sleep(1)
  except KeyboardInterrupt:
    loraTunning.lora.stop()

if __name__ == "__main__":
  asyncio.run(main())
