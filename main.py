import asyncio
from dotenv import load_dotenv
import os
from src import LoraTunning

load_dotenv(override=True)

PORT_FILTER = os.getenv('PORT_FILTER')

async def main():
  loraTunning = LoraTunning(PORT_FILTER)
  result = await loraTunning.sf_test()
  print(result)
  try:
    while True:
      await asyncio.sleep(1)  # щоб не жрав CPU
  except KeyboardInterrupt:
    loraTunning.lora.stop_listener()

if __name__ == "__main__":
  asyncio.run(main())

  