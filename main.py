from dotenv import load_dotenv
import os
from lora_tunning import LoraTunning

load_dotenv(override=True)

PORT_FILTER = os.getenv('PORT_FILTER')

if __name__ == "__main__":
  loraTunning = LoraTunning(PORT_FILTER)
  try:
    while True:
      pass 
  except KeyboardInterrupt:
    loraTunning.stop_listener()


  