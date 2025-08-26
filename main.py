import serial.tools.list_ports
from dotenv import load_dotenv
import os
import logging

load_dotenv(override=True)

PORT_FILTER = os.getenv('PORT_FILTER')

class LoraTunning:
  def __init__(self) -> None:
    self.logger: logging.Logger = self.getLogger()
    serial_port = self.find_serial_port(PORT_FILTER)
    self.read_serial(serial_port)

  def getLogger(self) -> logging.Logger:
    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s [%(levelname)s] %(message)s'
    )

    return logging.getLogger(__name__)
  
  def find_serial_port(self, filter: str) -> str:
    ports = serial.tools.list_ports.comports()
    for port in ports:
      if port.device.startswith(filter):
        return port.device
    return None
  
  def read_serial(self, port: str):
    ser = serial.Serial(port=port, baudrate=9600, timeout=1)

    self.logger.info(f"Connected to {port}")

    try:
      while True:
        if ser.in_waiting > 0:
          line = ser.readline().decode('utf-8', errors='ignore').strip()
          self.logger.info("Received:", line)
    except KeyboardInterrupt:
      self.logger.info("Done")
    finally:
      ser.close()


def main() -> None:
  LoraTunning()

if __name__ == "__main__":
    main()


  