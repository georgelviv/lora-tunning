import serial.tools.list_ports
import logging
import threading
from .utils import parse_msg

class Lora:
  def __init__(self, logger: logging.Logger, port_filter: str) -> None:
    self.logger: logging.Logger = logger
    self.ser = None
    self.thread = None
    self.running = False

    self.serial_port = self.find_serial_port(port_filter)

    if self.serial_port:
      self.start_listener()
    else:
      self.logger.error("Serial port not found")
  
  def find_serial_port(self, filter: str) -> str:
    ports = serial.tools.list_ports.comports()
    for port in ports:
      if port.device.startswith(filter):
        return port.device
    return None
  
  def start_listener(self):
    self.ser = serial.Serial(port=self.serial_port, baudrate=115200, timeout=1)
    self.running = True
    self.thread = threading.Thread(target=self.read_serial, daemon=True)
    self.thread.start()
    self.logger.info(f"Listening thread started on {self.serial_port}")
  
  def read_serial(self):
    self.logger.info(f"Connected to {self.serial_port}")

    try:
      while True:
        if self.ser.in_waiting > 0:
          line = self.ser.readline().decode("utf-8", errors="ignore").strip()
          if line:
            self.serial_handler(line)
    except Exception as e:
      self.logger.error(f"Error in read_serial: {e}")
    finally:
      self.ser.close()
      self.logger.info("Serial connection closed")

  def stop_listener(self):
    self.running = False
    if self.thread and self.thread.is_alive():
      self.thread.join(timeout=2)
    self.logger.info("Listener stopped")

  def write_serial(self, data: str) -> None: 
    if self.ser and self.ser.is_open:
      try:
        self.ser.write((data + "\r\n").encode("utf-8"))
        self.logger.info(f"Sent: {data}")
      except Exception as e:
        self.logger.error(f"Error writing to serial: {e}")
    else:
      self.logger.warning("Serial port is not open")

  def config_get(self) -> None:
    self.write_serial("CONFIG_GET")

  def serial_handler(self, msg: str):
    command, params = parse_msg(msg)
    if command:
      match command:
        case "CONFIG_GET":
          print("Handle config get", params)
        case _:
          self.logger.warning(f"Unknown command {command}")

      