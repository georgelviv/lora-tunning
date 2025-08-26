import serial.tools.list_ports
import logging
import threading

class LoraTunning:
  def __init__(self, port_filter) -> None:
    self.logger: logging.Logger = self.getLogger()

    self.ser = None
    self.thread = None
    self.running = False

    self.serial_port = self.find_serial_port(port_filter)

    if self.serial_port:
      self.start_listener()
    else:
      self.logger.error("Serial port not found")

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
  
  def start_listener(self):
    self.ser = serial.Serial(port=self.serial_port, baudrate=9600, timeout=1)
    self.running = True
    self.thread = threading.Thread(target=self.read_serial, daemon=True)
    self.thread.start()
    self.logger.info(f"Listening thread started on {self.serial_port}")
  
  def read_serial(self):
    self.logger.info(f"Connected to {self.serial_port}")

    try:
      while True:
        if self.ser.in_waiting > 0:
          line = self.ser.readline().decode('utf-8', errors='ignore').strip()
          if line:
            self.logger.info("Received:", line)
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