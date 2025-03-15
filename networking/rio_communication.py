import os
import json
import logging
from typing import Optional, Dict, Any

from networktables import NetworkTables

##############################################################################

ROBOT_IP_ADDRESS: str = "10.37.56.2"
NETWORK_TABLE_NAME: str = "AIPipeline"
DATA_ENTRY_NAME: str = "data"

script_name: str = os.path.splitext(os.path.basename(__file__))[0]
log_file: str = os.path.join("logs", f'{script_name}.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

##############################################################################

class RoboRio:
    def __init__(self) -> None:
        NetworkTables.initialize(server=ROBOT_IP_ADDRESS)
        self.table = NetworkTables.getTable(NETWORK_TABLE_NAME)
        logging.info(f'NetworkTables initialized with server: {ROBOT_IP_ADDRESS}')

    def send_data(self, data: Dict[str, Any]) -> None:
        """Posts JSON data to NetworkTables."""
        if not self.network_tables_connection():
            logging.warning("NetworkTables is not connected. Data not sent.")
            return
        
        try:
            json_data = json.dumps(data)
            self.table.putString(DATA_ENTRY_NAME, json_data)
            logging.info(f'Data successfully posted to NetworkTables: {json_data}')
        except (TypeError, ValueError) as e:
            logging.error(f'Failed to serialize data to JSON: {e}')

    def get_data(self, data) -> Optional[str]:
        """Pulls JSON data from NetworkTables and returns it as a dictionary."""
        if not self.network_tables_connection():
            logging.warning("NetworkTables is not connected. Returning None.")
            return None
        
        json_data = self.table.getString(data, "{}")
        try:
            return json.loads(json_data)
        except json.JSONDecodeError as e:
            logging.error(f'Error decoding JSON from NetworkTables: {e}')
            return None

    @staticmethod
    def network_tables_connection() -> bool:
        """Checks if NetworkTables is connected to the robot."""
        connected = NetworkTables.isConnected()
        logging.info(f'NetworkTables connection status: {connected}')
        return connected
