import requests
import sys
import logging
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging to stderr (stdout is reserved for MCP protocol)
logging.basicConfig(
    level=logging.DEBUG,
    format='[MCP] %(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)

mcp = FastMCP("OpenF1")

BASE_URL = "https://api.openf1.org/v1"
DEFAULT_SESSION_KEY = 9159  # Default to this session

def _get(url: str, params: Optional[Dict[str,Any]]=None, headers: Optional[Dict[str,Any]]=None) -> Any:
    """Internal helper: make a GET request, parse JSON, raise on error."""
    logger.info(f"Making request to {url} with params: {params}")
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    logger.debug(f"API Response: {len(data) if isinstance(data, list) else 'single'} records")
    return data

@mcp.tool()
def get_car_data(driver_number: int, min_speed: Optional[int]=None, session_key: Optional[int]=None) -> Dict:
    """
    Retrieve telemetry data for a specific car/driver in a session.
    Calls OpenF1 /car_data endpoint.
    ---
    Parameters:
      driver_number: the driver's number (int) - e.g., 44 for Hamilton, 55 for Sainz
      min_speed: optional filter to include only records where speed >= this value (in km/h)
      session_key: optional session key (defaults to 9159 if not provided)
    Returns a nicely translated dict summary.
    """
    if session_key is None:
        session_key = DEFAULT_SESSION_KEY
    
    logger.info(f"get_car_data called: driver_number={driver_number}, session_key={session_key}, min_speed={min_speed}")
    
    params = {
        "driver_number": driver_number,
        "session_key": session_key,
    }
    if min_speed is not None:
        params["speed>="] = min_speed
    
    url = f"{BASE_URL}/car_data"
    raw = _get(url, params=params)
    
    if not raw:
        logger.warning(f"No car data found for driver {driver_number} in session {session_key}")
        return {"message": f"No car data found for driver {driver_number} in session {session_key}."}
    
    latest = raw[-1]
    return {
        "driver_number": latest.get("driver_number"),
        "session_key": latest.get("session_key"),
        "speed": latest.get("speed"),
        "gear": latest.get("n_gear"),
        "rpm": latest.get("rpm"),
        "throttle_pct": latest.get("throttle"),
        "brake_pct": latest.get("brake"),
        "time": latest.get("date"),
        "records_found": len(raw),
    }

@mcp.tool()
def get_driver_info(driver_number: int, session_key: Optional[int]=None) -> Dict:
    """
    Retrieve driver information for a given session.
    Calls OpenF1 /drivers endpoint.
    ---
    Parameters:
      driver_number: the driver's number (int) - e.g., 44 for Hamilton, 55 for Sainz
      session_key: optional session key (defaults to 9159 if not provided)
    Returns a friendly summary of that driver in that session.
    """
    if session_key is None:
        session_key = DEFAULT_SESSION_KEY
        
    logger.info(f"get_driver_info called: driver_number={driver_number}, session_key={session_key}")
    
    url = f"{BASE_URL}/drivers"
    params = {"driver_number": driver_number, "session_key": session_key}
    
    try:
        raw = _get(url, params=params)
        
        if isinstance(raw, list) and len(raw) > 0:
            info = raw[0]
            result = {
                "driver_number": info.get("driver_number"),
                "name": info.get("full_name", info.get("name_acronym")),
                "team": info.get("team_name"),
                "nationality": info.get("country_code"),
                "session_key": info.get("session_key"),
            }
            return {k: v for k, v in result.items() if v is not None}
        
        logger.warning(f"No driver info found for driver {driver_number} in session {session_key}")
        return {"message": f"No driver info found for driver {driver_number} in session {session_key}."}
    except Exception as e:
        logger.error(f"Error fetching driver info: {str(e)}", exc_info=True)
        return {"message": f"Error fetching driver info: {str(e)}"}

@mcp.tool()
def get_lap_data(driver_number: int, lap_number: Optional[int]=None, session_key: Optional[int]=None) -> Dict:
    """
    Retrieve lap data for a driver in a session (optionally specifying lap number).
    Calls OpenF1 /laps endpoint.
    ---
    Parameters:
      driver_number: the driver's number (int)
      lap_number: optional specific lap number to query
      session_key: optional session key (defaults to 9159 if not provided)
    Returns a summary of lap(s).
    """
    if session_key is None:
        session_key = DEFAULT_SESSION_KEY
        
    logger.info(f"get_lap_data called: driver_number={driver_number}, session_key={session_key}, lap_number={lap_number}")
    
    url = f"{BASE_URL}/laps"
    params = {"driver_number": driver_number, "session_key": session_key}
    if lap_number is not None:
        params["lap_number"] = lap_number
    raw = _get(url, params=params)
    if not raw:
        return {"message": "No lap data found for given parameters."}
    record = raw[0] if lap_number is not None else raw[-1]
    return {
        "driver_number": record.get("driver_number"),
        "lap_number": record.get("lap_number"),
        "lap_time": record.get("lap_duration"),
        "session_key": record.get("session_key"),
    }

@mcp.tool()
def get_session_info(meeting_key: int, session_key: int) -> Dict:
    """
    Retrieve information about a particular session (practice/qualifying/race).
    Calls OpenF1 /sessions endpoint.
    ---
    Parameters:
      meeting_key: int
      session_key: int
    Returns human-friendly session metadata.
    """
    logger.info(f"get_session_info called: meeting_key={meeting_key}, session_key={session_key}")
    
    url = f"{BASE_URL}/sessions"
    params = {"meeting_key": meeting_key, "session_key": session_key}
    raw = _get(url, params=params)
    if not raw:
        return {"message": "No session info found for given parameters."}
    info = raw[0]
    return {
        "meeting_key": info.get("meeting_key"),
        "session_key": info.get("session_key"),
        "type": info.get("session_type"),
        "start_time": info.get("date_start"),
        "end_time": info.get("date_end"),
        "track": info.get("circuit_short_name"),
        "location": info.get("location"),
    }

@mcp.tool()
def get_weather_data(meeting_key: Optional[int]=None, session_key: Optional[int]=None) -> Dict:
    """
    Retrieve weather conditions for a given session.
    Calls OpenF1 /weather endpoint.
    ---
    Parameters:
      meeting_key: optional meeting key (uses default session if not provided)
      session_key: optional session key (defaults to 9159 if not provided)
    Returns a summary of track weather.
    """
    if session_key is None:
        session_key = DEFAULT_SESSION_KEY
    if meeting_key is None:
        meeting_key = 1219  # Default meeting for session 9159
        
    logger.info(f"get_weather_data called: meeting_key={meeting_key}, session_key={session_key}")
    
    url = f"{BASE_URL}/weather"
    params = {"meeting_key": meeting_key, "session_key": session_key}
    raw = _get(url, params=params)
    if not raw:
        return {"message": "No weather data found for given parameters."}
    latest = raw[-1]
    return {
        "meeting_key": latest.get("meeting_key"),
        "session_key": latest.get("session_key"),
        "track_temp": latest.get("track_temperature"),
        "air_temp": latest.get("air_temperature"),
        "wind_speed": latest.get("wind_speed"),
        "time": latest.get("date"),
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")