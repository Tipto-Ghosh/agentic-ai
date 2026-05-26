import os 
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP(name = "Weather_Server")


@mcp.tool(name_or_fn = "get_weather")
async def get_weather(city: str):
    """Get a detailed current weather update for a given city including temp, humidity, and wind.

    Args:
        city (str): Name of the city (e.g., "Dhaka", "New York")

    Returns:
        str: Comprehensive weather summary string.
    """
    api_key = os.getenv("OPEN_WEATHER_API")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10).json()
        if str(response.get("cod")) != "200":
            error_msg = response.get("message", "Unknown error").capitalize()
            return f"Could not find weather data for '{city}'. Reason: {error_msg}."
        
        weather_desc = response["weather"][0]["description"].capitalize()
        temp = response["main"]["temp"]
        feels_like = response["main"]["feels_like"]
        humidity = response["main"]["humidity"]
        pressure = response["main"]["pressure"]
        wind_speed = response["wind"]["speed"]  
        visibility_km = response.get("visibility", 0) / 1000 # Convert meters to km
        country = response["sys"]["country"]
        city_name = response["name"]
        
        detailed_report = (
            f"Current Weather Report for {city_name}, {country}:\n"
            f"- Condition: {weather_desc}\n"
            f"- Temperature: {temp}°C (Feels like: {feels_like}°C)\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind Speed: {wind_speed} m/s\n"
            f"- Barometric Pressure: {pressure} hPa\n"
            f"- Visibility: {visibility_km:.1f} km"
        )
        return detailed_report
    except Exception as e:
        return f"An error occurred while fetching the weather update: {str(e)}"

if __name__ == "__main__":
    mcp.run(
        transport = "streamable-http",
        host = "127.0.0.1",
        port = 8000
    )