import os
import requests
from typing import Literal
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API", "")


# Weather Tool 
@tool(name_or_callable = "weather_tool", description = "Get current weather update for a given city.")
def get_weather(city: str) -> str:
    """Get a detailed current weather update for a given city including temp, humidity, and wind.

    Args:
        city: Name of the city (e.g., "Dhaka", "New York")

    Returns:
        Comprehensive weather summary string.
    """
    api_key = os.getenv("OPEN_WEATHER_API")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url, timeout = 10).json()
        if str(response.get("cod")) != "200":
            error_msg = response.get("message", "Unknown error").capitalize()
            return f"Could not find weather data for '{city}'. Reason: {error_msg}."

        weather_desc  = response["weather"][0]["description"].capitalize()
        temp = response["main"]["temp"]
        feels_like = response["main"]["feels_like"]
        humidity = response["main"]["humidity"]
        pressure = response["main"]["pressure"]
        wind_speed = response["wind"]["speed"]
        visibility_km = response.get("visibility", 0) / 1000
        country = response["sys"]["country"]
        city_name = response["name"]
        return (
            f"Current Weather Report for {city_name}, {country}:\n"
            f"- Condition: {weather_desc}\n"
            f"- Temperature: {temp}°C (Feels like: {feels_like}°C)\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind Speed: {wind_speed} m/s\n"
            f"- Barometric Pressure: {pressure} hPa\n"
            f"- Visibility: {visibility_km:.1f} km"
        )
    except Exception as e:
        return f"An error occurred while fetching the weather update: {str(e)}"


# Web Search Tool
class StrictTavilySchema(BaseModel):
    query: str = Field(description="The search query string")
    topic: Literal["general", "news", "finance"] = Field(
        default="general",
        description="Category of search. Use 'news' for sports/current events.",
    )

search_tool = TavilySearch(max_results=3)
search_tool.args_schema = StrictTavilySchema

#Exported tool list
tools = [search_tool, get_weather]