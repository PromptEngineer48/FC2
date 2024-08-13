import json
import ollama
import asyncio
import requests
import time


def weather_place(place_name: str) -> str:

    url = "https://weatherapi-com.p.rapidapi.com/forecast.json"

    querystring = {"q": place_name, "days": "3"}

    headers = {
        "x-rapidapi-key": "b6dc2081ccmsh282bec1ffdb6254p10b966jsn0c43c4330ada",
        "x-rapidapi-host": "weatherapi-com.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=querystring)
    return json.dumps(
        f"The weather in {response.json()['location']['name']} is {response.json()['current']}"
    )


# rapidapi.com
def confirmed_cases(country_name: str) -> str:
    url = "https://covid-19-data.p.rapidapi.com/country/code"

    querystring = {"format": "json", "code": country_name[:2].upper()}

    headers = {
        "x-rapidapi-key": "b6dc2081ccmsh282bec1ffdb6254p10b966jsn0c43c4330ada",
        "x-rapidapi-host": "covid-19-data.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=querystring)
    return json.dumps(response.json()[0]["confirmed"])


# for key in ["India", "USA", "Italy", "Australia", "Germany"]:
#     a = confirmed_cases(key)
#     print(a)
#     print(type(a))
#     time.sleep(2)


def get_antonyms(word: str) -> str:
    "Get the antonyms of the any given word"

    words = {
        "hot": "cold",
        "small": "big",
        "weak": "strong",
        "light": "dark",
        "lighten": "darken",
        "dark": "bright",
    }

    return json.dumps(words.get(word, "Not available in database"))


# In a real application, this would fetch data from a live database or API
def get_flight_times(departure: str, arrival: str) -> str:
    flights = {
        "NYC-LAX": {
            "departure": "08:00 AM",
            "arrival": "11:30 AM",
            "duration": "5h 30m",
        },
        "LAX-NYC": {
            "departure": "02:00 PM",
            "arrival": "10:30 PM",
            "duration": "5h 30m",
        },
        "LHR-JFK": {
            "departure": "10:00 AM",
            "arrival": "01:00 PM",
            "duration": "8h 00m",
        },
        "JFK-LHR": {
            "departure": "09:00 PM",
            "arrival": "09:00 AM",
            "duration": "7h 00m",
        },
        "CDG-DXB": {
            "departure": "11:00 AM",
            "arrival": "08:00 PM",
            "duration": "6h 00m",
        },
        "DXB-CDG": {
            "departure": "03:00 AM",
            "arrival": "07:30 AM",
            "duration": "7h 30m",
        },
    }

    key = f"{departure}-{arrival}".upper()
    return json.dumps(flights.get(key, {"error": "Flight not found"}))


async def run(model: str, user_input: str):
    client = ollama.AsyncClient()
    # Initialize conversation with a user query
    messages = [
        {
            "role": "user",
            "content": user_input,
            # "content": "What is the capital of India?",
        }
    ]

    # First API call: Send the query and function description to the model
    response = await client.chat(
        model=model,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_flight_times",
                    "description": "Get the flight times between two cities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "departure": {
                                "type": "string",
                                "description": "The departure city (airport code)",
                            },
                            "arrival": {
                                "type": "string",
                                "description": "The arrival city (airport code)",
                            },
                        },
                        "required": ["departure", "arrival"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_antonyms",
                    "description": "Get the antonyms of any given words",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "word": {
                                "type": "string",
                                "description": "The word for which the opposite is required.",
                            },
                        },
                        "required": ["word"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "weather_place",
                    "description": "Get the weather condition of any particular place",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "place_name": {
                                "type": "string",
                                "description": "The place for which the weather data is required",
                            },
                        },
                        "required": ["country_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "confirmed_cases",
                    "description": "Get the number of confirmed COVID cases for this particular country",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "country_name": {
                                "type": "string",
                                "description": "The country for which the number of confirmed COVID cases is required.",
                            },
                        },
                        "required": ["country_name"],
                    },
                },
            },
        ],
    )

    # print(f"Response: {response}")

    # Add the model's response to the conversation history
    messages.append(response["message"])

    # print(f"Conversation history:\n{messages}")

    # Check if the model decided to use the provided function
    if not response["message"].get("tool_calls"):
        print("\nThe model didn't use the function. Its response was:")
        print(response["message"]["content"])
        return

    if response["message"].get("tool_calls"):
        # print(f"\nThe model used some tools")
        available_functions = {
            "get_flight_times": get_flight_times,
            "get_antonyms": get_antonyms,
            "weather_place": weather_place,
            "confirmed_cases": confirmed_cases,
        }
        # print(f"\navailable_function: {available_functions}")
        for tool in response["message"]["tool_calls"]:
            # print(f"available tools: {tool}")
            # tool: {'function': {'name': 'get_flight_times', 'arguments': {'arrival': 'LAX', 'departure': 'NYC'}}}
            function_to_call = available_functions[tool["function"]["name"]]
            print(f"Function Invoked: {function_to_call}")

            if function_to_call == get_flight_times:
                function_response = function_to_call(
                    tool["function"]["arguments"]["departure"],
                    tool["function"]["arguments"]["arrival"],
                )

            elif function_to_call == get_antonyms:
                function_response = function_to_call(
                    tool["function"]["arguments"]["word"],
                )

            elif function_to_call == confirmed_cases:
                function_response = function_to_call(
                    tool["function"]["arguments"]["country_name"],
                )

            elif function_to_call == weather_place:
                function_response = function_to_call(
                    tool["function"]["arguments"]["place_name"],
                )

            print(f"function response: {function_response}")

            messages.append(
                {
                    "role": "tool",
                    "content": function_response,
                }
            )


while True:
    user_input = input("\n Please ask=> ")
    if not user_input:
        # user_input = "What is the flight time from NYC to LAX?"
        # user_input = "What is the number of COVID confirmed cases in India"
        user_input = "Whats up with the weather in New York?"
    if user_input.lower() == "exit":
        break

    asyncio.run(run("llama3.1", user_input))
