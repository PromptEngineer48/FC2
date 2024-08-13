import json
import ollama
import asyncio
import requests


def finance_data(company_name: str) -> str:
    url = "https://real-time-finance-data.p.rapidapi.com/stock-time-series-source-2"

    querystring = {"symbol": company_name, "period": "1D"}

    headers = {
        "x-rapidapi-key": "b6dc2081ccmsh282bec1ffdb6254p10b966jsn0c43c4330ada",
        "x-rapidapi-host": "real-time-finance-data.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=querystring)

    return json.dumps(
        f"The current stock price of {response.json()['data']['symbol']} is {response.json()['data']['price']}"
    )


# # Save the result to a text file
# with open("finance_data.txt", "w") as file:
#     json.dump(a, file, indent=4)


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
    return json.dumps(
        f"The number of confirmed COVID cases in {country_name} is {response.json()[0]['confirmed']}"
    )


#########################################################################
async def run(model: str, user_input: str):
    client = ollama.AsyncClient()

    messages = [
        {
            "role": "user",
            "content": user_input,
        }
    ]

    response = await client.chat(
        model=model,
        messages=messages,
        tools=[
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
                        "required": ["place_name"],
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
            {
                "type": "function",
                "function": {
                    "name": "finance_data",
                    "description": "Get the finance data of a particular company",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_name": {
                                "type": "string",
                                "description": "The company for which the finance data is required",
                            },
                        },
                        "required": ["company_name"],
                    },
                },
            },
        ],
    )

    messages.append(response["message"])

    # Check if the model decided to use the provided function
    if not response["message"].get("tool_calls"):
        print("\nThe model didn't use the function. Its response was:")
        print(response["message"]["content"])
        return

    if response["message"].get("tool_calls"):
        available_functions = {
            "weather_place": weather_place,
            "confirmed_cases": confirmed_cases,
            "finance_data": finance_data,
        }
        for tool in response["message"]["tool_calls"]:
            # print(f"Tool is {tool}")
            function_to_call = available_functions[tool["function"]["name"]]
            print(f"Function Invoked: {function_to_call}")

            if function_to_call == confirmed_cases:
                function_response = function_to_call(
                    tool["function"]["arguments"]["country_name"],
                )

            elif function_to_call == weather_place:
                function_response = function_to_call(
                    tool["function"]["arguments"]["place_name"],
                )

            elif function_to_call == finance_data:
                function_response = function_to_call(
                    tool["function"]["arguments"]["company_name"],
                )

            print(f"Function Response: {function_response}")

            messages.append(
                {
                    "role": "tool",
                    "content": function_response,
                }
            )
            return function_response


while True:
    user_input = input("\n Please ask=> ")
    if not user_input:
        # user_input = "What is the number of COVID confirmed cases in India"
        # user_input = "Whats up with the weather in Kolkata?"
        user_input = "What is the current stock price of TSLA?"

    if user_input.lower() == "exit":
        break

    function_response = asyncio.run(run("llama3.1", user_input))
