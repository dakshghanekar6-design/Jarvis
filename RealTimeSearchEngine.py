from googlesearch import search
from groq import Groq
from json import load, dump
import datetime
from dotenv import dotenv_values


env_vars = dotenv_values(".env")


Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIkey = env_vars.get("GroqAPIkey")


client = Groq(api_key=GroqAPIkey)

System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which has real-time up-to-date information from the internet.
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***
*** Just answer the question from the provided data in a professional way. ***"""


try:
    with open(r"Data\ChatLog.json", "r") as f:
        messages = load(f)
except:
    with open(r"Data\ChatLog.json", "w") as f:
        dump([], f)


def GoogleSearch(query):
    results = list(search(query, num_results=5))
    Answer =  f"The Search Result For {query} are:\n[start]\n"

    for i in results:
        Answer +=  f"URL: {i}\n"

    Answer += "[end]"
    return Answer


def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer


SystemChatBot = [
    {"role": "system", "content": System},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello, How Can I Help You?"}
]


def Information():
    data = ""
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")
    data += f"Use This Real Time Information If Needed:\n"
    data += f"Day: {day}\n"
    data += f"Date: {date}\n"
    data += f"Month: {month}\n"
    data += f"Year: {year}\n"
    data += f"Time: {hour} hours, {minute} minutes, {second} seconds.\n"
    return data


def RealtimeSearchEngine(prompt):
    global SystemChatBot, messages


    with open(r"Data\ChatLog.json", "r") as f:
        messages = load(f)
    messages.append({"role": "user", "content": f"{prompt}"})

    SystemChatBot.append({"role": "system", "content": GoogleSearch(prompt)})



    completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=SystemChatBot + [{"role": "system", "content": Information()}] + messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
    )



    Answer = ""
    

    for chunk in completion:
        if chunk.choices[0].delta.content:
            Answer += chunk.choices[0].delta.content


    Answer = Answer.strip().replace("</s>", "")
    messages.append({"role": "assistant", "content": Answer})


    with open(r"Data\ChatLog.json", "w") as f:
        dump(messages, f, indent=4)


    SystemChatBot.pop()
    return AnswerModifier(Answer=Answer)


if __name__ == "__main__":
    while True:
        prompt = input("Enter Your Query: ")
        print(RealtimeSearchEngine(prompt))
