import openai
import json
import os

class GPT3ChatBot:
    API_KEY = None
    DEFAULT_MODEL = "gpt-3.5-turbo"
    SYSTEM_PROMPT = "You are primarily my insightful coding assistant. Responsd in HTML Structure. HTML should display beautifully. Make sure it really colourful too. Use blue,green,red maybe even purple to illustrate"
    SESSIONS_FOLDER = "ChatGPT_Sessions"
    session_id = None
    session_messages = []

    @staticmethod
    def initialize(api_key, default_model=None):
        GPT3ChatBot.API_KEY = api_key
        GPT3ChatBot.DEFAULT_MODEL = default_model or GPT3ChatBot.DEFAULT_MODEL
        openai.api_key = GPT3ChatBot.API_KEY

    @staticmethod
    def set_system_prompt(system_prompt):
        GPT3ChatBot.SYSTEM_PROMPT = system_prompt

    @staticmethod
    def create_session():
        # Create a new session and initialize session messages
        '''
        response = openai.chat.completions.create(
            model=GPT3ChatBot.DEFAULT_MODEL,
            messages=[{"role": "system", "content": GPT3ChatBot.SYSTEM_PROMPT}],
            temperature=0,
            stop=None,
        )
        print(response)
        GPT3ChatBot.session_id = response.id #Session ID is actually useless rn
        print(response.id)
        '''
        GPT3ChatBot.session_messages = [{"role": "system", "content": GPT3ChatBot.SYSTEM_PROMPT}]

    @staticmethod
    def chat(user_input, model=None):
        if not GPT3ChatBot.session_id:
            GPT3ChatBot.create_session()

        # Add user message to the session messages
        GPT3ChatBot.session_messages.append({"role": "user", "content": user_input})

        # Get chat completion using OpenAI API
        response = openai.chat.completions.create(
            model=model or GPT3ChatBot.DEFAULT_MODEL,
            messages=GPT3ChatBot.session_messages,  # Use the last message
            temperature=0,
            stop=None,
        )

        # Extract and return the bot's reply
        bot_reply = response.choices[0].message.content

        return bot_reply

    @staticmethod
    def save_session(filename, topic_heading):
        # Ensure the sessions folder exists
        os.makedirs(GPT3ChatBot.SESSIONS_FOLDER, exist_ok=True)

        # Save session data to a file with a given topic heading
        session_data = {
            "topic_heading": topic_heading,
            "session_id": GPT3ChatBot.session_id,
            "session_messages": GPT3ChatBot.session_messages
        }

        filepath = os.path.join(GPT3ChatBot.SESSIONS_FOLDER, filename)

        with open(filepath, 'w') as file:
            json.dump(session_data, file)

    @staticmethod
    def load_session(filename):
        # Load session data from a file
        filepath = os.path.join(GPT3ChatBot.SESSIONS_FOLDER, filename)

        with open(filepath, 'r') as file:
            session_data = json.load(file)

        GPT3ChatBot.session_id = session_data["session_id"]
        GPT3ChatBot.session_messages = session_data["session_messages"]

    @staticmethod
    def end_session():
        # End the current session
        GPT3ChatBot.session_id = None
        GPT3ChatBot.session_messages = []

# Example Usage
api_key = ""  # Replace with your actual API key
GPT3ChatBot.initialize(api_key)
