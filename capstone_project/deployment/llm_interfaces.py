import os
from openai import OpenAI
import google.generativeai as genai

from common import CONNECT_TO_CHATGPT_API, DEBUG_LEVEL, CONNECT_TO_GEMINI_API, GEMINI_MODEL_NAME, GEMINI_TEMPERATURE, \
    GEMINI_TOP_P, GEMINI_TOP_K, GEMINI_MAX_TOKENS

OPENAI_CLIENT:  OpenAI = None
GEMINI_SESSION:  genai.GenerativeModel = None

def initialize_llm_apis():
  global OPENAI_CLIENT, GEMINI_SESSION
  # Connect to the ChatGPT API and Gemini API

  if CONNECT_TO_CHATGPT_API:
    OPENAI_CLIENT = OpenAI(
      api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Briefly test the connection
    response = OPENAI_CLIENT.responses.create(
      model="gpt-5.5",
      instructions="You are a math teacher assistant.",
      input="What is 2 + 2?"
    )

    if DEBUG_LEVEL > 0:
      print("OpenAI API basic test response:  %s" % response.output_text)

  if CONNECT_TO_GEMINI_API:
    if DEBUG_LEVEL > 0:
      print("About to connect to Gemini.")

    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if DEBUG_LEVEL > 0:
      print("GEMINI_API_KEY is:  %s" % gemini_api_key)

    try:
      genai.configure(api_key=gemini_api_key)

      if DEBUG_LEVEL > 0:
        print("Gemini connect point 1")

      model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        generation_config={
          "temperature": GEMINI_TEMPERATURE,
          "top_p": GEMINI_TOP_P,
          "top_k": GEMINI_TOP_K,
          "max_output_tokens": GEMINI_MAX_TOKENS,
        },
        safety_settings=[]
      )

      if DEBUG_LEVEL > 0:
        print("Gemini connect point 2")

      GEMINI_SESSION = model.start_chat(history=[])

    except Exception as e:
      print("ERROR:  Gemini got an error:  %s" % str(e))
