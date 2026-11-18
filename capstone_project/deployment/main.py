import getpass
import os
from pyngrok import ngrok, conf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
import pandas as pd

from flask import Flask, render_template, request, session
from pathlib import Path
import threading
import io
import os
import base64
import re

from openai import OpenAI
import google.generativeai as genai

import io
import json
import numpy as np
import logging

# Setup debugging and connection parameters
CONNECT_TO_CHATGPT_API = True

CONNECT_TO_GEMINI_API = False
GEMINI_MODEL_NAME = "gemini-3.6-flash"
GEMINI_TEMPERATURE = 0.9
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 40
GEMINI_MAX_TOKENS = 2048
GEMINI_TIMEOUT = 30

REASONING_THRESHOLD = 0.70
EVALUATION_DEBUG_LEVEL = 2
DEBUG_LEVEL = 1 # EVALUATION_DEBUG_LEVEL
VAGUE_WRONG_ANSWER = "dunno"
VAGUE_WRONG_REASONING = "dunno"

CONTROL_RUN = False

# Note:  In terms of "scaling data," adding the following related problem and related properties,
#        to math problem 2 below, raised the cosine similarity of a student response
#        "4 * 250 = 4 * (25 * 10) = (4 * 25) * 10 = 100 * 10 = 1000" to response content wanted by the teacher
#        from around 41% to around 85%.
#
#   ...
#   [
#   ...
#   { "problem":  "4 * 250",
#     "answer":  "1000",
#     related_problems":  [
#     ...
#       { "problem":  "25 * 10",
#         "answer": 250
#       }
#     ],
#     "related_properties": [ { "property":  "associative",
#                               "usage":  "4 * (25 * 10) = (4 * 25) * 10"
#                             }
#                           ]
#   }
#   ...
#   ]

# 1.  Read a list of math arithmetic problems from a JSON file
#
# For example:
# [
# {
#   "num_candidate_answers":  10,
#   "retry_max_times": 10,
#   "problems":  [
#   { "problem":  "4 * 25",
#     "answer":  "100",
#     "related_problems":  [ { "problem":  "How many quarters are in a dollar",
#                              "answer":  4
#                            }
#                          ]
#   },
#   { "problem":  "4 * 250",
#     "answer":  "1000",
#     "related_problems":  [ { "problem":  "4 * 25",
#                              "answer":  100
#                            }
#                          ]
#   },
#   { "problem":  "1000 / 4",
#     "answer":  "250",
#     "related_problems":  [ { "problem":  "4 * 250",
#                              "answer":  1000
#                            }
#                          ]
#   },
#   { "problem":  "996 / 4",
#     "answer":  "249",
#     "related_problems":  [ { "problem":  "1000 / 4",
#                              "answer":  250
#                            }
#                          ],
#     "related_properties": [ { "property":  "distributive",
#                               "usage":  "996 / 4 = (1000 - 4 ) / 4 = (1000 + (-4)) * (1/4) = (1000 * (1/4)) + ((-4) * (1/4)) = (1000 / 4) + (-4/4) = 250 - 1 = 249"
#                             }
#                           ],
#     "common_errors": [ { "error":  "996 / 4 = 6",
#                          "reasoning":  "9 + 9 + 6 = 24, 24 / 4 = 6" }
#                      ]
#   }
#   ]
# }
# ]
#
# 2.  Connect to the ChatGPT API and Gemini API
#
# 3.  Iteratively looping over the math problems read from step 1, do the following steps:
#   (a)  Ask a student the math problem from the current list iteration.  For example:  996 / 4.
#   (b)  Pass the student's answer to ChatGPT and ask ChatGPT to verify if the student is correct (correct answers can
#        come in different forms, and this allows ChatGPT to resolve whether the student has provided an equivalent
#        correct answer in a different form.  For instance, 996 / 4 = 249 but it also equals "249.0 and 0/4".)
#   (c)  Whether the student was correct or incorrect in step 3(b), ask the student for what reasoning went into the answer.
#        If the answer came in a format unexpected for the question (for instance if a decimal number answer was provided when
#        a fraction was expected), ask the student why they chose a particular format for the answer.
#      (i)  If the student's answer was correct and also contains elements expected in reasoning (determined by cosine
#           similarity between the student's answer and reasoning provided for the problem in the JSON problem inputs), break
#           and go into the next loop iteration (for the next math problem in the list).
#      (ii)  If the student's answer was incorrect and/or did not contain reasoning elements expected for
#            an answer to the problem, prepare text melding the student's answer with input text from the JSON file
#            and ask ChatCPT to formulate a set of productive questions to help lead the student in a good direction.
#            Specify ask ChatGPT to provide n candidate answers to pick from, where n was specified in the JSON setup
#            file read in step 1 above.  For example:
#            “A child incorrectly thinks that 996 / 4 = 6
#             when correctly 996 / 4 = (1000 - 4 ) / 4 = (1000 + (-4)) * (1/4)
#             = (1000 * (1/4)) + ((-4) * (1/4)) = (1000 / 4) + (-4/4) = 250 - 1 = 249,
#             using the distributive property.  Ask a question that asks the child a
#             productive question (not an answer) containing correct reasoning.  Provide 10 possible
#             productive questions.  The child might have incorrectly reasoned that
#             996 / 4 = 6 because 9 + 9 + 6 = 24 and 24 / 4 = 6.”
#      (iii)  Read the ChatGPT answers and pick the "best" one to convey back to the student, using TF-IDF and cosine similarity
#             between notes in the JSON input file, text originating from the student, and the answers provided by ChatGPT.
#      (iv)  Continue steps (c)(i) through (c)(iii) until the student gives a correct answer containing at least some elements
#            of reasoning specified as desired in the JSON input file, or if the student wants to give up on the problem
#            (the correct answer should not be provided) or if a certain number of attempts t, read in from the JSON file
#             in step 1, was exceeded).


os.environ["FLASK_DEBUG"] = "development"

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY")

port = 5000

chatgpt_output_correlation_vector = np.array([0, 0, 0])
perfect_correlation_vector = np.array([0, 0, 0])

math_string_dfs = []
for posixPath in Path.cwd().glob("*.json"):
  print("File path:  %s" % posixPath)
  math_string_df = pd.read_json(posixPath)
  math_string_dfs.append(math_string_df)

if DEBUG_LEVEL > 0 and len(math_string_dfs) > 0:
  print(math_string_dfs[0].head())

# Note:  Python logging output is automatically
#        synchronized between Flask threads, so logging
#        statements in different user sessions
#        cannot interleave mid-line.
logging.basicConfig(
  filename="MYangMathBot.log",
  encoding="utf-8",
  filemode="a",  # "a" for append, "w" would be to rewrite/truncate
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s"
)

conf.get_default().auth_token = os.getenv("NGROK_AUTHTOKEN")

# Connect to the ChatGPT API and Gemini API

if CONNECT_TO_CHATGPT_API:
  client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
  )

  # Briefly test the connection
  response = client.responses.create(
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

    gemini_session = model.start_chat(history=[])

  except Exception as e:
    print("ERROR:  Gemini got an error:  %s" % str(e))


def execute_main():
  # The following code executes an interactive session with a user of this math bot
  if DEBUG_LEVEL < EVALUATION_DEBUG_LEVEL:
    run_state_machine(math_string_dfs)
    return

  # The following code executes an evaluation (scoring) of this math bot
  execute_evaluation(math_string_dfs)

  return


def run_state_machine(math_string_dfs):
  # Open a ngrok tunnel to the HTTP server
  public_url = ngrok.connect(port).public_url
  print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

  # Update any base URLs to use the public ngrok URL
  app.config["BASE_URL"] = public_url

  if __name__ == '__main__':
    # Start the Flask server in a new thread
    threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()

  return


@app.route('/', methods=['GET', 'POST'])
def index():
  math_chatbot_text = None
  math_chatbot_problem = ""
  math_chatbot_supplementary_text = ""
  show_student_response = ""
  show_next_problem_button = ""

  if request.method == 'POST':
    student_response = request.form['student_response']

    math_string_index = session['math_string_df_index']

    (math_chatbot_problem, math_chatbot_text) = state_machine_step(math_string_dfs[math_string_index], student_response)

    show_student_response = True
    show_next_problem_button = False
    math_chatbot_supplementary_text = ""
    if session['math_string_df_index'] >= len(math_string_dfs):
      # 4.  Say "thank you for the conversation" to the student and assure them that good elements of the math conversation
      #     have been recorded (logged) and will be passed to a human teacher for review.  Allow the student to sign off the session.
      #
      math_chatbot_supplementary_text = "Thank you for having this math conversation!  Your responses will be reviewed by human teachers to help you to improve."
      show_student_response = False
      show_next_problem_button = False

      session['state_machine_state'] = 0
      session['math_string_df_index'] = 0
      session['math_string_problem_index'] = 0
      session['initialized'] = False

      # 5.  Disconnect from the ChatGPT API.
      if CONNECT_TO_CHATGPT_API:
        client.close()

    elif (session['state_machine_state'] == 0
          and (session['student_tries'] >= session['retry_max_times'] or session['student_tries'] < 0)):
      math_chatbot_supplementary_text = "Please push the button to go on to the next problem."
      show_student_response = False
      show_next_problem_button = True
      session['student_tries'] = 0

  if request.method == 'GET':
    if not 'initialized' in session:
      initialize_to_login_state()

    math_string_index = session['math_string_df_index']
    math_chatbot_supplementary_text = ""
    show_student_response = True
    show_next_problem_button = False

    (math_chatbot_problem, math_chatbot_text) = state_machine_step(math_string_dfs[math_string_index], "")

  logging.info(
    "student_login:  %s, index():  math_chatbot_problem:  %s, math_chatbot_text:  %s, math_chatbot_supplementary_text:  %s, show_student_response:  %d, show_next_problem_button:  %d"
    % (session['student_login'], math_chatbot_problem, math_chatbot_text, math_chatbot_supplementary_text,
       show_student_response, show_next_problem_button)
  )

  return render_template("index.html", math_chatbot_text=math_chatbot_text, math_chatbot_problem=math_chatbot_problem,
                         math_chatbot_supplementary_text=math_chatbot_supplementary_text,
                         show_student_response=show_student_response, show_next_problem_button=show_next_problem_button)


@app.route('/next', methods=['POST'])
def next():
  math_string_index = session['math_string_df_index']
  session['student_tries'] = 0
  math_chatbot_supplementary_text = ""
  show_student_response = True
  show_next_problem_button = False

  (math_chatbot_problem, math_chatbot_text) = state_machine_step(math_string_dfs[math_string_index], "")

  logging.info(
    "student_login:  %s, next():  math_chatbot_problem:  %s, math_chatbot_text:  %s, math_chatbot_supplementary_text:  %s, show_student_response:  %d, show_next_problem_button:  %d" % (
      session['student_login'], math_chatbot_problem, math_chatbot_text, math_chatbot_supplementary_text,
      show_student_response, show_next_problem_button))
  return render_template("index.html", math_chatbot_text=math_chatbot_text, math_chatbot_problem=math_chatbot_problem,
                         math_chatbot_supplementary_text=math_chatbot_supplementary_text,
                         show_student_response=show_student_response, show_next_problem_button=show_next_problem_button)


@app.route('/reset', methods=['GET'])
def reset():
  initialize_to_login_state()

  math_chatbot_supplementary_text = ""
  math_string_index = 0
  show_student_response = True
  show_next_problem_button = False

  (math_chatbot_problem, math_chatbot_text) = state_machine_step(math_string_dfs[math_string_index], "")

  logging.info(
    "student_login:  %s, reset():  math_chatbot_problem:  %s, math_chatbot_text:  %s, math_chatbot_supplementary_text:  %s, show_student_response:  %d, show_next_problem_button:  %d" % (
      session['student_login'], math_chatbot_problem, math_chatbot_text, math_chatbot_supplementary_text,
      show_student_response, show_next_problem_button))
  return render_template("index.html", math_chatbot_text=math_chatbot_text, math_chatbot_problem=math_chatbot_problem,
                         math_chatbot_supplementary_text=math_chatbot_supplementary_text,
                         show_student_response=show_student_response, show_next_problem_button=show_next_problem_button)


@app.route('/login', methods=['GET', 'POST'])
def login():
  session['student_login'] = "not logged in"

  math_chatbot_supplementary_text = ""
  show_student_response = ""
  show_next_problem_button = ""

  if request.method == 'POST':
    student_login = request.form['student_login']
    session['student_login'] = student_login

    initialize_to_login_state()

    math_chatbot_supplementary_text = ""
    math_string_index = 0
    show_student_response = True
    show_next_problem_button = False

    (math_chatbot_problem, math_chatbot_text) = state_machine_step(math_string_dfs[math_string_index], "")

    logging.info(
      "student_login:  %s, login() A:  math_chatbot_problem:  %s, math_chatbot_text:  %s, math_chatbot_supplementary_text:  %s, show_student_response:  %d, show_next_problem_button:  %d" % (
        session['student_login'], math_chatbot_problem, math_chatbot_text, math_chatbot_supplementary_text,
        show_student_response, show_next_problem_button))
    return render_template("index.html", math_chatbot_text=math_chatbot_text, math_chatbot_problem=math_chatbot_problem,
                           math_chatbot_supplementary_text=math_chatbot_supplementary_text,
                           show_student_response=show_student_response,
                           show_next_problem_button=show_next_problem_button)

  if request.method == 'GET':
    initialize_to_login_state()

    math_chatbot_supplementary_text = ""
    show_student_response = False
    show_next_problem_button = False

  logging.info("student_login:  %s, login() B:  math_chatbot_problem:  , math_chatbot_text:  , math_chatbot_supplementary_text:  %s, show_student_response:  %d, show_next_problem_button:  %d, math_chatbot_login:  Login"
               % (session['student_login'], math_chatbot_supplementary_text, show_student_response, show_next_problem_button)
              )
  return render_template("index.html", math_chatbot_text="", math_chatbot_problem="",
                         math_chatbot_supplementary_text="", show_student_response=show_student_response,
                         show_next_problem_button=show_next_problem_button, math_chatbot_login="Login")


def initialize_to_login_state():
  session['state_machine_state'] = 0
  session['math_string_df_index'] = 0
  session['math_string_problem_index'] = 0
  session['student_tries'] = 0
  session['initialized'] = True


def execute_evaluation(math_string_dfs):

  for local_math_string_df in math_string_dfs:

    if DEBUG_LEVEL > 0:
      print(local_math_string_df.head())

    (chatgpt_output_correlation_vector, perfect_correlation_vector) = \
      evaluate_one_number_talk_string(local_math_string_df, chatgpt_output_correlation_vector, perfect_correlation_vector)

  perfect_correlation_distance = vector_tip_distance(perfect_correlation_vector, np.array([0, 0, 0]))

  chatgpt_output_correlation_distance = perfect_correlation_distance - \
                                        vector_tip_distance(chatgpt_output_correlation_vector,
                                                            perfect_correlation_vector)

  print("The distance between the tip of the chatgpt_output_correlation_overall vector and the perfect_correlation_vector is %f" % chatgpt_output_correlation_distance)
  if perfect_correlation_distance != 0:
    print("This is a performance score of %s %%" % (
              (chatgpt_output_correlation_distance / perfect_correlation_distance) * 100.0))
  else:
    print("This is a performance score of 100 %%")

  # 4.  Say "thank you for the conversation" to the student and assure them that good elements of the math conversation
  #     have been recorded (logged) and will be passed to a human teacher for review.  Allow the student to sign off the session.
  #
  print(
    "Thank you for having this math conversation!  Your responses will be reviewed by human teachers to help you to improve.")

  # 5.  Disconnect from the ChatGPT API.
  if CONNECT_TO_CHATGPT_API:
    client.close()

  return


def get_text_attributes_from_problem(problem_df):
  text_attributes_from_problem = ""

  if 'related_properties' in problem_df.columns:
    related_properties_df = pd.DataFrame.from_dict(problem_df['related_properties']).dropna()
    if len(related_properties_df) > 0:
      for local_index in range(0, len(related_properties_df)):
        property_text = "; %s:  %s" %  (related_properties_df['related_properties'][local_index]['property'], related_properties_df['related_properties'][local_index]['usage'])
        text_attributes_from_problem += property_text

  if 'related_problems' in problem_df.columns:
    related_problems_df = pd.DataFrame.from_dict(problem_df['related_problems']).dropna()
    if len(related_problems_df) > 0:
      for local_index in range(0, len(related_problems_df)):
        property_text = "; %s = %s" %  (related_problems_df['related_problems'][local_index]['problem'], related_problems_df['related_problems'][local_index]['answer'])
        text_attributes_from_problem += property_text

  return text_attributes_from_problem


def get_content_similarity(input_text, texts_to_correlate):
  # This function uses TF-IDF and cosine similarity

  # Note:  This function uses just TF-IDF and cosine similarity to characterize how relevant
  #        student reasoning is to a math problem.
  #
  #        As previously discussed with my UMGC AI bootcamp mentor, interactions with the
  #        ChatGPT API are more than sufficient for implementing my math chatbot capstone project.
  #
  #        But it is possible that this get_content_similarity() function can be improved by
  #        exploring classifiers or regressors in scikit-learn, or deep neural networks in keras,
  #        to see if inputs of student reasoning are really relevant to a math problem.
  #        Training data for such a model could be the fields "question" and "conversation" (for X)
  #        and "self-correctness" (for y) in the math conversation dataset at:
  #        https://github.com/eth-nlped/mathdial

  texts_list = [ input_text ]
  texts_list.extend(texts_to_correlate)

  if DEBUG_LEVEL > 0:
    print("input_text:  %s, texts_to_correlate:  %s, texts_list:  %s" % (input_text, texts_to_correlate, texts_list))

  vectorizer = TfidfVectorizer()
  vectorized_data = vectorizer.fit_transform(texts_list)

  tfidf_df = pd.DataFrame(vectorized_data.toarray(),
                          columns=vectorizer.get_feature_names_out())
  tfidf_df.index = texts_list

  # Compute the cosine similarity
  cosine_similarity_array = cosine_similarity(tfidf_df)

  # Get the indices of similar texts based on cosine similarity
  cosine_similarity_df = pd.DataFrame(cosine_similarity_array, index=tfidf_df.index, columns=tfidf_df.index)

  return cosine_similarity_df


def student_reasoning_is_relevant_to_problem(text_attributes_from_problem, student_reasoning):
  # Find similarities between student reasoning (and a list of previous student reasoning) and
  # teacher notes in the text_attributes_from_problem (that were read from the problem set .JSON file).

  if student_reasoning == "" or text_attributes_from_problem == "":
    return 0.0

  if DEBUG_LEVEL > 0:
    print("student_reasoning:  %s" % student_reasoning)
    print("text_attributes_from_problem:  %s" % text_attributes_from_problem)

  content_similarity_df = get_content_similarity(student_reasoning, [ text_attributes_from_problem ])
  student_reasoning_similarity_series = content_similarity_df[student_reasoning][1:]

  if DEBUG_LEVEL > 0:
    print("student_reasoning_similarity_series:  %s" % str(student_reasoning_similarity_series))
    print("student_reasoning_similarity_series.max().size:  %s" % str(student_reasoning_similarity_series.max().size))
    print("student_reasoning_similarity_series.max().shape:  %s" % str(student_reasoning_similarity_series.max().shape))

  student_reasoning_similarity_percentage = student_reasoning_similarity_series.max()
  if student_reasoning_similarity_series.max().size > 1:   # Remove max duplicates if necessary
    if DEBUG_LEVEL > 0:
      print("student_reasoning_similarity_series.max().iloc[0]:  %s" % str(student_reasoning_similarity_series.max().iloc[0]))
    student_reasoning_similarity_percentage = student_reasoning_similarity_series.max().iloc[0]

  return student_reasoning_similarity_percentage >= REASONING_THRESHOLD


def stub_to_correlate_phrases_in_a_0_to_10_ranking(text_phrase_0, text_phrase_1):

  #return 10
  #return 0

  content_similarity_df = get_content_similarity(text_phrase_0, [ text_phrase_1 ])
  reasoning_similarity_series = content_similarity_df[text_phrase_0][1:]

  if DEBUG_LEVEL > 0:
    print("reasoning_similarity_series:  %s" % str(reasoning_similarity_series))
    print("reasoning_similarity_series.max().size:  %s" % str(reasoning_similarity_series.max().size))
    print("reasoning_similarity_series.max().shape:  %s" % str(reasoning_similarity_series.max().shape))

  reasoning_similarity_percentage = reasoning_similarity_series.max()
  if reasoning_similarity_series.max().size > 1:   # Remove max duplicates if necessary
    if DEBUG_LEVEL > 0:
      print("reasoning_similarity_series.max().iloc[0]:  %s" % str(reasoning_similarity_series.max().iloc[0]))
    reasoning_similarity_percentage = reasoning_similarity_series.max().iloc[0]

  if DEBUG_LEVEL > 0:
    print("text_phrase_0:  %s" % text_phrase_0)
    print("text_phrase_1:  %s" % text_phrase_1)
    print("reasoning_similarity_percentage is:  %f" % reasoning_similarity_percentage)
      
  return reasoning_similarity_percentage*10.0

from parse import parse

def ask_gemini_to_correlate_phrases_in_a_0_to_10_ranking(text_phrase_0, text_phrase_1):

  if text_phrase_0 == "" or text_phrase_1 == "":
    return 5 # If either text_phrase is just empty, return a median correlation rating 5 (out of 10), basically signifying no knowledge either way

  gemini_response = stub_to_correlate_phrases_in_a_0_to_10_ranking(text_phrase_0, text_phrase_1)

  if CONNECT_TO_GEMINI_API:
    try:
      if DEBUG_LEVEL > 0:
        print("Gemini debug point 3")

      message = "Rate on a scale of 0 to 10 the relevance of ideas contained in phrase \"" + text_phrase_0 + "\" to ideas in phrase \"" + text_phrase_1 + ".\"  Answer only with one integer number."

      if DEBUG_LEVEL > 0:
        print("Gemini debug point 4")

      gemini_raw_response = gemini_session.send_message(message)

      if DEBUG_LEVEL > 0:
        print("Gemini debug point 5")
        print("Gemini response rating for (%s, %s) was:  %s"  % (text_phrase_0, text_phrase_1, gemini_response))

      gemini_text_response = gemini_raw_response.candidates[0].content.parts[0].text

      if DEBUG_LEVEL > 0:
        print("Gemini debug point 6")
        print("Gemini response text was:  %s" % gemini_text_response)

      result = parse("{:d}", gemini_text_response)

      if DEBUG_LEVEL > 0:
        print("Gemini debug point 7")
        print("Gemini result was:  %s" % result)
        gemini_response = result[0]
        print("Gemini rating was:  %s" % gemini_response)

    except Exception as e:
      print("ERROR:  Gemini got an error:  %s" % str(e))
      return 0

  return gemini_response


def pick_best_chatgpt_feedback(problem_df, student_reasoning, chatgpt_feedback_list, n):
  # Find similarities between student reasoning + teacher notes, against a list
  # of chatgpt feedback n items long.

  input_baseline_text = student_reasoning 
  input_baseline_text += get_text_attributes_from_problem(problem_df)

  if DEBUG_LEVEL > 0:
    print("input_baseline_text:  %s" % input_baseline_text)
    print("chatgpt_feedback_list:  %s" % chatgpt_feedback_list)

  content_similarity_df = get_content_similarity(input_baseline_text, chatgpt_feedback_list)
  reasoning_similarity_series = content_similarity_df.iloc[0, 1:]

  if DEBUG_LEVEL > 0:
    print("reasoning_similarity_series:  %s" % reasoning_similarity_series)
    print("reasoning_similarity_series.max():  %s" % reasoning_similarity_series.max())
    print("reasoning_similarity_series.idxmax():  %s" % reasoning_similarity_series.idxmax())

  return reasoning_similarity_series.idxmax()


import re

def chatgpt_api_supplies_leading_question(problem_df, student_answer, student_reasoning, num_leading_questions_to_supply):

  leading_question_prompt = "A child incorrectly thinks that %s = %s when correctly %s = %s" \
                            % (problem_df['problem'][0], student_answer, problem_df['problem'][0], problem_df['answer'][0])

  if 'related_properties' in problem_df.columns:
    related_properties_df = pd.DataFrame.from_dict(problem_df['related_properties']).dropna()
    if len(related_properties_df) > 0:
      related_property = related_properties_df['related_properties'][0]
      related_property_property = related_property['property']
      related_property_usage = related_property['usage']
      leading_question_prompt += " because %s from the %s property" % (related_property_usage, related_property_property)

  leading_question_prompt += ".  Ask the child %s candidate productive questions (not answers) containing correct reasoning.  " \
                             % num_leading_questions_to_supply

  leading_question_prompt += "  The child reasoned that '%s'" % student_reasoning

  if 'common_errors' in problem_df.columns:
    common_errors_df = pd.DataFrame.from_dict(problem_df['common_errors']).dropna()
    if len(common_errors_df) > 0:
      leading_question_prompt += ", and might have also incorrectly reasoned that: "
      for local_index in range(0, len(common_errors_df), 1):
        common_error = common_errors_df['common_errors'][local_index]
        common_error_reasoning = common_error['reasoning']
        if local_index > 0:
         leading_question_prompt += ", and "
        leading_question_prompt += "  %s" % common_error_reasoning 
      leading_question_prompt += ".  "

  if 'related_problems' in problem_df.columns:
    related_problems_df = pd.DataFrame.from_dict(problem_df['related_problems']).dropna()
    if len(related_problems_df) > 0:
      leading_question_prompt += "  Related problems:  "
      for local_index in range(0, len(related_problems_df), 1):
        leading_question_addendum = "%s = %s" %  (related_problems_df['related_problems'][local_index]['problem'], related_problems_df['related_problems'][local_index]['answer'])
        leading_question_prompt += leading_question_addendum
        leading_question_prompt += ".  "

  if 'related_properties' in problem_df.columns:
    related_properties_df = pd.DataFrame.from_dict(problem_df['related_properties']).dropna()
    if len(related_properties_df) > 1:
      leading_question_prompt += "  Other related properties:  "
      for local_index in range(1, len(related_properties_df), 1):
        leading_question_addendum = "%s due to the %s property" %  (related_properties_df['related_properties'][local_index]['property'], related_properties_df['related_properties'][local_index]['usage'])
        leading_question_prompt += leading_question_addendum
        leading_question_prompt += ".  "

  if DEBUG_LEVEL > 0:
    print("Question giving to ChatGPT:  %s" % leading_question_prompt)

  response_output = "1. If (996 div 4) meant 4 equal groups, would each group having 6 make a total of (4 times 6)? Is that total close to 996?" \
                    "2. Since (1000 div 4 = 250), and 996 is 4 less than 1000, how much less should each group get when we split 996 into 4 equal groups?" \
                    "3. Can we write (996) as (1000 - 4)? Then what would ((1000 - 4) div 4) become using the distributive property?" \
                    "4. What is (1000 div 4), and what is (4 div 4)? So what should ((1000 div 4) - (4 div 4)) be?" \
                    "5. When you added (9 + 9 + 6), did you keep the place values of 996, where the first 9 means 900 and the second 9 means 90?" \
                    "6. Is (996) the same number as (9 + 9 + 6), or is (996 = 900 + 90 + 6)? How does that change the division?" \
                    "7. If (24 div 4 = 6), what number are we dividing there: 24 or 996? Are those the same amount?" \
                    "8. Could adding the digits tell us something about divisibility, but not the actual answer to (996 div 4)? How could we check the actual quotient?" \
                    "9. If each group had 249, what would (249 times 4) give? How does that help check (996 div 4)?" \
                    "10. Can we split (996) into parts that are easy to divide by 4, like (800 + 196) or (1000 - 4), and divide each part correctly?"

  if CONNECT_TO_CHATGPT_API:
    local_response = client.responses.create(model="gpt-5.5",
                                       instructions="You are a math teacher assistant.",
                                       input=leading_question_prompt
                                      )
    response_output = local_response.output_text

  chatgpt_feedback_list = re.split(r'[0-9]+\.[ \t]+', response_output)
  if len(chatgpt_feedback_list) > 0:
    chatgpt_feedback_list = chatgpt_feedback_list[1:]

  best_chatgpt_feedback = pick_best_chatgpt_feedback(problem_df, student_reasoning, chatgpt_feedback_list, num_leading_questions_to_supply)

  if DEBUG_LEVEL > 0:
    print("chatgpt_feedback_list:  %s" % chatgpt_feedback_list)
    print("best_chatgpt_feedback:  %s" % best_chatgpt_feedback)

  return best_chatgpt_feedback


def justify_current_problem_solution_with_answers_to_previous_problems(math_problems):
    return_string = ""
    for math_problem in math_problems[:len(math_problems)-1]:
      return_string += math_problem['problem'] + ' = ' + math_problem['answer'] + ', '

    return return_string


def evaluate_one_number_talk_string(number_talk_string_df, chatgpt_output_correlation_vector, perfect_correlation_vector):

  previous_student_reasoning = []
    
  if DEBUG_LEVEL > 0:
    print(number_talk_string_df)
    
  num_candidate_answers = number_talk_string_df.num_candidate_answers[0]
  retry_max_times = 2
  if DEBUG_LEVEL < EVALUATION_DEBUG_LEVEL:
    retry_max_times = number_talk_string_df.retry_max_times[0]
  math_problems = number_talk_string_df.problems[0]

  last_math_problem_in_string = math_problems[len(math_problems)-1]

  if DEBUG_LEVEL > 0:
    print("last_math_problem_in_string:  %s" % last_math_problem_in_string)

  # ---------------------------------------------------------------------------
  # The following fulfills bullet point 3(a) in the block comment above.
  # problem_df = pd.DataFrame.from_dict(math_problem)
  problem_df = pd.DataFrame({key: pd.Series(value) for key, value in last_math_problem_in_string.items()})

  correct_answer_from_teacher = problem_df['answer'][0]

  if DEBUG_LEVEL > 0:
    print("correct_answer_from_teacher:  %s" % correct_answer_from_teacher)
      
  answer_good_enough = False
  student_tries = 0
  while not answer_good_enough and student_tries < retry_max_times:
    if 'instructions' in number_talk_string_df.keys():
      instructions = number_talk_string_df.instructions[0]
      print(instructions)

    student_answer = VAGUE_WRONG_ANSWER
    if CONTROL_RUN:
      student_answer = "Something around like " + correct_answer_from_teacher + " but not exactly ?"

    if DEBUG_LEVEL < EVALUATION_DEBUG_LEVEL:
      student_answer = input("What is %s ?" % problem_df['problem'][0])

    # ---------------------------------------------------------------------------
    # The following fulfills bullet point 3(c) in the block comment above.
    if student_tries == 0 and not CONTROL_RUN:
      student_reasoning = VAGUE_WRONG_REASONING
    else:
      student_reasoning = str(justify_current_problem_solution_with_answers_to_previous_problems(math_problems))
      if 'related_properties' in problem_df.keys() and len(problem_df['related_properties']) > 0:
        student_reasoning += " and " + str(problem_df['related_properties'][0]['property'] + " is relevant.")

    if DEBUG_LEVEL < EVALUATION_DEBUG_LEVEL:
      student_reasoning = input("And what is your reasoning for your answer?")

    if DEBUG_LEVEL > 0:
      print("Problem:  %s, Answer:  %s, Student Answer:  %s, Student Reasoning:  %s" \
            % (problem_df['problem'][0], correct_answer_from_teacher, student_answer, student_reasoning))

    student_answer_is_correct = False

    if correct_answer_from_teacher == student_answer:
      student_answer_is_correct = True

    if not student_answer_is_correct:
      # ---------------------------------------------------------------------------
      # The following fulfills bullet point 3(b) in the block comment above.
      verification_query = "A student answered that %s = %s.  Verify the answer is incorrect, yes or no." % (problem_df['problem'][0], student_answer)

      if DEBUG_LEVEL > 1:
        print(verification_query)

      if CONNECT_TO_CHATGPT_API:
        local_response = client.responses.create(
          model="gpt-5.5",
          instructions="You are a math teacher assistant.",
          input=verification_query
        )

        if local_response.output_text[:3].lower() != "yes":
          student_answer_is_correct = True

    previous_student_reasoning.append(student_reasoning)
    student_tries += 1

    text_attributes_from_problem = get_text_attributes_from_problem(problem_df)

    if not student_answer_is_correct:
      leading_question_from_chatgpt = chatgpt_api_supplies_leading_question(problem_df, student_answer, student_reasoning, num_candidate_answers)
      print(leading_question_from_chatgpt)

      if DEBUG_LEVEL >= EVALUATION_DEBUG_LEVEL:
        chatgpt_output_correlation_to_student_reasoning = ask_gemini_to_correlate_phrases_in_a_0_to_10_ranking(student_reasoning, leading_question_from_chatgpt)
        chatgpt_output_correlation_to_correct_answer = ask_gemini_to_correlate_phrases_in_a_0_to_10_ranking(correct_answer_from_teacher, leading_question_from_chatgpt)  
        chatgpt_output_correlation_to_related_properties = ask_gemini_to_correlate_phrases_in_a_0_to_10_ranking(text_attributes_from_problem, leading_question_from_chatgpt)  

        perfect_correlation_vector = perfect_correlation_vector + np.array([1, 1, 1])
        chatgpt_output_correlation_vector = chatgpt_output_correlation_vector + \
                                              np.array([chatgpt_output_correlation_to_student_reasoning/10.0, chatgpt_output_correlation_to_correct_answer/10.0, chatgpt_output_correlation_to_related_properties/10.0])

        perfect_correlation_distance = vector_tip_distance(np.array([1, 1, 1]), np.array([0, 0, 0]))

        chatgpt_output_correlation_distance = perfect_correlation_distance - \
                                              vector_tip_distance(np.array([chatgpt_output_correlation_to_student_reasoning/10.0, chatgpt_output_correlation_to_correct_answer/10.0, chatgpt_output_correlation_to_related_properties/10.0]),
                                                                  np.array([1, 1, 1]))

        print("For this one problem, the distance between the tip of the chatgpt_output_correlation_overall vector and the perfect_correlation_vector is %f" % chatgpt_output_correlation_distance)
        print("For this one problem %s, student_answer %s, text_attributes %s, and chatgpt output %s, this is a performance score of %s %%" \
              % (last_math_problem_in_string, student_reasoning, text_attributes_from_problem, leading_question_from_chatgpt, (chatgpt_output_correlation_distance/perfect_correlation_distance)*100.0))

      print("Let's try to think of another answer to this problem.")
      if student_reasoning_is_relevant_to_problem(text_attributes_from_problem, student_reasoning):
        print("Note: The reasoning you have provided was actually already along the lines of what the teacher was looking for.")
    else:
      print("Your answer was correct!")
      answer_good_enough = True
      student_tries = 0 # Since the student has given a correct answer, now allow the student a full set of tries to create good reasoning now.

      reasoning_good_enough = False
      while not reasoning_good_enough and student_tries < retry_max_times:
        if student_reasoning_is_relevant_to_problem(text_attributes_from_problem, student_reasoning):
          print("Importantly, your reasoning was along the lines of what the teacher was looking for.")
          reasoning_good_enough = True
        else:
          if DEBUG_LEVEL < EVALUATION_DEBUG_LEVEL:
            student_reasoning = input("Please add other reasoning for your answer to this problem.")
            previous_student_reasoning.append(student_reasoning)
          student_tries += 1

    if DEBUG_LEVEL > 0:
      print("previous_student_reasoning:  %s" % previous_student_reasoning)

  return chatgpt_output_correlation_vector, perfect_correlation_vector


def vector_tip_distance(vector0, vector1):
  return np.sqrt((vector1[0] - vector0[0])**2 + \
                 (vector1[1] - vector0[1])**2 + \
                 (vector1[2] - vector0[2])**2)


def state_machine_step(param_math_string_df, student_response):
  match (session['state_machine_state']):
    case 0:
      return state_problem_0(param_math_string_df)

    case 1:
      return get_student_answer_1(param_math_string_df, student_response)

    case 2:
      return get_student_reasoning_2(param_math_string_df, student_response)

    case 3:
      return check_student_reasoning_3(param_math_string_df, student_response)

    case _:
      return state_problem_0(param_math_string_df)


def state_problem_0(param_math_string_df):

  problem_df = get_problem_df(param_math_string_df)

  session['num_candidate_answers'] = int(param_math_string_df.num_candidate_answers[0])
  session['retry_max_times'] = int(param_math_string_df.retry_max_times[0])

  session['state_machine_state'] = 1

  logging.info("student_login:  %s, state_problem_0():  math_chatbot_problem:  What is %s ?, math_chatbot_text:  " % (session['student_login'], problem_df['problem'][0]))

  instructions_text = ""
  if 'instructions' in param_math_string_df.keys():
    instructions_text = param_math_string_df.instructions[0] + " ...  "

  return instructions_text + "What is %s ?" % problem_df['problem'][0], ""


def get_student_answer_1(param_math_string_df, student_response):

  problem_df = get_problem_df(param_math_string_df)

  session['state_machine_state'] = 2
  session['student_answer'] = student_response
  session['text_attributes_from_problem'] = get_text_attributes_from_problem(problem_df)

  logging.info("student_login:  %s, get_student_answer_1():  math_chatbot_problem:  Your problem answer was:  %s, math_chatbot_text:  And what is your reasoning for your answer?" % (session['student_login'], session['student_answer']))
  return "Your problem answer was:  %s" % session['student_answer'], "And what is your reasoning for your answer?"


def get_student_reasoning_2(param_math_string_df, student_response):

  problem_df = get_problem_df(param_math_string_df)

  session['student_reasoning'] = student_response
  session['student_tries'] += 1

  correct_answer_from_teacher = problem_df['answer'][0]

  if DEBUG_LEVEL > 0:
    print("get_student_reasoning_2():  student_answer is %s and correct_answer_from_teacher is %s" % (session['student_answer'], correct_answer_from_teacher))

  student_answer_is_correct = False

  if session['student_answer'] != correct_answer_from_teacher:
    if DEBUG_LEVEL > 0:
      print("get_student_reasoning_2():  student_response was *not* the same as correct_answer_from_teacher")

    # ---------------------------------------------------------------------------
    # The following fulfills bullet point 3(b) in the block-comment close to the top of this file.
    verification_query = "A student answered that %s = %s.  Verify the answer is incorrect, yes or no." % (problem_df['problem'][0], session['student_answer'])

    if DEBUG_LEVEL > 0:
      print(verification_query)

    if CONNECT_TO_CHATGPT_API:
      local_response = client.responses.create(
          model="gpt-5.5",
          instructions="You are a math teacher assistant.",
          input=verification_query
      )

      if local_response.output_text[:3].lower() != "yes":
        student_answer_is_correct = True
        session['student_tries'] = 0
  else:
    student_answer_is_correct = True
    session['student_tries'] = 0
    if DEBUG_LEVEL > 0:
      print("get_student_reasoning_2():  student_response was correct!")

  if not student_answer_is_correct:
    if session['student_tries'] < session['retry_max_times']:
      leading_question_from_chatgpt = chatgpt_api_supplies_leading_question(problem_df, session['student_answer'], session['student_reasoning'], session['num_candidate_answers'])
      student_reasoning_relevance_text = ""
      if student_reasoning_is_relevant_to_problem(session['text_attributes_from_problem'], session['student_reasoning']):
        student_reasoning_relevance_text = "Note: The reasoning you have provided was actually already along the lines of what the teacher was looking for.  "
      session['state_machine_state'] = 1
      logging.info("student_login:  %s, get_student_reasoning_2() A:  Student Reasoning:  %s, Student Tries:  %s, math_chatbot_problem:  What is %s ?, math_chatbot_text:  Clue:  %s" % (session['student_login'], session['student_reasoning'], session['student_tries'], problem_df['problem'][0], student_reasoning_relevance_text + leading_question_from_chatgpt))
      return "What is %s ?" % problem_df['problem'][0], "Clue:  " + student_reasoning_relevance_text + leading_question_from_chatgpt
    else:
      session['state_machine_state'] = 0
      session['math_string_problem_index'] += 1
      if session['math_string_problem_index'] >= len(param_math_string_df.problems[0]):
        session['math_string_df_index'] += 1
        session['math_string_problem_index'] = 0
      logging.info("student_login:  %s, get_student_reasoning_2() B:  Student Reasoning:  %s, Student Tries:  %s, math_chatbot_problem:  What is %s ?, math_chatbot_text:  " % (session['student_login'], session['student_reasoning'], session['student_tries'], problem_df['problem'][0]))
      return "", ""

  if student_reasoning_is_relevant_to_problem(session['text_attributes_from_problem'], session['student_reasoning']):
    session['state_machine_state'] = 0
    session['math_string_problem_index'] += 1
    if session['math_string_problem_index'] >= len(param_math_string_df.problems[0]):
      session['math_string_df_index'] += 1
      session['math_string_problem_index'] = 0
    session['student_tries'] = -1
    logging.info("student_login:  %s, get_student_reasoning_2() C:  Student Reasoning:  %s, Student Tries:  %s, math_chatbot_problem:  Your answer %s was correct!, math_chatbot_text:  Importantly, your reasoning was along the lines of what the teacher was looking for." % (session['student_login'], session['student_reasoning'], session['student_tries'], session['student_answer']))
    return "Your answer %s was correct!" % session['student_answer'], "Importantly, your reasoning was along the lines of what the teacher was looking for."

  session['state_machine_state'] = 3
  session['student_tries'] = 1
  logging.info("student_login:  %s, get_student_reasoning_2() D:  Student Reasoning:  %s, Student Tries:  %s, math_chatbot_problem:  Your answer %s was correct!, math_chatbot_text:  Please add other reasoning for your answer to this problem." % (session['student_login'], session['student_reasoning'], session['student_tries'], session['student_answer']))
  return "Your answer %s was correct!" % session['student_answer'], "Please add other reasoning for your answer to this problem."


def check_student_reasoning_3(param_math_string_df, student_response):

  student_reasoning = student_response
  session['student_tries'] += 1

  if student_reasoning_is_relevant_to_problem(session['text_attributes_from_problem'], student_reasoning):
    session['math_string_problem_index'] += 1 
    if session['math_string_problem_index'] >= len(param_math_string_df.problems[0]):
      session['math_string_df_index'] += 1
      session['math_string_problem_index'] = 0
    session['state_machine_state'] = 0
    session['student_tries'] = -1
    logging.info("student_login:  %s, check_student_reasoning_3() A:  Student Reasoning:  %s, Student Tries:  %s, math_chatbot_problem:  You said:  %s, math_chatbot_text:  Importantly, your reasoning was along the lines of what the teacher was looking for." % (session['student_login'], session['student_reasoning'], session['student_tries'], session['student_reasoning']))
    return "You said:  %s" % student_reasoning, "Importantly, your reasoning was along the lines of what the teacher was looking for."

  if session['student_tries'] < session['retry_max_times']:
    if DEBUG_LEVEL > 0:
      print("session['student_tries']:  %d, session['retry_max_times']:  %d, session['text_attributes_from_problem']:  %s, student_reasoning:  %s" % (session['student_tries'], session['retry_max_times'], session['text_attributes_from_problem'], student_reasoning))

    if not student_reasoning_is_relevant_to_problem(session['text_attributes_from_problem'], student_reasoning):
      session['state_machine_state'] = 3
      logging.info("student_login:  %s, check_student_reasoning_3() B:  Student Reasoning:  %s, Student Tries:  %s, math_chatbot_problem:  You said:  %s, math_chatbot_text:  Please add other reasoning for your answer to this problem." % (session['student_login'], session['student_reasoning'], session['student_tries'], session['student_reasoning']))
      return "You said:  %s" % student_reasoning, "Please add other reasoning for your answer to this problem."

  session['math_string_problem_index'] += 1 
  if session['math_string_problem_index'] >= len(param_math_string_df.problems[0]):
    session['math_string_df_index'] += 1
    session['math_string_problem_index'] = 0
  session['state_machine_state'] = 0
  session['student_tries'] = -1
  logging.info("student_login:  %s, check_student_reasoning_3() C:  Student Reasoning:  %s, Student Tries:  %s, math_chatbot_problem:  , math_chatbot_text:  " % (session['student_login'], session['student_reasoning'], session['student_tries']))
  return "", ""


def get_problem_df(param_math_string_df):
  if DEBUG_LEVEL > 0:
    print("session['math_string_df_index'] is %s, session['math_string_problem_index'] is %s" % (session['math_string_df_index'], session['math_string_problem_index']))

  math_problems = param_math_string_df.problems[0]
  math_problem_index = session['math_string_problem_index']
  math_problem = math_problems[math_problem_index]
  problem_df = pd.DataFrame({key: pd.Series(value) for key, value in math_problem.items()})

  return problem_df


execute_main()
