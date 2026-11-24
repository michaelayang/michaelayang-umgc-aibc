import os
from pyngrok import ngrok, conf
from flask import Flask, render_template, request, session
import threading
import logging
import pandas as pd

from common import CONNECT_TO_CHATGPT_API, DEBUG_LEVEL, math_string_dfs
from utilities import get_text_attributes_from_problem, chatgpt_api_supplies_leading_question, \
     student_reasoning_is_relevant_to_problem
import llm_interfaces

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


os.environ["FLASK_DEBUG"] = "development"

conf.get_default().auth_token = os.getenv("NGROK_AUTHTOKEN")

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY")

port = 5000

def run_state_machine(math_string_dfs):
  # Open a ngrok tunnel to the HTTP server
  public_url = ngrok.connect(port).public_url
  print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

  # Update any base URLs to use the public ngrok URL
  app.config["BASE_URL"] = public_url

  # Start the Flask server in a new thread
  threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()

  return


@app.route('/', methods=['GET', 'POST'])
def index():
  math_chatbot_text = None
  math_chatbot_problem = ""
  math_chatbot_supplementary_text, show_next_problem_button, show_student_response = set_baseline_display_parameters()

  if request.method == 'POST':
    student_response = request.form['student_response']

    math_string_index = session['math_string_df_index']

    (math_chatbot_problem, math_chatbot_text) = state_machine_step(math_string_dfs[math_string_index], student_response)

    math_chatbot_supplementary_text, show_next_problem_button, show_student_response = set_baseline_display_parameters()
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
        llm_interfaces.OPENAI_CLIENT.close()

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
    math_chatbot_supplementary_text, show_next_problem_button, show_student_response = set_baseline_display_parameters()

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
  math_chatbot_supplementary_text, show_next_problem_button, show_student_response = set_baseline_display_parameters()

  (math_chatbot_problem, math_chatbot_text) = state_machine_step(math_string_dfs[math_string_index], "")

  logging.info(
    "student_login:  %s, next():  math_chatbot_problem:  %s, math_chatbot_text:  %s, math_chatbot_supplementary_text:  %s, show_student_response:  %d, show_next_problem_button:  %d" % (
      session['student_login'], math_chatbot_problem, math_chatbot_text, math_chatbot_supplementary_text,
      show_student_response, show_next_problem_button))
  return render_template("index.html", math_chatbot_text=math_chatbot_text, math_chatbot_problem=math_chatbot_problem,
                         math_chatbot_supplementary_text=math_chatbot_supplementary_text,
                         show_student_response=show_student_response, show_next_problem_button=show_next_problem_button)


def set_baseline_display_parameters():
  math_chatbot_supplementary_text = ""
  show_student_response = True
  show_next_problem_button = False
  return math_chatbot_supplementary_text, show_next_problem_button, show_student_response


@app.route('/reset', methods=['GET'])
def reset():
  initialize_to_login_state()

  math_string_index = 0
  math_chatbot_supplementary_text, show_next_problem_button, show_student_response = set_baseline_display_parameters()

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
  show_student_response = False
  show_next_problem_button = False

  if request.method == 'POST':
    student_login = request.form['student_login']
    session['student_login'] = student_login

    initialize_to_login_state()

    math_string_index = 0
    math_chatbot_supplementary_text, show_next_problem_button, show_student_response = set_baseline_display_parameters()

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
      local_response = llm_interfaces.OPENAI_CLIENT.responses.create(
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

