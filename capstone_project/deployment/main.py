from pathlib import Path
import pandas as pd

from common import EVALUATION_DEBUG_LEVEL, DEBUG_LEVEL, math_string_dfs
from evaluation import execute_evaluation
from flask_state_machine import run_state_machine
from llm_interfaces import initialize_llm_apis

# Setup debugging and connection parameters

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

def execute_main():
  for posixPath in Path.cwd().glob("*.json"):
    print("File path:  %s" % posixPath)
    math_string_df = pd.read_json(posixPath)
    math_string_dfs.append(math_string_df)

  if DEBUG_LEVEL > 0 and len(math_string_dfs) > 0:
    print(math_string_dfs[0].head())

  initialize_llm_apis()

  # The following code executes an interactive session with a user of this math bot
  if DEBUG_LEVEL < EVALUATION_DEBUG_LEVEL:
    if __name__ == '__main__':
      run_state_machine(math_string_dfs)
    return

  # The following code executes an evaluation (scoring) of this math bot
  execute_evaluation(math_string_dfs)

  return


execute_main()
