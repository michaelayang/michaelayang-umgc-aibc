from common import CONNECT_TO_CHATGPT_API, EVALUATION_DEBUG_LEVEL, DEBUG_LEVEL, VAGUE_WRONG_ANSWER, \
  VAGUE_WRONG_REASONING, CONTROL_RUN
from utilities import justify_current_problem_solution_with_answers_to_previous_problems, \
    get_text_attributes_from_problem, chatgpt_api_supplies_leading_question, \
    ask_gemini_to_correlate_phrases_in_a_0_to_10_ranking, student_reasoning_is_relevant_to_problem
import llm_interfaces

def execute_evaluation(math_string_dfs):

  chatgpt_output_correlation_vector = np.array([0, 0, 0])
  perfect_correlation_vector = np.array([0, 0, 0])

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
    llm_interfaces.OPENAI_CLIENT.close()

  return


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
        local_response = llm_interfaces.OPENAI_CLIENT.responses.create(
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
