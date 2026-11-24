import time
from parse import parse
import pandas as pd
import re

from common import CONNECT_TO_CHATGPT_API, CONNECT_TO_GEMINI_API, REASONING_THRESHOLD, EVALUATION_DEBUG_LEVEL, \
  DEBUG_LEVEL
import llm_interfaces


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

      gemini_raw_response = llm_interfaces.GEMINI_SESSION.send_message(message)

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


def pick_best_chatgpt_feedback(problem_df, student_reasoning, chatgpt_feedback_list):
  # Find similarities between student reasoning + teacher notes, against a list
  # of chatgpt feedback items.

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
    local_response = llm_interfaces.OPENAI_CLIENT.responses.create(model="gpt-5.5",
                                                    instructions="You are a math teacher assistant.",
                                                    input=leading_question_prompt
                                                    )
    response_output = local_response.output_text

  elif DEBUG_LEVEL > 0 and DEBUG_LEVEL < EVALUATION_DEBUG_LEVEL:
    time.sleep(5)

  chatgpt_feedback_list = re.split(r'[0-9]+\.[ \t]+', response_output)
  if len(chatgpt_feedback_list) > 0:
    chatgpt_feedback_list = chatgpt_feedback_list[1:]

  best_chatgpt_feedback = pick_best_chatgpt_feedback(problem_df, student_reasoning, chatgpt_feedback_list)

  if DEBUG_LEVEL > 0:
    print("chatgpt_feedback_list:  %s" % chatgpt_feedback_list)
    print("best_chatgpt_feedback:  %s" % best_chatgpt_feedback)

  return best_chatgpt_feedback


def justify_current_problem_solution_with_answers_to_previous_problems(math_problems):
    return_string = ""
    for math_problem in math_problems[:len(math_problems)-1]:
      return_string += math_problem['problem'] + ' = ' + math_problem['answer'] + ', '

    return return_string


