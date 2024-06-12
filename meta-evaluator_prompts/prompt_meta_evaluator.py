prompt_meta_evaluator_pairwise = lambda question, answer1,answer2,result1,result2 :f"""
Please act as an impartial judge and evaluate the quality of the evaluation results provided by two AI evaluators. 
You will be given the original question and two answers given to two evaluators, and the two evaluators' judgements about the two answers.The two answers are wriiten by human, not by two evaluators.
You should choose the evaluator which offers a more reasonable and more fair judgement to the two answers.
Begin your evaluation by comparing the better answer chosen by each AI evaluator and decide which evaluator did the right judgement. If they choose the same one,you should compare the two evaluators' reasons why they think that answer is better and choose the evaluator which has more reasonable explanation.
Avoid any position biases and ensure that the order in which the evaluator's judgements were presented does not influence your decision. 
Do not allow the length of the reason of evaluator to influence your evaluation. 
Be as objective as possible. 

Here are original question and answers:
### Question:
{question}

### Answer_1:
{answer1}

### Answer_2:
{answer2}

Here are judgements made by two evaluators:
### Judgement of evaluator 1:
{result1}

### Judgement of evaluator 2:
{result2}


Remember firstly you should provide a short explanation and then judge which evaluator is better. 
1 means evaluator is better, 2 means evaluator 2 is better, 0 means the two evaluator are equally good.
Your explanation should be between the tag <Explanation> and </Explanation> and your final judgement result should be between the tag <Result> and </Result>.

Remember that the two answers have nothing to do with the two evaluators. For example, Answer 1 is better DOESN'T mean evaluator 1 is better, Answer 2 is better DOESN'T mean evaluator 2 is better.You must judge two evaluators independently. 

Please output your judgement by strictly following this format:

<Explanation>
(your explanation for comparing the two evaluators' judgements )
</Explanation>

<Result>
(0 or 1 or 2)
</Result>
"""


# prompt_meta_evaluator_pairwise_new= lambda question, answer1,answer2,result1,result2 :f"""##INSTRUCTIONS
# Please act as an impartial judge and evaluate the quality of the evaluation results provided by two AI evaluators. 
# You will be given the original question and two answers given to two evaluators, and the two evaluators' assessments about the two answers. The two answers are wriiten by human, not by two evaluators.
# You should choose the evaluator which offers a more reasonable and more fair judgement to the two answers.

# ##STEPS
# 1.Read carefully the original question and the two answers given to two evaluators, choose the one you think is better and give your reason. 
# 2.Read the two evaluators' judgements. If both of them chose the right "Better answer", you should read their explanations why they made this choice, and judge which explanation is more reasonable. "equally good" means the two evaluators' explanations are equally reasonable, evaluator1 means the evaluator1's explanation is more reasonable than the evaluator's explanation. evaluator2 means the evaluator2's explannation is more reasonable than the evaluator1's explanation. If one of the evaluator chose the right "Better answer",and the other one chose the wrong one, you should choose the evaluator who chose the right "Better answer" as better evaluator and your reasoning should be that one evaluator made the right choice and the other one chose the wrong better response. If both of them chose the wrong "Better answer", you should output "equally good" and your reasoning should be that they both made the wrong judgement. If the "Better answer" is equally good, and both of two evaluators thought one answer is better than the other, no matter which answer they thought was better, you should output "equally good" because both of evaluatoes made the wrong judgement.

# Here are original question and answers:
# ### Question:
# {question}

# ### Answer_1:
# {answer1}

# ### Answer_2:
# {answer2}

# Here are assessments made by two evaluators:
# ### Assessment of evaluator 1:
# {result1}

# ### Assessment of evaluator 2:
# {result2}

# ##Output format
# <Reasoning_for_choosing_better_answer>
# (your reasoning process only for comparing the two answers)
# </Reasoning_for_choosing_better_answer>

# <Better_answer>
# (choose Answer1 or Answer2 or equally good) 
# </Better_answer>

# <Reasoning_for_choosing_better_evaluator>
# (your reasoning process only for comparing the two evaluators' assessments )
# </Reasoning_for_choosing_better_evaluator>

# <Better_evaluator>
# (your final judgement for which evaluator is better:choose evaluator1 or evaluator2 or equally good)
# </Better_evaluator>

# ## Attention
# 1.Avoid any position biases and ensure that the order in which the evaluator's judgements were presented does not influence your decision. 
# 2.Do not allow the length of the reason of evaluator to influence your evaluation. 
# 3.Be as objective as possible. 
# 4. Remember that the two answers have nothing to do with the two evaluators. For example, Answer 1 is better DOESN'T mean evaluator 1 is better, Answer 2 is better DOESN'T mean evaluator 2 is better.You must judge two evaluators independently. Your Reasoning_for_choosing_better_answer should only focus on the quality of two answers. Your Reasoning_for_choosing_better_evaluator should only focus on accuracy and correctness of two evaluators' evaluation results. 
# 5. Remember that your output must consists of four parts:

# <Reasoning_for_choosing_better_answer>
# (your reasoning process only for comparing the two answers)
# </Reasoning_for_choosing_better_answer>

# <Better_answer>
# (choose Answer1 or Answer2 or equally good) 
# </Better_answer>

# <Reasoning_for_choosing_better_evaluator>
# (your reasoning process only for comparing the two evaluators' assessments )
# </Reasoning_for_choosing_better_evaluator>

# <Better_evaluator>
# (your final judgement for which evaluator is better:choose evaluator1 or evaluator2 or equally good)
# </Better_evaluator>
# ."""

prompt_meta_evaluator_pairwise_mt= lambda question, answer1,answer2,result1,result2 :f"""##INSTRUCTIONS
Please act as an impartial judge and evaluate the quality of the assessments provided by two AI assistants. 
You will be given the original question and two answers(AnswerA and AnswerB) given to two assistants(assistant1 and assistant2), and the two assistants' assessments about the two answers. The two answers are wriiten by human, not by two assistants. 
You should choose the assistant which offers a more reasonable and more fair assessment to the two answers.

##STEPS
1.Read carefully the original question and the two answers given to two assistants, choose which answer you think is better and give your reason. 
2.Read the two assistants' assessments for two answers. If both of two assisstants chose the same better answer as you chose, you should read their explanations why they made this choice, and judge which explanation is more reasonable. "equally good" means the two assistsnts' explanations are equally reasonable, assistant1 means the assistant1's explanation is more reasonable than the assistant2's explanation. assistant2 means the assistant2's explannation is more reasonable than the assistant1's explanation. If one of the assistant chose the same better answer as you did ,and the other assistant chose differently from what you chose, you should choose the assistant who chose same better answer with you as better evaluator and your reasoning should be that one evaluator made the right choice and the other one chose the wrong better answer. If both of them chose differently from what you chose as better answer, no matter which answer they thought was better, you should output "equally good" and your reasoning should be that two assistants both made the wrong assessments. 

Here are original question and answers:
### Question:
{question}

### AnswerA:
{answer1}

### AnswerB:
{answer2}

Here are assessments made by two assistants:
### Assessment of assistant1:
{result1}

### Assessment of assistant2:
{result2}

## Attention
1.Avoid any position biases and ensure that the order in which the assistant's judgements were presented does not influence your decision. 
2.Do not allow the length of the reason of assistant to influence your evaluation. 
3.Be as objective as possible. 
4.The two answers have nothing to do with the two assistants.
5.Please output your judgement by strictly following this format:

<Reasoning_for_choosing_better_answer>
(your reasoning process only for comparing the two answers)
</Reasoning_for_choosing_better_answer>

<Better_answer>
(choose AnswerA or AnswerB or equally good) 
</Better_answer>

<Reasoning_for_choosing_better_assistant>
(your reasoning process only for comparing the two assistants' assessments )
</Reasoning_for_choosing_better_assistant>

<Better_assistant>
(your final judgement for which assistant is better:choose Assistant1 or Assistant2 or equally good)
</Better_assistant>
 """


prompt_meta_evaluator_pairwise_llmbar= lambda question, answer1,answer2,result1,result2 :f"""##INSTRUCTIONS
Please act as an impartial judge and evaluate the quality of the assessments provided by two AI assistants. 
You will be given the original question and two answers(AnswerA and AnswerB) given to two assistants(assistant1 and assistant2), and the two assistants' assessments about the two answers. The two answers are wriiten by human, not by two assistants. 
You should choose the assistant which offers a more reasonable and more fair assessment to the two answers.

##STEPS
1.Read carefully the original question and the two answers given to two assistants, choose which answer you think is better and give your reason. 
2.Read the two assistants' assessments for two answers. If both of two assisstants chose the same better answer as you chose, you should read their explanations why they made this choice, and judge which explanation is more reasonable. "equally good" means the two assistsnts' explanations are equally reasonable, assistant1 means the assistant1's explanation is more reasonable than the assistant2's explanation. assistant2 means the assistant2's explannation is more reasonable than the assistant1's explanation. If one of the assistant chose the same better answer as you did ,and the other assistant chose differently from what you chose, you should choose the assistant who chose same better answer with you as better evaluator and your reasoning should be that one evaluator made the right choice and the other one chose the wrong better answer. If both of them chose differently from what you chose as better answer, no matter which answer they thought was better, you should output "equally good" and your reasoning should be that two assistants both made the wrong assessments. 

Here are original question and answers:
### Question:
{question}

### AnswerA:
{answer1}

### AnswerB:
{answer2}

Here are assessments made by two assistants:
### Assessment of assistant1:
{result1}

### Assessment of assistant2:
{result2}

## Attention
1.Avoid any position biases and ensure that the order in which the assistant's judgements were presented does not influence your decision. 
2.Do not allow the length of the reason of assistant to influence your evaluation. 
3.Be as objective as possible. 
4.The two answers have nothing to do with the two assistants.
5.Please output your judgement by strictly following this format:

<Reasoning_for_choosing_better_answer>
(your reasoning process only for comparing the two answers)
</Reasoning_for_choosing_better_answer>

<Better_answer>
(choose AnswerA or AnswerB) 
</Better_answer>

<Reasoning_for_choosing_better_assistant>
(your reasoning process only for comparing the two assistants' assessments )
</Reasoning_for_choosing_better_assistant>

<Better_assistant>
(your final judgement for which assistant is better:choose Assistant1 or Assistant2 or equally good)
</Better_assistant>
 """

prompt_meta_evaluator_pointwise_story = lambda text,eval_result: f"""##INSTRUCTIONS
Please act as an impartial judge and evaluate the quality of the assessment provided by AI assistant. 
You will be given the original five-line story given to AI assistant, and the AI assistant's assessment including a score ranging from 1 to 5 and its reason for why giving such a score. 
You should assign a score from 1 to 5 to AI assistant's assesment, where 1 indicates the AI assistant's assessment is poor, and 5 indicates it is excellent.

##STEPS
1.Please read the story carefully and assign it a score from 1 to 5 for the story's overall quality,considering factors such as the coherence,fluency,consistency,interestingness and creativity of the short story.
2.Read the AI assistant's assessment carefully.Based on your own rating for story,you should judge whether the assistant's assessment is reasonable and assign a score from 1 to 5 to the AI assistant's assesment.

Here is the story:
### Story:
{text}

Here is the assessment made by AI assistant:
### Assessment:
{eval_result}

## Attention
1.Do not allow the length of the reason of assistant to influence your evaluation for AI assistant. 
2.Be as objective as possible. 
4.The five-line story has nothing to do with the AI assistant,the story is written by human,not by AI assistant.
5.Please output your judgement by strictly following this format:

<Reasoning_for_scoring_story>
(your reasoning process only for scoring the story)
</Reasoning_for_scoring_story>

<Score_for_story>
(assign a score from 1 to 5 to the story)  
</Score_for_story>

<Reasoning_for_scoring_Assistant>
(your reasoning process only for scoring assistant's assessment )
</Reasoning_for_scoring_Assistant>

<Score_for_Assistant>
(assign a score from 1 to 5 to the assistant's assessment)
</Score_for_Assistant>
 """


prompt_meta_evaluator_pointwise_summeval = lambda text,eval_result: f"""##INSTRUCTIONS
Please act as an impartial judge and evaluate the quality of the assessment provided by AI assistant. 
You will be given the original text given to AI assistant, and the AI assistant's assessment for the text's coherence aspect, including a score ranging from 1 to 5 and its reason for why giving such a score. 
You should assign a score from 1 to 5 to AI assistant's assesment, where 1 indicates the AI assistant's assessment is poor, and 5 indicates it is excellent.

##STEPS
1.Please read the text carefully and assign it a score from 1 to 5 for the text's coherence,You should focus solely on coherence and do not take into account any other factors.
2.Read the AI assistant's assessment carefully and assign a score from 1 to 5 to the AI assistant's assesment for the text's coherence

Here is the text:
### Text:
{text}

Here is the assessment made by AI assistant:
### Assessment:
{eval_result}

## Attention
1.Do not allow the length of the reason of assistant to influence your evaluation for AI assistant. 
2.Be as objective as possible. 
4.The text has nothing to do with the AI assistant,the text is written by human,not by AI assistant.
5.Please output your judgement by strictly following this format:

<Reasoning_for_scoring_text_coherence>
(your reasoning process only for scoring the story)
</Reasoning_for_scoring_text_coherence>

<Score_for_text_coherence>
(assign a score from 1 to 5 to the story)  
</Score_for_text_coherence>

<Reasoning_for_scoring_Assistant_assessment>
(your reasoning process only for scoring assistant's assessment )
</Reasoning_for_scoring_Assistant_assessment>

<Score_for_Assistant_assessment>
(assign a score from 1 to 5 to the assistant's assessment)
</Score_for_Assistant_assessment>
 """

prompt_meta_evaluator_pointwise_branch_story = f"""
We want to evaluate the quality of the assessments provided by two AI assistants for a five-line story. Your task is to propose an evaluation plan that can be executed to compare the two ssessments. The
evaluation plan should consist of a list of up to five factors that one should consider such as helpfulness,
accuracy, etc. In each line, write an evaluation criterion along with a short description of how we
should evaluate that criterion.

<Evaluation Plan>
(output your evaluation plan)
</Evaluation Plan>
"""
prompt_meta_evaluator_pointwise_solve_story = lambda text, evaluation,criterion: f"""
You are given a five-line story and assessment for the story provided by two AI assistant. Your task is to evaluate and score the quality of the assessment based on a single evaluation criterion displayed below. Make sure to evaluate only based on the criterion specified and none other. 
Begin your evaluation by providing a short explanation. After providing your explanation, you must rate the assessment on a scale of 1 to 5 based on the criterion specified.

Story:
{text}

Assessment provided by AI assistant:
{evaluation}

Evaluation Criterion:
{criterion}

Please output your judgement by strictly following this format:

<Explanation>
(write your explanation for evaluating the assessment)
</Explanation>

<Score>
(assign a score from 1 to 5 to the assessment based on the criterion specified)  
</Score>
"""
