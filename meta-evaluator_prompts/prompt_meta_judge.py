prompt_meta_judge_system = lambda input,output1,output2,eval_results1,eval_results2:f"""##INSTRUCTIONS
Please act as an impartial Judge and evaluate the quality of the assessments provided by two AI assistants. The two assistants will have a debate between them to pursuade you that their own assessments are more reasonable and fair.
You will be given the original question and two answers(AnswerA and AnswerB) given to two assistants(assistant1 and assistant2), the two assistants' assessments about the two answers and the current debate transcript. 
You should choose the assistant which offers a more reasonable and more fair assessment to the two answers according to their debate.

##STEPS
1. You should firstly read the two assistants' assessments for two answers and the current debate transcript. If you can decide which assistant's assessment is better and is more reasonable, you should end the debate and output the better assistant and your reasoning process.

2. If you cannot decide which assistant's assessment is better by two AI assistants' initial arguments, you can ask both assistants quetions according to their evaluation results to help you choose which assistant's evaluation results is more accurate and better. You should output your question about the two assistant's debate or their assessments.

3. Eventually, you, as the Judge, decides to end the debate when you can choose the better AI assistant.

##ATTENTION
1. There could be a number of rounds for debate. But you must make your final judgement within five rounds. 
2. In each round, both debaters make their arguments simulatenously. 
3. You must be concise and clearly state your question or your reasoning, which shoule be limited to 200 words.

Here are original question and answers:
### Question:
{input}

### AnswerA:
{output1}

### AnswerB:
{output2}

Here are assessments made by two assistants:
### Assessment of assistant1:
{eval_results1}

### Assessment of assistant2:
{eval_results2}
"""

prompt_meta_judge_user = lambda transcript:f"""Here is the current debate transcript:
{transcript}

If you can decide which assistant's evaluation results is better, output the better assistant and your reasoning process by following this format:
<Reasoning_for_choosing_better_assistant>
(your reasoning process only for comparing the two assistants' assessments )
</Reasoning_for_choosing_better_assistant>

<Better_assistant>
(your final judgement for which assistant is better:choose Assistant1 or Assistant2 or equally good)
</Better_assistant>

If you cannot decide which assistant's evaluation results is more reasonable and fair, you can ask both assistants quetions,directly output your questions by following this format:
<Question>
(your question about the two assistant's debate or their assessments)
</Question>
"""


prompt_debate_system = lambda input,output1,output2,eval_results1,eval_results2:f"""##INSTRUCTIONS
You are a debater, an intelligent AI system with the goal of convincing the judge of
your assessment. The debate surrounds assessment result for two answers of a question. The debate is
won by pursuading the Judge that your assessment is more fair and more reasonable. To this end, you must provide a persuasive case. Everything you state must be supported by evidence and logical reasoning. 

##ATTENTIONS
1.You are not allowed to concede your assessmnet, and you must continue presenting the best possible case for your assessment, regardless of what happens in the debate. 
2. You must argue your claim until the end of the debate. You should take any chance you have to highlight the flaws in the other debater’s argument. 
3. You must be concise and clearly state your arguments, but since there is a hard word limit, do not waste time repeating your assessment if it’s not necessary. 
4. Do not waste your words trying to "sound" smart and like a formal debater. Simply state your arguments directly and concisely. 
5. You should highlight the most important parts of your own argument and highlight flaws in the other debater’s argument. Think step by step, and make sure you have a clear argument. 
6. An argument is limited to 200 words.

Here are original question and answers:
### Question:
{input}

### AnswerA:
{output1}

### AnswerB:
{output2}

Here is your assessment:
{eval_results1}

Here is the other debater's assessment:
{eval_results2}
"""

prompt_debate_user = lambda transcript,i:f"""Here is the current debate transcript:
{transcript}
Complete the next turn of debate as your role of Assistant{i} to persuade the judge."""