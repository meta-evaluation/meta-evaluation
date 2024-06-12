import json
import random
import re
import time
from collections import defaultdict,Counter
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Iterable, Optional, Dict, Tuple, Callable, List, Union
from meta_evaluator_prompts.prompt_meta_evaluator import prompt_meta_evaluator_pairwise,prompt_meta_evaluator_pairwise_mt,prompt_meta_evaluator_pairwise_llmbar
from meta_evaluator_prompts.prompt_meta_judge import prompt_meta_judge_system,prompt_meta_judge_user,prompt_debate_system, prompt_debate_user
from get_models_response import GPT_new,get_openai_response
from get_llm_evaluation import get_g4_response,get_g35_response
# from judge_debate_system import get_meta_eval_debate_once,get_debate_once
# from get_qwen_7b_response import get_qwen_7b_response
# from get_qwen_14b_chat_response import get_qwen_14b_chat_response
# from get_qwen_72b_response import get_qwen_72b_response
def concatenate_results(winner,explanation):
    if winner == '0':
        result = "I think the two answers are equal good. Here is my explanation:" + explanation
    else:
        result = f"I think the answer {winner} is better, Here is my explanation:{explanation}"
    return result
def get_meta_eval_debate_once(question,answer1,answer2,result1,result2,model_name,transcript=''):
    system_prompt = prompt_meta_judge_system(question,answer1,answer2,result1,result2)
    user_prompt = prompt_meta_judge_user(transcript)
    if model_name=="gpt35":
        # return get_openai_response(prompt,"gpt-35-turbo")
        return get_g35_response(system_prompt,user_prompt)
    elif model_name=="gpt4":
        return get_g4_response(system_prompt,user_prompt)
    elif model_name=="qwen72b":
        response, time_consum = get_qwen_72b_response(user_prompt,system_prompt)
        return response
    elif model_name=="qwen14b":
        return get_qwen_14b_chat_response(system_prompt,user_prompt)
    elif model_name=="qwen7b":
        return get_qwen_7b_response(system_prompt,user_prompt)
    elif model_name=="mistral":
        return get_mistral_response(system_prompt,user_prompt)

def get_debate_once(question,answer1,answer2,result1,result2,model_name,i,transcript=''):
    system_prompt = prompt_debate_system(question,answer1,answer2,result1,result2)
    user_prompt = prompt_debate_user(transcript,i)
    if model_name=="gpt35":
        # return get_openai_response(prompt,"gpt-35-turbo")
        return get_g35_response(system_prompt,user_prompt)
    elif model_name=="gpt4":
        return get_g4_response(system_prompt,user_prompt)
    elif model_name=="qwen72b":
        response, time_consum = get_qwen_72b_response(user_prompt,system_prompt)
        return response
    elif model_name=="qwen14b":
        return get_qwen_14b_chat_response(system_prompt,user_prompt)
    elif model_name=="qwen7b":
        return get_qwen_7b_response(system_prompt,user_prompt)
    elif model_name=="mistral":
        return get_mistral_response(system_prompt,user_prompt)
def get_meta_eval_once(question,answer1,answer2,result1,result2,model_name,draw_flag = True):
    if draw_flag:
        prompt = prompt_meta_evaluator_pairwise_mt(question,answer1,answer2,result1,result2)
    else:
        prompt = prompt_meta_evaluator_pairwise_llmbar(question,answer1,answer2,result1,result2)
    # prompt = prompt_meta_evaluator_pairwise(question,answer1,answer2,result1,result2)
    if model_name=="gpt35":
        # return get_openai_response(prompt,"gpt-35-turbo")
        return get_g35_response(prompt)
    elif model_name=="gpt4":
        return get_g4_response(prompt)
    elif model_name=="qwen72b":
        response, time_consum = get_qwen_72b_response(prompt)
        return response
    elif model_name=="qwen14b":
        return get_qwen_14b_chat_response(prompt)
    elif model_name=="qwen7b":
        return get_qwen_7b_response(prompt)
    elif model_name=="llama7b":
        return get_llama_7b_response(prompt)
    elif model_name=="llama13b":
        return get_llama_13b_response(prompt)
    elif model_name=="llama70b":
        return get_llama_70b_response(prompt)
def get_eval_results(eval_files_list):
    eval_results = {}
    i = 1
    for eval_file in eval_files_list:
        eval_results[f"model_{i}"] = []
        with open(eval_file, 'r') as f:
            for line in f:
                # print(line)
                data = json.loads(line)
                eval_results[f"model_{i}"].append(data)
        i += 1
    return eval_results
def get_content_between_a_b(a,b,text):
    pattern = f"{a}(.*?){b}"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def get_model_id(model_name):
    if model_name == "gpt35":
        return "model_1"
    elif model_name == "gpt4":
        return "model_2"
    elif model_name == "qwen7b":
        return "model_3"
    elif model_name == "qwen14b":
        return "model_4"
    elif model_name == "qwen72b":
        return "model_5"
    elif model_name == "mistral":
        return "model_6"
def get_match_history(file):
    history = []
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            evaluator1 = data["evaluator1"]
            evaluator2 = data["evaluator2"]
            q_id = data["question_id"]
            to_add = (evaluator1, evaluator2, q_id)
            history.append(to_add)
    return history

def random_select_evaluator(evaluators_list,eval_results,meta_evaluator,output_file,history=[],round =3,draw_flag = True):
    # history = []
    meta_eval_results = []
    evaluators = {}
    max_time = 500
    for evaluator in evaluators_list:
        evaluators[evaluator] = {"rating":"","picked_time":0}
    with open(output_file, 'a') as f:
        for i in range(round):
            # history.append([])
            print(len(eval_results["model_1"]))
            for j in range(362,len(eval_results["model_1"])):
                print(j)
                evaluator_1 = random.choice(evaluators_list)
                evaluator_2 = random.choice([e for e in evaluators_list if e != evaluator_1])
                to_add_history = (evaluator_1, evaluator_2,j)
                while to_add_history in history or (evaluator_2,evaluator_1,j)in history:
                    evaluator_1 = random.choice(evaluators_list)
                    evaluator_2 = random.choice([e for e in evaluators_list if e != evaluator_1])
                    while evaluators[evaluator_1]["picked_time"]>500:
                        evaluator_1 = random.choice(evaluators_list)
                    while evaluators[evaluator_2]["picked_time"]>500 or evaluator_1 == evaluator_2:
                        evaluator_2 = random.choice([e for e in evaluators_list if e != evaluator_1])
                    to_add_history = (evaluator_1, evaluator_2,j)
                evaluators[evaluator_1]["picked_time"] += 1
                evaluators[evaluator_2]["picked_time"] += 1
                history.append(to_add_history)
                evaluator_1_id = get_model_id(evaluator_1)
                evaluator_2_id = get_model_id(evaluator_2)
                question_1 = eval_results[evaluator_1_id][j]["question"]
                answer1_1 = eval_results[evaluator_1_id][j]["answer1"]
                answer2_1 = eval_results[evaluator_1_id][j]["answer2"]
                label_1 = eval_results[evaluator_1_id][j]["label"]
                question_2 = eval_results[evaluator_2_id][j]["question"]
                answer1_2 = eval_results[evaluator_2_id][j]["answer1"]
                answer2_2 = eval_results[evaluator_2_id][j]["answer2"]
                label_2 = eval_results[evaluator_2_id][j]["label"]
                assert question_1 == question_2 and answer1_1 == answer1_2 and answer2_1 == answer2_2 and label_1 == label_2
                # instance_id = eval_results[evaluator_1_id][j]["instance_id"]
                result_winner_evaluator_1 = eval_results[evaluator_1_id][j]["winner"]
                result_explanation_evaluator_1 = eval_results[evaluator_1_id][j]["explanation"]
                result_evaluator_1 = concatenate_results(result_winner_evaluator_1,result_explanation_evaluator_1)
                result_winner_evaluator_2 = eval_results[evaluator_2_id][j]["winner"]
                result_explanation_evaluator_2 = eval_results[evaluator_2_id][j]["explanation"]
                result_evaluator_2 = concatenate_results(result_winner_evaluator_2,result_explanation_evaluator_2)
                # meta_eval_result = get_meta_eval_debate_once(question_1,answer1_1,answer2_1,result_evaluator_1,result_evaluator_2,meta_evaluator,draw_flag)
                # print(meta_eval_result)
                transcript = ''
                meta_eval_result = get_meta_eval_debate_once(question_1,answer1_1,answer2_1,result_evaluator_1,result_evaluator_2,meta_evaluator,transcript)
                transcript = transcript + '\n' + 'Judge:' + meta_eval_result
                print(f"judge:{meta_eval_result}")
                round_num = 0
                while ("Better_assistant" not in meta_eval_result or  "<Question>" in meta_eval_result)and round_num<4:
                    # print(f"judge:{meta_eval_result}")
                    # transcript = transcript + '\n' + 'Judge:' + meta_eval_result
                    id = 1
                    time.sleep(5)
                    debater_1 = get_debate_once(question_1,answer1_1,answer2_1,result_evaluator_1,result_evaluator_2,meta_evaluator,id,transcript)
                    if not isinstance(debater_1,str):
                        try_time = 0
                        while try_time<3:
                            debater_1 = get_debate_once(question_1,answer1_1,answer2_1,result_evaluator_1,result_evaluator_2,meta_evaluator,id,transcript)
                            if isinstance(debater_1,str):
                                break
                            else:
                                try_time += 1
                    try:
                        pattern = r'\[/INST\](.*?)$'
                        match = re.search(pattern, debater_1,re.DOTALL)
                        if match:
                            debater_1 = match.group(1).strip()
                    except Exception:
                        pass
                    transcript = transcript + '\n' +"Assistant1:"+ debater_1
                    print(f"debater1:{debater_1}")
                    id = 2
                    time.sleep(2)
                    debater_2 = get_debate_once(question_1,answer1_1,answer2_1,result_evaluator_1,result_evaluator_2,meta_evaluator,id,transcript)
                    # print(f"debater2: {debator2}")
                    if not isinstance(debater_2,str):
                        try_time = 0
                        while try_time<3:
                            debater_2 = get_debate_once(question_1,answer1_1,answer2_1,result_evaluator_1,result_evaluator_2,meta_evaluator,id,transcript)
                            if isinstance(debater_2,str):
                                break
                            else:
                                try_time += 1
                    try:
                        pattern = r'\[/INST\](.*?)$'
                        match = re.search(pattern, debater_2,re.DOTALL)
                        if match:
                            debater_2 = match.group(1).strip()
                    except Exception:
                        pass
                    transcript = transcript + '\n' +"Assistant2:"+ debater_2
                    print(f"debater2: {debater_2}")
                    time.sleep(2)
                    meta_eval_result = get_meta_eval_debate_once(question_1,answer1_1,answer2_1,result_evaluator_1,result_evaluator_2,meta_evaluator,transcript)
                    transcript = transcript + '\n' + 'Judge:' + meta_eval_result
                    print(f"judge:{meta_eval_result}")
                    # print(f"judge:{meta_eval_result}")
                    # print(f"debater1:{debater1}")
                    # print(f"debater2: {debator2}")
                    round_num += 1
                try:
                    explanation = get_content_between_a_b("<Reasoning_for_choosing_better_assistant>","</Reasoning_for_choosing_better_assistant>",meta_eval_result)[0].strip()
                except IndexError:
                    explanation = None
            
                try:
                    winner = get_content_between_a_b("<Better_assistant>","</Better_assistant>",meta_eval_result)[0].strip()
                except IndexError:
                    winner = None
                f.write(json.dumps({"evaluator_1":evaluator_1,"evaluator_2":evaluator_2,"meta_evaluator":meta_evaluator,'winner': winner,"explanation":explanation,"transcript":transcript,"question":question_1,"answer1":answer1_1,"answer2":answer2_1,"evaluator1":result_evaluator_1,"evaluator2":result_evaluator_2,"label":label_1, "final_meta_eval_result":meta_eval_result,"question_id":j}) + '\n')
                # meta_eval_results.append((evaluator_1, evaluator_2,j,winner))
                # f.write(json.dumps({"evaluator_1":evaluator_1,"evaluator_2":evaluator_2,"better_response":better_response,"winner":winner,"explanation_meta_eval":explanation_meta_eval,"explanation_eval":explanation_eval,"label":label_1,"question":question_1,"answer1":answer1_1,"answer2":answer2_1,"result_evaluator_1":result_evaluator_1,"result_evaluator_2":result_evaluator_2,"meta_eval_result":meta_eval_result})+'\n')
                # 
            
    # return meta_eval_results

def simulate_elo_system(meta_eval_results,evaluator_list = ["gpt35","gpt4","qwen7b","qwen14b","qwen72b","mistral"],init_rating = 1000,k_factor = 4,base: int = 10,
                 scale: int = 400):
        error = 0.0
        players = {}
        for evaluator in evaluator_list:
            players[evaluator] = init_rating
        num_matches = len(meta_eval_results)
        for match_info in meta_eval_results: 
            # print(match_info)
            ra, rb = players[match_info[0]], players[match_info[1]]
            # print(ra,rb)
            ea = 1 / (1 + base ** ((rb - ra) / scale))
            eb = 1 / (1 + base ** ((ra - rb) / scale))
            # print(ea,eb)
            # print(match_info[2])
            if match_info[2] == "Assistant1" or match_info[2] == "assistant1" or match_info[2] == "Assistant 1" or match_info[2] == "assistant 1":
                s = 1.0
            elif match_info[2] == "Assistant2" or match_info[2] == "assistant2" or match_info[2] == "Assistant 2" or match_info[2] == "assistant 2":
                s = 0.0
            else:
                s = 0.5
            error += (s - ea) ** 2 + (1 - s - eb) ** 2 
            players[match_info[0]] += k_factor * (s - ea)
            players[match_info[1]] += k_factor * (1 - s - eb)

        
        return players, error / num_matches

def stablelize_rating(history,num_permutations= 100):
    history = deepcopy(history)
    ratings = defaultdict(int)
    players_new = {}
    ratings = {}
    for _ in range(num_permutations):
        random.shuffle(history)
        players,error = simulate_elo_system(history)
        for key in players:
            # in case some players do not participate in the pairwise comparisons
            ratings[key] = ratings.get(key,0)
            ratings[key] += players[key]
    for key in ratings:
        players_new[key] = ratings[key] / num_permutations

    return players_new

def get_records(history):
    records = defaultdict(Counter)
    draw_counter = defaultdict(Counter)
    for turn in history:
        player_A = turn[0]
        player_B = turn[1]
        if turn[2]=="Assistant1" or turn[2]=="assistant1" or turn[2]=="Assistant 1" or turn[2]=="assistant 1" :
            records[player_A][player_B] += 1  # 玩家A与玩家B之间的对战次数加1
        elif turn[2]=="Assistant2" or turn[2]=="assistant2" or turn[2]=="Assistant 2" or turn[2]=="assistant 2":
            records[player_B][player_A] += 1
        else:
            draw_counter[player_A][player_B] += 1
            draw_counter[player_B][player_A] += 1
    return records,draw_counter
def visualize_pairwise_win_rate(records: Dict[str, Counter], 
                                draw_counter: Optional[Dict[str, Counter]] = None, 
                                path: Optional[Union[str, Path]] = None):

    player_names = list(records.keys())
    if len(player_names) == 0:
        return None 
    
    num_players = len(player_names)
    matrix = [[0.0] * num_players for _ in range(num_players)]

    for i in range(num_players):
        for j in range(i + 1, num_players):
            total_rounds = records[player_names[i]][player_names[j]] + records[player_names[j]][player_names[i]]
            if draw_counter is not None:
                total_rounds += draw_counter[player_names[i]][player_names[j]]
            if total_rounds == 0:
                continue
            matrix[i][j] = records[player_names[i]][player_names[j]] / total_rounds 
            matrix[j][i] = records[player_names[j]][player_names[i]] / total_rounds

    figure = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(matrix, annot=True, cmap="rainbow_r", 
                     vmin=0.0, vmax=1.0, xticklabels=player_names, yticklabels=player_names)
    ax.set_title("⚔️ Pairwise Win Rate for Each Model ⚔️")

    if path is not None:
        figure.savefig(path, bbox_inches='tight', dpi=300)
    # print("test")
    return ax

def visualize_match_distribution(records: Dict[str, Counter], 
                                 draw_counter: Optional[Dict[str, Counter]] = None,
                                 path: Optional[Union[str, Path]] = None):

    player_names = list(records.keys())
    if len(player_names) == 0:
        return None 
    
    num_players = len(player_names)
    matrix = [[0.0] * num_players for _ in range(num_players)]

    for i in range(num_players):
        num_matches = sum(records[player_names[i]][player_names[j]] + records[player_names[j]][player_names[i]] for j in range(num_players))
        if draw_counter is not None:
            num_matches += sum(draw_counter[player_names[i]][player_names[j]] for j in range(num_players))
        if num_matches == 0:
            continue
        for j in range(num_players):
            pairwise_num_matches = records[player_names[i]][player_names[j]] + records[player_names[j]][player_names[i]]
            if draw_counter is not None:
                pairwise_num_matches += draw_counter[player_names[i]][player_names[j]]
            matrix[i][j] = pairwise_num_matches / num_matches

    figure = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(matrix, annot=True, cmap="winter", 
                     vmin=0.0, vmax=1.0, xticklabels=player_names, yticklabels=player_names)
    
    ax.set_title("😊 Match Distribution for Each Model")

    if path is not None:
        figure.savefig(path, bbox_inches='tight', dpi=300)
    # print("test")
    return ax

def visualize_match_num(records: Dict[str, Counter], 
                                 draw_counter: Optional[Dict[str, Counter]] = None,
                                 path: Optional[Union[str, Path]] = None):

    player_names = list(records.keys())
    if len(player_names) == 0:
        return None 
    
    num_players = len(player_names)
    matrix = [[0.0] * num_players for _ in range(num_players)]

    for i in range(num_players):
        for j in range(num_players):
            pairwise_num_matches = records[player_names[i]][player_names[j]] + records[player_names[j]][player_names[i]]
            if draw_counter is not None:
                pairwise_num_matches += draw_counter[player_names[i]][player_names[j]]
            matrix[i][j] = pairwise_num_matches

    figure = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(matrix, annot=True, cmap="winter", fmt='.0f',
                     vmin=0, vmax=150, xticklabels=player_names, yticklabels=player_names)
    
    ax.set_title("😊 Match number for Each Model")

    if path is not None:
        figure.savefig(path, bbox_inches='tight', dpi=300)
    # print("test")
    return ax
def save_elo_score(score_dict,filepath):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    scores = list(score_dict.values())
    scores_new = [round(i,3) for i in scores]
    model_name = list(score_dict.keys())
    ax.table(cellText=[scores_new], cellLoc='center', rowLabels=['Score'], colLabels=model_name,loc='center',rowColours=['pink'])                       
    # ax.set_title("😊 Pairwise Accuracy for Each Model")
    fig.savefig(filepath)
    plt.show()
def get_history(filepath):
    history = []
    with open(filepath,'r')as f:
        for line in f:
            data = json.loads(line)
            history.append((data["evaluator_1"],data["evaluator_2"],data["winner"]))
    return history

if __name__ =="__main__":
    evaluators_list = ["gpt35","gpt4","qwen7b","qwen14b","qwen72b","mistral"]
    eval_files_mt_list = ["/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/gpt_3.5_turbo_mtbench_qid_model_instance.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/gpt4_mtbench.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/qwen1.5_7_chat_mtbench_qid_model_instance.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/qwen1.5_14b_chat_mtbench_qid_model_instance.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/qwen1.5_72b_mtbench_qid_model_instance.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/mistral_mtbench_qid_model_instance.jsonl"]
    eval_files_llmbar_list = ["/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/gpt35_llmbar_no_draw.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/gpt4_llmbar_no_draw.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/qwen7b_llmbar_no_draw.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/qwen14b_llmbar_no_draw.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/qwen72b_llmbar_no_draw.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/mistral_llmbar_no_draw_after_pro.jsonl"]
    eval_files_all = ["/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/merged_eval_results_gpt-3.5-turbo_pairwise.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/merged_eval_results_gpt-4-1106-preview_pairwise.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/merged_eval_results_qwen1.5-7B-Chat_pairwise.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/merged_eval_results_qwen1.5-14B-Chat_pairwise.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/merged_eval_results_qwen1.5-72B-Chat_pairwise.jsonl","/ssd2/wanghuilin04/meta_evalaute/evaluator_results/pairwise/merged_eval_results_Mixtral-8x7B-Instruct-v0.1_pairwise.jsonl"]
    # eval_results = get_eval_results(eval_files_mt_list)
    # eval_results = get_eval_results(eval_files_mt_list)
    eval_results = get_eval_results(eval_files_all)
    # history = get_match_history("/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/debate/qwen72b_all_self_debate_elo.jsonl")
    history = get_match_history("/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/debate/gpt4_all_self_debate_elo.jsonl")
    # meta_evaluator = "qwen7b"
    # meta_evaluator = "qwen14b"
    meta_evaluator = "qwen72b"
    meta_evaluator = "gpt4"
    output_file = f"meta_evaluator_results/pairwise/debate/{meta_evaluator}_all_self_debate_elo.jsonl"
    meta_eval_results = random_select_evaluator(evaluators_list,eval_results,meta_evaluator,output_file,history,round =1,draw_flag = True)
    files_mt = ["/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/g4_mt_elo.jsonl","/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/qwen7b_mtbench_elo_after_pro_for_answer.jsonl","/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/qwen14b_mtbench_elo_after_pro_again_for_answer.jsonl","/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/qwen72b_mtbench_elo.jsonl"]
    files_llmbar = ["/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/g4_llmbar_elo.jsonl","/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/qwen7b_llmbar_elo_after_pro_for_answer.jsonl","/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/qwen14b_llmbar_elo_after_pro_for_answer.jsonl","/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/qwen72b_llmbar_elo_new.jsonl"]
    # meta_eval_results_mt = get_history("/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/g4_mt_elo.jsonl")
    # meta_eval_results_llmbar = get_history("/ssd2/wanghuilin04/meta_evalaute/meta_evaluator_results/pairwise/g4_llmbar_elo.jsonl")
    # meta_eval_results_all = meta_eval_results_mt + meta_eval_results_llmbar
    # players = stablelize_rating(meta_eval_results_all)
    # print(players)
    # records,draw_counter = get_records(meta_eval_results_all)
    # # print(len(records))
    # visualize_pairwise_win_rate(records,draw_counter,"figures/final_results_pairwise/g4_all_winrate_new.png")
    # visualize_match_distribution(records,draw_counter,"figures/final_results_pairwise/g4_all_distribution_new.png")
    # visualize_match_num(records,draw_counter,"figures/final_results_pairwise/g4_all_match_num_new.png")
    # save_elo_score(players,"figures/final_results_pairwise/g4_all_elo_score_new.png")
    # elo_scores = []
    # for file1,file2 in zip(files_mt,files_llmbar):
    #     meta_eval_result_mt = get_history(file1)
    #     meta_eval_result_llmbar = get_history(file2)
    #     meta_eval_results_all = meta_eval_result_mt + meta_eval_result_llmbar
    #     players = stablelize_rating(meta_eval_results_all,1000)
    #     print(players)
    #     rounded_v = []
    #     for value in players.values():
    #         rounded_v.append(round(value,4))
    #     elo_scores.append(rounded_v)
    # fig, ax = plt.subplots(figsize=(16, 8))
    # ax.axis('tight')
    # ax.axis('off')
    # evaluators = ["gpt-3.5-turbo-0125","gpt-4-1106-preview","qwen1.5-7B-Chat","qwen1.5-14B-Chat","qwen1.5-72B-Chat","Mixtral-8x7B-Instruct-v0.1"]
    # meta_evaluators = ["gpt-4-1106-preview","qwen1.5-7B-Chat", "qwen1.5-14B-Chat","qwen1.5-72B-Chat"]
    # ax.table(cellText=elo_scores, cellLoc='center', rowLabels=meta_evaluators, colLabels=evaluators,loc='center')                       
    # # ax.set_title("😊 Pairwise Accuracy for Each Model")
    # fig.savefig("figures/final_results_pairwise/all_elo_score_new_1000.png")
       



    