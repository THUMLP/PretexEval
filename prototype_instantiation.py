'''
    ### Instantiation of the prototypes using triplets from the MedLAMA dataset
'''

import pandas as pd
import json
import random
import os
if not os.path.exists("data"):
    os.makedirs("data")
random.seed(48)
prototypes = pd.read_csv('medlama/data/medlama/prompts_pretex.csv')   # Predefined 
pids = prototypes['pid'].tolist()   # pid refers to a specific relation
prototypes_dict = {}
# Including the original prototype from MedLAMA and other 7 prototypes derived from the corresponding predicate equivalent transformations
for one in ['human_prompt','prompt2', 'prompt3', 'prompt4', 'prompt5','prompt6','prompt7','prompt8']:   
    for i, text in enumerate(prototypes[one].tolist()):
        pid = pids[i]   
        if pid not in prototypes_dict:
            prototypes_dict[pid] = []
        prototypes_dict[pid].append(text)

outf = open(f'data/medlamatf_test.jsonl','w')   # test set
# dev_outf = open(f'data/medlamaqa_dev.jsonl','w')
dev_out = {}

for name in pids:
    data = pd.read_csv(f'medlama/data/medlama/2021AA/{name}_1000.csv')
    head_names = data['head_name'].tolist()
    tail_names_list = data['tail_names'].tolist()
    all_tail_set = set()
    tail_names_list_1 = []
    for one in tail_names_list:
        tail_list = one.split('||')
        tail_list = [item.strip() for item in tail_list]
        for item in tail_list:
            all_tail_set.add(item)
        tail_names_list_1.append(tail_list)
    
    all_outs = []
    for head_name, tail_names in zip(head_names, tail_names_list_1):
        neg_pool = []
        tail_name = random.choice(tail_names)
        # for tail_name in tail_names:
        others = list(all_tail_set-set(tail_names))
        neg_option = random.choice(others)
        
        questions, neg_questions = [], []
        labels = ['T','F','T','F','T','F','T','F']  # labels decided by the predicate transformation
        neg_labels = ['F','T','F','T','F','T','F','T']
        
        # Statements generated from (head_name, relation, tail_name)
        for i in range(8):
            question = prototypes_dict[name][i]
            question = 'Statement: \": ' + question.replace('[X]',head_name) + '\", is the statement above true or false? Please answer true/false.'
            question = question.replace('[Y]',tail_name)
            questions.append(question)
        
        # Statements generated from (head_name, relation, neg_option)
        for i in range(8):
            question = prototypes_dict[name][i]
            question = 'Statement: \": ' + question.replace('[X]',head_name) + '\", is the statement above true or false? Please answer true/false.'
            question = question.replace('[Y]', neg_option)
            neg_questions.append(question)
    
        outs = [head_name, tail_name, name, questions, neg_questions, labels, neg_labels]
        all_outs.append(outs)
    random.shuffle(all_outs)
    dev_out[name] = all_outs[:5]
    for outs in all_outs[5:]:
        outf.write(json.dumps(outs)+'\n')
outf.close()
json.dump(dev_out, open(f'data/medlamatf_dev.json','w'),indent=4)
            
            