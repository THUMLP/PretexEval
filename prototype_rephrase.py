'''
    ### generate the medlama questions based on each question prototype and LLM.
'''
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import re

dev_data = json.load(open('data/medlamatf_dev.json','r'))

# Load LLM for rephrasing the instantiated prototypes
model_name = 'YOUR_MODEL_NAME'
model_path = 'YOUR_MODEL_PATH'
tokenizer = AutoTokenizer.from_pretrained(model_path)
num_gpus = 2
llm = LLM(model=model_path,
    tensor_parallel_size=num_gpus,gpu_memory_utilization=0.95,max_model_len=4096
    )

# Prompt for rephrasing
prompt = 'Please paraphrase the following statement to present the same concept in a different way. DO NOT change the basic sentence structure. Directly output the paraphrased statement without other text. Statement: {}.'
terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
sampling_params = SamplingParams(temperature=0, stop_token_ids=terminators, max_tokens=2048)
gen_dev = True
if gen_dev:
    # dev
    print('Gen dev')
    new_dev = {}
    for key in tqdm(dev_data):
        item_list = dev_data[key]
        new_item_list = []
        for item in item_list:
            questions = item[3]
            neg_questions = item[4]
            option_parts = []
            prompts = []
            for question in questions:
                ques_part = re.search(r"Statement: \": (.*?)\", is the statement above true or false\? Please answer true/false.", question).group(1)
                input_text = prompt.format(ques_part)
                messages = [
                            {"role": "system", "content": "You are an experienced clinical expert."},
                            {"role": "user", "content": input_text}
                        ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)
            for question in neg_questions:
                ques_part = re.search(r"Statement: \": (.*?)\", is the statement above true or false\? Please answer true/false.", question).group(1)
                input_text = prompt.format(ques_part)
                messages = [
                            {"role": "system", "content": "You are an experienced clinical expert."},
                            {"role": "user", "content": input_text}
                        ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)
            responses = llm.generate(prompts,sampling_params)
            generated_text = [output.outputs[0].text for output in responses]
            new_questions = []
            for response in generated_text:
                ques = f"Statement: \": {response}\", is the statement above true or false\? Please answer true/false."
                new_questions.append(ques)
            new_item = item
            new_item[3] = new_questions[:8]
            new_item[4] = new_questions[8:]
            new_item_list.append(new_item)
        new_dev[key] = new_item_list
    with open(f'data/medlamatf_dev_gen.json','w') as f:
        json.dump(new_dev,f,indent=4)
# test
test_data = []
start = 0
with open('data/medlamatf_test.jsonl','r') as f:
    for line in f:
        test_data.append(json.loads(line.strip()))
outf = open(f'data/medlamatf_test_gen.jsonl','a')
print('Gen test')
cnt = -1
buffer = []
prompts = []
for item in tqdm(test_data):
    head = item[0]
    tail = item[1]
    rel = item[2]
    # if head == 'Stage 0 Oropharyngeal Carcinoma AJCC v6 and v7' and tail == 'Malignant Cell' and rel == 'disease_has_abnormal_cell':
    #     print('y')
    # elif head == 'Stage II Ureter Cancer AJCC v8' and tail == 'Neoplastic Epithelial Cell' and rel == 'disease_has_abnormal_cell':
    #     print('y')
    # else:
    #     continue
    questions = item[3]
    neg_questions = item[4]
    cnt += 1
    if cnt < start:
        continue
    option_parts = []
    for question in questions:
        ques_part = re.search(r"Statement: \": (.*?)\", is the statement above true or false\? Please answer true/false.", question).group(1)
        input_text = prompt.format(ques_part)
        messages = [
                    {"role": "system", "content": "You are an experienced clinical expert."},
                    {"role": "user", "content": input_text}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)
    for question in neg_questions:
        ques_part = re.search(r"Statement: \": (.*?)\", is the statement above true or false\? Please answer true/false.", question).group(1)
        input_text = prompt.format(ques_part)
        messages = [
                    {"role": "system", "content": "You are an experienced clinical expert."},
                    {"role": "user", "content": input_text}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)
    buffer.append(item)
    if len(prompts) >= 2048:
        counter = 0
        responses = llm.generate(prompts,sampling_params)
        generated_text = [output.outputs[0].text for output in responses]
        new_questions = []
        for response in generated_text:
            ques = f"Statement: \": {response}\", is the statement above true or false\? Please answer true/false."
            new_questions.append(ques)
        for one in buffer:
            one[3] = new_questions[counter:counter+8]
            one[4] = new_questions[counter+8:counter+16]
            outf.write(json.dumps(one)+'\n')
            outf.flush()
            counter += 16
        buffer = []
        prompts = []
counter = 0
responses = llm.generate(prompts,sampling_params)
generated_text = [output.outputs[0].text for output in responses]
new_questions = []
for response in generated_text:
    ques = f"Statement: \": {response}\", is the statement above true or false\? Please answer true/false."
    new_questions.append(ques)
for one in buffer:
    one[3] = new_questions[counter:counter+8]
    one[4] = new_questions[counter+8:counter+16]
    outf.write(json.dumps(one)+'\n')
    outf.flush()
    counter += 16
buffer = []
prompts = []    
outf.close()
