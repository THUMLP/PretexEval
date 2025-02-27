import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
from tqdm import tqdm
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from vllm import LLM, SamplingParams

# prepare input text for each test case
def prepare_text(tmp_input, dev_data, neg, key, num_examples,tokenizer):
    new_len = -1
    while new_len <= 0:
        demonstration = prepare_examples(dev_data, neg, key, num_examples)
        input_text = demonstration + tmp_input
        new_len = 2048-len(tokenizer.tokenize(input_text))
        if new_len > 10:
            new_len = 10
        num_examples -= 1
    return input_text

# generate responses based on the input texts
def generate(prompts, model):
    sampling_params = SamplingParams(
    temperature=0,
    max_tokens=50,
    stop=['\nStatement']
    )
    # preds= tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    outputs = model.generate(
        prompts,
        sampling_params
        )
    replys = []
    for output in outputs:
        generated_text = output.outputs[0].text
        replys += [generated_text]
    print(prompts[0])
    print(replys[0])
    return replys


def load_data(path):
    return json.load(open(path,'r'))

# prepare few shot examples for each test case
def prepare_examples(dev_data,idx,key,nums):
    examples = dev_data[key]
    tmp = ''
    for i,one in enumerate(examples[:nums]):
        ques = one[3][idx]
        ans = one[4][idx]
        
        if isinstance(ans,list):
            ans = ['true' if w == 'T' else 'false' for w in ans]
            ans = '、'.join(ans)
        else:
            ans = 'true' if ans == 'T' else 'false'
        tmp += ques + '\nAnswer: '+ans+'\n\n'

    return tmp

def main(args):
    # Load the evaluated model and tokenizer
    model = LLM(model=args.model,tensor_parallel_size=args.num_cuda, gpu_memory_utilization=0.95,max_model_len=2048,swap_space=8,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    
    # Load the dev and test data
    dev_path = os.path.join('data/medlamatf_dev_gen.json')
    forward_path = os.path.join('data/medlamatf_test_gen.jsonl')
    dev_data = load_data(dev_path)
    test_data = []
    with open(forward_path,'r',encoding='utf8') as f:
        for line in f:
            while '\x00' in line:
                line = line.replace('\x00','')
            test_data.append(json.loads(line.strip()))

    out_data = []
    
    # Prepare the output file
    if not os.path.exists('results/{}'.format(args.type)):
        os.makedirs('results/{}'.format(args.type))
    f = open('results/{1}/{0}_{1}_results.json'.format(args.model_name,args.type),'a',encoding='utf8')
    cnt = -1
    # test_data = {'forward':forward_data}
    texts_list = []
    item_list = []
    # Prepare the input texts for each test case
    for item in tqdm(test_data):
        cnt += 1
        if cnt < args.start:
            continue
        item_list.append(item)
        key = item[2]
        questions = item[3]
        neg_questions = item[4]
        replys = []
        for j, ques in enumerate(questions):
            input_text = ques + '\nAnswer:'
            prompt = prepare_text(input_text, dev_data, j, key, args.ntrain, tokenizer)
            texts_list.append(prompt)
        for j, ques in enumerate(neg_questions):
            input_text = ques + '\nAnswer:'
            prompt = prepare_text(input_text, dev_data, j, key, args.ntrain, tokenizer)
            texts_list.append(prompt)
    cnt = 0
    temp_test_data = item_list
    # Generate responses for each test case
    res = generate(texts_list, model)
    # Output the results
    while len(res) >= 16:
        item = temp_test_data[cnt]
        replys = res[:8]
        neg_replys = res[8:16]
        out_data = item + [replys, neg_replys]
        res = res[16:]
        f.write(json.dumps(out_data,ensure_ascii=False)+'\n')
        f.flush()
        cnt += 1
    f.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--type",type=str, default='medlamatf_gen')
    # parser.add_argument("--tokenizer_path", type=str, default="/home/zhouyx/llama2/vicuna_13B")
    parser.add_argument("--model", type=str, default='YOUR_MODEL_PATH')
    parser.add_argument("--model_name", type=str, default='YOUR_MODEL_NAME')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_cuda", type=int, default=1)
    # parser.add_argument("--max_tokens",type=int, default=2048)
    # parser.add_argument("--subjects", type=list, default=['cat_tf','isa_tf1'])
    args = parser.parse_args()
    main(args)

