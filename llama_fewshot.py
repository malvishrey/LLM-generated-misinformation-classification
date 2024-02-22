from transformers import LlamaForCausalLM, LlamaTokenizer

hf_access_token = "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"

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_access_token)
model = LlamaForCausalLM.from_pretrained("output_llama2", load_in_8bit=True)

def llama2_talk(text):
    gen_len = 2000
    generation_kwargs = {
          "max_new_tokens": gen_len,
          "top_p": 0.9,
          "temperature": 0.8,
          "repetition_penalty": 1.2,
          "do_sample": True,
      }

    B_INST, E_INST = "[INST]", "[/INST]"

    prompt_text = text

    prompt = f"{B_INST} {prompt_text} {E_INST}"  # Special format reuired by the Llama2 Chat Model

    prompt_ids = tokenizer(prompt, return_tensors="pt")

    prompt_size = prompt_ids['input_ids'].size()[1]

    generate_ids = model.generate(prompt_ids.input_ids.to(model.device), **generation_kwargs)

    generate_ids = generate_ids.squeeze()

    response = tokenizer.decode(generate_ids.squeeze()[prompt_size+1:], skip_special_tokens=True).strip()

    return response

import pandas as pd
df = pd.read_csv('synthetic-gpt-3.5-turbo_politifact_paraphrase_generation_processed.csv')

from sklearn.model_selection import train_test_split
train_df, test_df, _,_ = train_test_split(df, df["label"], test_size=0.2, stratify=df['label'],random_state=42)
# test_df_h = pd.read_csv('synthetic-gpt-3.5-turbo_politifact_hallucination_processed.csv')

pos_sample = train_df[train_df['label']==1]
neg_sample = train_df[train_df['label']==0]
pos_subset = list(pos_sample.sample(n=5)['synthetic_misinformation'])
neg_subset = list(neg_sample.sample(n=5)['synthetic_misinformation'])

FS_prompt = ""
prompt = "Given a 'passage' determine whether or not it is a piece of misinformation. Only output 'YES' or 'NO'. The 'passage' is: "
count = 0
for i in range(4):
    print(len(pos_subset[i]))
    FS_prompt += "Example "+str(count)+": "+prompt + pos_subset[i][:400] + " . The output is 'YES'.\n"
    count += 1
    FS_prompt += "Example "+str(count)+": "+prompt + neg_subset[i][:400] + ". The output is 'NO'.\n"
    count += 1
    print(len(FS_prompt))
FS_prompt += "Using the above 8 annotated examples, answer the following question: \n"

fs_res = []
# prompt = "Given a 'passage', please think step by step and then determine whether or not it is a piece of misinformation. You need to output your thinking process and answer “YES” or “NO”. The “passage” is:"
prompt = "Given a 'passage' determine whether or not it is a piece of misinformation. Only output 'YES' or 'NO'. The 'passage' is: "
print(FS_prompt+ prompt+test_df.iloc[0]['synthetic_misinformation'] )
print()
print()
for i in range(len(test_df)):
    # print(f"{prompt}{n}")
    fs_res.append(llama2_talk(FS_prompt+ prompt+test_df.iloc[i]['synthetic_misinformation'] ))
    print(i,fs_res[i])

print(fs_res)

for i in range(len(fs_res)):
    temp = fs_res[i]
    if(temp.find('YES')>temp.find('NO')):
        fs_res[i] = 'YES'
    else:
        fs_res[i] = 'NO'

labels = []
for x in fs_res:
    if(x=='YES'):
        labels.append(1)
    else:
        labels.append(0)

from sklearn.metrics import accuracy_score

print(accuracy_score(test_df['label'].values, labels))
