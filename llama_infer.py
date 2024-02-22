from transformers import LlamaForCausalLM, LlamaTokenizer

hf_access_token = "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"

tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", token=hf_access_token)
model = LlamaForCausalLM.from_pretrained("llama-2-7b-miniguanaco_5", load_in_8bit=True)

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
df = pd.read_csv('synthetic-gpt-3.5-turbo_politifact_partially_arbitrary_generation_politics_rumors_processed.csv')

from sklearn.model_selection import train_test_split
train_df, test_df, _,_ = train_test_split(df, df["label"], test_size=0.2, stratify=df['label'],random_state=42)
# test_df_h = pd.read_csv('synthetic-gpt-3.5-turbo_politifact_hallucination_processed.csv')

zero_shot = []
print('s')
prompt = "Given a 'passage' determine whether or not it is a piece of misinformation. Only output 'YES' or 'NO'. The 'passage' is: "
for i,n in enumerate(list(test_df["synthetic_misinformation"])):
    # print(f"{prompt}{n}")
    print(i)
    zero_shot.append(llama2_talk(f"{prompt}\n{n}"))
    # break


# for i in range(len(zero_shot)):
#     temp = zero_shot[i]
#     if(temp.find('YES')>temp.find('NO')):
#         zero_shot[i] = 'YES'
#     else:
#         zero_shot[i] = 'NO'

labels = []
for x in zero_shot:
    if(x.strip()=='YES'):
        labels.append(1)
    else:
        labels.append(0)

from sklearn.metrics import accuracy_score

print(accuracy_score(test_df['label'].values, labels))
