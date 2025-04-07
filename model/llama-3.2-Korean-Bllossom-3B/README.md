---
base_model:
- meta-llama/Meta-Llama-3.2-3B
language:
- en
- ko
library_name: transformers
license: llama3.2
---
 

<a href="https://github.com/MLP-Lab/Bllossom">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64a90711c05da19ca834f690/a0VE5UCY1HCEhaHtp3mGa.png" alt="image" width="30%" height="30%">
</a>

# Update!
* [2024.12.06] 훨씬 강력한 최신 Bllossom-AICA-5B로 업데이트 되었습니다 [링크](https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-AICA-5B)
* [2024.10.08] Bllossom-3B 모델이 최초 업데이트 되었습니다.


# Bllossom | [Demo]() | [Homepage](https://www.bllossom.ai/) | [Github](https://github.com/MLP-Lab/Bllossom) |

```bash
저희 Bllossom 팀에서 Bllossom-3B 모델을 공개합니다.
llama3.2-3B가 나왔는데 한국어가 포함 안되었다구?? 이번 Bllossom-3B는 한국어가 지원되지 않는 기본 모델을 한국어-영어로 강화모델입니다.
 - 100% full-tuning으로 150GB의 정제된 한국어로 추가 사전학습 되었습니다. (GPU많이 태웠습니다)
 - 굉장히 정제된 Instruction Tuning을 진행했습니다.
 - 영어 성능을 전혀 손상시키지 않은 완전한 Bilingual 모델입니다.
 - Instruction tuning만 진행했습니다. DPO 등 성능 올릴 방법으로 튜닝해보세요.
 - MT-Bench, LogicKor 등 벤치마크 점수를 잘받기 위해 정답데이터를 활용하거나 혹은 벤치마크를 타겟팅 해서 학습하지 않았습니다. (해당 벤치마크 타게팅해서 학습하면 8점도 나옵니다...)

언제나 그랬듯 해당 모델은 상업적 이용이 가능합니다.

1. Bllossom은 AAAI2024, NAACL2024, LREC-COLING2024 (구두) 발표되었습니다.
2. 좋은 언어모델 계속 업데이트 하겠습니다!! 한국어 강화를위해 공동 연구하실분(특히논문) 언제든 환영합니다!! 
```



```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
instruction = "철수가 20개의 연필을 가지고 있었는데 영희가 절반을 가져가고 민수가 남은 5개를 가져갔으면 철수에게 남은 연필의 갯수는 몇개인가요?"

messages = [
    {"role": "user", "content": f"{instruction}"}
    ]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=1024,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))
```
```
철수가 20개의 연필을 가지고 있었고 영희가 절반을 가져가면, 영희가 가져간 연필의 갯수는 20 / 2 = 10개입니다.

이제 철수가 남은 연필의 갯수를 계산해보겠습니다. 영희가 10개를 가져간 후 철수가 남은 연필의 갯수는 20 - 10 = 10개입니다.

민수가 남은 5개를 가져갔으므로, 철수가 남은 연필의 갯수는 10 - 5 = 5개입니다. 

따라서 철수가 남은 연필의 갯수는 5개입니다.
```

## Supported by

 - AICA  <img src="https://aica-gj.kr/images/logo.png" width="20%" height="20%">

## Citation
**Language Model**
```text
@misc{bllossom,
  author = {ChangSu Choi, Yongbin Jeong, Seoyoon Park, InHo Won, HyeonSeok Lim, SangMin Kim, Yejee Kang, Chanhyuk Yoon, Jaewan Park, Yiseul Lee, HyeJin Lee, Younggyun Hahm, Hansaem Kim, KyungTae Lim},
  title = {Optimizing Language Augmentation for Multilingual Large Language Models: A Case Study on Korean},
  year = {2024},
  journal = {LREC-COLING 2024},
  paperLink = {\url{https://arxiv.org/pdf/2403.10882}},
 },
}
```

**Vision-Language Model**
```text
@misc{bllossom-V,
  author = {Dongjae Shin, Hyunseok Lim, Inho Won, Changsu Choi, Minjun Kim, Seungwoo Song, Hangyeol Yoo, Sangmin Kim, Kyungtae Lim},
  title = {X-LLaVA: Optimizing Bilingual Large Vision-Language Alignment},
  year = {2024},
  publisher = {GitHub},
  journal = {NAACL 2024 findings},
  paperLink = {\url{https://arxiv.org/pdf/2403.11399}},
 },
}
```

## Contact
 - 임경태(KyungTae Lim), Professor at Seoultech. `ktlim@seoultech.ac.kr`
 - 함영균(Younggyun Hahm), CEO of Teddysum. `hahmyg@teddysum.ai`
 - 김한샘(Hansaem Kim), Professor at Yonsei. `khss@yonsei.ac.kr`

## Contributor
 - **유한결(Hangyeol Yoo)**, hgyoo@seoultech.ac.kr
 - 최창수(Chansu Choi), choics2623@seoultech.ac.kr