---
license: mit
language:
- ko
- en
base_model:
- deepseek-ai/DeepSeek-R1-Distill-Llama-8B
library_name: transformers
---
# DeepSeek-llama3.1-Bllossom-8B

DeepSeek-Bllossom Series는 기존 DeepSeek-R1-Distill Series 모델의 language mixing, 다국어 성능 저하 문제를 해결하기 위해 추가로 학습된 모델입니다.

DeepSeek-llama3.1-Bllossom-8B는 DeepSeek-R1-distill-Llama-8B 모델을 베이스로 구축된 모델로, 한국어 환경에서의 추론 성능 향상을 목표로 개발되었습니다.

본 모델은 UNIVA와 Bllossom팀이 합작으로 제작한 첫 번째 모델입니다.

<div align="center">

| **Model** | **Base Model** | **Download** |
| :------------: | :------------: | :------------: |
| DeepSeek-qwen-Bllossom-1.5B  | [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 공개예정  |
| DeepSeek-qwen-Bllossom-7B  | [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 공개예정  |
| DeepSeek-llama3.1-Bllossom-8B  | [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | [🤗 HuggingFace](https://huggingface.co/UNIVA-Bllossom/DeepSeek-llama3.1-Bllossom-8B)   |
| DeepSeek-qwen-Bllossom-14B   | [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) | 공개예정   |
| DeepSeek-qwen-Bllossom-32B  | [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | 공개예정   |
| DeepSeek-llama3.3-Bllossom-70B  | [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | [🤗 HuggingFace](https://huggingface.co/UNIVA-Bllossom/DeepSeek-llama3.3-Bllossom-70B )   |

</div>




## 1. Introduction

DeepSeek-llama3.1-Bllossom-8B은 DeepSeek-R1-distill-Llama-8B 모델을 베이스로 구축된 모델로, 기존 베이스 모델이 영어와 중국어 위주의 데이터로 학습된 한계를 극복하고자 개발되었습니다. 특히, 기존 DeepSeek-R1-distill-Llama-8B의 경우 한국어로 추론 시 모델 성능이 크게 하락하는 문제가 있었는데, DeepSeek-Bllossom은 이 문제를 해결하기 위해 내부 사고 과정은 영어로 수행하고 최종 사용자에게 제공되는 응답은 입력 언어에 따라 출력되도록 추가로 학습되었습니다. 이를 통해 한국어 환경에서의 추론 성능이 크게 개선되었습니다.

학습에는 한국어, 영어 reasoning 데이터를 사용하였으며, 기존 DeepSeek-R1 모델 학습에 주로 사용된 STEM 분야 데이터 외에도 다양한 분야의 데이터가 포함되었습니다. 데이터셋 설계와 모델 학습 과정에서 DeepSeek-llama3.1-Bllossom-8B는 한국어 사용 환경에서 더 정확하고 신뢰할 수 있는 추론 결과를 제공하는 것을 주된 목표로 개발되었습니다.

---

## 2. Post-training

DeepSeek-llama3.1-Bllossom-8B는 자체적으로 제작한 다양한 reasoning 데이터를 활용하여 post-training 과정을 진행하였습니다. 이 과정에서는 대규모 모델이 보유한 우수한 reasoning 능력과 한국어 처리 능력을 DeepSeek-R1-distill-Llama-8B 모델에 효과적으로 distillation하는 방법을 적용하였습니다. 이를 통해 기존 모델의 성능을 보완하고, 복합적인 추론 문제에 대해 더 정확하며 신뢰할 수 있는 응답을 생성할 수 있도록 최적화하였습니다.

---

## 3. inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "UNIVA-Bllossom/DeepSeek-llama3.1-Bllossom-8B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("UNIVA-Bllossom/DeepSeek-llama3.3-Bllossom-70B")

system='''
You are a highly capable assistant. For every user question, follow these instructions exactly:
	1.	First, think through the problem step-by-step in English. Enclose all of your internal reasoning between <think> and </think> tags. This chain-of-thought should detail your reasoning process.
	2.	After the closing </think> tag, provide your final answer.
	3.	Do not include any additional text or commentary outside of this format.
	4.	Your output should strictly follow this structure:

<think>
[Your detailed step-by-step reasoning in English]
</think>
[Your final answer]
'''

text="철수, 영희, 민수가 3회의 게임에서 점수를 받았습니다. 영희의 점수는 민수의 점수의 두 배이며, 민수의 점수는 철수의 4배입니다. 철수가 10점을 받았다면 이 3명의 평균 점수를 계산하세요."
chat = [
    {"role": "system", "content": system},
    {"role": "user", "content": text}
]

prompt=tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=True
)

if "token_type_ids" in model_inputs:
    del model_inputs["token_type_ids"]

model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192,
)

```

## 4. 모델 응답 비교

| 입력 (Prompt) | DeepSeek-R1-distill-Llama-70B | DeepSeek-llama3.3-Bllossom-70B |
| ------------- | ----------------------------- | -------------------------- |
| **Alice, Bob, Charlie가 3회의 게임에서 점수를 받았습니다.<br>Alice의 점수는 Bob의 점수의 두 배이며, Bob의 점수는 Charlie의 4배입니다.<br>Charlie가 10점을 받았다면 이 3명의 평균 점수를 계산하세요.** |<br>Charlie의 점수는 10점입니다.<br>Bob의 점수는 Charlie의 4배이므로, Bob의 점수는 <br>$$<br>4 \\times 10 = 40<br>$$<br>점입니다.<br>Alice의 점수는 Bob의 점수의 두 배이므로, Alice의 점수는 <br>$$<br>2 \\times 40 = 80<br>$$<br>점입니다.<br>세人的 총 점수는 <br>$$<br>10 + 40 + 80 = 130<br>$$<br>점입니다.<br>평균 점수는 <br>$$<br>\\frac{130}{3} \\approx 43.33$점입니다.<br>$$<br>\\boxed{43.33}<br>$$<br>Charlie의 점수는 10점입니다. | <br>Charlie의 점수는 10점으로 주어졌습니다. Bob의 점수는 Charlie의 점수의 4배이며, 이는 다음과 같이 계산됩니다:<br>$$<br>4 \times 10 = 40<br>$$<br>Alice의 점수는 Bob의 점수의 두 배이며, 이는 다음과 같습니다:<br>$$<br>2 \times 40 = 80<br>$$<br>세 사람의 총 점수는 다음과 같습니다:<br>$$<br>10 + 40 + 80 = 130<br>$$<br>이 세 사람의 평균 점수는 총 점수를 사람 수로 나누어 계산됩니다:<br>$$<br>\frac{130}{3}<br>$$<br>따라서, 이 세 사람의 평균 점수는 <br>$$<br>\boxed{\dfrac{130}{3}}<br>$$<br>입니다. |
| **소수가 무한하다는 사실을 3가지 방법으로 증명하시오.** | <br>以下是用三种方法证明素数无限的详细步骤：<br>### 方法一：欧几里得的反证法<br>1. **假设**：假设素数是有限的，且所有素数为 $ p_1, p_2, p_3, \\ldots, p_n $。<br>2. **构造数**：构造一个新数 $ N = (p_1 \\times p_2 \\times p_3 \\times \\ldots \\times p_n) + 1 $。<br>3. **分析N的性质**：<br>   - 如果N是素数，则N不在原假设的素数列表中，与假设矛盾。<br>   - 如果N是合数，则存在某个素数 $ p $ 整除N。但由于N除以任何 $ p_i $ 都余1，故 $ p $ 不在原列表中，同样矛盾。<br>4. **结论**：因此，素数必须是无限的。<br>### 方法二：费马数的性质<br>1. **定义**：费马数形如 $ F_k = 2^{2^k} + 1 $，其中 $ k $ 为非负整数。<br>2. **假设**：假设素数是有限的，且所有素数为 $ p_1, p_2, \\ldots, p_n $。<br>3. **构造费马数**：对于每个 $ k $，构造 $ F_k $。如果 $ F_k $ 是素数，则不在原列表中；如果是合数，其素因数不在原列表中。<br>4. **结论**：因此，素数必须是无限的。<br>### 方法三：欧拉函数与素数无限性<br>1. **欧拉函数**：欧拉函数 $ \\phi(n) $ 表示小于 $ n $ 且与 $ n $ 互质的数的个数。<br>2. **假设**：假设素数有限，记为 $ p_1, p_2, \\ldots, p_n $。<br>3. **构造数**：考虑形如 $ N = p_1^{k_1} p_2^{k_2} \\ldots p_n^{k_n} $ 的数，其欧拉函数为 $ \\phi(N) = N \\prod_{i=1}^{n} \\left(1 - \\frac{1}{p_i}\\right) $。<br>4. **分析**：如果素数有限，$ \\prod_{p \\text{素数}} \\left(1 - \\frac{1}{p}\\right) $ 收敛，但实际上该积发散，导致矛盾。<br>5. **结论**：因此，素数必须是无限的。<br>这三种方法各自从不同的角度证明了素数的无限性，展示了数论中素数无限性的深刻性。 | <br>소수가 무한하다는 것을 3가지 방법으로 증명하기 위해, 우리는 다음과 같은 접근 방식을 사용할 수 있습니다:<br>1. **유클리드의 모순 증명**:<br>   - **가정**: 소수는 유한한 개수 $ p_1, p_2, \\dots, p_n $이라고 가정합니다.<br>   - **구성**: $ P = p_1 \\times p_2 \\times \\dots \\times p_n + 1 $으로 정의합니다.<br>   - **결론**: $ P $가 소수이면, 그것은 목록에 없는 새로운 소수입니다. 만약 $ P $가 합성수이면, 그것은 목록에 없는 소수를 포함해야 합니다. 두 경우 모두 가정에 모순됩니다. 따라서, 소수는 무한합니다.<br>2. **팩토리얼 접근법**:<br>   - **가정**: 소수는 유한한 개수 $ p_1, p_2, \\dots, p_n $이라고 가정합니다.<br>   - **구성**: $ n! + 1 $을 고려하십시오, 여기서 $ n $은 가장 큰 소수입니다.<br>   - **결론**: $ n! $은 모든 $ p_i $로 나누어 떨어지므로, $ n! + 1 $은 $ p_i $ 중 어느 것도 포함하지 않습니다. 만약 $ n! + 1 $이 소수이면, 그것은 $ n $보다 큰 새로운 소수입니다. 만약 $ n! + 1 $이 합성수이면, 그것은 $ n $보다 큰 소수를 포함해야 합니다. 두 경우 모두 가정에 모순됩니다. 따라서, 소수는 무한합니다.<br>3. **오일러의 제타 함수 증명**:<br>   - **개념**: 오일러는 제타 함수 $ \\zeta(s) = \\sum_{k=1}^\\infty \\frac{1}{k^s} $가 소수에 대한 곱으로 표현될 수 있음을 보였습니다: $ \\zeta(s) = \\prod_{p \\text{ 소수}} \\frac{1}{1 - \\frac{1}{p^s}} $.<br>   - **결론**: 만약 소수가 유한하면, 곱은 유한해야 할 것입니다. 그러나 $ s $가 1보다 큰 값을 향해 접근할 때, $ \\zeta(s) $는 무한대로 발산합니다. 이것은 곱이 발산해야 함을 요구하며, 따라서 소수는 무한해야 합니다.<br>각 방법은 소수가 무한하다는 것을 확증하며, 서로 다른 수학적 원칙에 기반을 둡니다. 유클리드의 증명은 모순을 사용합니다, 팩토리얼 접근법은 간단한 수론을 사용하며, 오일러의 증명은 해석적 수론의 도구를 사용합니다. 세 방법 모두 소수 무한성에 대한 이해를 강화합니다. |

## 5. Benchmark
- dtype을 float16으로 추론을 진행하였습니다.
- max_tokens: 32786
- temperature: 0.7
- 평가 방식: 각 벤치마크를 3회 반복 실행한 후 평균 점수를 산출하였습니다.
- _en 벤치마크: 원본 벤치마크 질문을 그대로 사용하였습니다.
- _ko 벤치마크: 원본 벤치마크 질문을 한국어로 고품질 번역하여 사용하였습니다.
  
| Model                                 | AIME24_ko | AIME24_en | MATH500_ko | MATH500_en |
|---------------------------------------|-----------|-----------|------------|------------|
| DeepSeek-R1-Distill-Llama-8B          | 25.56     | 46.67     | 63.40      | 88.87      |
| DeepSeek-llama3.1-Bllossom-8B         | 36.67     | 40.00     | 78.07      | 87.80      |
| DeepSeek-R1-Distill-Llama-70B         | 58.89     | 70.00     | 88.53      | 93.73      |
| DeepSeek-llama3.3-Bllossom-70B        | 62.22     | 65.56     | 88.40      | 93.33      |

## 6. License

This code repository and the model weights are licensed under the MIT License.
DeepSeek-Bllossom series support commercial use, allow for any modifications and derivative works, including, but not limited to, distillation for training other LLMs. Please note that:
- **DeepSeek-R1-Distill-Llama-8B** is derived from Llama3.1-8B-Instruct and is originally licensed under llama3.1 license.
- **DeepSeek-llama3.1-Bllossom-8B** is derived from DeepSeek-R1-Distill-Llama-8B and is originally licensed under llama3.1 license.

## 7. Contributor
- **UNIVA AI Team** ([UNIVA](https://univa.co.kr), Main contributor)
- 최창수 (서울과학기술대학교, [MLP연구실](https://sites.google.com/view/aailab) 석사과정)
- **임경태** (KAIST, [MLP연구실](https://sites.google.com/view/aailab) 교수)

## 8. Contact
If you have any questions, please raise an issue or contact us at [frodobaggins@univa.co.kr](frodobaggins@univa.co.kr) or [ktlim@seoultech.ac.kr](ktlim@seoultech.ac.kr).