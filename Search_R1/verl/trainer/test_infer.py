from openai import OpenAI

base_url = "http://10.52.99.19:8199/v1"
PRE_PROMPT = """
你是一个答案验证器, 我会给出Ground Truth和Solution, 请判断答案和回复是否相符, 二者可能存在格式或表述的不一致.
请只输出是或者否来判断是否匹配, 下面是一个例子:
Ground Truth: 12
Solution: 答案是24
你的输出:
<res>否</res>
下面请参考以上格式进行判断:
"""

golden_answer_str = "是的，2019年12月30日，京张高铁正式开通。"
extract_solution_str = "不是"

# 初始化客户端
client = OpenAI(
    api_key="",
    base_url=base_url
)

messages = [
    {"role": "user", "content": PRE_PROMPT + f"Ground Truth: {golden_answer_str}\nSolution: {extract_solution_str}\n你的输出:\n"}
]

response = client.chat.completions.create(
    model="qwen",
    messages=messages,
    max_tokens=1024,
)

reply = response.choices[0].message.content.strip()
print(f"模型回复: {reply}")