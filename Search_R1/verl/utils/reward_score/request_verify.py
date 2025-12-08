import asyncio
import re
from openai import AsyncOpenAI
# from openai import OpenAI
import random

SYSTEM_PROMPT = """
你是一个答案验证器, 下面我会给出Ground Truth和Solution, 请判断答案和回复是否相符, 二者可能存在格式或表述的不一致.
请只输出是或者否来判断是否匹配, 下面是一个例子:
Ground Truth: 12
Solution: 答案是24
你的输出:
<res>否</res>
"""

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        print(f"模型输出不正确, 没拆出来: \n{solution_str}")
        return ""
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


async def compute_score(solution_str, ground_truth, base_url: str = "http://10.52.99.19:8199/v1", method='strict', format_score=0., score=1.) -> float:
    """
    使用 OpenAI 官方异步 SDK 比较 solution_str 和 ground_truth，返回一个分数。
    """
    extract_solution_str = extract_solution(solution_str)
    golden_answer_str = ground_truth['target']

    # 初始化客户端
    client = AsyncOpenAI(
        api_key="",
        base_url=base_url
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ground Truth: {golden_answer_str}\nSolution: {extract_solution_str}\n你的输出:\n"}
    ]
    print(f"Ground Truth: {golden_answer_str}\nSolution: {extract_solution_str}\n")
    # do_print = random.randint(1, 64) == 1
    
    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Golden answers: {ground_truth['target']}")
    #     print(f"Extracted answer: {extract_solution_str}")
    #     print(f"Solution string: {solution_str}")

    try:
        response = await client.chat.completions.create(
            model="qwen",
            messages=messages,
            max_tokens=15,
        )

        reply = response.choices[0].message.content.strip()
        print(f"模型回复: {reply}")
        # 尝试提取分数
        match = re.search(r"<res>(.*?)</res>", reply)
        if match:
            content = match.group(1).strip()
            if "是" in content:
                return 1.0
            elif "否" in content:
                return 0.0
            
        # 兜底逻辑：如果模型没输出标签，直接判断关键字
        if "是" in reply:
            return 1.0
        elif "否" in reply:
            return 0.0
        else:
            print(f"模型胡言乱语, 没辙了: {reply}")
            return 0.0
            
    except Exception as e:
        print(f"OpenAI API 请求失败: {e}")
        return 0.0

# 测试方法
def test_compute_score():
    ground_truth = {"target": "42"}
    solution_correct = "<answer>答案是 42</answer>"
    solution_wrong = "<answer>答案是 100</answer>"
    
    print("--- 开始测试 ---")
    
    print(f"正在测试正确答案:")
    score1 = compute_score(solution_correct, ground_truth)
    print(f"评分结果: {score1}")
    
    print(f"\n正在测试错误答案:")
    score2 = compute_score(solution_wrong, ground_truth)
    print(f"评分结果: {score2}")

if __name__ == "__main__":
    test_compute_score()