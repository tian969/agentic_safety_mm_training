import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import time
import hashlib
import datetime
import json
import asyncio
import aiohttp

# --- 配置部分 ---
AURORA_URL = "http://10.138.162.80:8820/baizhong/eb118/multi-search"
AK = "f0b204f0-4fc9-4ffb-672d-a3c1dedfe323"
SK = "b313ccb8-9f10-4e81-4e8c-9f9c3662805d"


class SearchRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 3
    return_scores: Optional[bool] = False


app = FastAPI(title="Baidu Search Server (Async)")


# --- 鉴权与工具函数 (保持不变) ---
def get_auth_token():
    timestamp = str(int(time.time() * 1000))
    ak_t_sk = "/".join([AK, timestamp, SK])
    aftersha256 = hashlib.sha256()
    aftersha256.update(ak_t_sk.encode('utf-8'))
    token = "/".join([AK, timestamp, aftersha256.hexdigest()])
    return token


def format_publish_time(timestamp_str):
    if not timestamp_str:
        return ""
    try:
        ts = int(timestamp_str)
        return str(datetime.datetime.fromtimestamp(ts).date())
    except Exception:
        return str(timestamp_str)


# --- 异步检索函数 ---
async def search_open_domain_async(session: aiohttp.ClientSession, query: str, size: int = 10) -> List[Dict]:
    """
    异步版本的检索函数
    """
    headers = {
        'Authorization': get_auth_token(),
        'Content-Type': 'application/json',
    }
    payload = {
        'queryList': [[query]],
        'style': 'all',
        'isSafe': '1',
        'size': size,
        'minSScore': '0.0',
        "passway": "max_aiapi"
    }

    try:
        # 使用 aiohttp 发送异步 POST 请求
        async with session.post(AURORA_URL, json=payload, headers=headers, timeout=5) as response:
            response.raise_for_status()
            resp_json = await response.json()  # 异步读取 JSON

            if "baizhong" not in resp_json:
                print(f"Warning: 'baizhong' key not found for query: {query}")
                return []

            data = resp_json["baizhong"]
            results = []

            for item in data:
                source_data = item.get("_source", {})
                content_text = source_data.get("answer") or source_data.get("content") or ""

                entry = {
                    "title": source_data.get("title", ""),
                    "content": content_text,
                    "url": item.get("_id", ""),
                    "publish_time": format_publish_time(item.get("_publish_time", "")),
                    "score": item.get("_score", 0),
                    "source_name": item.get("_data_source_name", "OpenWeb")
                }
                results.append(entry)
            return results

    except Exception as e:
        print(f"Async Request Error for {query}: {e}")
        return []


# --- 异步 API 端点 ---
@app.post("/retrieve")
async def retrieve_endpoint(request: SearchRequest):
    # 创建一个 ClientSession 用于复用连接
    async with aiohttp.ClientSession() as session:
        tasks = []
        # 1. 创建所有查询的任务
        for query in request.queries:
            tasks.append(search_open_domain_async(session, query, size=request.topk))

        # 2. 并发执行所有任务 (并行等待)
        raw_results_list = await asyncio.gather(*tasks)

        # 3. 格式化结果 (CPU 密集型操作，很快，无需异步)
        batch_results = []
        for raw_results in raw_results_list:
            formatted_results = []
            for item in raw_results:
                title = item.get("title", "No Title")
                content = item.get("content", "")
                if not content:
                    content = "No content available."

                doc_entry = {
                    "document": {
                        "contents": f"\"{title}\"\n{content}"
                    }
                }
                if request.return_scores:
                    doc_entry["score"] = item["score"]
                formatted_results.append(doc_entry)
            batch_results.append(formatted_results)

    return {"result": batch_results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)