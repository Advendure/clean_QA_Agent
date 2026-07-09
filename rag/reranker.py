import os
import requests
from typing import List
from langchain_core.documents import Document
from utils.logger_handler import logger
from utils.config_handler import rag_conf


class RerankerService:
    def __init__(self):
        reranker_conf = rag_conf.get("reranker", {})
        self.enable = reranker_conf.get("enable", True)
        self.provider = reranker_conf.get("provider", "dashscope")
        self.model = reranker_conf.get("model", "qwen3-rerank")
        self.top_k = reranker_conf.get("top_k", 3)
        self.top_n = reranker_conf.get("top_n", 20)
        self.api_key = None
        self._init_client()

    def _init_client(self):
        if not self.enable:
            logger.info("[ReRanker]精排未启用")
            return

        if self.provider == "dashscope":
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("[ReRanker]未配置DASHSCOPE_API_KEY，精排功能将不可用")
                self.enable = False
                return
            logger.info(f"[ReRanker]通义千问精排初始化完成，模型：{self.model}")
        else:
            logger.warning(f"[ReRanker]不支持的提供商：{self.provider}")
            self.enable = False

    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        if not self.enable or not self.api_key:
            logger.warning("[ReRanker]精排不可用，直接返回原始文档")
            return documents[: (top_k or self.top_k)]

        if not documents:
            return []

        target_k = top_k or self.top_k
        if len(documents) <= target_k:
            return documents

        try:
            return self._rerank_dashscope(query, documents, target_k)
        except Exception as e:
            logger.error(f"[ReRanker]精排失败：{str(e)}", exc_info=True)
            logger.warning("[ReRanker]精排失败，返回原始文档")
            return documents[:target_k]

    def _rerank_dashscope(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        doc_texts = [doc.page_content for doc in documents]

        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": doc_texts,
            },
            "parameters": {
                "top_n": min(top_k, len(documents)),
                "return_documents": False,
            },
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            try:
                error_info = response.json()
                error_msg = error_info.get("message", str(response.status_code))
            except Exception:
                error_msg = f"HTTP {response.status_code}"
            logger.error(f"[ReRanker]通义API返回错误：{error_msg}")
            return documents[:top_k]

        data = response.json()
        results = data.get("output", {}).get("results", [])

        reranked_docs = []
        for item in results:
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0)
            if idx < len(documents):
                doc = documents[idx]
                metadata = dict(doc.metadata) if doc.metadata else {}
                metadata["rerank_score"] = float(score)
                reranked_docs.append(Document(page_content=doc.page_content, metadata=metadata))

        if not reranked_docs:
            logger.warning("[ReRanker]精排结果为空，返回原始文档")
            return documents[:top_k]

        logger.info(f"[ReRanker]精排完成，{len(documents)} → {len(reranked_docs)}")
        return reranked_docs

    def is_available(self) -> bool:
        return self.enable and self.api_key is not None


if __name__ == "__main__":
    reranker = RerankerService()
    print(f"ReRanker可用: {reranker.is_available()}")

    test_docs = [
        Document(page_content="扫地机器人故障排查方法"),
        Document(page_content="如何保养扫地机器人"),
        Document(page_content="扫地机器人选购指南"),
    ]

    results = reranker.rerank("扫地机器人坏了怎么办", test_docs, top_k=2)
    for i, doc in enumerate(results):
        print(f"\n--- 结果{i+1} (score: {doc.metadata.get('rerank_score', 'N/A')}) ---")
        print(doc.page_content)
