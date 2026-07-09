import os
import pickle
import hashlib
from typing import List
from langchain_core.documents import Document
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.config_handler import chroma_conf


class BM25StoreService:
    def __init__(self, bm25_index_path: str = None):
        self.bm25_index_path = bm25_index_path or get_abs_path("bm25_index.pkl")
        self.documents: List[Document] = []
        self.bm25 = None
        self._tokenized_corpus = []
        self._init_bm25()

    def _init_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            self._BM25Okapi = BM25Okapi
        except ImportError:
            logger.warning("[BM25]rank_bm25未安装，BM25检索将不可用")
            self._BM25Okapi = None
            return

        if os.path.exists(self.bm25_index_path):
            self._load_index()
        else:
            logger.info("[BM25]索引文件不存在，待构建")

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def build_index(self, documents: List[Document]):
        if self._BM25Okapi is None:
            logger.warning("[BM25]rank_bm25未安装，跳过索引构建")
            return

        if not documents:
            logger.warning("[BM25]文档列表为空，跳过索引构建")
            return

        self.documents = documents
        self._tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = self._BM25Okapi(self._tokenized_corpus)
        self._save_index()
        logger.info(f"[BM25]索引构建完成，共{len(documents)}篇文档")

    def _save_index(self):
        try:
            data = {
                "documents": self.documents,
                "tokenized_corpus": self._tokenized_corpus,
            }
            with open(self.bm25_index_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"[BM25]索引已保存到 {self.bm25_index_path}")
        except Exception as e:
            logger.error(f"[BM25]保存索引失败：{str(e)}")

    def _load_index(self):
        try:
            with open(self.bm25_index_path, "rb") as f:
                data = pickle.load(f)
            self.documents = data["documents"]
            self._tokenized_corpus = data["tokenized_corpus"]
            if self._BM25Okapi:
                self.bm25 = self._BM25Okapi(self._tokenized_corpus)
            logger.info(f"[BM25]索引加载成功，共{len(self.documents)}篇文档")
        except Exception as e:
            logger.error(f"[BM25]加载索引失败：{str(e)}")
            self.documents = []
            self._tokenized_corpus = []
            self.bm25 = None

    def retrieve(self, query: str, top_n: int = 20) -> List[Document]:
        if self.bm25 is None or not self.documents:
            logger.warning("[BM25]索引未构建，返回空结果")
            return []

        try:
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = self.documents[idx]
                    metadata = dict(doc.metadata) if doc.metadata else {}
                    metadata["bm25_score"] = float(scores[idx])
                    results.append(Document(page_content=doc.page_content, metadata=metadata))
            logger.info(f"[BM25]检索完成，返回{len(results)}篇文档")
            return results
        except Exception as e:
            logger.error(f"[BM25]检索失败：{str(e)}", exc_info=True)
            return []

    def rebuild_if_needed(self, documents: List[Document]):
        current_md5 = hashlib.md5(
            "".join(doc.page_content for doc in documents).encode("utf-8")
        ).hexdigest()

        md5_file = get_abs_path("bm25_md5.text")
        stored_md5 = ""
        if os.path.exists(md5_file):
            with open(md5_file, "r", encoding="utf-8") as f:
                stored_md5 = f.read().strip()

        if current_md5 != stored_md5:
            logger.info("[BM25]文档内容有变化，重建索引")
            self.build_index(documents)
            with open(md5_file, "w", encoding="utf-8") as f:
                f.write(current_md5)
        else:
            logger.info("[BM25]文档无变化，复用已有索引")

    def is_available(self) -> bool:
        return self.bm25 is not None and len(self.documents) > 0


if __name__ == "__main__":
    from rag.vector_store import VectorStoreService
    vs = VectorStoreService()
    all_docs = vs.get_all_documents()
    print(f"获取到 {len(all_docs)} 篇文档")

    bm25_service = BM25StoreService()
    bm25_service.rebuild_if_needed(all_docs)

    results = bm25_service.retrieve("扫地机器人 故障", top_n=5)
    for i, doc in enumerate(results):
        print(f"\n--- 结果{i+1} (score: {doc.metadata.get('bm25_score', 'N/A')}) ---")
        print(doc.page_content[:100])
