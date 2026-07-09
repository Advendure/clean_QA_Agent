"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
优化版本：混合检索(BM25+稠密向量) + RRF融合 + ReRanker精排
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from rag.bm25_store import BM25StoreService
from rag.reranker import RerankerService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.config_handler import rag_conf, chroma_conf
from utils.logger_handler import logger


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()

        self.hybrid_conf = rag_conf.get("hybrid_search", {})
        self.hybrid_enable = self.hybrid_conf.get("enable", True)
        self.bm25_top_n = self.hybrid_conf.get("bm25_top_n", 50)
        self.dense_top_n = self.hybrid_conf.get("dense_top_n", 50)
        self.fusion_top_n = self.hybrid_conf.get("fusion_top_n", 20)

        self.rrf_conf = rag_conf.get("rrf", {})
        self.rrf_k = self.rrf_conf.get("k", 60)

        self.bm25_service = BM25StoreService()
        self.reranker_service = RerankerService()
        self._bm25_built = False

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def _ensure_bm25_built(self):
        if not self._bm25_built and self.hybrid_enable:
            try:
                all_docs = self.vector_store.get_all_documents()
                if all_docs:
                    self.bm25_service.rebuild_if_needed(all_docs)
                    self._bm25_built = True
                    logger.info(f"[RAG]BM25索引就绪，共{len(all_docs)}篇文档")
            except Exception as e:
                logger.error(f"[RAG]BM25索引构建失败：{str(e)}", exc_info=True)

    def _dense_retrieve(self, query: str, top_n: int = None) -> list[Document]:
        k = top_n or self.dense_top_n
        retriever = self.vector_store.get_retriever(k=k)
        docs = retriever.invoke(query)
        for i, doc in enumerate(docs):
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata["dense_rank"] = i + 1
            doc.metadata = metadata
        return docs

    def _bm25_retrieve(self, query: str) -> list[Document]:
        if not self.bm25_service.is_available():
            return []
        return self.bm25_service.retrieve(query, top_n=self.bm25_top_n)

    def _rrf_fusion(
        self,
        dense_docs: list[Document],
        bm25_docs: list[Document],
    ) -> list[Document]:
        doc_scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense_docs):
            key = self._doc_key(doc)
            score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + score
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = self._doc_key(doc)
            score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + score
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        top_n = min(self.fusion_top_n, len(sorted_keys))

        fused_docs = []
        for key in sorted_keys[:top_n]:
            doc = doc_map[key]
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata["rrf_score"] = doc_scores[key]
            fused_docs.append(Document(page_content=doc.page_content, metadata=metadata))

        logger.info(
            f"[RAG]RRF融合完成：稠密{len(dense_docs)}篇 + BM25{len(bm25_docs)}篇 → {len(fused_docs)}篇"
        )
        return fused_docs

    def _doc_key(self, doc: Document) -> str:
        source = doc.metadata.get("source", "") if doc.metadata else ""
        page = doc.metadata.get("page", "") if doc.metadata else ""
        content = doc.page_content[:50]
        return f"{source}_{page}_{content}"

    def retriever_docs(self, query: str) -> list[Document]:
        self._ensure_bm25_built()

        if not self.hybrid_enable or not self.bm25_service.is_available():
            logger.info("[RAG]混合检索未启用，使用纯稠密向量检索")
            dense_docs = self._dense_retrieve(query)
            if self.reranker_service.is_available():
                return self.reranker_service.rerank(query, dense_docs)
            return dense_docs[:chroma_conf.get("k", 3)]

        dense_docs = self._dense_retrieve(query)
        bm25_docs = self._bm25_retrieve(query)

        if not bm25_docs:
            logger.warning("[RAG]BM25无结果，降级为纯稠密检索")
            if self.reranker_service.is_available():
                return self.reranker_service.rerank(query, dense_docs)
            return dense_docs[:chroma_conf.get("k", 3)]

        fused_docs = self._rrf_fusion(dense_docs, bm25_docs)

        if self.reranker_service.is_available():
            return self.reranker_service.rerank(query, fused_docs)

        top_k = chroma_conf.get("k", 3)
        return fused_docs[:top_k]

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
