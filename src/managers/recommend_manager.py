from typing import Dict, List, Any, Optional

import pandas as pd

from ..models import VllmEmbedding, VllmReranker
from ..databases import FaissIndex


class RecommendationManager:
    def __init__(
        self,
        embedding: VllmEmbedding,
        reranker: VllmReranker,
        index: FaissIndex,
        score_column_name: str,
        rerank_top_k: int,
    ) -> None:
        self.embedding = embedding
        self.reranker = reranker

        self.index = index
        self.index.load()

        self.score_column_name = score_column_name
        self.rerank_top_k = rerank_top_k

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedding(query=query)
        candidates = self.index.search(query_embedding=query_embedding)
        return candidates

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not candidates:
            return None

        target_candidates = [candidate["chunk"] for candidate in candidates]
        scores = self.reranker(
            query=query,
            candidates=target_candidates,
        )

        for candidate, score in zip(candidates, scores):
            candidate[self.score_column_name] = float(score)

        candidates.sort(
            key=lambda x: x[self.score_column_name],
            reverse=True,
        )
        reranked_candidates = candidates[: self.rerank_top_k]
        return reranked_candidates

    def retrieve_and_rerank(
        self,
        input_value: str,
    ) -> Dict[str, Any]:
        query = input_value

        candidates = self.retrieve(query=query)
        if candidates is None:
            reranked_result = {
                "reranked_candidates": None,
                "query": query,
            }
            return reranked_result

        reranked_candidates = self.rerank(
            query=query,
            candidates=candidates,
        )
        if reranked_candidates is None:
            reranked_result = {
                "reranked_candidates": None,
                "query": query,
            }
            return reranked_result

        reranked_result = {
            "reranked_candidates": reranked_candidates,
            "query": query,
        }
        return reranked_result

    def recommend(
        self,
        input_value: str,
    ) -> str:
        reranked_result = self.retrieve_and_rerank(
            input_value=input_value,
        )

        reranked_candidates = reranked_result["reranked_candidates"]

        if reranked_candidates is None:
            return None

        return reranked_candidates
