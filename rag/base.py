from __future__ import annotations

from rag.runtime import RetrievalRuntime


class RAGModel:
    def __init__(self) -> None:
        self.runtime = RetrievalRuntime()

    def predict(self, questions: list[str]) -> list[str]:
        if not questions:
            return []
        retrieved = self.runtime.retrieve_many(questions)
        return self.runtime.answer_many(questions, retrieved)
