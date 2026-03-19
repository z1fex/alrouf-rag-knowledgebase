"""Answer generation for the RAG pipeline.

Provides two implementations:
- MockGenerator: Pattern-matches queries against pre-written responses in JSON.
- OpenAIGenerator: Calls the OpenAI Chat API with retrieved context and citation
  instructions.
"""

import json
import os
from abc import ABC, abstractmethod

from app.models.schemas import ChunkMetadata, Citation


class Generator(ABC):
    """Abstract base class for answer generators."""

    @abstractmethod
    def generate(
        self,
        query: str,
        chunks: list[tuple[ChunkMetadata, float]],
        language: str = "en",
    ) -> tuple[str, list[Citation]]:
        """Generate an answer with citations from retrieved chunks.

        Args:
            query: The user's question.
            chunks: Retrieved chunks with similarity scores.
            language: Response language ('en' or 'ar').

        Returns:
            Tuple of (answer_text, list_of_citations).
        """
        ...


class MockGenerator(Generator):
    """Pattern-matching generator using pre-written responses from JSON.

    Falls back to concatenating retrieved chunk texts if no pattern matches.
    """

    def __init__(self, mocks_dir: str = "mocks"):
        self.responses_path = os.path.join(mocks_dir, "sample_responses.json")
        self._patterns: list[dict] = []
        self._out_of_scope: dict = {}
        self._load_responses()

    def _load_responses(self) -> None:
        """Load pattern-response mappings from JSON file."""
        if os.path.exists(self.responses_path):
            with open(self.responses_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._patterns = data.get("patterns", [])
            self._out_of_scope = data.get("out_of_scope", {})

    def _find_best_pattern(self, query: str) -> dict | None:
        """Find the pattern with the most keyword matches for a query."""
        query_lower = query.lower()
        best_match = None
        best_score = 0

        for pattern in self._patterns:
            score = sum(1 for kw in pattern["keywords"] if kw in query_lower)
            if score > best_score:
                best_score = score
                best_match = pattern

        return best_match if best_score > 0 else None

    def generate(
        self,
        query: str,
        chunks: list[tuple[ChunkMetadata, float]],
        language: str = "en",
    ) -> tuple[str, list[Citation]]:
        """Generate a mock answer by pattern matching or chunk concatenation."""
        # Build citations from retrieved chunks
        citations = [
            Citation(
                document_title=chunk.document_title,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                source_file=chunk.source_file,
                relevance_score=round(score, 4),
            )
            for chunk, score in chunks
        ]

        # Try pattern matching
        pattern = self._find_best_pattern(query)
        if pattern:
            key = f"response_{language}"
            answer = pattern.get(key, pattern.get("response_en", ""))
            return answer, citations

        # Fallback: summarize from chunk texts
        if chunks:
            context_parts = []
            for i, (chunk, _score) in enumerate(chunks, 1):
                context_parts.append(f"[{i}] {chunk.text[:500]}")
            context = "\n\n".join(context_parts)

            if language == "ar":
                answer = f"بناءً على المعلومات المتاحة:\n\n{context}"
            else:
                answer = f"Based on the available information:\n\n{context}"
            return answer, citations

        # No chunks at all
        return self._out_of_scope.get(f"response_{language}", "No information found."), citations


class OpenAIGenerator(Generator):
    """OpenAI Chat API-based generator with citation instructions."""

    SYSTEM_PROMPT_EN = """You are a helpful assistant for Alrouf Lighting Technology (الروف لتقنية الإضاءة), a Saudi Arabian outdoor lighting manufacturer.

Rules:
- Answer based ONLY on the provided context chunks. Do not use external knowledge.
- Cite your sources using [1], [2], [3] markers that reference the context chunks.
- If the context does not contain enough information to answer, say so honestly.
- Be concise but thorough. Use bullet points and tables when helpful.
- Respond in English."""

    SYSTEM_PROMPT_AR = """أنت مساعد ذكي لشركة الروف لتقنية الإضاءة، شركة سعودية لتصنيع الإضاءة الخارجية.

القواعد:
- أجب بناءً على السياق المقدم فقط. لا تستخدم معلومات خارجية.
- استشهد بمصادرك باستخدام علامات [1]، [2]، [3] التي تشير إلى أجزاء السياق.
- إذا لم يحتوِ السياق على معلومات كافية للإجابة، قل ذلك بصراحة.
- كن موجزاً وشاملاً.
- أجب باللغة العربية."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _build_context(self, chunks: list[tuple[ChunkMetadata, float]]) -> str:
        """Build the context block for the prompt."""
        parts = []
        for i, (chunk, score) in enumerate(chunks, 1):
            parts.append(
                f"[{i}] (Source: {chunk.document_title}, "
                f"Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}, "
                f"File: {chunk.source_file}):\n{chunk.text}"
            )
        return "\n\n".join(parts)

    def generate(
        self,
        query: str,
        chunks: list[tuple[ChunkMetadata, float]],
        language: str = "en",
    ) -> tuple[str, list[Citation]]:
        """Generate an answer using the OpenAI Chat API."""
        system_prompt = self.SYSTEM_PROMPT_AR if language == "ar" else self.SYSTEM_PROMPT_EN
        context = self._build_context(chunks)

        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content or ""

        citations = [
            Citation(
                document_title=chunk.document_title,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                source_file=chunk.source_file,
                relevance_score=round(score, 4),
            )
            for chunk, score in chunks
        ]

        return answer, citations


def get_generator(
    use_mock: bool = True,
    api_key: str = "",
    model: str = "gpt-4o-mini",
    mocks_dir: str = "mocks",
) -> Generator:
    """Factory function to create the appropriate generator.

    Args:
        use_mock: If True, use MockGenerator; otherwise use OpenAI.
        api_key: OpenAI API key (required if use_mock is False).
        model: OpenAI model name.
        mocks_dir: Directory for mock response files.

    Returns:
        A Generator instance.
    """
    if use_mock:
        return MockGenerator(mocks_dir=mocks_dir)
    else:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when USE_MOCK_LLM=false")
        return OpenAIGenerator(api_key=api_key, model=model)
