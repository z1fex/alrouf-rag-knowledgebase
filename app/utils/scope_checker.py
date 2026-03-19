"""Out-of-scope detection for the RAG pipeline.

Uses a two-layer approach:
1. Similarity threshold — if the best chunk score is too low, refuse.
2. Keyword heuristic — check for domain-specific terms in the query and
   retrieved chunks. If neither contains any, and similarity is moderate, refuse.
"""

import re

from app.models.schemas import ChunkMetadata

# Strong domain keywords — specific to Alrouf/lighting domain.
# Used for query-level keyword matching. Excludes generic location names
# that could appear in unrelated queries (e.g., "weather in Riyadh").
QUERY_KEYWORDS_EN = {
    "alrouf", "al-rouf", "al rouf",
    "street light", "streetlight", "street lighting",
    "floodlight", "flood light", "garden light",
    "luminaire", "led", "lighting",
    "alr-sl", "alr-fl", "alr-gl",
    "pole", "lamp", "watt", "wattage",
    "ip rating", "ip65", "ip66", "ip67",
    "lumen", "luminous", "photometric",
    "warranty", "maintenance", "installation",
    "saso", "iso 9001", "iso 14001", "iso 45001",
    "certification", "certified",
    "surge protection", "dimming", "dali",
    "mounting", "spigot", "bracket",
    "color temperature", "cri",
    "product catalog", "spare parts",
    "after-sales", "service center",
}

QUERY_KEYWORDS_AR = {
    "الروف",
    "إضاءة", "اضاءة",
    "شوارع", "شارع",
    "كشاف", "كشافات",
    "حدائق", "حديقة",
    "ضمان",
    "صيانة",
    "تركيب",
    "شهادات", "شهادة",
    "جودة",
    "ساسو",
    "منتجات", "منتج",
    "مصنع",
    "أعمدة", "عمود",
    "إنارة", "انارة",
    "مواصفات",
    "فنية",
    "كهربائية",
    "قطع غيار",
}

# Combined set for query-level matching
ALL_KEYWORDS = QUERY_KEYWORDS_EN | QUERY_KEYWORDS_AR


def _text_contains_keywords(text: str) -> bool:
    """Check if text contains any domain keyword."""
    text_lower = text.lower()
    for keyword in ALL_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


def is_in_scope(
    query: str,
    results: list[tuple[ChunkMetadata, float]],
    similarity_threshold: float = 0.3,
) -> bool:
    """Determine whether a query is within the scope of Alrouf's knowledge base.

    Uses a two-layer approach:
    1. Hard similarity threshold — below this, always refuse.
    2. Keyword heuristic — the query must contain at least one domain keyword,
       OR the similarity must be very high (>0.85). This prevents false positives
       from TF-IDF matching on common substrings like city names.

    Args:
        query: The user's question.
        results: Retrieved chunks with similarity scores.
        similarity_threshold: Minimum score below which we always refuse.

    Returns:
        True if the query is in-scope, False otherwise.
    """
    # No results at all — out of scope
    if not results:
        return False

    best_score = results[0][1]

    # Layer 1: Hard similarity threshold
    if best_score < similarity_threshold:
        return False

    # Layer 2: Query must contain domain keywords, unless similarity is very high
    query_has_keywords = _text_contains_keywords(query)
    if query_has_keywords:
        return True

    # No query keywords — require very high similarity (likely exact domain match)
    if best_score >= 0.85:
        return True

    return False


def get_refusal_message(language: str = "en") -> str:
    """Return a polite out-of-scope refusal message.

    Args:
        language: 'en' for English, 'ar' for Arabic.

    Returns:
        Refusal message string.
    """
    if language == "ar":
        return (
            "يمكنني فقط الإجابة على الأسئلة المتعلقة بمنتجات وخدمات شركة الروف "
            "لتقنية الإضاءة. يرجى السؤال عن منتجاتنا أو التركيب أو الضمان أو "
            "معلومات الشركة."
        )
    return (
        "I can only answer questions related to Alrouf Lighting Technology "
        "products and services. Please ask about our products, installations, "
        "warranties, or company information."
    )
