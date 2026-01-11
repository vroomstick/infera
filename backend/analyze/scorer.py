# analyze/scorer.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Prompt/Query to simulate "riskiness"
RISK_PROMPT = (
    "lawsuits litigation disruption economic downturn regulation compliance product recalls "
    "cybersecurity natural disaster pandemic war inflation supply chain fraud data breach labor shortage"
)

def score_paragraphs(paragraphs):
    corpus = [RISK_PROMPT] + paragraphs
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked = sorted(zip(paragraphs, similarities), key=lambda x: x[1], reverse=True)
    return ranked

def score_sections(text, top_n=5):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        return []
    scored = score_paragraphs(paragraphs)
    return [p for p, _ in scored[:top_n]]
