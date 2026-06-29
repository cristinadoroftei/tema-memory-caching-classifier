"""
Compare Intent Classification: sklearn vs LLM

Classifies the same queries with both methods and compares:
- Accuracy (vs known labels)
- Latency (ms)
- Token usage (LLM only)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skillab-py" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from skillab import get_llm
from intent_classifier import IntentClassifier

# === Test queries with known labels (NOT in training data) ===
TEST_QUERIES = [
    ("Găsește documente despre licitații publice din 2024", "search"),
    ("Ce avem în bază legat de contracte de mentenanță?", "search"),
    ("Arată-mi tot ce ține de achiziții de vehicule", "search"),
    ("Câte contracte au fost semnate luna trecută?", "extract"),
    ("Care este suma totală cheltuită pe servicii de curățenie?", "extract"),
    ("Câți furnizori au contracte active?", "extract"),
    ("Care este media valorică a achizițiilor din 2023?", "extract"),
    ("Fă un rezumat al documentelor despre achiziții de echipamente", "summarize"),
    ("Sintetizează informațiile despre contractele de transport", "summarize"),
    ("Vreau un overview al achizițiilor directe din acest an", "summarize"),
]

VALID_LABELS = {"search", "extract", "summarize"}

# === LLM classification prompt ===
LLM_CLASSIFY_PROMPT = """Clasifică următoarea întrebare într-una din cele 3 categorii:
- search: utilizatorul vrea să caute/găsească documente
- extract: utilizatorul vrea date numerice, statistici, totaluri (necesită interogare SQL)
- summarize: utilizatorul vrea un rezumat/sinteză a informațiilor

Răspunde DOAR cu eticheta (search, extract, sau summarize), fără alte explicații.

Întrebare: {query}"""


def classify_with_llm(llm, query: str) -> tuple[str, float, dict]:
    """Classify using LLM. Returns (label, latency_ms, usage_dict)."""
    prompt = LLM_CLASSIFY_PROMPT.format(query=query)
    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()
    response = llm.generate_sync(messages)
    latency_ms = (time.perf_counter() - start) * 1000

    label = response.strip().lower()
    # Clean up LLM response — sometimes it adds extra text
    for valid in VALID_LABELS:
        if valid in label:
            label = valid
            break

    # Get token usage if available (Anthropic stores it in last_usage)
    usage = {}
    if hasattr(llm, "last_usage") and llm.last_usage:
        u = llm.last_usage
        usage = {
            "input": getattr(u, "input_tokens", 0),
            "output": getattr(u, "output_tokens", 0),
        }

    return label, latency_ms, usage


def classify_with_sklearn(clf: IntentClassifier, query: str) -> tuple[str, float]:
    """Classify using sklearn. Returns (label, latency_ms)."""
    start = time.perf_counter()
    result = clf.predict(query)
    latency_ms = (time.perf_counter() - start) * 1000
    return result["label"], latency_ms


def main():
    print("=" * 80)
    print("COMPARAȚIE: Intent Classification — sklearn vs LLM")
    print("=" * 80)

    # Init
    llm = get_llm()
    clf = IntentClassifier()
    print(f"LLM: {llm.name} / {llm.model}")
    print(f"sklearn: TF-IDF + LogisticRegression")
    print(f"Test queries: {len(TEST_QUERIES)}\n")

    # Classify all queries with both methods
    results = []
    total_llm_tokens = 0

    for query, expected in TEST_QUERIES:
        sk_label, sk_ms = classify_with_sklearn(clf, query)
        llm_label, llm_ms, usage = classify_with_llm(llm, query)
        tokens = usage.get("input", 0) + usage.get("output", 0)
        total_llm_tokens += tokens

        results.append({
            "query": query,
            "expected": expected,
            "sk_label": sk_label,
            "sk_ms": sk_ms,
            "llm_label": llm_label,
            "llm_ms": llm_ms,
            "tokens": tokens,
        })

    # === Print results table ===
    print(f"{'Query':<55} {'Expected':<10} {'sklearn':<10} {'LLM':<10} {'sk ms':>7} {'LLM ms':>8} {'Tokens':>7}")
    print("-" * 110)

    sk_correct = 0
    llm_correct = 0
    sk_total_ms = 0
    llm_total_ms = 0

    for r in results:
        sk_mark = "+" if r["sk_label"] == r["expected"] else "X"
        llm_mark = "+" if r["llm_label"] == r["expected"] else "X"
        sk_correct += r["sk_label"] == r["expected"]
        llm_correct += r["llm_label"] == r["expected"]
        sk_total_ms += r["sk_ms"]
        llm_total_ms += r["llm_ms"]

        short_q = r["query"][:53] + ".." if len(r["query"]) > 55 else r["query"]
        print(
            f"{short_q:<55} {r['expected']:<10} "
            f"{r['sk_label']:<8}{sk_mark:<2} {r['llm_label']:<8}{llm_mark:<2} "
            f"{r['sk_ms']:>6.1f} {r['llm_ms']:>7.1f} {r['tokens']:>7}"
        )

    # === Summary ===
    n = len(TEST_QUERIES)
    print("\n" + "=" * 80)
    print("SUMAR")
    print("=" * 80)
    print(f"{'Metric':<30} {'sklearn':>15} {'LLM':>15}")
    print("-" * 62)
    print(f"{'Accuracy':<30} {sk_correct}/{n} ({sk_correct/n:.0%}){'':<6} {llm_correct}/{n} ({llm_correct/n:.0%})")
    print(f"{'Total latency':<30} {sk_total_ms:>12.1f} ms {llm_total_ms:>12.1f} ms")
    print(f"{'Avg latency per query':<30} {sk_total_ms/n:>12.1f} ms {llm_total_ms/n:>12.1f} ms")
    print(f"{'Total tokens':<30} {'0':>15} {total_llm_tokens:>15}")
    print(f"{'Cost':<30} {'$0 (free)':>15} {'~${:.4f}'.format(total_llm_tokens * 0.000003):>15}")
    print(f"{'Speedup':<30} {llm_total_ms/sk_total_ms:>14.0f}x {'(baseline)':>15}")


if __name__ == "__main__":
    main()
