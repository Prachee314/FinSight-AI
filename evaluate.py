# evaluate.py
import time
import json
from main import run
from src.embedder import search

# ============================================================
# GROUND TRUTH — verified from actual PDFs
# ============================================================
GROUND_TRUTH = {
    "simple_qa": [
        {
            "query": "What is Infosys revenue in Q2?",
            "expected_value": 5076,
            "expected_company": "infosys",
            "expected_quarter": "Q2_2025",
            "keywords": ["5,076", "million", "Q2"]
        },
        {
            "query": "What is Infosys revenue in Q1?",
            "expected_value": 4941,
            "expected_company": "infosys",
            "expected_quarter": "Q1_2025",
            "keywords": ["4,941", "million", "Q1"]
        },
        {
            "query": "What is Apple Q2 revenue?",
            "expected_value": 95359,
            "expected_company": "apple",
            "expected_quarter": "Q2_2025",
            "keywords": ["95,359", "million", "Q2"]
        },
        {
            "query": "What is Apple Q1 revenue?",
            "expected_value": 94036,
            "expected_company": "apple",
            "expected_quarter": "Q1_2025",
            "keywords": ["94,036", "million", "Q1"]
        },
        {
            "query": "What is Infosys operating margin in Q2?",
            "expected_company": "infosys",
            "expected_quarter": "Q2_2025",
            "keywords": ["21.0", "21%", "operating margin"]
        },
        {
            "query": "What is Infosys free cash flow in Q1?",
            "expected_company": "infosys",
            "expected_quarter": "Q1_2025",
            "keywords": ["884", "million", "cash flow"]
        },
        {
            "query": "What were Infosys large deal wins in Q3?",
            "expected_company": "infosys",
            "expected_quarter": "Q3_2025",
            "keywords": ["4.8", "billion", "deal"]
        },
    ],
    "comparison": [
        {
            "query": "Compare Infosys Q1 and Q2 revenue",
            "expected_company": "infosys",
            "expected_quarters": ["Q1_2025", "Q2_2025"],
            "keywords": ["4,941", "5,076", "135", "2.73"]
        },
        {
            "query": "Compare Apple Q1 and Q2 revenue",
            "expected_company": "apple",
            "expected_quarters": ["Q1_2025", "Q2_2025"],
            "keywords": ["94,036", "95,359"]
        },
    ],
    "reasoning": [
        {
            "query": "What drove Infosys revenue growth in Q2?",
            "expected_company": "infosys",
            "keywords": ["AI", "deal", "growth", "revenue"]
        },
        {
            "query": "Why did Apple revenue increase in Q2?",
            "expected_company": "apple",
            "keywords": ["iPhone", "services", "revenue", "growth"]
        },
    ],
    "summarization": [
        {
            "query": "Summarize Infosys Q3",
            "expected_company": "infosys",
            "expected_quarter": "Q3_2025",
            "keywords": ["5,099", "revenue", "margin", "deal"]
        },
        {
            "query": "Summarize Apple Q2",
            "expected_company": "apple",
            "expected_quarter": "Q2_2025",
            "keywords": ["95,359", "revenue", "net sales"]
        },
    ],
    "classifier": [
        {"query": "What is Infosys revenue?",           "expected_type": "simple_qa"},
        {"query": "Compare Infosys Q1 and Q2",          "expected_type": "comparison"},
        {"query": "Why did Infosys revenue increase?",  "expected_type": "reasoning"},
        {"query": "Summarize Infosys Q2",               "expected_type": "summarization"},
        {"query": "What is Infosys revenue trend?",     "expected_type": "trend"},
        {"query": "Compare Apple vs Infosys revenue",   "expected_type": "comparison"},
        {"query": "What caused Apple growth in Q2?",    "expected_type": "reasoning"},
        {"query": "Give me overview of Infosys Q1",     "expected_type": "summarization"},
    ],
    "invalid": [
        {"query": "What is revenue?",       "expected_type": "simple_qa"},
        {"query": "Compare Q1 and Q2",      "expected_type": "comparison"},
        {"query": "Why did revenue increase?", "expected_type": "reasoning"},
    ]
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def check_keywords(answer, keywords):
    answer_lower = answer.lower()
    found = [k for k in keywords if k.lower() in answer_lower]
    return len(found), len(keywords), found


def measure_time(query):
    time.sleep(1)
    start = time.time()
    result, qtype = run(query)
    elapsed = time.time() - start
    return result, qtype, elapsed


def measure_retrieval_time(query, company, quarter=None):
    start = time.time()
    chunks = search(query, company_filter=company, quarter_filter=quarter, top_k=5)
    elapsed = time.time() - start
    return chunks, elapsed


def check_numeric_accuracy(answer, expected_value, tolerance=0.05):
    import re
    clean_answer = answer.replace(",", "")
    numbers = re.findall(r"\d+\.?\d*", clean_answer)
    for n in numbers:
        try:
            val = float(n)
            if expected_value * (1 - tolerance) <= val <= expected_value * (1 + tolerance):
                return True
        except Exception:
            pass
    return False


def score_label(passed, total):
    pct = (passed / total * 100) if total > 0 else 0
    if pct >= 80:
        return "PASS", pct
    elif pct >= 50:
        return "PARTIAL", pct
    else:
        return "FAIL", pct


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def eval_classifier():
    print("\n" + "="*60)
    print("1. CLASSIFIER ACCURACY")
    print("="*60)

    all_cases = GROUND_TRUTH["classifier"] + GROUND_TRUTH["invalid"]
    passed = 0
    results = []

    for case in all_cases:
        query = case["query"]
        expected = case["expected_type"]
        try:
            result, qtype, elapsed = measure_time(query)
            correct = qtype == expected
            if correct:
                passed += 1
            status = "PASS" if correct else "FAIL"
            results.append({
                "query": query,
                "expected": expected,
                "got": qtype,
                "status": status,
                "time": elapsed
            })
            print(f"  [{status}] '{query[:45]}'")
            print(f"         Expected: {expected} | Got: {qtype} | Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"  [ERROR] '{query}' → {e}")
            results.append({"query": query, "status": "ERROR", "error": str(e)})

    label, pct = score_label(passed, len(all_cases))
    print(f"\n  Result: {passed}/{len(all_cases)} correct ({pct:.1f}%) → {label}")
    return {"passed": passed, "total": len(all_cases), "pct": pct, "details": results}


def eval_simple_qa():
    print("\n" + "="*60)
    print("2. SIMPLE QA — KEYWORD + NUMERIC ACCURACY")
    print("="*60)

    passed = 0
    total = len(GROUND_TRUTH["simple_qa"])
    results = []

    for case in GROUND_TRUTH["simple_qa"]:
        query = case["query"]
        keywords = case["keywords"]
        expected_value = case.get("expected_value")

        try:
            result, qtype, elapsed = measure_time(query)
            found_count, total_kw, found = check_keywords(result, keywords)
            kw_score = found_count / total_kw

            numeric_ok = True
            if expected_value:
                numeric_ok = check_numeric_accuracy(result, expected_value)

            overall_pass = kw_score >= 0.6 and numeric_ok
            if overall_pass:
                passed += 1

            status = "PASS" if overall_pass else "FAIL"
            results.append({
                "query": query,
                "status": status,
                "keyword_score": f"{found_count}/{total_kw}",
                "numeric_ok": numeric_ok,
                "time": elapsed
            })
            print(f"  [{status}] '{query[:45]}'")
            print(f"         Keywords: {found_count}/{total_kw} found {found}")
            if expected_value:
                print(f"         Numeric: {'OK' if numeric_ok else 'WRONG'} (expected ~{expected_value})")
            print(f"         Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"  [ERROR] '{query}' → {e}")
            results.append({"query": query, "status": "ERROR", "error": str(e)})

    label, pct = score_label(passed, total)
    print(f"\n  Result: {passed}/{total} correct ({pct:.1f}%) → {label}")
    return {"passed": passed, "total": total, "pct": pct, "details": results}


def eval_comparison():
    print("\n" + "="*60)
    print("3. COMPARISON ACCURACY")
    print("="*60)

    passed = 0
    total = len(GROUND_TRUTH["comparison"])
    results = []

    for case in GROUND_TRUTH["comparison"]:
        query = case["query"]
        keywords = case["keywords"]

        try:
            result, qtype, elapsed = measure_time(query)
            found_count, total_kw, found = check_keywords(result, keywords)
            kw_score = found_count / total_kw

            overall_pass = kw_score >= 0.5
            if overall_pass:
                passed += 1

            status = "PASS" if overall_pass else "FAIL"
            results.append({
                "query": query,
                "status": status,
                "keyword_score": f"{found_count}/{total_kw}",
                "time": elapsed
            })
            print(f"  [{status}] '{query[:45]}'")
            print(f"         Keywords: {found_count}/{total_kw} found {found}")
            print(f"         Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"  [ERROR] '{query}' → {e}")

    label, pct = score_label(passed, total)
    print(f"\n  Result: {passed}/{total} correct ({pct:.1f}%) → {label}")
    return {"passed": passed, "total": total, "pct": pct, "details": results}


def eval_reasoning():
    print("\n" + "="*60)
    print("4. REASONING QUALITY")
    print("="*60)

    passed = 0
    total = len(GROUND_TRUTH["reasoning"])
    results = []

    for case in GROUND_TRUTH["reasoning"]:
        query = case["query"]
        keywords = case["keywords"]

        try:
            result, qtype, elapsed = measure_time(query)
            found_count, total_kw, found = check_keywords(result, keywords)
            kw_score = found_count / total_kw

            has_direct_answer = "direct answer" in result.lower()
            has_numbers = any(c.isdigit() for c in result)
            has_sources = any(w in result.lower() for w in ["source", "press", "pdf"])

            overall_pass = kw_score >= 0.5 and has_numbers
            if overall_pass:
                passed += 1

            status = "PASS" if overall_pass else "FAIL"
            results.append({
                "query": query,
                "status": status,
                "keyword_score": f"{found_count}/{total_kw}",
                "has_numbers": has_numbers,
                "has_sources": has_sources,
                "time": elapsed
            })
            print(f"  [{status}] '{query[:45]}'")
            print(f"         Keywords: {found_count}/{total_kw} | Numbers: {has_numbers} | Sources: {has_sources}")
            print(f"         Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"  [ERROR] '{query}' → {e}")

    label, pct = score_label(passed, total)
    print(f"\n  Result: {passed}/{total} correct ({pct:.1f}%) → {label}")
    return {"passed": passed, "total": total, "pct": pct, "details": results}


def eval_summarization():
    print("\n" + "="*60)
    print("5. SUMMARIZATION QUALITY")
    print("="*60)

    passed = 0
    total = len(GROUND_TRUTH["summarization"])
    results = []

    for case in GROUND_TRUTH["summarization"]:
        query = case["query"]
        keywords = case["keywords"]

        try:
            result, qtype, elapsed = measure_time(query)
            found_count, total_kw, found = check_keywords(result, keywords)
            kw_score = found_count / total_kw

            has_highlights = "highlight" in result.lower()
            has_financial = "financial" in result.lower()
            has_quarter = case.get("expected_quarter", "")[:2] in result

            overall_pass = kw_score >= 0.5 and has_quarter
            if overall_pass:
                passed += 1

            status = "PASS" if overall_pass else "FAIL"
            results.append({
                "query": query,
                "status": status,
                "keyword_score": f"{found_count}/{total_kw}",
                "has_highlights": has_highlights,
                "time": elapsed
            })
            print(f"  [{status}] '{query[:45]}'")
            print(f"         Keywords: {found_count}/{total_kw} | Highlights: {has_highlights}")
            print(f"         Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"  [ERROR] '{query}' → {e}")

    label, pct = score_label(passed, total)
    print(f"\n  Result: {passed}/{total} correct ({pct:.1f}%) → {label}")
    return {"passed": passed, "total": total, "pct": pct, "details": results}


def eval_retrieval():
    print("\n" + "="*60)
    print("6. RETRIEVAL METRICS")
    print("="*60)

    test_cases = [
        {"query": "Infosys revenue Q2 2025",     "company": "infosys", "quarter": "Q2_2025", "expected_keywords": ["revenue", "5,076", "44,490"]},
        {"query": "Apple net sales Q1 2025",     "company": "apple",   "quarter": "Q1_2025", "expected_keywords": ["net sales", "94,036"]},
        {"query": "Infosys operating margin Q3", "company": "infosys", "quarter": "Q3_2025", "expected_keywords": ["margin", "21.2"]},
    ]

    total_precision = 0
    results = []

    for case in test_cases:
        chunks, elapsed = measure_retrieval_time(
            case["query"],
            case["company"],
            case["quarter"]
        )

        relevant = 0
        for chunk in chunks:
            text = chunk["text"].lower()
            if any(kw.lower() in text for kw in case["expected_keywords"]):
                relevant += 1

        precision = relevant / len(chunks) if chunks else 0
        total_precision += precision

        results.append({
            "query": case["query"],
            "chunks_retrieved": len(chunks),
            "relevant_chunks": relevant,
            "precision": precision,
            "retrieval_time": elapsed
        })

        print(f"  Query: '{case['query']}'")
        print(f"         Chunks: {len(chunks)} | Relevant: {relevant} | Precision: {precision:.2f} | Time: {elapsed:.3f}s")

    avg_precision = total_precision / len(test_cases)
    label = "PASS" if avg_precision >= 0.5 else "FAIL"
    print(f"\n  Average Retrieval Precision: {avg_precision:.2f} → {label}")
    return {"avg_precision": avg_precision, "details": results}


def eval_response_time():
    print("\n" + "="*60)
    print("7. RESPONSE TIME")
    print("="*60)

    test_queries = [
        ("What is Infosys revenue in Q2?",      "simple_qa"),
        ("Summarize Infosys Q3",                "summarization"),
        ("Compare Infosys Q1 and Q2 revenue",   "comparison"),
        ("What drove Infosys growth?",          "reasoning"),
    ]

    times = []
    for query, qtype in test_queries:
        try:
            _, _, elapsed = measure_time(query)
            times.append(elapsed)
            speed = "FAST" if elapsed < 3 else "OK" if elapsed < 6 else "SLOW"
            print(f"  [{speed}] {qtype}: {elapsed:.2f}s — '{query[:40]}'")
        except Exception as e:
            print(f"  [ERROR] {query} → {e}")

    if times:
        avg = sum(times) / len(times)
        print(f"\n  Average response time: {avg:.2f}s")
        print(f"  Fastest: {min(times):.2f}s | Slowest: {max(times):.2f}s")
        return {"avg_time": avg, "min": min(times), "max": max(times)}
    return {}


# ============================================================
# MAIN — RUN ALL EVALUATIONS
# ============================================================
def run_evaluation():
    print("\n" + "="*60)
    print("  FINANCIAL INTELLIGENCE SYSTEM — EVALUATION REPORT")
    print("="*60)

    scores = {}

    scores["classifier"]    = eval_classifier()
    scores["simple_qa"]     = eval_simple_qa()
    scores["comparison"]    = eval_comparison()
    scores["reasoning"]     = eval_reasoning()
    scores["summarization"] = eval_summarization()
    scores["retrieval"]     = eval_retrieval()
    scores["response_time"] = eval_response_time()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)

    metric_scores = {
        "Classifier Accuracy":     scores["classifier"]["pct"],
        "Simple QA Accuracy":      scores["simple_qa"]["pct"],
        "Comparison Accuracy":     scores["comparison"]["pct"],
        "Reasoning Quality":       scores["reasoning"]["pct"],
        "Summarization Quality":   scores["summarization"]["pct"],
        "Retrieval Precision":     scores["retrieval"]["avg_precision"] * 100,
    }

    for metric, score in metric_scores.items():
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        label = "PASS" if score >= 70 else "PARTIAL" if score >= 50 else "FAIL"
        print(f"  {metric:<25} {bar} {score:.1f}% [{label}]")

    overall = sum(metric_scores.values()) / len(metric_scores)
    print(f"\n  Overall Score: {overall:.1f}%")

    if overall >= 80:
        grade = "EXCELLENT — Ready for placement demo"
    elif overall >= 65:
        grade = "GOOD — Minor improvements needed"
    elif overall >= 50:
        grade = "FAIR — Some components need fixing"
    else:
        grade = "NEEDS WORK — Review failed components"

    print(f"  Grade: {grade}")

    # Save results to JSON
    with open("evaluation_results.json", "w") as f:
        json.dump(scores, f, indent=2, default=str)
    print(f"\n  Full results saved to: evaluation_results.json")
    print("="*60)


if __name__ == "__main__":
    run_evaluation()