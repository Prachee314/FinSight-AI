#main.py
import re
from src.classifier import classify
from src.embedder import search
from src.reasoning import explain
from src.summarizer import summarize
from src.qa import answer
from src.compare import compare_quarters
from src.trend import get_trend

LATEST_QUARTER = {
    "infosys": "Q4_2025",
    "apple": "Q2_2025"
}


def extract_company(query):
    query_lower = query.lower()
    
    # Aliases map
    aliases = {
        "infy": "infosys",
        "infosys ltd": "infosys",
        "aapl": "apple",
        "apple inc": "apple",
        "msft": "microsoft",
    }
    # Check aliases first
    for alias, company in aliases.items():
        if alias in query_lower:
            return company

    companies = ["apple", "infosys", "microsoft", "tcs", "wipro"]
    for company in companies:
        if company in query_lower:
            return company
    return None

def extract_company_quarter_pairs(query):
    """
    Extract (company, quarter) pairs in order they appear in query.
    Example: 'compare infosys Q2 and apple Q1'
    → [('infosys', 'Q2_2025'), ('apple', 'Q1_2025')]
    """
    query_lower = query.lower()
    companies = ["apple", "infosys", "microsoft", "tcs", "wipro"]
    quarters = re.findall(r"Q[1-4]", query.upper())

    # Find position of each company in query
    found = []
    for company in companies:
        pos = query_lower.find(company)
        if pos != -1:
            found.append((pos, company))

    # Sort by position in query
    found.sort(key=lambda x: x[0])
    companies_in_order = [c for _, c in found]

    # Pair company with quarter in order
    pairs = []
    for i, company in enumerate(companies_in_order):
        quarter = f"Q{quarters[i][1]}_2025" if i < len(quarters) else None
        pairs.append((company, quarter))

    return pairs

def extract_companies(query):
    query_lower = query.lower()
    companies = ["apple", "infosys", "microsoft", "tcs", "wipro"]
    found = []
    for company in companies:
        if company in query_lower:
            found.append(company)
    return found  # returns list, e.g. ["infosys", "apple"]


def remove_duplicates(chunks):
    seen = set()
    unique_chunks = []
    for c in chunks:
        text = c["text"]
        if text not in seen:
            seen.add(text)
            unique_chunks.append(c)
    return unique_chunks


def extract_quarters(query):
    query_upper = query.upper()
    
    # Handle "latest" / "most recent" / "recent"
    if any(w in query.lower() for w in ["latest", "most recent", "recent", "last quarter", "current"]):
        company = extract_company(query)
        if company and company in LATEST_QUARTER:
            return [LATEST_QUARTER[company]]
        
    matches = re.findall(
        r"Q([1-4])(?:[_\s]?(\d{4}))?",
        query_upper
    )
    result = []
    for quarter_num, year in matches:
        y = year if year else "2025"
        result.append(f"Q{quarter_num}_{y}")
    return result


def run(query, quarter_filter=None):

    # STEP 1: Classification
    query_type = classify(query)
    company_filter = extract_company(query)
    companies_found = extract_companies(query)  # noqa: F841
    quarter_filters = extract_quarters(query)


    # Bug 2 fix — use passed quarter_filter as fallback
    if not quarter_filters and quarter_filter:
        quarter_filters = [quarter_filter]

    # STEP 2: Query Enhancement
    enhanced_query = query
    if quarter_filters:
        enhanced_query += " " + " ".join(quarter_filters)

    if any(word in query.lower() for word in [
        "revenue", "sales", "income", "profit", "expenses"
    ]):
        enhanced_query += (
            " total revenue net sales profit expenses "
            "financial results quarterly performance"
        )

    # STEP 3: Routing

    # ========= REASONING =========
    if query_type == "reasoning":
        chunks = []

        if quarter_filters:
            expanded_quarters = quarter_filters.copy()

            # Bug 3 fix — dedup outside loop
            for q in quarter_filters:
                q_num = int(q[1])
                if q_num > 1:
                    prev_q = f"Q{q_num - 1}_2025"
                    expanded_quarters.append(prev_q)

            expanded_quarters = list(set(expanded_quarters))

            for q in expanded_quarters:
                quarter_specific_query = (
                    f"{company_filter} total revenue net sales financial results "
                    f"revenue crore million billion"
                )
                q_chunks = search(
                    quarter_specific_query,
                    quarter_filter=q,
                    company_filter=company_filter,
                    top_k=5
                )
                chunks.extend(q_chunks)

        else:
            chunks1 = search(query, company_filter=company_filter, top_k=6)
            chunks2 = search("revenue profit expenses growth", company_filter=company_filter, top_k=4)
            chunks3 = search("financial performance quarterly results", company_filter=company_filter, top_k=4)
            chunks = chunks1 + chunks2 + chunks3

        chunks = remove_duplicates(chunks)
        result = explain(query, chunks)
        return result, query_type, chunks


    # ========= COMPARISON =========
    elif query_type == "comparison":
        chunks = []
        companies_found = extract_companies(query)

        if quarter_filters and len(companies_found) == 2:
            # Cross-company with quarters specified — use ordered pairs
            pairs = extract_company_quarter_pairs(query)
            for company, q in pairs:
                if not q:
                    continue
                press_chunks = search(
                    f"delivered million revenues net sales {company}",
                    quarter_filter=q,
                    company_filter=company,
                    top_k=5
                )
                report_chunks = search(
                    f"total revenue from operations net sales crore {company}",
                    quarter_filter=q,
                    company_filter=company,
                    top_k=5
                )
                chunks.extend(press_chunks)
                chunks.extend(report_chunks)

        elif quarter_filters:
            # Same company multiple quarters
            for q in quarter_filters:
                press_chunks = search(
                    f"delivered million revenues {company_filter}",
                    quarter_filter=q,
                    company_filter=company_filter,
                    top_k=5
                )
                report_chunks = search(
                    f"total revenue from operations crore {company_filter}",
                    quarter_filter=q,
                    company_filter=company_filter,
                    top_k=5
                )
                chunks.extend(press_chunks)
                chunks.extend(report_chunks)

        else:
            # No quarters — use latest for each company
            if len(companies_found) == 2:
                for company in companies_found:
                    q = LATEST_QUARTER.get(company)
                    if not q:
                        continue
                    press_chunks = search(
                        f"delivered million revenues net sales {company}",
                        quarter_filter=q,
                        company_filter=company,
                        top_k=5
                    )
                    report_chunks = search(
                        f"total revenue from operations crore {company}",
                        quarter_filter=q,
                        company_filter=company,
                        top_k=5
                    )
                    chunks.extend(press_chunks)
                    chunks.extend(report_chunks)
            else:
                chunks = search(
                    f"delivered million revenues total revenue {company_filter}",
                    quarter_filter=None,
                    company_filter=company_filter,
                    top_k=10
                )

        chunks = remove_duplicates(chunks)
        if not chunks:
            return "Not enough data to compare.", query_type, []
        result = compare_quarters(query, chunks)
        return result, query_type, chunks
    
    elif query_type == "trend":
        result = get_trend(
            query=query,
            company=company_filter,
            quarters=quarter_filters if quarter_filters else None
    )
        return result, query_type, []  # ← add this

        
    # ========= SUMMARIZATION =========  Bug 1 fix
    elif query_type == "summarization":
        # Prefer press releases for summaries — they have clean highlights
        summary_query = f"{company_filter} revenues million delivered quarterly results highlights"
        if quarter_filters:
            summary_query += " " + " ".join(quarter_filters)

        chunks = search(
            summary_query,
            quarter_filter=quarter_filters[0] if quarter_filters else quarter_filter,
            company_filter=company_filter,
            top_k=6
        )
        chunks = remove_duplicates(chunks)
    # Prefer press chunks first
        press_chunks = [c for c in chunks if c["metadata"].get("doc_type") == "press"]
        report_chunks = [c for c in chunks if c["metadata"].get("doc_type") != "press"]
        if not press_chunks and quarter_filters and "Q4" in quarter_filters[0]:
            chunks = report_chunks[:6]
        else:
            chunks = (press_chunks + report_chunks)[:6]

        if not chunks:
            return "No relevant data found.", query_type, []
        
        result = summarize(query, chunks)
        return result, query_type, chunks

    # ========= SIMPLE QA =========
    else:  # simple_qa

        # For latest queries use targeted quarterly search
        if any(w in query.lower() for w in ["latest", "most recent", "recent"]):
            chunks = search(
                f"Q4 revenues were million {company_filter}",
                quarter_filter=quarter_filters[0] if quarter_filters else None,
                company_filter=company_filter,
                top_k=8
            )

        elif quarter_filters and "Q4" in quarter_filters[0]:
            metric_chunks = search(
                f"{company_filter} operating margin EPS free cash flow Q4",
                quarter_filter="Q4_2025",
                company_filter=company_filter,
                top_k=5
            )
            base_chunks = search(
                enhanced_query,
                quarter_filter="Q4_2025",
                company_filter=company_filter,
                top_k=5
            )
            chunks = remove_duplicates(metric_chunks + base_chunks)

        else:
            chunks = search(
                enhanced_query,
                quarter_filter=quarter_filters[0] if quarter_filters else quarter_filter,
                company_filter=company_filter,
                top_k=8
            )

        # Prefer press release chunks for revenue queries
        if any(w in query.lower() for w in ["revenue", "sales", "results"]):
            press_chunks = [c for c in chunks if c["metadata"].get("doc_type") == "press"]
            report_chunks = [c for c in chunks if c["metadata"].get("doc_type") != "press"]
            chunks = (press_chunks + report_chunks)[:5]

        chunks = remove_duplicates(chunks)
        if not chunks:
            return "No relevant answer found.", query_type, []
        result = answer(query, chunks)
        return result, query_type, chunks

# TERMINAL INTERFACE
if __name__ == "__main__":

    print("📊 Financial Intelligence System")
    print("Type 'exit' to quit\n")

    while True:
        q = input("Ask: ")

        if q.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        # Safety catch — if run() returns None
        response = run(q)
        if response is None:
            print("⚠️ Something went wrong. Please try again.")
            continue

        if len(response) == 3:
            result, qtype, sources = response
        else:
            result, qtype = response
            sources = []

        print(f"\n🔎 Query Type: {qtype}")
        print("\n💬 Answer:\n")
        print(result)
        print("\n" + "=" * 70 + "\n")