# src/utils.py
from groq import Groq
import os
import re
from dotenv import load_dotenv

load_dotenv()


def get_groq_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not found")
    return Groq(api_key=key)


def quarter_order(q):
    if not q:
        return 0
    match = re.search(r"Q([1-4])", q)
    if match:
        return int(match.group(1))
    return 0


def extract_quarters_from_query(query):
    if not query:
        return []
    matches = re.findall(
        r"Q([1-4])(?:[_\s]?(\d{4}))?",
        query.upper()
    )
    result = []
    for quarter_num, year in matches:
        y = year if year else "2025"
        result.append(f"Q{quarter_num}_{y}")
    return result


def extract_revenue(chunks):
    values = []
    
    for c in chunks:
        raw_text = c["text"]           # keep original for regex
        text = raw_text.lower()        # lowercase only for keyword detection
        metadata = c["metadata"]
        source = metadata.get("source", "")
        doc_type = metadata.get("doc_type", "report")           
        

    # Pattern 2 — USD: run on RAW text to preserve $
        matches_usd = re.findall(
            r"[$]\s?([\d]{1,3}(?:,[\d]{3})*(?:\.\d+)?)\s*million",
            raw_text      # ← raw not lowercased
        )
        
        # Pattern 2b — Apple report: "net sales 95,359" with "(in millions)" header
        if not matches_usd and "(in millions" in text:
            matches_usd = re.findall(
                "(?:net sales|total net sales|revenues?)"
                r"[\s\d\.]{0,15}"
                r"([\d]{2,3}(?:,[\d]{3})+(?:\.\d+)?)",
                text
        )

    # Pattern 1 — INR: run on lowercased text
        matches_inr = re.findall(
            r"(?:total revenue from operations"
            r"|total revenue"
            r"|revenues"
            r"|revenue from software"
            r"|net sales"
            r"|revenue)"
            r"[\s\d\.]{0,15}"
            r"[₹]?\s?"
            r"([\d]{2,3}(?:,[\d]{2,3})+(?:\.\d+)?)",
            text          # ← lowercased is fine here
        )
        
        
        # Skip INR matches if this is an Apple chunk — Apple uses millions not crores
        if metadata.get("company") == "apple":
            matches_inr = []

    # Process USD
        if matches_usd:
            company = metadata.get("company", "")
            max_usd = 200000 if company == "apple" else 15000

            
            quarterly_match = re.search(
                r"\$([\d,]+)\s*million\s+in\s+Q[1-4]\s+revenues"
                r"|Q[1-4]\s+revenues\s+were\s+\$\s*([\d,]+)\s*million"
                r"|Q[1-4]\s+revenue\s+was\s+\$\s*([\d,]+)\s*million",
                raw_text,
                re.IGNORECASE
            )

            if quarterly_match:
                
                matched_value = next(
                    g for g in quarterly_match.groups() if g is not None
                )
                try:
                    number = float(matched_value.replace(",", ""))
                    if 1000 <= number <= max_usd:
                        values.append({
                            "value": number,
                            "quarter": metadata.get("quarter"),
                            "source": source,
                            "company": company,
                            "currency": "USD",
                            "doc_type": doc_type
                        })
                except Exception:
                    pass
            else:
                for match in matches_usd:
                    try:
                        number = float(str(match).replace(",", ""))
                        if 1000 <= number <= max_usd:
                            match_pos = raw_text.lower().find(match.lower())
                            context_start = max(0, match_pos - 30)
                            context_end = min(len(raw_text), match_pos + 100)
                            context = raw_text[context_start:context_end].lower()
                        
                            if any(w in context for w in [
                                "fy26", "fy25", 
                                "fy 26", "fy 25","full year",
                                "annual", "twelve months", "12 months"
                            ]):
                                continue
            
                            values.append({
                                "value": number,
                                "quarter": metadata.get("quarter"),
                                "source": source,
                                "company": company,
                                "currency": "USD",
                                "doc_type": doc_type
                            })
                            break
                    except Exception:
                        pass

    # Process INR
        if matches_inr:
            for match in matches_inr:
                try:
                    number = float(str(match).replace(",", ""))
                    if 10000 <= number <= 9999999:
                        # Detect segment table — has multiple segment names
                        is_segment_table = (
                            "financial services" in text
                            and "manufacturing" in text
                            and "total revenue" in text
                        )
                        # Check if from total revenue line
                        is_total = (
                            not is_segment_table
                            and (
                                "total revenue from operations" in text
                                or ("revenues" in text and "2.16" in text)
                                or "total net sales" in text
                            )
                        )
                        values.append({
                            "value": number,
                            "quarter": metadata.get("quarter"),
                            "source": source,
                            "company": company,
                            "currency": "INR",
                            "doc_type": doc_type,
                            "is_total": is_total
                        })
                        break
                except Exception:
                    pass


    # Separate USD and INR per quarte
    # Separate USD and INR per quarter — use quarter+company as key
    usd_unique = {}
    inr_unique = {}

    for v in values:
        q = v["quarter"]
        company = v.get("company", "")
        key = f"{q}_{company}"          # ← add company to key

        if v["currency"] == "USD":
            if key not in usd_unique or v["value"] > usd_unique[key]["value"]:
                usd_unique[key] = v
        else:
            if key not in inr_unique:
                inr_unique[key] = v
            else:
                existing = inr_unique[key]
                existing_is_total = existing.get("is_total", False)
                current_is_total = v.get("is_total", False)
                if current_is_total and not existing_is_total:
                    inr_unique[key] = v
                elif current_is_total and existing_is_total:
                    if v["value"] > existing["value"]:
                        inr_unique[key] = v
                elif not existing_is_total and not current_is_total:
                    if v["value"] > existing["value"]:
                        inr_unique[key] = v

    # Combine both currencies — use quarter+company as key
    all_keys = set(list(usd_unique.keys()) + list(inr_unique.keys()))
    result = []

    for key in all_keys:
        usd = usd_unique.get(key)
        inr = inr_unique.get(key)

        result.append({
            "quarter": (usd or inr)["quarter"],
            "usd": usd,
            "inr": inr,
            "value": usd["value"] if usd else inr["value"],
            "currency": "USD" if usd else "INR",
            "source": (usd or inr)["source"],
            "company": (usd or inr)["company"]
        })

    return result
    

def extract_metrics(chunks):
    """
    Extract non-revenue metrics from chunks:
    - Operating margin
    - Net profit
    - EPS
    - Free cash flow
    - Deal wins (TCV)
    - Headcount
    """
    metrics = {}

    for c in chunks:
        raw_text = c["text"]
        text = raw_text.lower()
        metadata = c["metadata"]
        quarter = metadata.get("quarter")
        company = metadata.get("company")

        if quarter not in metrics:
            metrics[quarter] = {
                "quarter": quarter,
                "company": company,
                "operating_margin": None,
                "net_profit_usd": None,
                "net_profit_inr": None,
                "eps_usd": None,
                "eps_inr": None,
                "free_cash_flow": None,
                "deal_wins_tcv": None,
                "headcount": None
            }

        # Operating margin — "operating margin at 21.0%"
        if not metrics[quarter]["operating_margin"]:
            m = re.search(
                r"operating margin[^\d]{0,15}([\d]{1,2}\.[\d]{1,2})%",
                text
            )
            if m:
                metrics[quarter]["operating_margin"] = float(m.group(1))

        # Free cash flow USD — "$884 million" near "free cash flow"
        if not metrics[quarter]["free_cash_flow"]:
            m = re.search(
                r"free cash flow[^\d$]{0,20}\$\s?([\d,]+\.?[\d]*)\s*(?:million|billion)",
                text
            )
            if not m:
                m = re.search(
                    r"\$([\d,]+\.?[\d]*)\s*(?:million|billion)[^\d]{0,30}free cash",
                    text
                )
            if m:
                val = float(m.group(1).replace(",", ""))
                # convert billion to million
                if "billion" in text[m.start():m.start()+60]:
                    val *= 1000
                metrics[quarter]["free_cash_flow"] = val

        # Deal wins TCV — "$3.8 billion" near "large deal" or "tcv"
        if not metrics[quarter]["deal_wins_tcv"]:
            m = re.search(
                r"(?:tcv|large deal)[^\d$]{0,30}\$\s?([\d,]+\.?[\d]*)\s*billion",
                text
            )
            if not m:
                m = re.search(
                    r"\$([\d,]+\.?[\d]*)\s*billion[^\d]{0,30}(?:tcv|large deal|deal wins)",
                    text
                )
            if m:
                metrics[quarter]["deal_wins_tcv"] = float(
                    m.group(1).replace(",", "")
                ) * 1000  # convert to millions

        # EPS USD — "$0.XX" near "earnings per share" or "eps"
        if not metrics[quarter]["eps_usd"]:
            m = re.search(
                r"(?:eps|earnings per share)[^\d$]{0,20}\$\s?([\d]+\.[\d]+)",
                text
            )
            if m:
                metrics[quarter]["eps_usd"] = float(m.group(1))

        # Net profit INR — "net profit 6,924 crore"
        if not metrics[quarter]["net_profit_inr"]:
            m = re.search(
                r"net profit[^\d]{0,15}([\d]{1,3}(?:,[\d]{2,3})+)",
                text
            )
            if m:
                val = float(m.group(1).replace(",", ""))
                if val > 1000:
                    metrics[quarter]["net_profit_inr"] = val

        # Headcount — "employee count" or "employees" near a large number
        if not metrics[quarter]["headcount"]:
            m = re.search(
                r"(?:headcount|employees|workforce)[^\d]{0,20}([\d,]{5,})",
                text
            )
            if m:
                val = int(m.group(1).replace(",", ""))
                if val > 10000:  # must be > 10k to be total headcount
                    metrics[quarter]["headcount"] = val

    return list(metrics.values())