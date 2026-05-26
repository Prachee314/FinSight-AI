import fitz  # PyMuPDF
import os
import re


# -------------------------------
# SECTION DETECTION (RULE-BASED)
# -------------------------------
def detect_section(text):
    
    text_lower = text.lower()

    # -----------------------------------
    # MANAGEMENT DISCUSSION FIRST
    # -----------------------------------
    if any(word in text_lower for word in [
        "management discussion",
        "management’s discussion",
        "md&a"
    ]):
        return "mda"

    # -----------------------------------
    # REVENUE
    # -----------------------------------
    elif any(word in text_lower for word in [
        "net sales",
        "total revenue",
        "revenue"
    ]):
        return "revenue"

    # -----------------------------------
    # EXPENSES
    # -----------------------------------
    elif any(word in text_lower for word in [
        "operating expenses",
        "expenses",
        "cost"
    ]):
        return "expenses"

    # -----------------------------------
    # BALANCE SHEET
    # -----------------------------------
    elif any(word in text_lower for word in [
        "balance sheet",
        "assets",
        "liabilities"
    ]):
        return "balance_sheet"

    # -----------------------------------
    # RISK LAST
    # -----------------------------------
    elif any(word in text_lower for word in [
        "risk factors",
        "market risk",
        "uncertainty"
    ]):
        return "risk"

    else:
        return "general"
    


# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text


# -------------------------------
# FILENAME PARSING
# -------------------------------
def parse_filename(filename):
    
    name = (
        filename
        .replace(".pdf", "")
        .replace("-", "_")
    )

    parts = name.split("_")

    company = parts[0].lower()

    quarter = "Unknown"
    year = "Unknown"

    for part in parts:

        part_upper = part.upper()

        # Detect quarter
        if part_upper in ["Q1", "Q2", "Q3", "Q4"]:
            quarter = part_upper

        # Detect year
        if part.isdigit() and len(part) == 4:
            year = part

    quarter_full = f"{quarter}_{year}"

    # Detect doc type
    doc_type = (
        "press"
        if "press" in name.lower()
        else "report"
    )

    return company, quarter_full, doc_type

# -------------------------------
# PDF EXTRACTION
# -------------------------------
def extract_pdf(pdf_path):
    filename = os.path.basename(pdf_path)

    # Parse metadata from filename
    company, quarter, doc_type = parse_filename(filename)
    print(f"  Company: {company}")
    print(f"  Quarter: {quarter}")
    print(f"  Doc Type: {doc_type}")

    # Try opening PDF safely
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Failed to open {filename}: {e}")
        return []

    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        if not text.strip():
            continue

        cleaned = clean_text(text)

        # Skip very small / noisy pages
        if len(cleaned) < 50:
            continue

        section = detect_section(cleaned)

        pages.append({
            "text": cleaned,
            "page": page_num + 1,
            "section": section,
            "company": company,
            "quarter": quarter,
            "doc_type": doc_type,
            "source": filename
        })

    return pages


# -------------------------------
# EXTRACT ALL FILES
# -------------------------------
def extract_all_pdfs(folder="data/reports/"):
    all_pages = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            print(f"📄 Extracting: {file}")

            pages = extract_pdf(path)

            # Safety check
            if pages:
                all_pages.extend(pages)

    print(f"\n✅ Total pages extracted: {len(all_pages)}")
    return all_pages


# -------------------------------
# TEST
# -------------------------------
if __name__ == "__main__":
    pages = extract_all_pdfs()

    print("\n🔍 Sample Output:\n")

    for p in pages[:3]:
        print({
            "page": p["page"],
            "section": p["section"],
            "company": p["company"],
            "quarter": p["quarter"],
            "doc_type": p["doc_type"]
        })
        print(p["text"][:300])
        print("-" * 50)