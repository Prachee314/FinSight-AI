from src.embedder import search
import re

chunks = search('delivered million revenues', company_filter='infosys', quarter_filter='Q1_2025', top_k=8)

for c in chunks:
    if 'press' not in c['metadata']['source']:
        continue
    
    raw_text = c["text"]
    lower_text = raw_text.lower()
    
    print("SOURCE:", c['metadata']['source'])
    print("$ in raw:", "$" in raw_text)
    print("$ in lower:", "$" in lower_text)
    print("Raw sample:", raw_text[:200])
    print("---")
    
    # Test regex on both
    m1 = re.findall(r"[$]\s?([\d]{1,3}(?:,[\d]{3})*(?:\.\d+)?)\s*million", raw_text)
    m2 = re.findall(r"[$]\s?([\d]{1,3}(?:,[\d]{3})*(?:\.\d+)?)\s*million", lower_text)
    print("Regex on raw:", m1)
    print("Regex on lower:", m2)
    print("===")