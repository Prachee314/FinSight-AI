def classify(query):
        q = query.lower()

        comparison_words = [
        "compare", "vs", "versus", "difference",
        "change", "trend", "between", "higher",
        "lower", "better", "worse"
    ]

        reasoning_words = [
        "why", "reason", "cause", "explain",
        "how did", "what caused", "what led",
        "what drove", "behind", "factor",
        "due to", "because", "result of",
        "increase", "decrease", "growth",
        "decline", "improve", "drop", "fall",
        "rise", "went up", "went down"
    ]

        summarization_words = [
        "summarize", "summary", "overview",
        "highlight", "brief", "key points",
        "main points", "what happened",
        "tell me about", "give me", "overall",
        "guidance", "outlook", "forecast",    # ← add these
        "fy26", "fy25", "full year"
    ]
        trend_words = [
        "trend", "all quarters",
        "q1 to q4", "quarterly trend",
        "over the year", "throughout"
    ]

     # Add before comparison check:
        if any(w in q for w in trend_words):
            return "trend"

        if any(w in q for w in comparison_words):
            return "comparison"
        elif any(w in q for w in summarization_words):
            return "summarization"
        elif any(w in q for w in reasoning_words):
            return "reasoning"
        else:
            return "simple_qa"