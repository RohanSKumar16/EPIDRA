"""EPIDRA Chatbot Test"""
import requests

API = "http://127.0.0.1:8000"

TESTS = [
    ("dharwad", "risk_check"),
    ("bijapur", "risk_check"),
    ("meghalaya", "risk_check"),
    ("Sikkim", "risk_check"),
    ("bombay", "risk_check"),
    ("what is cholera?", "disease_info"),
    ("symptoms of dengue", "disease_info"),
    ("why is guwahati risky?", "why_risk"),
    ("prevention tips", "prevention"),
    ("hello", "greeting"),
    ("help", "help"),
    ("overall risk", "overall_risk"),
    ("Chennai", "risk_check"),
]

results = []
for query, expected in TESTS:
    r = requests.post(f"{API}/chat", json={"message": query, "language": "en"}, timeout=15)
    d = r.json()
    ok = d["intent"] == expected
    results.append((query, expected, d["intent"], ok))

for q, exp, got, ok in results:
    tag = "OK" if ok else "XX"
    print(f"[{tag}] {q:30s} exp={exp:15s} got={got}")

p = sum(1 for _,_,_,x in results if x)
f = len(results) - p
print(f"\n{p}/{len(results)} passed  {f} failed")

h = requests.get(f"{API}/health").json()
print(f"v{h['version']} aliases={h['city_aliases']} kb={h['disease_kb_entries']} gemini={h['gemini_enabled']}")
