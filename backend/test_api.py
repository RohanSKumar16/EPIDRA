"""EPIDRA v3.0 — Full API test suite."""
import requests
import json

BASE = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE}/health")
    d = r.json()
    print("=== HEALTH ===")
    print(json.dumps(d, indent=2))
    assert d["status"] == "ok"
    assert d["cities"] == 75
    assert "calibrated" in d["model"].lower()
    print("PASS\n")

def test_districts():
    r = requests.get(f"{BASE}/districts")
    d = r.json()
    print(f"=== GET /districts === ({len(d)} cities)")
    assert len(d) == 75
    # Check renamed field
    assert "model_confidence" in d[0]
    assert "confidence" not in d[0]
    # Check calibration: no 100% confidence
    confs = [x["model_confidence"] for x in d]
    print(f"  Confidence range: {min(confs):.2%} – {max(confs):.2%}")
    assert max(confs) <= 0.93, f"FAIL: max confidence is {max(confs)} (should be <= 0.93)"
    assert min(confs) >= 0.55, f"FAIL: min confidence is {min(confs)} (should be >= 0.55)"
    print("  Realistic range confirmed (no 100%)!")
    print("PASS\n")

def test_district_detail():
    r = requests.get(f"{BASE}/districts/1")
    d = r.json()
    print("=== GET /districts/1 ===")
    print(f"  City: {d['city']}, Risk: {d['risk']}")
    print(f"  Confidence: {d['model_confidence']:.2%}")
    # New structured SHAP
    assert "shap_explanations" in d
    assert len(d["shap_explanations"]) == 3
    for s in d["shap_explanations"]:
        assert "feature" in s
        assert "display_name" in s
        assert "impact" in s
        assert s["impact"] in ("increase", "decrease")
        assert "strength" in s
        assert s["strength"] in ("high", "medium", "low")
        assert "explanation" in s
        assert len(s["explanation"]) > 10  # meaningful text
        print(f"    {s['display_name']}: {s['impact']} ({s['strength']}) — {s['explanation'][:60]}…")
    assert "dominant_driver" in d
    assert "interaction_note" in d
    assert "prevention_tips" in d
    print(f"  Dominant driver: {d['dominant_driver']}")
    print(f"  Interaction: {d['interaction_note'][:70]}…")
    print(f"  Tips: {len(d['prevention_tips'])}")

    # 404 test
    r2 = requests.get(f"{BASE}/districts/999")
    assert r2.status_code == 404
    print("PASS\n")

def test_predict():
    print("=== POST /predict ===")
    r1 = requests.post(f"{BASE}/predict", json={"rainfall": 80, "temperature": 28, "humidity": 85})
    d1 = r1.json()
    print(f"  High scenario: {d1['risk']} ({d1['model_confidence']:.2%})")
    assert "model_confidence" in d1
    assert d1["model_confidence"] < 1.0

    r2 = requests.post(f"{BASE}/predict", json={"rainfall": 0, "temperature": 10, "humidity": 30})
    d2 = r2.json()
    print(f"  Low scenario:  {d2['risk']} ({d2['model_confidence']:.2%})")
    assert d2["model_confidence"] < 1.0
    print("PASS\n")

def test_chat():
    print("=== POST /chat ===")
    tests = [
        ({"message": "hello", "language": "en"}, "greeting"),
        ({"message": "Guwahati", "language": "en"}, "city"),
        ({"message": "risk", "language": "en"}, "risk"),
        ({"message": "help", "language": "en"}, "help"),
        ({"message": "precaution", "language": "en"}, "tips"),
        ({"message": "Imphal", "language": "hi"}, "hindi"),
        ({"message": "নমস্কাৰ", "language": "as"}, "assamese"),
    ]
    for body, label in tests:
        r = requests.post(f"{BASE}/chat", json=body)
        d = r.json()
        assert len(d["reply"]) > 5
        print(f"  {label:10s} → {d['reply'][:65]}…")
    print("PASS\n")

if __name__ == "__main__":
    test_health()
    test_districts()
    test_district_detail()
    test_predict()
    test_chat()
    print("=" * 50)
    print("ALL v3.0 TESTS PASSED")
    print("=" * 50)
