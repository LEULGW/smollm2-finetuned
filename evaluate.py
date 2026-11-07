
import json
from collections import Counter

# 1. Load data
def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

test = load("test.jsonl")
predictions = load("smollm2_predictions.jsonl")

ids = [x["id"] for x in predictions]
assert len(ids) == len(set(ids)), "Duplicate prediction IDs"
assert len({x["id"] for x in test}) == len(test), "Duplicate test IDs"

lookup = {x["id"]: x.get("prediction") for x in predictions}

CATEGORIES = set("""
PLACE_OF_BIRTH RACE HEIGHT MARITAL_STATUS COUNTRY_OF_CITIZENSHIP
SHOPPING_BEHAVIOR ZIP_CODE_PLUS_4 NUMBER_OF_CHILDREN JOB_TITLE HOMETOWN
INCOME_LEVEL OCCUPATION GENDER BIRTH_DATE PERSON_NAME AGE ONLINE_SCREEN_NAME
RELIGION WEIGHT POLITICAL_AFFILIATION SEXUAL_PREFERENCE
DOCUMENTATION_OF_GRIEVANCES SIGNED_PETITIONS SURVEY_ANSWERS
EMAIL_ADDRESS IP_ADDRESS HOME_PHONE_NUMBER CELL_PHONE_NUMBER
WORK_ADDRESS WORK_CONTACT_INFORMATION WORK_PHONE_NUMBER
FAMILY_FRIEND_CONTACT_INFORMATION SOCIAL_NETWORK_PROFILE
LAW_ENFORCEMENT_FILES MOTHERS_MAIDEN_NAME DRIVERS_LICENSE_NUMBER
VEHICLE_REGISTRATION_NUMBER LICENSE_PLATE_NUMBER CREDIT_CARD_NUMBER
CREDIT_SCORE FINANCIAL_ACCOUNT_NUMBER MEDICAL_HISTORY HOME_ADDRESS
SOCIAL_SECURITY_NUMBER GPS_LOCATION SECURITY_ACCESS_CODE
HEALTH_INSURANCE_ID PASSPORT_NUMBER
""".split())

# 2. Validate output
def valid(p, prompt):
    if not isinstance(p, dict):
        return False
    if set(p) != {"spans", "has_sensitive_info"}:
        return False
    if not isinstance(p["spans"], list):
        return False
    if type(p["has_sensitive_info"]) is not bool:
        return False
    if p["has_sensitive_info"] != bool(p["spans"]):
        return False

    for s in p["spans"]:
        if not isinstance(s, dict) or set(s) != {"text", "category"}:
            return False
        if not isinstance(s["text"], str) or not s["text"]:
            return False
        if s["text"] not in prompt:
            return False
        if not isinstance(s["category"], str) or s["category"] not in CATEGORIES:
            return False
    return True

# 3. Metrics
def metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return p, r, f1

text_tp = text_fp = text_fn = 0
joint_tp = joint_fp = joint_fn = 0
boolean_correct = exact_correct = valid_count = 0

for x in test:
    pred = lookup.get(x["id"])
    ok = valid(pred, x["prompt"])
    valid_count += int(ok)

    gold = x["spans"]
    predicted = pred["spans"] if ok else []

    # Text-only matching
    gt = Counter(s["text"] for s in gold)
    pt = Counter(s["text"] for s in predicted)
    tp = sum((gt & pt).values())

    text_tp += tp
    text_fp += sum(pt.values()) - tp
    text_fn += sum(gt.values()) - tp

    # Exact text + category matching
    gj = Counter((s["text"], s["category"]) for s in gold)
    pj = Counter((s["text"], s["category"]) for s in predicted)
    tp = sum((gj & pj).values())

    joint_tp += tp
    joint_fp += sum(pj.values()) - tp
    joint_fn += sum(gj.values()) - tp

    if ok:
        bool_ok = pred["has_sensitive_info"] == x["has_sensitive_info"]
        boolean_correct += int(bool_ok)
        exact_correct += int(bool_ok and gj == pj)

# 4. Results
n = len(test)
assert n > 0, "Empty test dataset"

p, r, f1 = metrics(text_tp, text_fp, text_fn)
jp, jr, jf1 = metrics(joint_tp, joint_fp, joint_fn)
cat_acc = joint_tp / text_tp if text_tp else 0

print("\n===== SMOLLM2 EVALUATION =====")
print(f"Total prompts: {n}")
print(f"Valid predictions: {valid_count}")
print(f"Failed/null predictions: {n - valid_count}")
print(f"Valid output rate: {valid_count/n:.2%}")

print("\n--- TEXT DETECTION ---")
print(f"TP: {text_tp} | FP: {text_fp} | FN: {text_fn}")
print(f"Precision: {p:.2%}")
print(f"Recall: {r:.2%}")
print(f"F1 Score: {f1:.2%}")

print("\n--- TEXT + CATEGORY ---")
print(f"TP: {joint_tp} | FP: {joint_fp} | FN: {joint_fn}")
print(f"Joint Precision: {jp:.2%}")
print(f"Joint Recall: {jr:.2%}")
print(f"Joint F1: {jf1:.2%}")
print(f"Category Accuracy: {cat_acc:.2%}")

print("\n--- PROMPT LEVEL ---")
print(f"Boolean Accuracy: {boolean_correct/n:.2%}")
print(f"Exact Match Accuracy: {exact_correct/n:.2%}")
