
import json

with open("test.jsonl", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]

with open("smollm2_predictions.jsonl", encoding="utf-8") as f:
    predictions = [json.loads(line) for line in f if line.strip()]

text_tp = 0
text_fp = 0
text_fn = 0

category_correct = 0
category_total = 0

boolean_correct = 0
exact_matches = 0

successful = 0
failed = 0

for example in test_data:

    prediction = None

    for item in predictions:
        if item["id"] == example["id"]:
            prediction = item["prediction"]
            break

    if not isinstance(prediction, dict):
        failed += 1
        continue

    if (
        not isinstance(prediction.get("spans"), list)
        or type(prediction.get("has_sensitive_info")) != bool
    ):
        failed += 1
        continue

    if not all(
        isinstance(span, dict)
        and isinstance(span.get("text"), str)
        and isinstance(span.get("category"), str)
        for span in prediction["spans"]
    ):
        failed += 1
        continue

    successful += 1

    gold_spans = example["spans"]
    predicted_spans = prediction["spans"]

    gold_texts = {span["text"] for span in gold_spans}
    predicted_texts = {span["text"] for span in predicted_spans}

    matching_texts = gold_texts & predicted_texts

    text_tp += len(matching_texts)
    text_fp += len(predicted_texts - gold_texts)
    text_fn += len(gold_texts - predicted_texts)

    gold_categories = {
        span["text"]: span["category"] for span in gold_spans
    }

    predicted_categories = {
        span["text"]: span["category"] for span in predicted_spans
    }

    for text in matching_texts:
        category_total += 1

        if gold_categories[text] == predicted_categories[text]:
            category_correct += 1

    if prediction["has_sensitive_info"] == example["has_sensitive_info"]:
        boolean_correct += 1

    gold_pairs = {
        (span["text"], span["category"]) for span in gold_spans
    }

    predicted_pairs = {
        (span["text"], span["category"]) for span in predicted_spans
    }

    if (
        gold_pairs == predicted_pairs
        and prediction["has_sensitive_info"] == example["has_sensitive_info"]
    ):
        exact_matches += 1


precision = (
    text_tp / (text_tp + text_fp) * 100
    if text_tp + text_fp else 0
)

recall = (
    text_tp / (text_tp + text_fn) * 100
    if text_tp + text_fn else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

category_accuracy = (
    category_correct / category_total * 100
    if category_total else 0
)

boolean_accuracy = (
    boolean_correct / successful * 100
    if successful else 0
)

exact_match_accuracy = (
    exact_matches / successful * 100
    if successful else 0
)


print("\n===== SMOLLM2 EVALUATION RESULTS =====")

print(f"Total test cases: {len(test_data)}")
print(f"Successful predictions: {successful}")
print(f"Failed predictions: {failed}")

print("\n--- Prompt-Level Evaluation ---")
print(f"Correct has_sensitive_info predictions: {boolean_correct}/{successful}")
print(f"Boolean Accuracy: {boolean_accuracy:.2f}%")
print(f"Completely correct predictions: {exact_matches}/{successful}")
print(f"Exact Match Accuracy: {exact_match_accuracy:.2f}%")

print("\n--- Individual Span Evaluation ---")
print(f"Total sensitive spans in correct answers: {text_tp + text_fn}")
print(f"Total spans identified by SmolLM2: {text_tp + text_fp}")
print(f"Correctly identified spans: {text_tp}")
print(f"Incorrectly identified spans: {text_fp}")
print(f"Sensitive spans SmolLM2 missed: {text_fn}")

print(f"\nPrecision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")

print(f"\nCorrect category assignments: {category_correct}/{category_total}")
print(f"Category Accuracy: {category_accuracy:.2f}%")
