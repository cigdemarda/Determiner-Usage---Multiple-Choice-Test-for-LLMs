import csv
import json
import re
import time
import requests
import pandas as pd
import numpy as np
import os
from pathlib import Path
import unicodedata



# Default: Local LM Studio server, but can be overridden by environment variable
LM_STUDIO_URL = os.getenv(
    "LM_STUDIO_URL", 
    "http://localhost:1234/v1/chat/completions"
)
MODEL_ID      = "Phi-4-mini-instruct"  
# Model: lmstudio-community/Phi-4-mini-instruct-GGUF, 
# Quantization: Q4_K_M,
# Model Card: https://huggingface.co/lmstudio-community/Phi-4-mini-instruct-GGUF


INPUT_CSV  = "first1k.csv"
OUTPUT_CSV = "combined_with_model.csv"
LOG_FILE   = "phi_raw_log.txt"

SYSTEM_PROMPT = (
    'You are answering multiple-choice questions about noun phrase grammaticality.\n'
    'For each question:\n'
    '1. Generate several reasoning paths step by step (do NOT show your reasoning in the final answer).\n'
    '2. Compare the different reasoning paths and select the most consistent final answer.\n'
    '3. Provide only the final answer in the required format:\n'
    '   - If grammatical: output only "A"\n'
    '   - If ungrammatical: output "B" on one line, then the corrected noun phrase on the next line.\n\n'

    'Now proceed with the questions using this method.'
)



QUESTION_TEMPLATE = (
    'In the sentence "{sentence}", is the noun phrase "{noun_phrase}" grammatical?\n'
    'A: yes\n'
    'B: no\n'
    'Think through multiple reasoning paths before deciding, then output only the final answer:\n'
    '- If A: output only "A"\n'
    '- If B: output "B" on one line, then the corrected noun phrase on the next line.'
)




def asking_phi(prompt_text):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 50   #changable
    }
    response = requests.post(
        LM_STUDIO_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=60
    )
    response.raise_for_status()
    reply_text = response.json()["choices"][0]["message"]["content"].strip()
    return reply_text

def parse_reply(raw_text):
    if not raw_text or not str(raw_text).strip():
        return "", "", ""

    lines = str(raw_text).strip().splitlines()
    first_line = lines[0].strip()

    def normalize_text(text):
        import re, unicodedata
        if text is None:
            return ""
        text = unicodedata.normalize("NFKC", str(text))
        text = text.replace("\u00A0", " ")
        text = text.strip().casefold()
        text = re.sub(r"\s+", " ", text)
        return text

    # A detection
    if re.match(r"^\s*A(\b|[^A-Za-z0-9])", first_line, flags=re.IGNORECASE):
        return "A", "", ""

    # B detection
    if re.match(r"^\s*B(\b|[^A-Za-z0-9])", first_line, flags=re.IGNORECASE):
        after_b = re.sub(r"^\s*B\s*[:)\-]?\s*", "", first_line, flags=re.IGNORECASE)
        if after_b:
            return "B", after_b, normalize_text(after_b)
        for line in lines[1:]:
            if re.match(r"^\s*reformulation\s*[:\-]", line, flags=re.IGNORECASE):
                fix_part = re.sub(r"^\s*reformulation\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
                return "B", fix_part, normalize_text(fix_part)
            if line.strip():
                return "B", line.strip(), normalize_text(line.strip())
        return "B", "", ""

    # Yes/No fallback
    if first_line.lower().startswith("yes"):
        return "A", "", ""
    if first_line.lower().startswith("no"):
        for line in lines[1:]:
            if re.match(r"^\s*reformulation\s*[:\-]", line, flags=re.IGNORECASE):
                fix_part = re.sub(r"^\s*reformulation\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
                return "B", fix_part, normalize_text(fix_part)
            if line.strip():
                return "B", line.strip(), normalize_text(line.strip())
        return "B", "", ""

    return "", "", ""


data = pd.read_csv(INPUT_CSV)
print("Loaded rows:", len(data))



answer_A = []
answer_B = []
model_fixes = []
model_fixes_norm = []

start_all = time.time()

with Path(LOG_FILE).open("w", encoding="utf8") as log_file:
    for index in range(len(data)):
        row = data.iloc[index]

        prompt_text = QUESTION_TEMPLATE.format(
            sentence = row["sentence"],
            noun_phrase = row["noun_phrase"]
        )

        try:
            raw_reply = asking_phi(prompt_text)
        except Exception as exc:
            raw_reply = "ERROR: " + str(exc)

        letter, fix_raw, fix_norm = parse_reply(raw_reply)

        if letter == "A":
            answer_A.append("A")
            answer_B.append("")
        elif letter == "B":
            answer_A.append("")
            answer_B.append("B")
        else:
            answer_A.append("")
            answer_B.append("")

        model_fixes.append(fix_raw)
        model_fixes_norm.append(fix_norm)
        log_file.write(str(index) + "\t" + letter + "\t" + fix_raw + "\t" + raw_reply + "\n")

        if (index + 1) % 50 == 0 or index == len(data) - 1:
            print(index + 1, "/", len(data), "rows processed")

elapsed_all = time.time() - start_all
print("Phi run time (sec):", round(elapsed_all, 1))



data["A"] = answer_A
data["B"] = answer_B
data["model_fix"] = model_fixes
data["model_fix_norm"] = model_fixes_norm

data.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print("Model answers saved to", OUTPUT_CSV)




# Accuracy
# Determine the gold_label (Yes/No for original NP grammaticality)
# Create the 'gold_label' column based on the presence of a 'gold_fix'
llm_answers = []
for _, row in data.iterrows():
    if str(row.get("A", "")).strip().upper() == "A":
        llm_answers.append("A")
    elif str(row.get("B", "")).strip().upper() == "B":
        llm_answers.append("B")
    else:
        llm_answers.append("")   # unusable/empty model output
data["llm_answer"] = llm_answers
data["llm_answer_norm"] = data["llm_answer"].astype(str).str.strip().str.upper()

# Create the gold_label from gold_fix (empty column -> A, not empty column -> B)
gold_labels = []
for _, row in data.iterrows():
    if pd.isna(row["gold_fix"]) or str(row["gold_fix"]).strip() == "":
        gold_labels.append("A")
    else:
        gold_labels.append("B")
data["gold_label"] = pd.Series(gold_labels).str.strip().str.upper()


data["is_correct"] = data["llm_answer_norm"] == data["gold_label"]


num_correct = int(data["is_correct"].sum())
num_total = len(data)   # or 1000 
accuracy = num_correct / num_total



# Checking the "fix" correctness for the 'No' rows

# True where the correct label is B (gold_fix is filled and not just spaces)
gold_requires_fix = ~data["gold_fix"].isna() & (data["gold_fix"].astype(str).str.strip() != "")
# True where the model predicted B
model_predicted_B = data["llm_answer_norm"] == "B"

# Rows where a fix was required AND the model actually attempted a fix
rows_with_fix_attempt = data.loc[gold_requires_fix & model_predicted_B].copy()
rows_with_fix_attempt["gold_fix_norm"] = rows_with_fix_attempt["gold_fix"].astype(str).str.strip().str.casefold()

# Mark correct fixes
rows_with_fix_attempt["fix_is_correct"] = rows_with_fix_attempt["model_fix_norm"] == rows_with_fix_attempt["gold_fix_norm"]

# Metrics
fix_attempt_count = len(rows_with_fix_attempt)   # cases where fix was required and model attempted it
fix_correct_count = int(rows_with_fix_attempt["fix_is_correct"].sum())
fix_accuracy_when_attempted = (fix_correct_count / fix_attempt_count) if fix_attempt_count > 0 else float("nan")

# Coverage: How often the model tried a fix when it should have
total_gold_B = int(gold_requires_fix.sum())
fix_attempt_coverage = (fix_attempt_count / total_gold_B) if total_gold_B > 0 else float("nan")

# Fix recall: proportion of all gold = B cases where the fix was both attempted and correct
fix_recall_over_gold_B = (fix_correct_count / total_gold_B) if total_gold_B > 0 else float("nan")



# For dumb models
gold_A = data["gold_label"] == "A"
gold_B = data["gold_label"] == "B"
pred_A = data["llm_answer_norm"] == "A"
pred_B = data["llm_answer_norm"] == "B"

# Confusion counts 
TP_A = int((gold_A & pred_A).sum()) # gold A, predicted A
FN_A = int((gold_A & ~pred_A).sum()) # gold A, predicted not A 

TP_B = int((gold_B & pred_B).sum())  # gold B, predicted B
FN_B = int((gold_B & ~pred_B).sum())# gold B, predicted not B 

# Denominators per class
n_A = TP_A + FN_A  # number of actual A's
n_B = TP_B + FN_B  # number of actual B's

# Per-class recall
recall_A = TP_A / n_A if n_A > 0 else float("nan")
recall_B = TP_B / n_B if n_B > 0 else float("nan")

# Balanced accuracy (average of per-class recalls) 
balanced_accuracy = np.nanmean([recall_A, recall_B])


print(f"Correct: {num_correct}/{num_total}")
print(f"Label Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Per-Class Recall_A: {recall_A:.3f}  |  Recall_B: {recall_B:.3f}")
print(f"Balanced Label Accuracy: {balanced_accuracy:.3f}")

print(f"Total gold=B cases (requires fix): {total_gold_B}")
print(f"Fix attempts (model predicted B): {fix_attempt_count}")
print(f"Correct fixes (when attempted): {fix_correct_count}")
print(f"Fix Accuracy (when attempted): {fix_accuracy_when_attempted:.3f} ({fix_accuracy_when_attempted*100:.1f}%)")
print(f"Fix Attempt Coverage (predicted B | gold B): {fix_attempt_coverage:.3f} ({fix_attempt_coverage*100:.1f}%)")
print(f"Fix Recall over gold=B: {fix_recall_over_gold_B:.3f} ({fix_recall_over_gold_B*100:.1f}%)")
