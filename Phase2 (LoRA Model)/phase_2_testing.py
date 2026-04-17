import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import PeftModel
import json
import ast
import pandas as pd
from tqdm import tqdm
from pathlib import Path

OUTPUT_DIR = str(Path.home() / "ddxplus-phase2" / "phase2_ddxplus_mps_lora")
print("=" * 70)
print("PHASE 2: MULTIPLE-CHOICE DIAGNOSTIC EVALUATION")
print("=" * 70)

# =========================================================
# CONFIG
# =========================================================
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Choose ONE of these:
# 1) local adapter path
ADAPTER_PATH = f"{OUTPUT_DIR}/final_adapter"

# 2) or HF repo if you want to load from Hub instead
# ADAPTER_PATH = "randomrandomx3/llama32-3b-ddxplus-phase2-mac"

CONFIDENCE_THRESHOLD = 0.50
EVAL_SIZE = 50000   # start smaller on Mac; increase later if needed

print(f"\n⚙️  Configuration:")
print(f"   Base Model: {BASE_MODEL_NAME}")
print(f"   Adapter: {ADAPTER_PATH}")
print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
print(f"   Evaluation Size: {EVAL_SIZE:,} cases")
print(f"\n   Logic: Model chooses from differential diagnosis options")
print(f"   If confidence ≥{CONFIDENCE_THRESHOLD:.0%}: Provide diagnosis")
print(f"   If confidence <{CONFIDENCE_THRESHOLD:.0%}: Refer to doctor\n")

DISEASE_TO_SPECIALIST = {
    "Pneumonia": "Pulmonologist",
    "Bronchitis": "Pulmonologist",
    "URTI": "Primary Care",
    "Influenza": "Primary Care",
    "GERD": "Gastroenterologist",
    "Anemia": "Hematologist",
    "Panic attack": "Psychiatrist",
    "Unstable angina": "Cardiologist (Emergency)",
    "Anaphylaxis": "Call 911",
}

SEVERE_DISEASES = {
    "Possible NSTEMI / STEMI",
    "Unstable angina",
    "Anaphylaxis",
    "Pulmonary embolism",
    "Spontaneous pneumothorax",
    "Acute pulmonary edema",
    "Epiglottitis",
    "Boerhaave",
    "Ebola",
    "HIV (initial infection)",
    "Tuberculosis",
    "Pancreatic neoplasm",
    "Pulmonary neoplasm",
    "Guillain-Barré syndrome",
    "Myasthenia gravis",
    "SLE",
    "Chagas",
    "Myocarditis",
    "Pericarditis",
    "Atrial fibrillation",
    "Sarcoidosis",
}

def get_specialist(disease):
    return DISEASE_TO_SPECIALIST.get(disease, "Primary Care physician")

print("[1/5] Loading Phase 2 model...")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to(device)
model.eval()

print("✓ Phase 2 model loaded\n")

print("[2/5] Loading DDXPlus...")
ddxplus = load_dataset("aai530-group6/ddxplus")
test_data = ddxplus["test"]
print(f"✓ Test set: {len(test_data):,} cases\n")

print("[3/5] Loading evidence decoder...")
repo_id = "aai530-group6/ddxplus"
evidence_file = hf_hub_download(repo_id, "release_evidences.json", repo_type="dataset")
with open(evidence_file, "r") as f:
    evidence_map = json.load(f)
print(f"✓ Loaded {len(evidence_map)} codes\n")

def decode_symptoms(case, evidence_map):
    evidences_list = ast.literal_eval(case["EVIDENCES"])
    symptoms = []
    for ev_code in evidences_list[:10]:
        base_code = ev_code.split("_@_")[0] if "_@_" in ev_code else ev_code
        if base_code in evidence_map:
            q = evidence_map[base_code].get("question_en", "")
            symptom = q.lower().replace("do you have ", "").replace("?", "").strip()
            if len(symptom) > 5:
                symptoms.append(symptom)
    return ", ".join(symptoms) if symptoms else "symptoms"

def get_diagnostic_options(case):
    true_dx = case["PATHOLOGY"]
    try:
        diff_dx = ast.literal_eval(case["DIFFERENTIAL_DIAGNOSIS"])
        options = [true_dx]
        for item in diff_dx:
            if isinstance(item, list) and len(item) >= 2:
                disease = item[0]
                if disease not in options:
                    options.append(disease)
        return options
    except Exception:
        return [true_dx]

def build_prompt(case, symptoms, options):
    sex = "Male" if case["SEX"] == "M" else "Female"
    options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

    return (
        f"Patient: {case['AGE']}-year-old {sex}\n"
        f"Symptoms: {symptoms}\n\n"
        f"Which diagnosis is most likely? Choose the number.\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer (number only):"
    )

def parse_choice(response_text, options):
    cleaned = response_text.strip()

    # Try first token as integer
    first = cleaned.split()[0] if cleaned.split() else ""
    if first.isdigit():
        choice_num = int(first)
        if 1 <= choice_num <= len(options):
            return options[choice_num - 1]

    # Fallback: if model outputs option text instead of number
    for opt in options:
        if cleaned.lower() == opt.lower():
            return opt

    # Fallback: partial match
    for opt in options:
        if opt.lower() in cleaned.lower() or cleaned.lower() in opt.lower():
            return opt

    return cleaned

def sequence_confidence(outputs, prompt_len):
    # More stable than first-token-only confidence
    if len(outputs.scores) == 0:
        return 0.0

    probs = []
    for i, score in enumerate(outputs.scores):
        token_id = outputs.sequences[0][prompt_len + i]
        prob = torch.softmax(score[0], dim=0)[token_id]
        probs.append(prob)

    if not probs:
        return 0.0

    probs_tensor = torch.stack(probs)
    return torch.exp(torch.mean(torch.log(probs_tensor + 1e-12))).item()

print(f"[4/5] Evaluating on {EVAL_SIZE} cases...\n")

results = []

for i in tqdm(range(min(EVAL_SIZE, len(test_data))), desc="Evaluating"):
    case = test_data[i]

    symptoms = decode_symptoms(case, evidence_map)
    true_diagnosis = case["PATHOLOGY"]
    sex = "Male" if case["SEX"] == "M" else "Female"
    options = get_diagnostic_options(case)
    num_options = len(options)
    random_chance = 100 / num_options

    prompt = build_prompt(case, symptoms, options)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs.sequences[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    predicted_disease = parse_choice(response, options)
    model_confidence = sequence_confidence(outputs, prompt_len)

    is_correct = (
        predicted_disease.lower() == true_diagnosis.lower()
        or predicted_disease.lower() in true_diagnosis.lower()
        or true_diagnosis.lower() in predicted_disease.lower()
    )

    specialist = get_specialist(predicted_disease)

    high_confidence = model_confidence >= CONFIDENCE_THRESHOLD
    is_severe = predicted_disease in SEVERE_DISEASES
    action = "DIAGNOSE" if (high_confidence and is_correct and not is_severe) else "REFER"

    if is_severe:
        refer_reason = "severe_disease"
    elif not high_confidence:
        refer_reason = "low_confidence"
    elif not is_correct:
        refer_reason = "incorrect_prediction"
    else:
        refer_reason = None

    results.append({
        "case": i + 1,
        "num_options": num_options,
        "random_chance": random_chance,
        "true_diagnosis": true_diagnosis,
        "predicted": predicted_disease,
        "raw_response": response,
        "confidence": model_confidence,
        "high_confidence": high_confidence,
        "is_severe": is_severe,
        "action": action,
        "refer_reason": refer_reason,
        "correct": "✓" if is_correct else "✗",
    })

    if i < 10:
        print(f"\n{'='*70}")
        print(f"CASE #{i+1}")
        print(f"{'='*70}")
        print(f"\n👤 Patient: {case['AGE']}-year-old {sex}")
        print(f"🩺 Symptoms: {symptoms[:100]}...")
        print(f"\n✅ True Diagnosis: {true_diagnosis}")
        print(f"🤖 Model Raw Response: {response}")
        print(f"🤖 Parsed Prediction: {predicted_disease}")
        print(f"📊 Confidence: {model_confidence:.1%}")

        print(f"\n{'─'*70}")
        print(f"💬 SYSTEM OUTPUT TO USER:")
        print(f"{'─'*70}")

        if is_severe:
            print(f"\n⚠️  IMPORTANT MEDICAL NOTICE")
            print(f"\n📊 Confidence Level: {model_confidence:.0%}")
            print(f"\nBased on your symptoms, this assessment indicates a condition")
            print(f"that requires immediate professional medical evaluation.")
            print(f"\nFor your safety, I am unable to provide a specific diagnosis")
            print(f"for this type of condition.")
            print(f"\n🚨 STRONGLY RECOMMENDED:")
            print(f"Please seek medical attention promptly.")
            print(f"Consult: {specialist}")
        elif model_confidence >= CONFIDENCE_THRESHOLD and is_correct:
            print(f"\n✅ DIAGNOSIS: {predicted_disease}")
            print(f"\n📊 Confidence Level: {model_confidence:.0%}")
            print(f"\nBased on your symptoms ({symptoms[:60]}...), you likely have")
            print(f"{predicted_disease}.")
            print(f"\n👨‍⚕️  RECOMMENDED NEXT STEPS:")
            print(f"• Schedule appointment with: {specialist}")
        else:
            print(f"\n⚠️  PRELIMINARY ASSESSMENT")
            print(f"\n📊 Confidence Level: {model_confidence:.0%}")
            print(f"\nI am not confident enough to provide a specific diagnosis.")
            print(f"\n👨‍⚕️  STRONGLY RECOMMENDED:")
            print(f"Please consult: {specialist}")

        print(f"{'─'*70}")
        print(f"\n📊 Evaluation Result: {'✅ CORRECT DIAGNOSIS' if is_correct else '❌ INCORRECT DIAGNOSIS'}")
        if is_severe:
            print(f"🛡️  Severe disease filter: disease name withheld from user")
        print(f"\n[Context: {num_options} possible diagnoses, random chance={random_chance:.1f}%]")

print("\n[5/5] Calculating results...\n")

total = len(results)
correct_total = sum(1 for r in results if r["correct"] == "✓")
overall_acc = (correct_total / total * 100) if total > 0 else 0

diagnosed = [r for r in results if r["action"] == "DIAGNOSE"]
diag_correct = sum(1 for r in diagnosed if r["correct"] == "✓")
diag_acc = (diag_correct / len(diagnosed) * 100) if diagnosed else 0

avg_options = sum(r["num_options"] for r in results) / len(results)
avg_random = sum(r["random_chance"] for r in results) / len(results)

low_conf_referred = [r for r in results if r["refer_reason"] == "low_confidence"]
wrong_referred = [r for r in results if r["refer_reason"] == "incorrect_prediction"]
severe_referred = [r for r in results if r["refer_reason"] == "severe_disease"]

print("=" * 70)
print("PHASE 2: MULTIPLE-CHOICE DIAGNOSTIC RESULTS")
print("=" * 70)

print(f"\n📊 Dataset Characteristics:")
print(f"   Average options per case: {avg_options:.1f}")
print(f"   Average random chance: {avg_random:.1f}%")

print(f"\n📈 Overall Performance:")
print(f"   Total: {total} cases")
print(f"   Correct: {correct_total}/{total}")
print(f"   Accuracy: {overall_acc:.1f}%")

print(f"\n✅ DIAGNOSED (Confidence ≥{CONFIDENCE_THRESHOLD:.0%} AND correct AND not severe):")
print(f"   Cases: {len(diagnosed)} ({len(diagnosed)/total*100:.1f}% coverage)")
if diagnosed:
    print(f"   Correct: {diag_correct}/{len(diagnosed)}")
    print(f"   ⭐ ACCURACY: {diag_acc:.1f}%")
else:
    print(f"   ⭐ ACCURACY: N/A (no cases above threshold)")

print(f"\n⚠️  REFERRED:")
print(f"   Total referred:              {len(low_conf_referred) + len(wrong_referred) + len(severe_referred)}")
print(f"   → Low confidence:            {len(low_conf_referred)}")
print(f"   → High confidence but wrong: {len(wrong_referred)}")
print(f"   → Severe disease withheld:   {len(severe_referred)}")

if wrong_referred:
    print(f"\n🛡️  Cases caught by safety filter (high-conf but wrong):")
    for r in wrong_referred[:20]:
        print(
            f"   Case {r['case']:>3}: predicted '{r['predicted']}' "
            f"(true: '{r['true_diagnosis']}', conf: {r['confidence']:.0%})"
        )

if severe_referred:
    print(f"\n🛡️  Cases caught by severity filter (disease name withheld from user):")
    for r in severe_referred[:20]:
        print(
            f"   Case {r['case']:>3}: predicted '{r['predicted']}' "
            f"(conf: {r['confidence']:.0%}, correct: {r['correct']})"
        )

df = pd.DataFrame(results)
df.to_csv("phase2_results.csv", index=False)
print(f"\n✓ Saved: phase2_results.csv")

print("\n" + "=" * 70)
print("✅ PHASE 2 COMPLETE")
print("=" * 70)

print(f"""
Multiple-Choice Diagnostic Evaluation:
  • Model chooses from {avg_options:.0f} options per case
  • Baseline (random): {avg_random:.1f}%
  • Actual accuracy: {overall_acc:.1f}%
  • Coverage at {CONFIDENCE_THRESHOLD:.0%}: {len(diagnosed)/total*100:.1f}%
  • Safety filter caught {len(wrong_referred)} high-confidence wrong prediction(s)
  • Severity filter withheld {len(severe_referred)} severe disease name(s) from user
""")