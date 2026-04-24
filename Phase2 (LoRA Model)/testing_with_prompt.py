import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model
MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

print("Loaded locally")

SEVERE_DISEASES = {
    # Life-threatening emergencies
    "Possible NSTEMI / STEMI",
    "Unstable angina",
    "Anaphylaxis",
    "Pulmonary embolism",
    "Spontaneous pneumothorax",
    "Acute pulmonary edema",
    "Epiglottitis",
    "Boerhaave",
    "Ebola",

    # Serious chronic/life-altering
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

severe_disease_list = "\n".join(
    [f"- {disease}" for disease in sorted(SEVERE_DISEASES)]
)

"""
STEP 2: Phase 1 Baseline Evaluation (Simplified)
Just test the base Llama model - no training yet
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
import ast
import pandas as pd
import random
from tqdm import tqdm

print("="*70)
print("STEP 2: PHASE 1 BASELINE EVALUATION")
print("="*70)

# ============================================
# 1. Load the base model hf_mAYRrasGeiBczuQmNtokenNITALNuQLNEUkKnNW
# ============================================

print("\n[1/5] Loading base Llama model (no training)...")
print("This will download ~6GB if first time...\n")

print("\n[1/5] Loading base Llama model (no training)...")
print("This will download ~6GB if first time...\n")

print(f"Using model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto"
)
model.eval()

print("✓ Base model loaded (this is Meta's pre-trained model)")
print("✓ No medical fine-tuning yet - this is the baseline")

# ============================================
# 2. Load DDXPlus test set
# ============================================

print("\n[2/5] Loading DDXPlus test set...")

ddxplus = load_dataset("aai530-group6/ddxplus")
test_data = ddxplus['test']

print(f"✓ Test set: {len(test_data):,} cases")

# Choose evaluation size
print(f"\n⚠️  IMPORTANT: Choose evaluation size")
print(f"   • Full test set: {len(test_data):,} cases (~10-15 hours!)")
print(f"   • Random sample: 50,000 cases")
print(f"   • Quick test: 1,000 cases")

# Random sample size, without duplicate test cases
EVAL_SIZE = min(50000, len(test_data))
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
eval_indices = random.sample(range(len(test_data)), EVAL_SIZE)

print(f"\n✓ Will evaluate on {EVAL_SIZE:,} random non-duplicate cases")
print(f"✓ Random seed: {RANDOM_SEED}")

if EVAL_SIZE > 1000:
    print(f"\n⚠️  WARNING: This will take approximately {EVAL_SIZE * 0.06 / 60:.1f} hours")
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted. Change EVAL_SIZE to a smaller number.")
        exit()

# ============================================
# 3. Load evidence mapping
# ============================================

print("\n[3/5] Loading evidence decoder...")

repo_id = "aai530-group6/ddxplus"
evidence_file = hf_hub_download(repo_id, "release_evidences.json", repo_type="dataset")

with open(evidence_file, 'r') as f:
    evidence_map = json.load(f)

print(f"✓ Loaded {len(evidence_map)} evidence codes")

# ============================================
# 4. Helper function to decode symptoms
# ============================================

def decode_symptoms(case, evidence_map):
    """
    Convert evidence codes to readable text - IMPROVED VERSION
    """
    # Parse evidences (it's a string that looks like a list)
    evidences_str = case['EVIDENCES']
    evidences_list = ast.literal_eval(evidences_str)

    symptoms = []

    # Decode each evidence
    for ev_code in evidences_list[:12]:  # First 12 symptoms
        # Handle evidence@value format
        if '_@_' in ev_code:
            parts = ev_code.split('_@_')
            base_code = parts[0]
            value_code = parts[1] if len(parts) > 1 else None
        else:
            base_code = ev_code
            value_code = None

        # Skip if not in map
        if base_code not in evidence_map:
            continue

        ev_data = evidence_map[base_code]
        question = ev_data.get('question_en', '')

        if not question:
            continue

        # For binary questions (yes/no)
        if ev_data.get('data_type') == 'B':
            # Convert question to statement
            symptom = question.lower()
            symptom = symptom.replace('do you have ', '')
            symptom = symptom.replace('have you ', '')
            symptom = symptom.replace('are you ', '')
            symptom = symptom.replace('did you ', '')
            symptom = symptom.replace('?', '')
            symptom = symptom.strip()

            if len(symptom) > 5:  # Only add meaningful symptoms
                symptoms.append(symptom)

        # For questions with values
        elif value_code and 'value_meaning' in ev_data:
            value_meanings = ev_data.get('value_meaning', {})
            if value_code in value_meanings:
                value_text = value_meanings[value_code].get('en', '')
                if value_text and value_text != 'N' and value_text != 'NA':
                    # Extract key part of question
                    q_short = question.split('?')[0].lower()
                    q_short = q_short.replace('do you ', '').replace('have you ', '')
                    symptoms.append(f"{q_short}: {value_text}")

    # Return formatted string
    if len(symptoms) == 0:
        return "patient presents with medical complaint"

    return ", ".join(symptoms)

# ============================================
# 5. Test on a few cases
# ============================================

print(f"\n[4/5] Testing base model on {EVAL_SIZE:,} cases...")
print("Progress will be shown below. This may take a while...\n")

results = []
severe_disease_list = "\n".join(
    [f"- {disease}" for disease in sorted(SEVERE_DISEASES)]
)

for case_num, i in enumerate(tqdm(eval_indices, desc="Evaluating"), start=1):
    case = test_data[i]

    # Decode symptoms
    symptoms = decode_symptoms(case, evidence_map)
    true_diagnosis = case['PATHOLOGY']
    sex_text = "Male" if case['SEX'] == 'M' else "Female"

    # Parse evidences for the prompt
    evidences_list = ast.literal_eval(case['EVIDENCES'])
    
    # Parse differential diagnosis
    try:
        diff_dx_str = case['DIFFERENTIAL_DIAGNOSIS']
        if isinstance(diff_dx_str, str):
            diff_dx_list = ast.literal_eval(diff_dx_str)
        else:
            diff_dx_list = diff_dx_str
        # Extract disease names from differential (format: [["Disease", probability], ...])
        differential_diseases = []
        for item in diff_dx_list:
            if isinstance(item, list) and len(item) >= 2:
                differential_diseases.append(item[0])
            elif isinstance(item, str):
                differential_diseases.append(item)

        if true_diagnosis not in differential_diseases:
            differential_diseases.append(true_diagnosis)

        confidence_threshold = 2 / len(differential_diseases) if differential_diseases else 1.0
        diagnosis_options = "\n".join(
            [f"{idx + 1}. {disease}" for idx, disease in enumerate(differential_diseases)]
        )
    except Exception:
        differential_diseases = [true_diagnosis]
        confidence_threshold = 1.0
        diagnosis_options = f"1. {true_diagnosis}"

    # SYSTEM PROMPT (Model's role and instructions)
    system_prompt = """You are a medical diagnosis assistant for a research prototype.

Task:
Choose the top 3 most likely diseases from the provided diagnosis_options, then apply the final decision rule.

Rules:
1. Return valid JSON only.
2. Use only diseases from diagnosis_options.
3. Return exactly 3 predictions ranked by likelihood.
4. Each prediction must include rank, disease, and confidence.
5. Confidence must be between 0 and 1.
6. Apply the final decision rule exactly.
7. Do not provide treatment advice.

Final decision rule:
- If rank 1 disease is in severe_disease_list, final_prediction disease must be "Refer to specialist" and reason must be "severity".
- Else if rank 1 confidence < confidence_threshold, final_prediction disease must be "Refer to specialist" and reason must be "low_confidence".
- Else final_prediction disease must be rank 1 disease and reason must be "none".

Return JSON in this format:
{
  "top_3_predictions": [
    {"rank": 1, "disease": "Disease Name", "confidence": 0.6},
    {"rank": 2, "disease": "Disease Name", "confidence": 0.25},
    {"rank": 3, "disease": "Disease Name", "confidence": 0.15}
  ],
  "final_prediction": {
    "disease": "Disease Name or Refer to specialist",
    "reason": "none or severity or low_confidence",
    "confidence": 0.6
  }
}
"""

    # USER PROMPT (The actual patient case)
    user_prompt = f"""Patient Information:
- Age: {case['AGE']} years
- Sex: {sex_text}
- Symptoms: {symptoms}
- Number of evidences: {len(evidences_list)}
- confidence_threshold: {confidence_threshold}

severe_disease_list:
{severe_disease_list}



diagnosis_options:
{diagnosis_options}

Based on this patient's presentation, provide your diagnostic assessment with the top 3 most likely diagnoses and confidence scores."""

    # Format for chat model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Print detailed output only for first 5 cases
    if i < 5:
        print(f"\n{'='*70}")
        print(f"CASE #{case_num} - PROMPT SENT TO MODEL")
        print(f"{'='*70}")
        print(formatted_prompt)
        print(f"{'='*70}")

    # Get model prediction
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode prediction (FULL response, no cleanup)
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # PRINT THE RAW MODEL RESPONSE (only first 5 cases)
    if i < 5:
        print(f"\n{'='*70}")
        print(f"CASE #{case_num} - RAW MODEL RESPONSE")
        print(f"{'='*70}")
        print(prediction)
        print(f"{'='*70}\n")

    # Try to parse JSON response
    response_json = {}
    top_3 = []
    final_prediction = {}
    parse_ok = False
    parse_error = None

    try:
        json_text = prediction
        if '```json' in json_text:
            json_text = json_text.split('```json')[1].split('```')[0]
        elif '```' in json_text:
            json_text = json_text.split('```')[1].split('```')[0]

        response_json = json.loads(json_text.strip())
        parse_ok = True
        top_3 = response_json.get('top_3_predictions', [])
        final_prediction = response_json.get('final_prediction', {})

        if final_prediction:
            predicted_disease = final_prediction.get('disease', prediction[:50])
            model_confidence = final_prediction.get('confidence', 0.5)
        elif top_3:
            predicted_disease = top_3[0].get('disease', prediction[:50])
            model_confidence = top_3[0].get('confidence', 0.5)
        else:
            predicted_disease = prediction[:50]
            model_confidence = 0.5

    except Exception as e:
        parse_error = str(e)
        predicted_disease = prediction.split('\n')[0][:50]
        model_confidence = 0.5
        top_3 = []

    rank1 = top_3[0] if len(top_3) > 0 else {}
    rank2 = top_3[1] if len(top_3) > 1 else {}
    rank3 = top_3[2] if len(top_3) > 2 else {}
    rank1_disease = rank1.get('disease')
    rank1_confidence = rank1.get('confidence')
    rank2_disease = rank2.get('disease')
    rank2_confidence = rank2.get('confidence')
    rank3_disease = rank3.get('disease')
    rank3_confidence = rank3.get('confidence')
    final_reason = final_prediction.get('reason')

    # Check diagnostic accuracy against rank 1, not referral text.
    diagnosis_for_accuracy = rank1_disease or predicted_disease
    is_correct = False

    if diagnosis_for_accuracy.lower() == true_diagnosis.lower():
        is_correct = True
    elif diagnosis_for_accuracy.lower() in true_diagnosis.lower():
        is_correct = True
    elif true_diagnosis.lower() in diagnosis_for_accuracy.lower():
        is_correct = True

    # ============================================
    # BUG FIX: Append result to results list
    # ============================================
    results.append({
        'case_num': case_num,
        'original_test_index': i,
        'age': case['AGE'],
        'sex': sex_text,
        'symptoms': symptoms,
        'num_evidences': len(evidences_list),
        'raw_evidences': case['EVIDENCES'],
        'raw_differential_diagnosis': case['DIFFERENTIAL_DIAGNOSIS'],
        'true_diagnosis': true_diagnosis,
        'diagnosis_options': diagnosis_options,
        'num_diagnosis_options': len(differential_diseases),
        'confidence_threshold': confidence_threshold,
        'severe_disease_list': severe_disease_list,
        'rank1_disease': rank1_disease,
        'rank1_confidence': rank1_confidence,
        'rank2_disease': rank2_disease,
        'rank2_confidence': rank2_confidence,
        'rank3_disease': rank3_disease,
        'rank3_confidence': rank3_confidence,
        'final_prediction': predicted_disease,
        'final_confidence': model_confidence,
        'final_reason': final_reason,
        'final_is_referral': predicted_disease == 'Refer to specialist',
        'diagnosis_for_accuracy': diagnosis_for_accuracy,
        'diagnostic_correct': '✓' if is_correct else '✗',
        'correct': '✓' if is_correct else '✗',
        'parse_ok': parse_ok,
        'parse_error': parse_error,
        'raw_model_response': prediction,
        'parsed_model_json': json.dumps(response_json, ensure_ascii=False)
    })

    # Print summary only (detailed view only for first 5 cases)
    if i < 5:
        print(f"\n{'='*70}")
        print(f"CASE #{case_num} - SUMMARY")
        print(f"{'='*70}")

        print(f"\n👤 PATIENT:")
        print(f"   Age: {case['AGE']} years")
        print(f"   Sex: {sex_text}")

        print(f"\n🔍 DIFFERENTIAL DIAGNOSIS PROVIDED:")
        print(diagnosis_options)

        print(f"\n🩺 FULL SYMPTOMS:")
        print(f"   {symptoms}")

        print(f"\n✅ TRUE DIAGNOSIS:")
        print(f"   {true_diagnosis}")

        if top_3:
            print(f"\n📋 TOP 3 PREDICTIONS FROM MODEL:")
            for pred in top_3:
                print(f"   {pred.get('rank', '?')}.  {pred.get('disease', '?')} "
                      f"(confidence: {pred.get('confidence', 0):.0%})")
                if 'reason' in pred:
                    print(f"      Reason: {pred.get('reason', '')}")

        print(f"\n🎯 FINAL EXTRACTED DIAGNOSIS: {predicted_disease}")
        print(f"📊 MODEL CONFIDENCE: {model_confidence:.0%}")
        print(f"\n{'✅ CORRECT' if is_correct else '❌ WRONG'}")

        if is_correct:
            match_type = 'Exact' if diagnosis_for_accuracy.lower() == true_diagnosis.lower() else 'Partial'
            print(f"   Match type: {match_type}")

    # For cases 6+, just show progress every 50 cases
    elif case_num % 50 == 0:
        correct_so_far = sum(1 for r in results if r['correct'] == '✓')
        acc_so_far = (correct_so_far / len(results)) * 100 if results else 0
        print(f"\nProgress: {case_num}/{EVAL_SIZE} cases | Running accuracy: {acc_so_far:.1f}%")


# ============================================
# 6. Calculate baseline accuracy
# ============================================

print("\n[5/5] Calculating baseline accuracy...")

correct_count = sum(1 for r in results if r['correct'] == '✓')
total_count = len(results)
accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0

print("\n" + "="*70)
print("PHASE 1 BASELINE RESULTS")
print("="*70)

print(f"\n📊 Results on {total_count} cases:")
print(f"   Correct: {correct_count}/{total_count}")
print(f"   Accuracy: {accuracy:.1f}%")

print(f"\n💭 What this means:")
if accuracy < 10:
    print(f"   The base model has almost no diagnostic ability")
    print(f"   This is expected - it wasn't trained on medical diagnosis")
elif accuracy < 30:
    print(f"   The base model has minimal diagnostic ability")
    print(f"   Slightly better than random guessing (~2% for 49 diseases)")
else:
    print(f"   The base model has some diagnostic ability from pre-training")
    print(f"   But still needs fine-tuning for good performance")

print(f"\n✨ Next Steps:")
print(f"   Phase 2: Fine-tune on DDXPlus training data")
print(f"   Expected improvement: 20-30% accuracy → 60-70%")

# Save results
df = pd.DataFrame(results)
df.to_csv('phase1b_baseline_results.csv', index=False)
print(f"\n✓ Results saved to: phase1_baseline_results.csv")

print("\n" + "="*70)
print("✅ PHASE 1 COMPLETE!")
print("="*70)

print(f"""
Summary:
  • Tested base Llama model (no medical training)
  • Evaluated on {total_count} diagnostic cases
  • Baseline accuracy: {accuracy:.1f}%
  • This establishes your starting point

What we learned:
  • Base model cannot diagnose well without training
  • Justifies need for medical fine-tuning (Phase 2)
  • Sets baseline for measuring improvement
""")

print("\nReady for Step 3: Phase 2 Fine-tuning? (This will take 2-4 hours)")
print("Or increase sample_size to 100-1000 cases for more robust baseline")
