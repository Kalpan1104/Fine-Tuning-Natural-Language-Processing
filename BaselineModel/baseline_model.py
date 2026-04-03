"""
STEP 2: Phase 1 Baseline Evaluation (Simplified)
Just test the base Llama model - no training yet
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
import ast
from tqdm import tqdm

print("="*70)
print("STEP 2: PHASE 1 BASELINE EVALUATION")
print("="*70)

# ============================================
# 1. Load the base model
# ============================================

print("\n[1/5] Loading base Llama model (no training)...")
print("This will download ~6GB if first time...\n")

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
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
print(f"✓ We'll test on 10 cases for demonstration")

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

print("\n[4/5] Testing base model on 10 cases...")
print("(This shows you what Phase 1 baseline looks like)\n")

results = []

for i in tqdm(range(10), desc="Evaluating"):
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
        for item in diff_dx_list[:5]:  # Top 5 differentials
            if isinstance(item, list) and len(item) >= 2:
                differential_diseases.append(item[0])
            elif isinstance(item, str):
                differential_diseases.append(item)

        differential_text = ", ".join(differential_diseases) if differential_diseases else "various conditions"
    except:
        differential_text = "various conditions"

    # SYSTEM PROMPT (Model's role and instructions)
    system_prompt = """You are a medical diagnosis assistant for a research prototype.

Your task: Given a patient's symptoms and demographics, return the top 3 most likely diseases with confidence scores.

Rules:
1. Output must be valid JSON only
2. No markdown, no extra text
3. Return exactly 3 predictions ranked by likelihood
4. Each prediction must have: rank, disease, confidence, reason
5. Confidence must be between 0 and 1
6. The 3 confidence values must sum to 1.0
7. Use concise disease names
8. Keep reasons to one sentence

Return JSON in this format:
{
  "top_3_predictions": [
    {"rank": 1, "disease": "Disease Name", "confidence": 0.6, "reason": "Brief explanation"},
    {"rank": 2, "disease": "Disease Name", "confidence": 0.25, "reason": "Brief explanation"},
    {"rank": 3, "disease": "Disease Name", "confidence": 0.15, "reason": "Brief explanation"}
  ],
  "final_prediction": {"disease": "Disease Name", "confidence": 0.6}
}"""

    # USER PROMPT (The actual patient case)
    user_prompt = f"""Patient Information:
- Age: {case['AGE']} years
- Sex: {sex_text}
- Symptoms: {symptoms}
- Number of evidences: {len(evidences_list)}

Common differential diagnoses to consider for similar presentations:
{differential_text}

Based on this patient's presentation, provide your diagnostic assessment with the top 3 most likely diagnoses and confidence scores."""

    # Format for chat model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # PRINT THE COMPLETE PROMPT BEING SENT TO MODEL
    print(f"\n{'='*70}")
    print(f"CASE #{i+1} - PROMPT SENT TO MODEL")
    print(f"{'='*70}")
    print(formatted_prompt)
    print(f"{'='*70}")

    # Get model prediction
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # More tokens for JSON response
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode prediction (FULL response, no cleanup)
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # PRINT THE RAW MODEL RESPONSE
    print(f"\n{'='*70}")
    print(f"CASE #{i+1} - RAW MODEL RESPONSE")
    print(f"{'='*70}")
    print(prediction)
    print(f"{'='*70}\n")

    # Try to parse JSON response
    try:
        # Remove markdown code blocks if present
        json_text = prediction
        if '```json' in json_text:
            json_text = json_text.split('```json')[1].split('```')[0]
        elif '```' in json_text:
            json_text = json_text.split('```')[1].split('```')[0]

        response_json = json.loads(json_text.strip())

        # Extract top prediction
        if 'final_prediction' in response_json:
            predicted_disease = response_json['final_prediction']['disease']
            model_confidence = response_json['final_prediction']['confidence']
        elif 'top_3_predictions' in response_json:
            predicted_disease = response_json['top_3_predictions'][0]['disease']
            model_confidence = response_json['top_3_predictions'][0]['confidence']
        else:
            predicted_disease = prediction[:50]  # Fallback
            model_confidence = 0.5

        top_3 = response_json.get('top_3_predictions', [])

    except:
        # If JSON parsing fails, use text as-is
        predicted_disease = prediction.split('\n')[0][:50]
        model_confidence = 0.5
        top_3 = []

    # Check if correct (fuzzy matching)

    # Check if correct (fuzzy matching)
    is_correct = False

    # Exact match
    if predicted_disease.lower() == true_diagnosis.lower():
        is_correct = True
    # Partial match
    elif predicted_disease.lower() in true_diagnosis.lower():
        is_correct = True
    elif true_diagnosis.lower() in predicted_disease.lower():
        is_correct = True

    results.append({
        'case': i+1,
        'symptoms': symptoms,  # FULL symptoms
        'true_diagnosis': true_diagnosis,
        'predicted_disease': predicted_disease,
        'model_confidence': model_confidence,
        'full_response': prediction,  # Complete model output
        'top_3': top_3,
        'correct': '✓' if is_correct else '✗'
    })

    # Print each case with FULL details
    print(f"\n{'='*70}")
    print(f"CASE #{i+1}")
    print(f"{'='*70}")

    print(f"\n👤 PATIENT:")
    print(f"   Age: {case['AGE']} years")
    print(f"   Sex: {sex_text}")

    print(f"\n🔍 DIFFERENTIAL DIAGNOSIS PROVIDED:")
    print(f"   {differential_text}")

    print(f"\n🩺 FULL SYMPTOMS:")
    print(f"   {symptoms}")

    print(f"\n✅ TRUE DIAGNOSIS:")
    print(f"   {true_diagnosis}")

    print(f"\n🤖 MODEL'S COMPLETE RESPONSE:")
    print(f"   {prediction}")

    if top_3:
        print(f"\n📋 TOP 3 PREDICTIONS:")
        for pred in top_3:
            print(f"   {pred.get('rank', '?')}. {pred.get('disease', '?')} "
                  f"(confidence: {pred.get('confidence', 0):.0%})")
            if 'reason' in pred:
                print(f"      Reason: {pred.get('reason', '')}")

    print(f"\n🎯 EXTRACTED DIAGNOSIS: {predicted_disease}")
    print(f"📊 MODEL CONFIDENCE: {model_confidence:.0%}")
    print(f"\n{'✅ CORRECT' if is_correct else '❌ WRONG'}")

    if is_correct:
        match_type = 'Exact' if predicted_disease.lower() == true_diagnosis.lower() else 'Partial'
        print(f"   Match type: {match_type}")

# ============================================
# 6. Calculate baseline accuracy
# ============================================

print("\n[5/5] Calculating baseline accuracy...")

correct_count = sum(1 for r in results if r['correct'] == '✓')
total_count = len(results)
accuracy = (correct_count / total_count) * 100

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
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('phase1_baseline_results.csv', index=False)
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
""")
