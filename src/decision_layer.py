import pandas as pd

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

def count_options(option):
    # Failsafe: if the cell is blank/NaN, return 0
    if not isinstance(option, str):
        return 0
        
    diagnosis_list = []
    lines = option.strip().split('\n') 
    
    for line in lines:
        if line.strip(): 
            clean_name = line.split('.', 1)[-1].strip() 
            diagnosis_list.append(clean_name)
            
    return len(diagnosis_list)
        
df['number_of_options'] = df['diagnosis_options'].apply(count_options)

df['threshold'] = df['number_of_options'].apply(lambda x: 2/x if x > 0 else 1.0)

import re

def extract_data_regex(text_input):
    # Safety check for empty rows
    if not isinstance(text_input, str):
        return None, None
        
    try:
        # 1. Search for the pattern: "disease": "ANY_TEXT"
        # The (.*?) captures whatever is inside the quotes
        disease_match = re.search(r'"disease":\s*"(.*?)"', text_input)
        
        # 2. Search for the pattern: "confidence": 0.XXX
        # The ([0-9.]+) captures the numbers and decimals
        confidence_match = re.search(r'"confidence":\s*([0-9.]+)', text_input)
        
        # 3. If both were successfully found, extract them
        if disease_match and confidence_match:
            rank_1_disease = disease_match.group(1) # Gets the captured text
            rank_1_confidence = float(confidence_match.group(1)) # Converts to decimal
            
            return rank_1_disease, rank_1_confidence
        else:
            return None, None

    except Exception as e:
        print(f"Failed to parse: {e}")
        return None, None

df[['preicted_rank1_disease', 'predicted_rank1_confidence']] = df['raw_model_response'].apply(
    lambda x: pd.Series(extract_data_regex(x))
)

df.head()

def decision_layer(disease, confidence, threshold):
    if disease in SEVERE_DISEASES or confidence < threshold:
        return "Refer to specialist", confidence
    else:
        return disease, confidence

df[['rank1_disease', 'rank1_confidence']] = df.apply(
    lambda row: pd.Series(
        decision_layer(
            row['preicted_rank1_disease'], 
            row['predicted_rank1_confidence'], 
            row['threshold']
        )
    ),
    axis=1 
)

df.to_csv("phase3_results.csv", index=False)
