import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. Load Results
# ============================================

print("\n[1/5] Loading Phase 3 results...")

df = pd.read_csv('phase3_results.csv')

print(f"✓ Loaded {len(df):,} cases")
print(f"\nColumns available: {list(df.columns)}")

TRUTHY = {'✓', '✔', '☑', 'true', 'True', '1', 'yes', 'Yes'}
df['correct'] = df['correct'].astype(str).str.strip().isin(TRUTHY)
df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
df['num_options'] = pd.to_numeric(df['num_options'], errors='coerce')

# Convert action column: REFER → True, DIAGNOSE → False
df['is_referral'] = df['action'].astype(str).str.strip().str.upper() == 'REFER'

# Convert is_severe and high_confidence (may be bool or string)
df['is_severe'] = df['is_severe'].astype(str).str.strip().isin(TRUTHY)
df['high_confidence'] = df['high_confidence'].astype(str).str.strip().isin(TRUTHY)

# Compute dynamic confidence threshold from num_options
df['confidence_threshold'] = 2 / df['num_options']

print("✓ Converted column types")
print(f"✓ Dynamic confidence thresholds computed from num_options")
print(f"  Threshold range: {df['confidence_threshold'].min():.3f} – {df['confidence_threshold'].max():.3f}")
print(f"  Mean threshold: {df['confidence_threshold'].mean():.3f}")

# ============================================
# 2. Calculate Pass@1 (strict criteria)
# ============================================

print("\n[2/5] Calculating Pass@1 metrics (strict criteria)...")

total_cases = len(df)

# Compute flags
df['above_threshold'] = df['confidence'] > df['confidence_threshold']
df['severe_not_referred'] = (df['is_severe']) & (~df['is_referral'])

# ── SUCCESS categories ──

# 1. Correct diagnosis: above threshold, not severe (or not applicable), and correct
df['success_correct_dx'] = (df['above_threshold']) & (~df['severe_not_referred']) & (df['correct'])

# 2. Severe disease correctly referred
df['success_severe_referred'] = (df['is_severe']) & (df['is_referral'])

# 3. Low confidence correctly referred
df['success_low_conf_referred'] = (~df['above_threshold']) & (df['is_referral'])

# Overall success: any of the three
df['pass_at_1_success'] = (df['success_correct_dx']) | (df['success_severe_referred']) | (df['success_low_conf_referred'])

# ── FAILURE categories (mutually exclusive, priority order) ──

# 1. Severe disease NOT referred (highest priority failure)
df['fail_severe_not_referred'] = df['severe_not_referred']

# 2. Low confidence NOT referred (excluding already counted severe cases)
df['fail_low_conf_not_referred'] = (~df['above_threshold']) & (~df['is_referral']) & (~df['fail_severe_not_referred'])

# 3. Wrong diagnosis — eligible but incorrect (excluding above two)
df['fail_wrong_dx'] = (~df['pass_at_1_success']) & (~df['fail_severe_not_referred']) & (~df['fail_low_conf_not_referred'])

# Counts
n_success_correct = df['success_correct_dx'].sum()
n_success_severe_ref = df['success_severe_referred'].sum()
n_success_low_conf_ref = df['success_low_conf_referred'].sum()
n_total_success = df['pass_at_1_success'].sum()

n_fail_wrong = df['fail_wrong_dx'].sum()
n_fail_severe_unref = df['fail_severe_not_referred'].sum()
n_fail_low_conf_unref = df['fail_low_conf_not_referred'].sum()
n_total_fail = total_cases - n_total_success

pass_at_1_accuracy = (n_total_success / total_cases) * 100 if total_cases > 0 else 0.0

# Sanity check: all categories must sum to total
assert n_total_success + n_fail_wrong + n_fail_severe_unref + n_fail_low_conf_unref == total_cases, \
    f"Category mismatch: {n_total_success} + {n_fail_wrong} + {n_fail_severe_unref} + {n_fail_low_conf_unref} != {total_cases}"

print(f"\n📊 Pass@1 Results (Strict Criteria):")
print(f"   Total cases: {total_cases:,}")
print(f"")
print(f"   ✅ SUCCESSES: {n_total_success:,} ({n_total_success/total_cases*100:.1f}%)")
print(f"      Correct diagnosis (eligible):    {n_success_correct:,}")
print(f"      Severe disease referred:         {n_success_severe_ref:,}")
print(f"      Low confidence referred:         {n_success_low_conf_ref:,}")
print(f"")
print(f"   ❌ FAILURES: {n_total_fail:,} ({n_total_fail/total_cases*100:.1f}%)")
print(f"      Wrong diagnosis (eligible):      {n_fail_wrong:,}")
print(f"      Severe NOT referred:             {n_fail_severe_unref:,}")
print(f"      Low confidence NOT referred:     {n_fail_low_conf_unref:,}")
print(f"")
print(f"   ⭐ Pass@1 Accuracy: {n_total_success:,}/{total_cases:,} = {pass_at_1_accuracy:.2f}%")

# ============================================
# 3. Breakdown by Referral Decision
# ============================================

print("\n[3/5] Analyzing by referral decision...")

diagnosed = df[df['is_referral'] == False]
referred = df[df['is_referral'] == True]

diag_acc = 0.0
ref_acc = 0.0

print(f"\n✅ DIAGNOSED CASES:")
print(f"   Count: {len(diagnosed):,} ({len(diagnosed)/total_cases*100:.1f}% coverage)")
if len(diagnosed) > 0:
    diag_correct = diagnosed['correct'].sum()
    diag_acc = (diag_correct / len(diagnosed)) * 100
    print(f"   Correct: {diag_correct}/{len(diagnosed)}")
    print(f"   Accuracy: {diag_acc:.2f}%")

print(f"\n⚠️  REFERRED CASES:")
print(f"   Count: {len(referred):,} ({len(referred)/total_cases*100:.1f}%)")
if len(referred) > 0:
    ref_correct = referred['correct'].sum()
    ref_acc = (ref_correct / len(referred)) * 100
    print(f"   Accuracy (if diagnosed): {ref_acc:.2f}%")

# Referral reason breakdown
if 'refer_reason' in df.columns:
    print(f"\n📋 Referral Reasons:")
    reasons = df[df['is_referral']]['refer_reason'].value_counts()
    for reason, count in reasons.items():
        print(f"   {reason}: {count:,} ({count/total_cases*100:.1f}%)")

if len(diagnosed) > 0 and len(referred) > 0:
    if diag_acc > ref_acc:
        print(f"\n🛡️ SAFETY CHECK: ✅ PASSED")
        print(f"   Diagnosed accuracy ({diag_acc:.1f}%) > Referred ({ref_acc:.1f}%)")
    else:
        print(f"\n🛡️ SAFETY CHECK: ⚠️ NEEDS IMPROVEMENT")

# ============================================
# 4. Confidence Analysis
# ============================================

print("\n[4/5] Confidence analysis...")

avg_conf_all = df['confidence'].mean()
avg_conf_correct = df[df['correct'] == True]['confidence'].mean()
avg_conf_wrong = df[df['correct'] == False]['confidence'].mean()

print(f"\n📊 Confidence Statistics:")
print(f"   Average (all cases): {avg_conf_all:.1%}")
print(f"   Average (correct): {avg_conf_correct:.1%}")
print(f"   Average (wrong): {avg_conf_wrong:.1%}")

if avg_conf_correct > avg_conf_wrong:
    print(f"\n✅ Good calibration: Model more confident when correct")
else:
    print(f"\n⚠️ Poor calibration: Model not more confident when correct")

# ============================================
# 5. Visualizations
# ============================================

print("\n[5/5] Creating visualizations...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Phase 3: Pass@1 Evaluation Results (Decision Layer in Code)',
             fontsize=16, fontweight='bold', y=1.02)

# Plot 1: Pass@1 Accuracy — Solid Correct + Stacked Wrong

# Bar 1: Correct 
correct_pct = n_total_success / total_cases * 100
ax1.bar('Correct', n_total_success, color='#2ecc71', alpha=0.85,
        edgecolor='black', linewidth=1.5, label=f'Correct ({correct_pct:.1f}%)')
ax1.text(0, n_total_success / 2, f'{correct_pct:.1f}%',
         ha='center', va='center', fontweight='bold', fontsize=12, color='white')

# Bar 2: Wrong (stacked — 3 failure types)
fail_segments = [n_fail_wrong, n_fail_severe_unref, n_fail_low_conf_unref]
fail_labels = ['Wrong Diagnosis', 'Severe Not Referred', 'Low Conf Not Referred']
fail_colors = ['#e74c3c', '#8e44ad', '#f39c12']

bottom = 0
for val, label, color in zip(fail_segments, fail_labels, fail_colors):
    pct = val / total_cases * 100
    ax1.bar('Wrong', val, bottom=bottom, color=color, alpha=0.85,
            edgecolor='black', linewidth=1.5, label=f'{label} ({pct:.1f}%)')
    if pct >= 3:
        ax1.text(1, bottom + val / 2, f'{pct:.1f}%',
                 ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    bottom += val

ax1.set_ylabel('')
ax1.set_title(f'Phase 3 Pass@1 Accuracy: {pass_at_1_accuracy:.1f}%',
              fontweight='bold', fontsize=13)
ax1.set_yticklabels([])
ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.2, axis='y')

# Plot 2: Decision Distribution
categories = ['Diagnosed', 'Referred']
counts = [len(diagnosed), len(referred)]
colors = ['#3498db', '#f39c12']
bars = ax2.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('')
ax2.set_title('Decision Distribution', fontweight='bold', fontsize=13)
ax2.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 10)
ax2.set_yticklabels([])

for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{count/total_cases*100:.1f}%',
            ha='center', va='bottom', fontweight='bold')
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('phase3_pass_at_1_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: phase3_pass_at_1_analysis.png")

# ============================================
# 6. Summary Report
# ============================================

diag_acc_str = f"{diag_acc:.2f}%" if len(diagnosed) > 0 else "N/A"
calibration_str = "✅ Good" if avg_conf_correct > avg_conf_wrong else "⚠️ Poor"
safety_str = "✅ Passed" if (len(diagnosed) > 0 and len(referred) > 0 and diag_acc > ref_acc) else "⚠️ Check"

print(f"""
📊 PHASE 3 RESULTS (DECISION LAYER IN CODE):

Pass@1 Criteria:
  • Dynamic confidence threshold: 2 / num_options (2x better than random)
  • Threshold range: {df['confidence_threshold'].min():.3f} – {df['confidence_threshold'].max():.3f} (mean: {df['confidence_threshold'].mean():.3f})
  • Correct referrals (severe + low confidence) count as SUCCESSES
  • Decision layer enforced deterministically in code (not prompt)
  • Denominator = ALL cases

Pass@1 Accuracy: {pass_at_1_accuracy:.2f}%
  • Total successes: {n_total_success:,}/{total_cases:,}

Decision Breakdown:
  • Diagnosed: {len(diagnosed):,} ({len(diagnosed)/total_cases*100:.1f}%)
  • Referred: {len(referred):,} ({len(referred)/total_cases*100:.1f}%)

Diagnosed Cases Accuracy: {diag_acc_str}

Confidence Calibration:
  • Avg confidence (correct): {avg_conf_correct:.1%}
  • Avg confidence (wrong): {avg_conf_wrong:.1%}
  • Calibration: {calibration_str}

Key Metrics:
  • Pass@1: {pass_at_1_accuracy:.2f}%
  • Safety: {safety_str}

Files Generated:
  ✓ phase3_pass_at_1_analysis.png (visualization)
""")
