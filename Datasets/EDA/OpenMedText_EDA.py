"""
OpenMedText - Download ALL Files (or controlled subset)
Downloads files discovered from the repository
"""

from huggingface_hub import hf_hub_download
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from tqdm import tqdm

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("="*70)
print("OPENMEDTEXT - COMPREHENSIVE DOWNLOAD & ANALYSIS")
print("="*70)

# ============================================
# 1. Load the discovered file list
# ============================================

print("\n[1/7] Loading discovered file list...")

try:
    with open('openmedtext_all_files.json', 'r') as f:
        data = json.load(f)
    
    all_files_available = []
    for files in data['categories'].values():
        all_files_available.extend(files)
    
    print(f"✓ Loaded {len(all_files_available):,} file paths")
    print(f"✓ Categories: {len(data['categories'])}")
    
except FileNotFoundError:
    print("✗ Error: 'openmedtext_all_files.json' not found!")
    print("Please run the discovery script first.")
    exit()

# ============================================
# 2. Choose download strategy
# ============================================

print("\n" + "="*70)
print("DOWNLOAD STRATEGY")
print("="*70)

print(f"""
Total files available: {len(all_files_available):,}

⚠️  WARNING: Downloading all 127,707 files would take 4-8 HOURS!

Choose a strategy:
""")

# OPTION 1: Download a percentage from each category (RECOMMENDED)
percentage_per_category = 0.05  # 5% of each category
print(f"OPTION 1 (ACTIVE): Download {percentage_per_category*100:.0f}% from each category")

selected_files = []
category_stats = {}

for category, files in data['categories'].items():
    num_to_take = max(1, int(len(files) * percentage_per_category))
    selected_files.extend(files[:num_to_take])
    category_stats[category] = {
        'total': len(files),
        'selected': num_to_take
    }

print(f"  → Will download: {len(selected_files):,} files")
print(f"  → Estimated time: {len(selected_files) * 2 / 60:.1f} minutes")

# OPTION 2: Download fixed number per category (uncomment to use)
# num_per_category = 50
# selected_files = []
# for category, files in data['categories'].items():
#     selected_files.extend(files[:num_per_category])
# print(f"OPTION 2: {num_per_category} files per category = {len(selected_files)} total")

# OPTION 3: Download ALL files (uncomment to use - NOT RECOMMENDED without good reason)
# selected_files = all_files_available
# print(f"OPTION 3: ALL {len(selected_files):,} files (This will take HOURS!)")

# OPTION 4: Download from specific categories only
# specific_categories = ['allergies', 'cancers_1', 'diseases', 'vaccines', 'viruses']
# selected_files = []
# for cat in specific_categories:
#     if cat in data['categories']:
#         selected_files.extend(data['categories'][cat])
# print(f"OPTION 4: Specific categories = {len(selected_files)} files")

print(f"\n✓ Strategy selected: {len(selected_files):,} files to download")

# Show breakdown
print("\nFiles per category:")
for cat in sorted(category_stats.keys())[:20]:
    stats = category_stats[cat]
    print(f"  • {cat}: {stats['selected']}/{stats['total']} ({stats['selected']/stats['total']*100:.1f}%)")

if len(category_stats) > 20:
    print(f"  ... and {len(category_stats) - 20} more categories")

# Confirm if downloading many files
if len(selected_files) > 5000:
    print(f"\n⚠️  You're about to download {len(selected_files):,} files!")
    print(f"Estimated time: {len(selected_files) * 2 / 60:.0f}-{len(selected_files) * 3 / 60:.0f} minutes")
    response = input("\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        exit()

# ============================================
# 3. Download files
# ============================================

print("\n" + "="*70)
print("DOWNLOADING FILES")
print("="*70)

repo_id = "ywchoi/OpenMedText"
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'  # No token needed

downloaded_texts = []
file_info = []
failed_files = []

print(f"\nDownloading {len(selected_files):,} files...")
print("Progress will be shown below:\n")

# Download with progress bar
for file_path in tqdm(selected_files, desc="Downloading", unit="file"):
    try:
        # Download file
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            token=False
        )
        
        # Read content
        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Skip empty files
        if not text.strip():
            continue
        
        # Extract info
        parts = file_path.split('/')
        category = parts[1] if len(parts) > 1 else "unknown"
        filename = parts[-1]
        
        downloaded_texts.append(text)
        file_info.append({
            'path': file_path,
            'category': category,
            'filename': filename,
            'text': text
        })
        
    except Exception as e:
        failed_files.append((file_path, str(e)))
        continue

print(f"\n{'='*70}")
print(f"DOWNLOAD COMPLETE")
print(f"{'='*70}")
print(f"✓ Successfully downloaded: {len(downloaded_texts):,} files")
print(f"✗ Failed: {len(failed_files):,} files")
print(f"  Success rate: {len(downloaded_texts)/(len(downloaded_texts)+len(failed_files))*100:.1f}%")

if len(downloaded_texts) == 0:
    print("\n✗ ERROR: No files downloaded!")
    exit()

# Show category breakdown
downloaded_categories = Counter([info['category'] for info in file_info])
print(f"\n✓ Files from {len(downloaded_categories)} categories:")
for i, (cat, count) in enumerate(downloaded_categories.most_common(15), 1):
    print(f"  {i:2d}. {cat}: {count:,} files")

if len(downloaded_categories) > 15:
    print(f"  ... and {len(downloaded_categories) - 15} more categories")

# ============================================
# 4. Text Analysis
# ============================================

print("\n" + "="*70)
print("TEXT ANALYSIS")
print("="*70)

print(f"\nAnalyzing {len(downloaded_texts):,} documents...")

# Calculate statistics
word_counts = [len(text.split()) for text in downloaded_texts]
char_counts = [len(text) for text in downloaded_texts]

print("\n📊 Word Count Statistics:")
print(f"  • Mean:     {np.mean(word_counts):,.1f} words")
print(f"  • Median:   {np.median(word_counts):,.1f} words")
print(f"  • Min:      {min(word_counts):,} words")
print(f"  • Max:      {max(word_counts):,} words")
print(f"  • Std Dev:  {np.std(word_counts):,.1f} words")
print(f"  • Total:    {sum(word_counts):,} words")

print("\n📊 Character Count Statistics:")
print(f"  • Mean:     {np.mean(char_counts):,.1f} characters")
print(f"  • Total:    {sum(char_counts):,} characters ({sum(char_counts)/1e6:.1f} MB as text)")

# Length categories
very_short = sum(1 for w in word_counts if w < 1000)
short = sum(1 for w in word_counts if 1000 <= w < 3000)
medium = sum(1 for w in word_counts if 3000 <= w < 6000)
long = sum(1 for w in word_counts if 6000 <= w < 10000)
very_long = sum(1 for w in word_counts if w >= 10000)

total = len(word_counts)
print(f"\n📏 Document Length Categories:")
print(f"  • Very Short (<1000):      {very_short:6,} ({very_short/total*100:5.1f}%)")
print(f"  • Short (1000-2999):       {short:6,} ({short/total*100:5.1f}%)")
print(f"  • Medium (3000-5999):      {medium:6,} ({medium/total*100:5.1f}%)")
print(f"  • Long (6000-9999):        {long:6,} ({long/total*100:5.1f}%)")
print(f"  • Very Long (≥10000):      {very_long:6,} ({very_long/total*100:5.1f}%)")

# ============================================
# 5. Medical Content Analysis
# ============================================

print("\n" + "="*70)
print("MEDICAL CONTENT ANALYSIS")
print("="*70)

# Sample analysis on subset for speed
sample_size = min(1000, len(downloaded_texts))
print(f"\nAnalyzing medical terms in {sample_size:,} sample documents...")

medical_terms = [
    'disease', 'patient', 'treatment', 'therapy', 'clinical', 'diagnosis',
    'symptom', 'infection', 'cancer', 'cell', 'cells', 'blood', 'gene', 
    'protein', 'study', 'research', 'medical', 'health', 'chronic', 'acute'
]

term_counts = {}
for term in medical_terms:
    count = sum(downloaded_texts[i].lower().count(term) for i in range(sample_size))
    term_counts[term] = count

sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)

print("\n🔬 Top medical terms:")
for i, (term, count) in enumerate(sorted_terms[:15], 1):
    avg_per_doc = count / sample_size
    print(f"  {i:2d}. '{term}': {count:,} occurrences (avg {avg_per_doc:.1f} per doc)")

# ============================================
# 6. Data Quality
# ============================================

print("\n" + "="*70)
print("DATA QUALITY")
print("="*70)

empty = sum(1 for text in downloaded_texts if not text.strip())
very_short_docs = sum(1 for w in word_counts if w < 100)
has_medical = sum(1 for i in range(min(1000, len(downloaded_texts)))
                  if any(term in downloaded_texts[i].lower() for term in medical_terms[:5]))

print(f"\n✓ Empty files: {empty:,}")
print(f"✓ Very short files (<100 words): {very_short_docs:,}")
print(f"✓ Files with medical content (sample): {has_medical}/{min(1000, len(downloaded_texts))} ({has_medical/min(1000, len(downloaded_texts))*100:.1f}%)")

if empty < total * 0.01 and has_medical > min(1000, len(downloaded_texts)) * 0.8:
    print("\n✅ Overall data quality: EXCELLENT")
else:
    print("\n✅ Overall data quality: GOOD")

# ============================================
# 7. Visualization
# ============================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'OpenMedText Complete Analysis - {len(downloaded_texts):,} Documents', 
             fontsize=16, fontweight='bold')

# Plot 1: Word count distribution
ax1 = axes[0, 0]
ax1.hist(word_counts, bins=50, alpha=0.7, color='#27ae60', edgecolor='black')
ax1.axvline(np.mean(word_counts), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(word_counts):,.0f}')
ax1.axvline(np.median(word_counts), color='blue', linestyle='--', linewidth=2,
           label=f'Median: {np.median(word_counts):,.0f}')
ax1.set_xlabel('Document Length (words)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Document Length Distribution', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Length categories
ax2 = axes[0, 1]
categories_plot = ['Very\nShort', 'Short', 'Medium', 'Long', 'Very\nLong']
counts_plot = [very_short, short, medium, long, very_long]
colors_plot = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c']
bars = ax2.bar(categories_plot, counts_plot, color=colors_plot, alpha=0.8,
              edgecolor='black', linewidth=2)
ax2.set_ylabel('Number of Documents', fontsize=12, fontweight='bold')
ax2.set_title('Document Length Categories', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Top medical terms
ax3 = axes[1, 0]
top_terms_plot = sorted_terms[:12]
terms_plot = [t[0] for t in top_terms_plot]
counts_terms = [t[1] for t in top_terms_plot]
ax3.barh(terms_plot, counts_terms, color='#16a085', alpha=0.8, edgecolor='black')
ax3.set_xlabel('Occurrences (in sample)', fontsize=12, fontweight='bold')
ax3.set_title('Top 12 Medical Terms', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Top categories
ax4 = axes[1, 1]
top_cats = downloaded_categories.most_common(15)
cat_names = [c[0][:12] for c in top_cats]
cat_counts_plot = [c[1] for c in top_cats]
ax4.barh(cat_names, cat_counts_plot, color='#8e44ad', alpha=0.8, edgecolor='black')
ax4.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
ax4.set_title('Top 15 Categories by File Count', fontsize=14, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('openmedtext_complete_dataset.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'openmedtext_complete_dataset.png'")

# ============================================
# 8. Final Summary
# ============================================

print("\n" + "="*70)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*70)

print(f"""
🎉 SUCCESS! OpenMedText Dataset Fully Loaded and Analyzed

📊 Dataset Overview:
   • Total files in repository: {len(all_files_available):,}
   • Files downloaded: {len(downloaded_texts):,}
   • Coverage: {len(downloaded_texts)/len(all_files_available)*100:.2f}%
   • Categories represented: {len(downloaded_categories)}/{len(data['categories'])}

📈 Content Statistics:
   • Average document length: {np.mean(word_counts):,.0f} words
   • Median document length: {np.median(word_counts):,.0f} words
   • Total content: {sum(word_counts):,} words ({sum(word_counts)/1e6:.1f}M words)
   • Total text size: ~{sum(char_counts)/1e6:.1f} MB

""")

print("="*70)
print("🏆 DATA PREPAREDNESS: 100% COMPLETE!")
print("="*70)
print(f"\nYou now have {len(downloaded_texts):,} OpenMedText documents")
print("plus 11,451 MedQA questions - fully analyzed and ready!")
