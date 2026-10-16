#!/usr/bin/env python
"""Create PowerPoint presentation for quantum vs classical ML project"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# Create presentation
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

# Define color scheme
TITLE_COLOR = RGBColor(0, 51, 102)       # Dark blue
ACCENT_COLOR = RGBColor(0, 102, 204)     # Blue
TEXT_COLOR = RGBColor(51, 51, 51)        # Dark gray
GOOD_COLOR = RGBColor(0, 153, 0)         # Green
WARNING_COLOR = RGBColor(255, 102, 0)    # Orange

def add_title_slide(prs, title, subtitle):
    """Add a title slide"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(0, 51, 102)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(1.5))
    title_frame = title_box.text_frame
    title_frame.word_wrap = True
    p = title_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(9), Inches(2))
    subtitle_frame = subtitle_box.text_frame
    subtitle_frame.word_wrap = True
    p = subtitle_frame.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(28)
    p.font.color.rgb = RGBColor(200, 200, 200)
    p.alignment = PP_ALIGN.CENTER
    
    return slide

def add_content_slide(prs, title, content_list):
    """Add a content slide with title and bullet points"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(255, 255, 255)
    
    # Title bar
    title_shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(1))
    title_shape.fill.solid()
    title_shape.fill.fore_color.rgb = TITLE_COLOR
    title_shape.line.color.rgb = TITLE_COLOR
    
    title_frame = title_shape.text_frame
    title_frame.clear()
    p = title_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.space_before = Pt(10)
    p.space_after = Pt(10)
    
    # Content
    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(9), Inches(5.7))
    text_frame = content_box.text_frame
    text_frame.word_wrap = True
    
    for i, item in enumerate(content_list):
        if i == 0:
            p = text_frame.paragraphs[0]
        else:
            p = text_frame.add_paragraph()
        
        p.text = item
        p.font.size = Pt(18)
        p.font.color.rgb = TEXT_COLOR
        p.space_before = Pt(6)
        p.space_after = Pt(6)
        p.level = 0
    
    return slide

# SLIDE 1: TITLE SLIDE
print('Creating Slide 1: Title...')
add_title_slide(prs, 
    'Quantum vs Classical ML',
    'Streaming Data Analytics: An Experimental Study\nOctober 2026')

# SLIDE 2: RESEARCH PROBLEM
print('Creating Slide 2: Research Problem...')
content = [
    '🎯 Research Question:',
    'Can quantum machine learning compete with classical methods on streaming medical data?',
    '',
    '📊 Dataset: Breast Cancer Wisconsin',
    '• 569 samples, 30 features | Binary classification: Malignant vs Benign',
    '• 60/20/20 stratified split: 341 train, 114 val, 114 test',
    '',
    '🔬 Approach:',
    '• Classical: SGDClassifier with partial_fit (incremental learning)',
    '• Quantum: 4-qubit PennyLane variational quantum classifier'
]
add_content_slide(prs, 'RESEARCH PROBLEM', content)

# SLIDE 3: METHODOLOGY
print('Creating Slide 3: Methodology...')
content = [
    '📋 Data Representation:',
    '• Classical: 30 original features (StandardScaler normalized)',
    '• Quantum: 4 PCA-reduced features (99.99% variance retained)',
    '',
    '🔄 Streaming Protocol:',
    '• Deterministic batch generation (seed=42)',
    '• Batch sizes: 16, 32, 64 records',
    '• Classical: partial_fit() per batch | Quantum: Train once, then predict-only',
    '',
    '🧪 Three Experiments:',
    '• Experiment A: Performance (42 runs)  |  B: Batch sensitivity  |  C: Stream length'
]
add_content_slide(prs, 'METHODOLOGY', content)

# SLIDE 4: KEY FINDINGS
print('Creating Slide 4: Key Findings...')
slide = prs.slides.add_slide(prs.slide_layouts[6])
background = slide.background
fill = background.fill
fill.solid()
fill.fore_color.rgb = RGBColor(255, 255, 255)

# Title bar
title_shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(1))
title_shape.fill.solid()
title_shape.fill.fore_color.rgb = TITLE_COLOR
title_shape.line.color.rgb = TITLE_COLOR
title_frame = title_shape.text_frame
title_frame.clear()
p = title_frame.paragraphs[0]
p.text = 'KEY FINDINGS'
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = RGBColor(255, 255, 255)
p.space_before = Pt(10)

# Results
content_box = slide.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(8.6), Inches(5.8))
text_frame = content_box.text_frame
text_frame.word_wrap = True

results = [
    ('Accuracy', 'Classical: 87.97%  |  Quantum: 75.44%  |  Advantage: +12.5 pp'),
    ('Precision', 'Classical: 89.04%  |  Quantum: 74.16%  |  Advantage: +14.9 pp'),
    ('Recall', 'Classical: 91.55%  |  Quantum: 92.96%  |  Note: Quantum higher'),
    ('F1-Score', 'Classical: 0.9028  |  Quantum: 0.8250  |  Advantage: +0.0778'),
    ('Speed', 'Classical: 0.012s  |  Quantum: 11.13s  |  Advantage: 912× faster'),
]

for idx, (metric, result) in enumerate(results):
    if idx == 0:
        p = text_frame.paragraphs[0]
    else:
        p = text_frame.add_paragraph()
    
    p.text = f'{metric}: {result}'
    p.font.size = Pt(16)
    p.font.color.rgb = TEXT_COLOR
    p.space_before = Pt(8)
    p.space_after = Pt(8)

# Summary
p = text_frame.add_paragraph()
p.text = '✅ Classical significantly outperformed on accuracy and speed (912× faster end-to-end)'
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = GOOD_COLOR
p.space_before = Pt(16)

# SLIDE 5: ANALYSIS - WHY CLASSICAL WON
print('Creating Slide 5: Analysis...')
content = [
    '❌ Why Quantum Underperformed:',
    '',
    '1️⃣  CPU Simulator Overhead',
    '   • Quantum circuits executed on classical CPU simulator',
    '   • Simulation is computationally expensive',
    '',
    '2️⃣  Different Training Strategies (Not Equivalent)',
    '   • Classical: Incremental updates (partial_fit) on every batch',
    '   • Quantum: Single training window, then inference-only',
    '',
    '3️⃣  Feature Dimensionality Reduction',
    '   • Quantum limited to 4 qubits → 4 features via PCA',
    '   • Information loss though 99.99% variance retained',
    '',
    '4️⃣  Honest Assessment',
    '   • Results specific to CPU simulator setup',
    '   • Cannot extrapolate to real quantum hardware'
]
add_content_slide(prs, 'ANALYSIS: WHY CLASSICAL WON', content)

# SLIDE 6: CONCLUSIONS & LIMITATIONS
print('Creating Slide 6: Conclusions...')
content = [
    '✅ What We Learned:',
    '• Classical: 87.97% accuracy, 0.012s end-to-end',
    '• Quantum: 75.44% accuracy, 11.13s end-to-end',
    '• 42 experiments completed (0 failures, 100% success)',
    '• Reproducibility verified with fixed seed (seed=42)',
    '',
    '⚠️  Known Limitations:',
    '• Simulated streaming (batch-based, not real-time)',
    '• Small dataset (569 samples)',
    '• CPU-only quantum simulation',
    '• Different model update strategies (not directly comparable)',
    '',
    '🚀 Future Work:',
    '• Real quantum hardware  •  Larger datasets  •  Periodic retraining  •  Concept drift'
]
add_content_slide(prs, 'CONCLUSIONS & LIMITATIONS', content)

# Save presentation
output_path = 'Quantum_vs_Classical_ML_Streaming_Data.pptx'
prs.save(output_path)
print(f'✅ PowerPoint created: {output_path}')
print(f'📊 6 slides ready for presentation')
