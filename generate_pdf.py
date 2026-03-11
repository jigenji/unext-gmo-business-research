#!/usr/bin/env python3
"""Generate a white-themed multi-page PDF from the presentation HTML."""

from weasyprint import HTML

# Read the original HTML
with open("gpu_model_presentation.html", "r") as f:
    html = f.read()

# 1. Make all slides visible with page breaks
# 2. Switch to white theme
css_overrides = """
<style>
  /* ===== White Theme Override ===== */
  :root {
    --bg: #ffffff !important;
    --surface: #f5f5f7 !important;
    --card: #ffffff !important;
    --border: #d2d2d7 !important;
    --text: #1d1d1f !important;
    --text-secondary: #6e6e73 !important;
  }

  body {
    background: #ffffff !important;
    color: #1d1d1f !important;
    overflow: visible !important;
    height: auto !important;
    width: auto !important;
  }

  /* Show ALL slides, one per page */
  .slide {
    display: flex !important;
    width: 100% !important;
    height: 100vh !important;
    max-height: 100vh !important;
    page-break-after: always !important;
    break-after: page !important;
    page-break-inside: avoid !important;
    break-inside: avoid !important;
    padding: 30px 40px !important;
    position: relative !important;
    overflow: hidden !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    background: #ffffff !important;
  }

  .slide:last-of-type {
    page-break-after: avoid !important;
    break-after: avoid !important;
  }

  /* Card styling for white theme */
  .card {
    background: #ffffff !important;
    border: 1px solid #d2d2d7 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
  }

  .card-glow-blue { box-shadow: 0 1px 8px rgba(41,151,255,0.12) !important; }
  .card-glow-green { box-shadow: 0 1px 8px rgba(48,209,88,0.12) !important; }
  .card-glow-orange { box-shadow: 0 1px 8px rgba(255,159,10,0.12) !important; }
  .card-glow-red { box-shadow: 0 1px 8px rgba(255,69,58,0.12) !important; }

  /* Text colors */
  h1, h2 { color: #1d1d1f !important; }
  h3 { color: #2997ff !important; }
  .subtitle { color: #6e6e73 !important; }
  .desc { color: #6e6e73 !important; }
  .desc strong { color: #1d1d1f !important; font-weight: 600 !important; }
  .param-name { color: #1d1d1f !important; }
  .param-source { color: #6e6e73 !important; }
  .slide-number { color: #6e6e73 !important; }
  .metric-label { color: #6e6e73 !important; }

  /* Formula boxes */
  .formula {
    background: rgba(41,151,255,0.05) !important;
    border: 1px solid rgba(41,151,255,0.2) !important;
    color: #0066cc !important;
  }

  /* Gradient text - fallback to solid color for PDF */
  .gradient-text {
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: #2997ff !important;
    background-clip: unset !important;
    color: #2997ff !important;
  }

  /* Tags */
  .tag-blue { background: rgba(41,151,255,0.1) !important; }
  .tag-green { background: rgba(48,209,88,0.1) !important; }
  .tag-orange { background: rgba(255,159,10,0.1) !important; }
  .tag-red { background: rgba(255,69,58,0.1) !important; }
  .tag-purple { background: rgba(191,90,242,0.1) !important; }

  /* Badges */
  .badge-high { background: rgba(48,209,88,0.12) !important; }
  .badge-medium { background: rgba(255,159,10,0.12) !important; }
  .badge-low { background: rgba(255,69,58,0.12) !important; }

  /* Param rows */
  .param-row {
    border-bottom: 1px solid #e5e5e7 !important;
  }

  /* List items */
  .list-clean { color: #1d1d1f !important; }

  /* SVG text colors - override dark theme fills */
  /* Note: SVG inline fills can't be overridden by CSS easily,
     but key structural elements will still be visible on white */

  /* Compact layout for PDF */
  h2 { font-size: 26px !important; margin-bottom: 10px !important; }
  h3 { font-size: 16px !important; margin-bottom: 8px !important; }
  .tag { margin-bottom: 8px !important; padding: 4px 12px !important; font-size: 11px !important; }
  .grid-2 { gap: 16px !important; }
  .card { padding: 8px 10px !important; margin-bottom: 4px !important; }
  .desc { font-size: 11px !important; line-height: 1.5 !important; }
  .diagram-container svg { max-height: 340px !important; }
  .param-row { padding: 6px 0 !important; font-size: 12px !important; }
  .param-name { font-size: 12px !important; }
  .param-value { font-size: 11px !important; }
  .param-source { font-size: 10px !important; }

  /* Hide navigation elements */
  .nav-hint, .progress-bar { display: none !important; }

  /* Page setup */
  @page {
    size: A4 landscape;
    margin: 15mm;
  }
</style>
"""

# Insert CSS overrides before </head>
html_modified = html.replace("</head>", css_overrides + "\n</head>")

# Also fix SVG colors that use dark theme fills
# Replace dark background fills in SVGs
svg_replacements = [
    ('fill="#1a1a1a"', 'fill="#f0f0f2"'),
    ('fill="#f5f5f7"', 'fill="#1d1d1f"'),
    ('stroke="#333"', 'stroke="#d2d2d7"'),
    ('stroke="#555"', 'stroke="#999999"'),
    ('fill="#555"', 'fill="#999999"'),
    ('fill="#a1a1a6"', 'fill="#6e6e73"'),
    ('fill="#fff"', 'fill="#ffffff"'),
    ('fill="rgba(255,255,255,0.7)"', 'fill="rgba(0,0,0,0.5)"'),
]

for old, new in svg_replacements:
    html_modified = html_modified.replace(old, new)

# Remove the JavaScript (not needed for PDF)
import re
html_modified = re.sub(r'<script>.*?</script>', '', html_modified, flags=re.DOTALL)

# Write and convert
HTML(string=html_modified).write_pdf('gpu_model_presentation.pdf')
print("PDF generated: gpu_model_presentation.pdf")
