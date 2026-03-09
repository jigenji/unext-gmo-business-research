#!/usr/bin/env python3
import markdown
from weasyprint import HTML

INPUT = "gmo_business_segment_analysis.md"
OUTPUT = "gmo_business_segment_analysis.pdf"

with open(INPUT, "r", encoding="utf-8") as f:
    md_text = f.read()

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

html_full = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
}}
body {{
    font-family: "Noto Sans CJK JP", "Noto Sans JP", "Hiragino Sans", "Yu Gothic", sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #222;
}}
h1 {{
    font-size: 18pt;
    border-bottom: 3px solid #0057b8;
    padding-bottom: 6px;
    margin-top: 0;
    color: #0057b8;
}}
h2 {{
    font-size: 14pt;
    color: #0057b8;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    margin-top: 24px;
}}
h3 {{
    font-size: 12pt;
    color: #333;
    margin-top: 18px;
}}
h4 {{
    font-size: 10.5pt;
    color: #444;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9pt;
}}
th, td {{
    border: 1px solid #bbb;
    padding: 5px 8px;
    text-align: left;
}}
th {{
    background-color: #0057b8;
    color: white;
    font-weight: bold;
}}
tr:nth-child(even) {{
    background-color: #f4f8fc;
}}
code, pre {{
    font-family: "Noto Sans Mono CJK JP", monospace;
    font-size: 8.5pt;
    background: #f5f5f5;
    padding: 2px 4px;
    border-radius: 3px;
}}
pre {{
    padding: 10px;
    overflow-x: auto;
    white-space: pre-wrap;
}}
a {{
    color: #0057b8;
    text-decoration: none;
}}
ul, ol {{
    padding-left: 20px;
}}
li {{
    margin-bottom: 3px;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

HTML(string=html_full).write_pdf(OUTPUT)
print(f"PDF generated: {OUTPUT}")
