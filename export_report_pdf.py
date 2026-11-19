import os
import re
import base64
import subprocess
import markdown

REPORT_MD_PATH = r"C:\Users\mriga\.gemini\antigravity\brain\41046365-f6e3-4673-86d3-072757b58664\PI_FNO_CFD_Pipeline_Report.md"
WORKSPACE_DIR = r"E:\Openfoam\PI-FNO-CFD-Pipeline"
HTML_OUT_PATH = os.path.join(WORKSPACE_DIR, "PI_FNO_CFD_Pipeline_Report.html")
PDF_OUT_PATH = os.path.join(WORKSPACE_DIR, "PI_FNO_CFD_Pipeline_Report.pdf")
ARTIFACT_DIR = r"C:\Users\mriga\.gemini\antigravity\brain\41046365-f6e3-4673-86d3-072757b58664"
ARTIFACT_PDF_PATH = os.path.join(ARTIFACT_DIR, "PI_FNO_CFD_Pipeline_Report.pdf")
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def get_base64_image(image_path):
    """Convert an image file to a base64 data URI."""
    clean_path = image_path.replace("file:///", "").replace("file://", "")
    if os.name == 'nt':
        clean_path = clean_path.replace("/", "\\")
    if os.path.exists(clean_path):
        with open(clean_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode("utf-8")
            return f"data:image/png;base64,{b64_data}"
    return image_path

def preprocess_markdown(text):
    """Process carousels, callouts, and protect LaTeX math."""
    
    # 1. Protect Math blocks from markdown parser mangling
    math_blocks = []
    def save_display_math(match):
        idx = len(math_blocks)
        math_blocks.append(match.group(0))
        return f"MATHBLOCKDISPLAY_{idx}_PLACEHOLDER"
    
    def save_inline_math(match):
        idx = len(math_blocks)
        math_blocks.append(match.group(0))
        return f"MATHBLOCKINLINE_{idx}_PLACEHOLDER"
    
    # Protect $$...$$
    text = re.sub(r'\$\$(.+?)\$\$', save_display_math, text, flags=re.DOTALL)
    # Protect $...$
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', save_inline_math, text)
    
    # 2. Process Carousel blocks -> High-quality Figure Grids
    def format_carousel(match):
        content = match.group(1)
        slides = re.split(r'<!--\s*slide\s*-->', content)
        
        figures_html = ['<div class="figures-gallery">']
        for i, slide in enumerate(slides):
            img_match = re.search(r'!\[(.*?)\]\((.*?)\)', slide.strip())
            if img_match:
                caption = img_match.group(1)
                img_url = img_match.group(2)
                b64_src = get_base64_image(img_url)
                figures_html.append(f"""
                <div class="figure-card">
                    <img src="{b64_src}" alt="{caption}" />
                    <div class="figure-caption"><strong>Figure {i+1}:</strong> {caption}</div>
                </div>
                """)
        figures_html.append('</div>')
        return "\n\n" + "\n".join(figures_html) + "\n\n"

    text = re.sub(r'````carousel\s*(.*?)\s*````', format_carousel, text, flags=re.DOTALL)
    
    # 3. Process GitHub-style Alert Callouts (> [!NOTE], > [!WARNING], etc.)
    def format_callout(match):
        callout_type = match.group(1).lower()
        body = match.group(2).strip()
        # Clean lines starting with >
        cleaned_body = "\n".join([line.lstrip("> ").strip() for line in body.splitlines()])
        
        icon = "ℹ️"
        title = "NOTE"
        if callout_type == "warning":
            icon = "⚠️"
            title = "WARNING"
        elif callout_type == "tip":
            icon = "💡"
            title = "TIP"
        elif callout_type == "important":
            icon = "❗"
            title = "IMPORTANT"
            
        return f"""
<div class="callout callout-{callout_type}">
    <div class="callout-header"><span class="callout-icon">{icon}</span> <strong>{title}</strong></div>
    <div class="callout-content">{cleaned_body}</div>
</div>
"""

    text = re.sub(r'>\s*\[!(NOTE|WARNING|TIP|IMPORTANT|CAUTION)\]\s*\n((?:>.*(?:\n|$))*)', 
                  format_callout, text, flags=re.IGNORECASE)
    
    # 4. Clean file:// links for clean print appearance
    def clean_file_links(match):
        link_text = match.group(1)
        return f'<span class="file-ref"><code>{link_text}</code></span>'
    text = re.sub(r'\[([^\]]+)\]\(file:///[^\)]+\)', clean_file_links, text)

    return text, math_blocks

def restore_math(html, math_blocks):
    """Restore protected LaTeX math blocks."""
    for idx, math_content in enumerate(math_blocks):
        display_placeholder = f"MATHBLOCKDISPLAY_{idx}_PLACEHOLDER"
        inline_placeholder = f"MATHBLOCKINLINE_{idx}_PLACEHOLDER"
        html = html.replace(display_placeholder, math_content)
        html = html.replace(inline_placeholder, math_content)
    return html

def build_html_document(body_html):
    """Wrap converted HTML body into a complete styled publication-grade document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PI-FNO-CFD Pipeline Technical Report</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  svg: {{
    fontCache: 'global'
  }},
  startup: {{
    typeset: true
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script" async></script>
<style>
@page {{
    size: A4 portrait;
    margin: 20mm 18mm 22mm 18mm;
    @bottom-center {{
        content: counter(page);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 9pt;
        color: #64748b;
    }}
}}

* {{
    box-sizing: border-box;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0 auto;
    max-width: 100%;
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
    font-weight: 700;
    margin-top: 1.8em;
    margin-bottom: 0.6em;
    page-break-after: avoid;
    break-after: avoid;
}}

h1 {{
    font-size: 22pt;
    color: #0f2942;
    border-bottom: 2.5px solid #0284c7;
    padding-bottom: 6px;
    margin-top: 0;
}}

h2 {{
    font-size: 14pt;
    color: #0369a1;
    border-bottom: 1.2px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 24pt;
}}

h3 {{
    font-size: 11.5pt;
    color: #0f172a;
    margin-top: 16pt;
}}

h4 {{
    font-size: 10.5pt;
    color: #334155;
}}

p {{
    margin-top: 0.6em;
    margin-bottom: 0.8em;
    text-align: justify;
    text-justify: inter-word;
}}

strong {{
    color: #0f172a;
}}

hr {{
    border: none;
    height: 1px;
    background-color: #cbd5e1;
    margin: 24pt 0;
}}

/* Tables */
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14pt 0;
    font-size: 8.8pt;
    page-break-inside: avoid;
    break-inside: avoid;
}}

table th {{
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    text-align: left;
    padding: 8px 10px;
    border-top: 1.5px solid #0f172a;
    border-bottom: 1.5px solid #0f172a;
}}

table td {{
    padding: 7px 10px;
    border-bottom: 1px solid #e2e8f0;
    color: #334155;
}}

table tr:nth-child(even) td {{
    background-color: #f8fafc;
}}

/* Preformatted ASCII and Code Blocks */
pre {{
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 14px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 7.8pt;
    line-height: 1.45;
    color: #0f172a;
    overflow-x: auto;
    margin: 12pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
}}

code {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 8.5pt;
    background-color: #f1f5f9;
    padding: 2px 5px;
    border-radius: 4px;
    color: #0f2942;
}}

pre code {{
    background-color: transparent;
    padding: 0;
    font-size: inherit;
    color: inherit;
}}

/* Callouts */
.callout {{
    border-left: 4px solid #0284c7;
    background-color: #f0f9ff;
    padding: 10px 14px;
    margin: 12pt 0;
    border-radius: 0 6px 6px 0;
    font-size: 9.2pt;
    page-break-inside: avoid;
    break-inside: avoid;
}}

.callout-header {{
    font-weight: 700;
    color: #0369a1;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}}

.callout-content {{
    color: #334155;
}}

/* Figures Gallery */
.figures-gallery {{
    display: flex;
    flex-direction: column;
    gap: 16pt;
    margin: 16pt 0;
}}

.figure-card {{
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
}}

.figure-card img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
    border-radius: 4px;
}}

.figure-caption {{
    margin-top: 8px;
    font-size: 8.8pt;
    color: #475569;
}}

/* Lists */
ul, ol {{
    margin-top: 0.4em;
    margin-bottom: 0.8em;
    padding-left: 22px;
}}

li {{
    margin-bottom: 0.3em;
}}

.file-ref code {{
    background-color: #e0f2fe;
    color: #0369a1;
    font-weight: 600;
}}

/* Clean up header decoration */
.header-sub {{
    color: #64748b;
    font-size: 12pt;
    margin-top: -8px;
    margin-bottom: 20px;
    font-weight: 500;
}}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

def main():
    print(f"[PDF Export] Reading report from: {REPORT_MD_PATH}")
    with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Preprocess
    print("[PDF Export] Preprocessing markdown (carousels, callouts, base64 images, math)...")
    preprocessed_md, math_blocks = preprocess_markdown(md_text)

    # Convert to HTML via python-markdown with tables & attr_list extensions
    print("[PDF Export] Converting markdown to HTML...")
    html_body = markdown.markdown(preprocessed_md, extensions=['tables', 'fenced_code', 'attr_list'])

    # Restore math
    print(f"[PDF Export] Restoring {len(math_blocks)} LaTeX math expressions...")
    html_body = restore_math(html_body, math_blocks)

    # Wrap in complete HTML
    full_html = build_html_document(html_body)

    # Save HTML file
    print(f"[PDF Export] Writing standalone printable HTML to: {HTML_OUT_PATH}")
    with open(HTML_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)

    # Render to PDF via Chrome headless
    print(f"[PDF Export] Rendering PDF via Chrome headless...")
    args = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={PDF_OUT_PATH}",
        HTML_OUT_PATH
    ]
    
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[PDF Export Error] Chrome failed: {proc.stderr}")
        return False

    if os.path.exists(PDF_OUT_PATH):
        pdf_size_mb = os.path.getsize(PDF_OUT_PATH) / (1024 * 1024)
        print(f"[PDF Export] Success! PDF created at: {PDF_OUT_PATH} ({pdf_size_mb:.2f} MB)")
        
        # Copy to artifact dir as well
        import shutil
        shutil.copy2(PDF_OUT_PATH, ARTIFACT_PDF_PATH)
        print(f"[PDF Export] Copied to artifact directory: {ARTIFACT_PDF_PATH}")
        return True
    else:
        print("[PDF Export Error] PDF file was not created.")
        return False

if __name__ == "__main__":
    main()
