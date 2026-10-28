import os
import time
import json
import PyPDF2
from flask import Flask, render_template, request, jsonify, Response, send_file
from dotenv import load_dotenv

from agents import MultiAgentSystem, AGENT_DEFINITIONS, DEFAULT_MODELS, test_api_key
from sample_data import SAMPLE_PAPERS

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max PDF size

DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


@app.route("/")
def index():
    """Renders the main stylish vibrant UI dashboard."""
    return render_template(
        "index.html",
        has_env_key=bool(DEFAULT_API_KEY),
        default_model=DEFAULT_MODEL,
        agents=AGENT_DEFINITIONS,
        default_models=DEFAULT_MODELS
    )


@app.route("/api/config", methods=["GET"])
def get_config():
    """Returns current configuration and agent definitions."""
    return jsonify({
        "has_env_key": bool(DEFAULT_API_KEY),
        "env_key_preview": (DEFAULT_API_KEY[:4] + "..." + DEFAULT_API_KEY[-4:]) if len(DEFAULT_API_KEY) > 8 else "",
        "default_model": DEFAULT_MODEL,
        "available_models": DEFAULT_MODELS,
        "agents": AGENT_DEFINITIONS
    })


@app.route("/api/validate-key", methods=["POST"])
def validate_key():
    """Validates the provided Gemini API key and fetches supported models."""
    data = request.get_json(silent=True) or {}
    api_key = data.get("api_key") or DEFAULT_API_KEY
    
    if not api_key:
        return jsonify({
            "valid": False, 
            "error": "No API key provided. Please input your Gemini API Key."
        }), 400

    is_valid, result = test_api_key(api_key)
    if is_valid:
        return jsonify({
            "valid": True,
            "models": result,
            "selected_model": result[0] if result else DEFAULT_MODEL
        })
    else:
        return jsonify({
            "valid": False,
            "error": str(result)
        }), 400


@app.route("/api/extract-pdf", methods=["POST"])
def extract_pdf():
    """Extracts text content and metadata from an uploaded PDF file."""
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided in request."}), 400
    
    file = request.files["pdf"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Uploaded file must be a PDF."}), 400

    try:
        pdf_reader = PyPDF2.PdfReader(file)
        pages_count = len(pdf_reader.pages)
        raw_text_parts = []
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            raw_text_parts.append(page_text)
            
        full_text = "\n".join(raw_text_parts).strip()
        
        if not full_text:
            return jsonify({
                "error": "Unable to extract readable text. The PDF might contain scanned images without OCR."
            }), 400

        # Safety cap to 100k characters for API performance
        truncated = False
        if len(full_text) > 100000:
            full_text = full_text[:100000]
            truncated = True

        word_count = len(full_text.split())
        char_count = len(full_text)

        return jsonify({
            "success": True,
            "filename": file.filename,
            "pages": pages_count,
            "char_count": char_count,
            "word_count": word_count,
            "truncated": truncated,
            "preview": full_text[:600] + ("..." if len(full_text) > 600 else ""),
            "text": full_text
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse PDF: {str(e)}"}), 500


@app.route("/api/sample-paper", methods=["GET"])
def get_sample_paper():
    """Provides the built-in sample research paper for 1-click exploration."""
    sample = SAMPLE_PAPERS.get("attention_is_all_you_need")
    return jsonify({
        "success": True,
        "sample": sample
    })


@app.route("/api/analyze-agent", methods=["POST"])
def analyze_agent():
    """
    Executes a single specified agent in the multi-agent pipeline.
    Enables concurrent parallel execution from the frontend with granular live state.
    """
    data = request.get_json(silent=True) or {}
    agent_id = data.get("agent_id")
    context = data.get("context", "")
    api_key = data.get("api_key") or DEFAULT_API_KEY
    model_name = data.get("model_name") or DEFAULT_MODEL

    if not api_key:
        return jsonify({"error": "Gemini API key is required."}), 400
    if not context:
        return jsonify({"error": "Context text is required for analysis."}), 400
    if agent_id not in AGENT_DEFINITIONS:
        return jsonify({"error": f"Invalid agent_id: {agent_id}"}), 400

    agent_conf = AGENT_DEFINITIONS[agent_id]
    mas = MultiAgentSystem(api_key=api_key, model_name=model_name)

    start_time = time.time()
    result_text = mas.run_agent(
        name=agent_conf["name"],
        role_prompt=agent_conf["role_prompt"],
        task=agent_conf["task"],
        context=context
    )
    elapsed = round(time.time() - start_time, 2)

    is_error = result_text.startswith("⚠️ **Error")

    return jsonify({
        "success": not is_error,
        "agent_id": agent_id,
        "agent_name": agent_conf["name"],
        "elapsed_seconds": elapsed,
        "output": result_text
    })


@app.route("/api/export-dossier", methods=["POST"])
def export_dossier():
    """Exports full research dossier formatted in Markdown or HTML."""
    data = request.get_json(silent=True) or {}
    paper_title = data.get("title", "Research Paper Analysis")
    summary = data.get("summary", "")
    methodology = data.get("methodology", "")
    critique = data.get("critique", "")
    briefing = data.get("briefing", "")
    model_name = data.get("model", DEFAULT_MODEL)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    markdown_content = f"""# Multi-Agent Research Dossier: {paper_title}
*Generated on {timestamp} using Multi-Agent Paper Analysis ({model_name})*

---

## 📊 1. Executive Intelligence Briefing (Synthesizer Agent)
{briefing}

---

## 📝 2. Academic Summary (Summarizer Agent)
{summary}

---

## 🔬 3. Methodology & Architecture Analysis (Methodologist Agent)
{methodology}

---

## ⚖️ 4. Rigorous Peer Review & Critique (Critic Agent)
{critique}

---
*Report generated by Agentic AI Research Pipeline.*
"""

    return jsonify({
        "success": True,
        "filename": f"research_analysis_{int(time.time())}.md",
        "markdown": markdown_content
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🌟 Starting Multi-Agent Paper Analysis on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
