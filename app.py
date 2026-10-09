"""Tender‑IQ Flask application – final version (Stage 2).

Features added in this update:
* File upload (PDF/TXT) and text‑area input handling.
* Sample‑tender loading via hidden form field.
* Text extraction using `nlp.preprocessing.extract_text_from_file` (PyMuPDF for PDFs).
* Rule‑based NLP pipeline (`nlp.extraction.extract_information`).
* Result is stored as a temporary JSON file identified by a UUID so the user can download it.
* The `results.html` template receives `result_id` to build a download link.
"""

import os
import uuid
import json
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    send_from_directory,
)

# import fitz  # optional PDF support – handled in preprocessing

# Local NLP utilities
from nlp.preprocessing import extract_text_from_file
from nlp.extraction import extract_information

app = Flask(__name__)
app.config["SECRET_KEY"] = "replace-with-a-secure-key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
app.config["UPLOAD_EXTENSIONS"] = [".pdf", ".txt"]
# Uploads folder inside the project root
app.config["UPLOAD_PATH"] = os.path.join(os.getcwd(), "uploads")
os.makedirs(app.config["UPLOAD_PATH"], exist_ok=True)

@app.context_processor
def inject_year():
    return {"current_year": datetime.utcnow().year}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/sample-tender')
def sample_tender():
    return send_from_directory("data", "sample_tender.txt", as_attachment=True, download_name="sample_tender.txt")

@app.route('/history')
def history():
    # Placeholder – in a full project this would list past analyses.
    return "History page – to be implemented."

@app.route('/analyze', methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        uploaded_file = request.files.get("tender_file")
        pasted_text = request.form.get("tender_text")
        source = None
        raw_text = ""
        filename = ""

        # ---------------------------------------------------
        # 1️⃣  Uploaded file handling
        # ---------------------------------------------------
        if uploaded_file and uploaded_file.filename:
            filename = uploaded_file.filename
            ext = os.path.splitext(filename)[1].lower()
            if ext not in app.config["UPLOAD_EXTENSIONS"]:
                flash("Unsupported file type – only PDF or TXT are allowed.", "danger")
                return redirect(request.url)
            # Store with a random UUID prefix to avoid name collisions
            safe_name = f"{uuid.uuid4().hex}{ext}"
            file_path = os.path.join(app.config["UPLOAD_PATH"], safe_name)
            uploaded_file.save(file_path)
            try:
                raw_text = extract_text_from_file(file_path)
                source = f"File: {filename}"
            except Exception as exc:
                flash(f"Error extracting text from the uploaded file: {exc}", "danger")
                return redirect(request.url)

        # ---------------------------------------------------
        # 2️⃣  Pasted text handling
        # ---------------------------------------------------
        elif pasted_text and pasted_text.strip():
            raw_text = pasted_text.strip()
            source = "Pasted text"

        # ---------------------------------------------------
        # 3️⃣  Sample tender loading (button on the form)
        # ---------------------------------------------------
        elif request.form.get("load_sample") == "true":
            sample_path = os.path.join("data", "sample_tender.txt")
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                source = "Sample tender"
            except FileNotFoundError:
                flash("Sample tender file not found on the server.", "danger")
                return redirect(request.url)

        else:
            flash("Please provide a file, paste text, or load the sample tender.", "warning")
            return redirect(request.url)

        # ---------------------------------------------------
        # 4️⃣  Run the NLP extraction pipeline
        # ---------------------------------------------------
        extraction_result = extract_information(raw_text)

        # Store the full result temporarily so the user can download it later.
        result_id = uuid.uuid4().hex
        result_path = os.path.join(app.config["UPLOAD_PATH"], f"result_{result_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {"source": source, "text": raw_text, "extraction": extraction_result},
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Render the results page – the template receives the UUID for the download link.
        return render_template(
            "results.html",
            source=source,
            extraction=extraction_result,
            result_id=result_id,
        )

    # GET – simply show the upload form.
    return render_template("analyze.html")

@app.route('/download/<result_id>')
def download_result(result_id):
    result_path = os.path.join(app.config["UPLOAD_PATH"], f"result_{result_id}.json")
    if not os.path.exists(result_path):
        flash("Requested analysis result not found.", "danger")
        return redirect(url_for("index"))
    return send_file(
        result_path,
        as_attachment=True,
        download_name="tender_analysis_result.json",
    )

if __name__ == "__main__":
    # Development server – set host to 0.0.0.0 so it can be accessed from other devices if needed.
    app.run(debug=True, host="0.0.0.0")
