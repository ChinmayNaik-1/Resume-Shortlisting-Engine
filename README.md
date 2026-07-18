# Resume Shortlisting Engine

The Resume Shortlisting Engine is a two-stage Python pipeline — an offline PDF parser feeding a three-tier scoring cascade — that ranks a folder of PDF resumes against one or more job descriptions and produces an explainable, confidence-rated shortlist. Every resume always appears in the output (scored or flagged for review), each result carries a score out of 100, a High/Medium/Low confidence level, three plain-language reasoning bullets, and a parse-quality flag, so a reviewer can trust and audit every decision.

## Architecture

**Stage 1 — Parser (100% offline, no API keys).**
Extracts text with PyMuPDF (primary) and pdfplumber (tables), reconstructs multi-column layouts by clustering text-block bounding boxes and reading the left column top-to-bottom then the right, and falls back to `pytesseract` OCR when a page yields too little text. It then extracts fields (name, contact, education, skills, projects), normalizes grades (CGPA / percentage / GPA) onto a 10-point scale with a recorded assumption when ambiguous, and classifies each resume as **Clean**, **Partial**, or **Failed**.

**Stage 2 — Scoring cascade.**
A rules baseline always runs offline first and establishes a guaranteed floor score. On top of that, an LLM enhancement cascade adds implicit-skill detection and richer matching:

1. **Rules baseline** — offline synonym/vocabulary matching (always runs).
2. **Groq `llama-3.3-70b-versatile`** — primary cloud tier.
3. **Gemini Flash** — backup cloud tier.
4. **Local Qwen2.5:7B via Ollama** — offline fallback tier.

All tiers run at `temperature=0` and results are cached by resume hash, so reruns on unchanged input are deterministic and free of network calls. The LLM can only raise a score above the offline floor, never silently lower it. Confidence is capped by both parse quality (Partial ⇒ at most Medium, Failed ⇒ no score) and which tier confirmed the result (rules-only or local ⇒ at most Medium).

## Setup (under 5 minutes)

**1. Python 3.11**

```bash
python --version   # expect 3.11.x
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. System prerequisite — Tesseract OCR (optional)**

`pytesseract` is only a wrapper; it needs the native Tesseract binary to OCR scanned resumes.

- Windows: install from [UB-Mannheim Tesseract builds](https://github.com/UB-Mannheim/tesseract/wiki)
- macOS: `brew install tesseract`
- Debian/Ubuntu: `sudo apt-get install tesseract-ocr`

Tesseract is **optional for a basic run**: if it is absent, the engine still runs end-to-end and simply flags scanned/image-only resumes as **Failed** (they appear in the output for manual review rather than being dropped).

**4. Optional API keys (`.env`)**

Copy `.env.example` to `.env` and fill in whatever you have:

```bash
cp .env.example .env
```

```dotenv
# .env
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
OLLAMA_HOST=http://localhost:11434
```

The engine runs **fully offline on the rules tier with no keys set** — it just caps confidence at Medium. Add `GROQ_API_KEY` (primary) and/or `GEMINI_API_KEY` (backup) to unlock cloud-tier enhancement and High confidence.

**5. Optional local fallback (Ollama)**

For an offline LLM tier without cloud keys:

```bash
ollama pull qwen2.5:7b
```

## Run a working example

```bash
python -m engine.cli --resumes ./resumes --jd ./jds/backend.json --out ./output
```

- `--resumes` — folder containing the PDF resumes to rank.
- `--jd` — a Job Description config file (JSON or YAML); **repeatable** to score against several JDs in one run.
- `--out` — directory where output artifacts are written (default `./output`).

Run against all five bundled JDs at once:

```bash
python -m engine.cli --resumes ./resumes \
  --jd ./jds/backend.json \
  --jd ./jds/frontend.json \
  --jd ./jds/fullstack.json \
  --jd ./jds/api.json \
  --jd ./jds/database.json \
  --out ./output
```

A missing required argument prints usage and exits non-zero; a missing or empty resume folder is reported and exits cleanly. The pipeline is **offline-first** — it produces a result even with no keys, no Tesseract, and no Ollama.

## Output

For each Job Description the engine produces:

- A **ranked shortlist** (top-N by `slots`) and a **reserve** list of the next-best candidates.
- Per candidate: a **score out of 100**, a **confidence** level (**High / Medium / Low**), **3 reasoning bullets** explaining the score, and a **parse-quality flag** (Clean / Partial / Failed).
- Candidates whose resume **Failed** to parse appear with no score, flagged for human review — never silently dropped.

Alongside the shortlist, two supporting artifacts are written to `--out`:

- **`sample_output`** — the ranked, explainable shortlist/reserve per JD (Markdown + JSON).
- **`parse_quality_report`** — a per-resume parse summary (Clean/Partial/Failed, OCR usage, failure reasons) so you can see exactly how each PDF was handled.

Every ingested PDF is guaranteed to appear in the output exactly once; the count of output records is asserted equal to the count of ingested PDFs before anything is written.

## Project layout

```
resume-shortlister/
├── engine/                 # pipeline package
│   ├── cli.py              # CLI entry point (--resumes / --jd / --out)
│   ├── config.py           # tunable constants + .env loading
│   ├── parser.py           # Stage 1: PDF extraction, columns, OCR, fields
│   ├── normalizer.py       # grade normalization to 10-pt scale
│   ├── jd.py               # JobDescription model + JSON/YAML loader
│   ├── synonyms.py         # skill synonym dictionary (rules tier)
│   ├── scorer.py           # Stage 2: rules baseline + cascade + scoring
│   ├── output.py           # ranking, shortlist/reserve, report writers
│   └── models.py           # dataclasses (ResumeData, ScoredCandidate, ...)
├── jds/                    # sample Job Description configs (5 roles)
│   ├── backend.json  frontend.json  fullstack.json  api.json  database.json
├── output/                 # generated shortlist + reports (created on run)
├── requirements.txt        # Python dependencies (Python 3.11)
├── .env.example            # template for optional API keys / Ollama host
└── README.md
```
