# AI Usage Log

**Tools used.** An AI coding assistant (Kiro) scaffolded the engine from a written spec (requirements/design/tasks). Separately, the running engine calls LLM APIs at runtime — Groq `llama-3.3-70b-versatile` (primary), Gemini Flash (backup), and local `Qwen2.5:7B` via Ollama — for skill extraction and synonym/implicit matching.

**Where each was used.** The AI assistant generated module scaffolding (parser, scorer, output, CLI), the synonym dictionary, and the regex field extractors from the spec. Runtime LLMs run in Stage 2 only: extracting skills from unstructured resume text and classifying matches (exact/synonym/partial/implicit) against the JD.

**Built and tuned by the developer, beyond generation.** The multi-column bounding-box reconstruction heuristic, the two-mode OCR failure handling, the grade-normalization rules, the weighted scoring formula and its component weights, the parse-quality→confidence coupling, the three-tier cascade with 3s timeouts and resume-hash caching for determinism, and the slot-aware shortlist/reserve logic were all designed and tuned manually. The rules tier runs fully offline, so the engine never depends on an LLM being reachable.
