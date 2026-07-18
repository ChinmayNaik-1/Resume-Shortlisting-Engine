"""Command-line entry point for the Resume Shortlisting Engine.

This module owns folder ingestion, per-resume error isolation, and run
orchestration. Task 3.1 lays the ingestion foundation:

- ``find_pdfs`` lists a folder and separates ``.pdf`` files from everything
  else, recording skipped non-PDF names for the run log (Req 1.2).
- ``ingest_resumes`` parses each PDF with per-resume try/except isolation so a
  single bad PDF becomes a ``Failed`` ``ResumeData`` and never aborts the run
  (Req 2.3).
- ``main`` parses CLI args and guards the resume folder: a missing/empty folder
  is reported and exits cleanly with code 0 (Req 1.3).

Scoring and output wiring arrive in later tasks (3.3 end-to-end slice, 11.7 arg
hardening); TODO markers below indicate where they slot in.
"""

import argparse
import os
import sys

from engine import jd as jd_module
from engine import output, parser, pipeline
from engine.models import ResumeData, ParseFlag


def find_pdfs(folder: str) -> tuple[list[str], list[str]]:
    """Split a folder's files into PDF paths and skipped non-PDF names.

    Returns ``(pdf_paths, skipped_non_pdf_names)``. PDF detection is
    case-insensitive on the ``.pdf`` extension. Only regular files are
    considered; sub-directories are ignored entirely. ``pdf_paths`` are full
    paths (suitable for ``parse_resume``); skipped entries are bare file names
    for the run log (Req 1.1, 1.2).

    A non-existent folder yields two empty lists so callers can treat "missing"
    and "empty" identically (Req 1.3).
    """
    pdf_paths: list[str] = []
    skipped_non_pdf_names: list[str] = []

    if not os.path.isdir(folder):
        return pdf_paths, skipped_non_pdf_names

    for name in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, name)
        if not os.path.isfile(full_path):
            continue  # ignore sub-directories and other non-file entries
        if name.lower().endswith(".pdf"):
            pdf_paths.append(full_path)
        else:
            skipped_non_pdf_names.append(name)

    return pdf_paths, skipped_non_pdf_names


def ingest_resumes(folder: str, log: list[str] | None = None) -> list[ResumeData]:
    """Parse every PDF in ``folder`` with per-resume error isolation.

    Each ``parse_resume`` call is wrapped in try/except so an unexpected failure
    on one PDF becomes a ``Failed`` ``ResumeData`` (with ``error_reason``) and
    the loop continues to the next file (Req 2.3). ``parse_resume`` already
    self-guards and never raises, but this outer guard is the contract-level
    isolation boundary required by the design.

    Progress and skip messages are appended to ``log`` when provided so the run
    log can surface skipped non-PDF files (Req 1.2) and per-resume failures.
    """
    pdf_paths, skipped = find_pdfs(folder)

    if log is not None:
        for name in skipped:
            log.append(f"Skipped non-PDF file: {name}")

    resumes: list[ResumeData] = []
    for pdf_path in pdf_paths:
        file_name = os.path.basename(pdf_path)
        try:
            resume = parser.parse_resume(pdf_path)
        except Exception as exc:  # Isolation boundary — one bad PDF never aborts the run (Req 2.3).
            resume = ResumeData(
                file_name=file_name,
                resume_hash="",
                raw_text="",
                parse_flag=ParseFlag.FAILED,
                error_reason=f"Ingestion error: {exc}",
            )

        resumes.append(resume)

        if log is not None:
            if resume.parse_flag == ParseFlag.FAILED and resume.error_reason:
                log.append(f"Parsed {file_name}: Failed ({resume.error_reason})")
            else:
                log.append(f"Parsed {file_name}: {resume.parse_flag.value}")

    return resumes


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser (task 11.7, Req 19.1-19.3).

    ``--jd`` is a repeatable JSON/YAML JD file path. ``--jd-text`` is a reserved
    hook for the live-JD bonus (bonus B); because live parsing is not yet
    enabled it is rejected cleanly at runtime rather than silently ignored.
    Argument-level validation (at least one JD source is required) is enforced
    in :func:`main` so a missing required arg exits non-zero without processing
    (Req 19.2).
    """
    arg_parser = argparse.ArgumentParser(
        prog="engine.cli",
        description="Resume Shortlisting Engine — rank PDF resumes against job descriptions.",
    )
    arg_parser.add_argument(
        "--resumes",
        required=True,
        metavar="DIR",
        help="Folder containing raw PDF resumes.",
    )
    arg_parser.add_argument(
        "--jd",
        action="append",
        default=None,
        metavar="FILE",
        help="Job Description config file (JSON or YAML). Repeatable for multiple JDs.",
    )
    arg_parser.add_argument(
        "--jd-text",
        default=None,
        metavar="TEXT",
        help=(
            "Raw Job Description text parsed live into a JobDescription "
            "(bonus B — not yet enabled)."
        ),
    )
    arg_parser.add_argument(
        "--out",
        default="./output",
        metavar="DIR",
        help="Output directory for the ranked shortlist artifacts (default: ./output).",
    )
    arg_parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the optional Streamlit UI (bonus).",
    )
    return arg_parser


def main(argv=None) -> int:
    """CLI entry point. Returns a process exit code.

    End-to-end run (Req 19.1-19.3):

    1. **Argument validation (Req 19.2).** ``--resumes`` is required by argparse
       (a missing value exits non-zero). At least one JD source is required: if
       neither ``--jd`` nor ``--jd-text`` is given, a usage error is printed and
       the process exits non-zero *without processing*. ``--jd-text`` is the
       live-JD bonus hook; since live parsing is not yet enabled, supplying it is
       rejected cleanly with a non-zero exit.
    2. **Folder guard (Req 1.3).** A missing folder or a folder with zero PDFs is
       reported and exits cleanly with code 0.
    3. **Ingestion (Req 1.1, 1.2, 2.3).** Every PDF is parsed once with
       per-resume isolation; non-PDF files are skipped and logged.
    4. **Scoring + artifacts (Req 16.x, 17.1, 17.2).** The already-parsed resume
       list is handed to :func:`pipeline.run_many` (parsed once, reused across
       JDs), which writes ``sample_output`` and the authoritative
       ``parse_quality_report`` to ``args.out``. A completeness listing is also
       written. Each JD's summary line and the artifact paths are printed.
    """
    arg_parser = _build_arg_parser()
    args = arg_parser.parse_args(argv)

    log: list[str] = []

    # --- Argument validation: at least one JD source required (Req 19.2) ---
    if args.jd_text is not None:
        # Live JD parsing is a bonus that is not yet enabled — reject cleanly
        # rather than silently ignoring the flag (Req 19.3, 20.2).
        print(
            "error: --jd-text (live JD parsing) is a bonus feature that is not "
            "yet enabled; supply a JD file via --jd instead.",
            file=sys.stderr,
        )
        return 2

    if not args.jd:
        arg_parser.error(
            "at least one --jd FILE is required (repeatable). "
            "Provide a JSON or YAML Job Description config file."
        )
        # arg_parser.error raises SystemExit(2); the return below is unreachable
        # but documents the non-zero contract (Req 19.2).
        return 2

    # --- Folder guard (Req 1.3) -------------------------------------------
    if not os.path.isdir(args.resumes):
        print(f"Resume folder not found: {args.resumes}")
        print("Nothing to process. Exiting.")
        return 0

    pdf_paths, skipped = find_pdfs(args.resumes)

    if not pdf_paths:
        print(f"No PDF files found in folder: {args.resumes}")
        if skipped:
            print(f"Skipped {len(skipped)} non-PDF file(s): {', '.join(skipped)}")
        print("Nothing to process. Exiting.")
        return 0

    # --- Ingestion (Req 1.1, 1.2, 2.3) ------------------------------------
    print(f"Found {len(pdf_paths)} PDF file(s) in {args.resumes}.")
    if skipped:
        print(f"Skipped {len(skipped)} non-PDF file(s): {', '.join(skipped)}")

    # Parse EXACTLY ONCE here; the parsed list is reused across every JD.
    resumes = ingest_resumes(args.resumes, log=log)

    # --- Completeness output (Req 2.1, 2.4) -------------------------------
    # Hard guard before writing: the number of ingested PDFs must equal the
    # number of resume records we are about to emit, so a resume can never be
    # silently dropped (Req 2.4).
    output.assert_completeness(len(pdf_paths), len(resumes))

    listing_path = output.write_completeness_listing(resumes, args.out)
    print(f"\nWrote completeness listing to: {listing_path}")

    # --- Summary: totals and counts by parse flag (Req 19.1) --------------
    clean = sum(1 for r in resumes if r.parse_flag == ParseFlag.CLEAN)
    partial = sum(1 for r in resumes if r.parse_flag == ParseFlag.PARTIAL)
    failed = sum(1 for r in resumes if r.parse_flag == ParseFlag.FAILED)

    print(f"Total resumes: {len(resumes)}")
    print(f"  Clean:   {clean}")
    print(f"  Partial: {partial}")
    print(f"  Failed:  {failed}")

    # --- Load JDs (Req 19.3) ----------------------------------------------
    jds = jd_module.load_jds(args.jd)

    # --- Score + write artifacts through the engine facade (Req 16.x, 17.x) -
    # Pass the already-parsed resume list so no PDF is parsed a second time.
    results = pipeline.run_many(resumes, jds, out_dir=args.out)

    print()
    for jd, result in zip(jds, results):
        # One-line per-JD summary from the contract dict (Req 16.3).
        cutoff = result["cutoff"]
        cutoff_str = "n/a" if cutoff is None else f"{cutoff:.1f}"
        print(
            f"JD '{jd.title}': {result['evaluated']} candidate(s) evaluated, "
            f"{result['shortlisted']} shortlisted, score cutoff {cutoff_str}, "
            f"{result['failures']} parse failure(s)."
        )

    sample_md = os.path.join(args.out, "sample_output.md")
    quality_md = os.path.join(args.out, "parse_quality_report.md")
    print(f"\nWrote sample output to: {sample_md}")
    print(f"Wrote parse quality report to: {quality_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
