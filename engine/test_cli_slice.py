"""End-to-end slice test for the CLI (task 3.3).

Proves the thin vertical slice runs to completion WITHOUT PyMuPDF (or any OCR
stack) installed: a folder holding a dummy (invalid) ``.pdf`` plus a non-PDF
file is processed so that

* ``main`` returns exit code 0,
* the dummy PDF appears exactly once as ``Failed`` in ``parse_listing.json``
  (failure is a first-class, visible outcome — never a silent drop, Req 2.1),
* the non-PDF file is skipped (does not appear in the listing, Req 1.2).

The dummy PDF is deliberately not a valid PDF, so extraction fails and the
parser downgrades the resume to ``Failed`` regardless of whether PyMuPDF is
present — which is exactly the DQ-protection guarantee this slice exists for.

Runnable directly with ``python engine/test_cli_slice.py`` (pytest optional).
"""

import json
import os
import tempfile

from engine import cli


def test_slice_runs_and_lists_failed_pdf_without_pymupdf() -> None:
    """Dummy PDF lands as Failed; non-PDF is skipped; exit code is 0."""
    with tempfile.TemporaryDirectory() as resumes_dir, \
            tempfile.TemporaryDirectory() as out_dir:
        # A file with a .pdf extension but bytes that are NOT a valid PDF.
        dummy_pdf = os.path.join(resumes_dir, "dummy.pdf")
        with open(dummy_pdf, "wb") as fh:
            fh.write(b"this is not a real pdf document")

        # A non-PDF file that must be skipped (Req 1.2).
        non_pdf = os.path.join(resumes_dir, "notes.txt")
        with open(non_pdf, "w", encoding="utf-8") as fh:
            fh.write("just some notes, not a resume")

        # A minimal JD file — the CLI now requires at least one --jd (Req 19.2).
        jd_path = os.path.join(resumes_dir, "backend.json")
        with open(jd_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "id": "backend",
                    "title": "Backend Intern",
                    "required_skills": ["Python", "SQL"],
                },
                fh,
            )

        exit_code = cli.main(
            ["--resumes", resumes_dir, "--jd", jd_path, "--out", out_dir]
        )
        assert exit_code == 0, f"expected exit code 0, got {exit_code}"

        listing_json = os.path.join(out_dir, "parse_listing.json")
        assert os.path.isfile(listing_json), "parse_listing.json was not written"

        with open(listing_json, encoding="utf-8") as fh:
            records = json.load(fh)

        # Exactly one record — only the PDF, the non-PDF is skipped (Req 2.1, 1.2).
        assert len(records) == 1, f"expected 1 record, got {len(records)}: {records}"

        record = records[0]
        assert record["file_name"] == "dummy.pdf"
        assert record["parse_flag"] == "Failed", (
            f"expected dummy.pdf to be Failed, got {record['parse_flag']}"
        )
        assert record["human_review"] is True
        assert record["score"] is None

        # The non-PDF must not appear anywhere in the listing (Req 1.2).
        file_names = {r["file_name"] for r in records}
        assert "notes.txt" not in file_names


if __name__ == "__main__":
    test_slice_runs_and_lists_failed_pdf_without_pymupdf()
    print("PASS: slice runs end-to-end; dummy.pdf -> Failed; notes.txt skipped")
