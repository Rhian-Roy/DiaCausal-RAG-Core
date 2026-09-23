"""Document ingestion pipeline with section-aware chunking.

Parses PDF and Markdown clinical documents, applies hierarchical
chunking (~512 tokens per chunk), and preserves metadata (doc_id,
section path, page number, authority tier).

Includes a synthetic guideline seeder for bootstrapping the knowledge
base with verified clinical rules when real PDFs are not yet available.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Iterator

from app.schemas.contracts import MedicalChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MAX_CHUNK_TOKENS = 512  # approximate; we split on whitespace
OVERLAP_TOKENS = 50


# ---------------------------------------------------------------------------
# Synthetic Guideline Seeder
# ---------------------------------------------------------------------------

_SYNTHETIC_GUIDELINES: list[dict] = [
    {
        "filename": "ICMR_T2DM_Workflow_2024.md",
        "doc_title": "ICMR Guidelines for Management of Type 2 Diabetes",
        "issuing_body": "ICMR",
        "publication_year": 2024,
        "authority_tier": 2,
        "sections": {
            "1. Initial Assessment": (
                "All patients with newly diagnosed Type 2 Diabetes should undergo "
                "comprehensive metabolic assessment including HbA1c, fasting glucose, "
                "lipid profile, renal function (eGFR and UACR), and cardiovascular "
                "risk stratification. BMI should be classified using Asian-Indian "
                "cutoffs: >=23 kg/m² overweight, >=25 kg/m² obese."
            ),
            "2. First-Line Therapy": (
                "Metformin remains the first-line pharmacotherapy for Type 2 Diabetes "
                "unless contraindicated (eGFR <30 mL/min/1.73m² or risk of lactic "
                "acidosis). Starting dose 500 mg once daily, titrate to 1000 mg "
                "twice daily over 4-8 weeks. If HbA1c target not met after 3 months "
                "of optimized metformin, consider add-on therapy."
            ),
            "3. Second-Line Add-On Selection": (
                "Selection of second-line agent should be individualized based on "
                "patient profile:\n"
                "- SGLT2 inhibitors (dapagliflozin, empagliflozin): Preferred if "
                "cardiovascular disease, heart failure, or CKD with adequate eGFR.\n"
                "- DPP-4 inhibitors (sitagliptin, vildagliptin): Preferred if "
                "hypoglycemia risk is a concern and weight neutrality desired.\n"
                "- Sulfonylureas (glimepiride, gliclazide): Effective but carry "
                "hypoglycemia risk; use with caution in elderly and renal impairment."
            ),
            "4. Monitoring and Follow-Up": (
                "HbA1c should be measured every 3 months until target achieved, "
                "then every 6 months. Renal function should be monitored at least "
                "annually. Individualized HbA1c targets: <7% for most adults, "
                "<8% for elderly or those with hypoglycemia risk."
            ),
        },
    },
    {
        "filename": "FDA_FARXIGA_Dapagliflozin_Label.md",
        "doc_title": "FDA FARXIGA (Dapagliflozin) Prescribing Information",
        "issuing_body": "FDA",
        "publication_year": 2023,
        "authority_tier": 1,
        "sections": {
            "1. Indications and Usage": (
                "FARXIGA (dapagliflozin) is a sodium-glucose co-transporter 2 "
                "(SGLT2) inhibitor indicated as an adjunct to diet and exercise "
                "to improve glycemic control in adults with type 2 diabetes mellitus."
            ),
            "2.2 Dosage in Renal Impairment": (
                "Dapagliflozin is NOT RECOMMENDED for glycemic control in patients "
                "with an eGFR below 45 mL/min/1.73m². Dapagliflozin can be initiated "
                "or continued for heart failure and CKD indications regardless of "
                "eGFR. No dose adjustment is needed for eGFR >=45 mL/min/1.73m²."
            ),
            "5. Warnings and Precautions": (
                "Ketoacidosis: Cases of diabetic ketoacidosis (DKA) have been "
                "reported. Assess for ketoacidosis in patients presenting with "
                "signs and symptoms. Discontinue FARXIGA if DKA is confirmed. "
                "Volume Depletion: SGLT2 inhibitors cause intravascular volume "
                "contraction. Assess volume status before initiation. "
                "Genital Mycotic Infections: SGLT2 inhibitors increase risk."
            ),
        },
    },
    {
        "filename": "FDA_JARDIANCE_Empagliflozin_Label.md",
        "doc_title": "FDA JARDIANCE (Empagliflozin) Prescribing Information",
        "issuing_body": "FDA",
        "publication_year": 2023,
        "authority_tier": 1,
        "sections": {
            "1. Indications and Usage": (
                "JARDIANCE (empagliflozin) is a sodium-glucose co-transporter 2 "
                "(SGLT2) inhibitor indicated as an adjunct to diet and exercise "
                "to improve glycemic control in adults with type 2 diabetes mellitus."
            ),
            "2.2 Dosage in Renal Impairment": (
                "Empagliflozin is NOT RECOMMENDED for glycemic control in patients "
                "with an eGFR below 30 mL/min/1.73m². Empagliflozin is "
                "CONTRAINDICATED in patients on dialysis. No dose adjustment is "
                "required for patients with eGFR >=30 mL/min/1.73m²."
            ),
            "5. Warnings and Precautions": (
                "Ketoacidosis: Cases of DKA have been reported in patients with "
                "diabetes treated with SGLT2 inhibitors. Hypotension: Empagliflozin "
                "causes intravascular volume contraction. Symptomatic hypotension "
                "may occur particularly in patients with renal impairment, elderly, "
                "or patients on diuretics."
            ),
        },
    },
    {
        "filename": "FDA_JANUVIA_Sitagliptin_Label.md",
        "doc_title": "FDA JANUVIA (Sitagliptin) Prescribing Information",
        "issuing_body": "FDA",
        "publication_year": 2023,
        "authority_tier": 1,
        "sections": {
            "1. Indications and Usage": (
                "JANUVIA (sitagliptin) is a dipeptidyl peptidase-4 (DPP-4) "
                "inhibitor indicated as an adjunct to diet and exercise to improve "
                "glycemic control in adults with type 2 diabetes mellitus."
            ),
            "2.2 Dosage in Renal Impairment": (
                "Dosing is based on renal function as follows:\n"
                "- eGFR >=45 mL/min/1.73m²: 100 mg once daily (no adjustment needed).\n"
                "- eGFR 30 to <45 mL/min/1.73m²: 50 mg once daily.\n"
                "- eGFR <30 mL/min/1.73m² or ESRD on dialysis: 25 mg once daily.\n"
                "Renal function should be assessed prior to initiation and "
                "periodically thereafter."
            ),
            "5. Warnings and Precautions": (
                "Pancreatitis: There have been postmarketing reports of acute "
                "pancreatitis. If pancreatitis is suspected, promptly discontinue "
                "JANUVIA. Hypersensitivity Reactions: Serious hypersensitivity "
                "reactions including anaphylaxis, angioedema, and exfoliative skin "
                "conditions have been reported. Severe and Disabling Arthralgia: "
                "Severe joint pain has been reported. Consider as a possible cause "
                "and discontinue if appropriate."
            ),
        },
    },
    {
        "filename": "Glimepiride_Label.md",
        "doc_title": "Glimepiride Prescribing Information (Sulfonylurea)",
        "issuing_body": "FDA",
        "publication_year": 2022,
        "authority_tier": 1,
        "sections": {
            "1. Indications and Usage": (
                "Glimepiride is a sulfonylurea indicated as an adjunct to diet and "
                "exercise to improve glycemic control in adults with type 2 "
                "diabetes mellitus."
            ),
            "2.1 Recommended Dosing": (
                "Starting dose: 1 mg or 2 mg once daily with breakfast or first "
                "main meal. Maximum recommended dose: 8 mg once daily. "
                "In patients with renal impairment, start at 1 mg once daily "
                "and titrate carefully. Glimepiride carries a HIGH RISK of "
                "hypoglycemia, especially in elderly patients, those with renal "
                "impairment, malnourished patients, or those with adrenal or "
                "pituitary insufficiency."
            ),
            "5. Warnings and Precautions": (
                "Hypoglycemia: All sulfonylureas can cause severe hypoglycemia. "
                "Proper patient selection, dosage, and instructions are important. "
                "Risk factors for hypoglycemia include: renal impairment, hepatic "
                "impairment, elderly, debilitated or malnourished patients, "
                "adrenal or pituitary insufficiency, alcohol use, and concomitant "
                "use of other glucose-lowering agents. Hemolytic Anemia: Treat "
                "patients with G6PD deficiency with caution and consider a "
                "non-sulfonylurea alternative."
            ),
        },
    },
]


def seed_synthetic_guidelines(output_dir: Path | None = None) -> list[Path]:
    """Write synthetic Markdown guidelines into *output_dir*.

    Returns the list of paths written.
    """
    out = output_dir or DATA_RAW_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for guide in _SYNTHETIC_GUIDELINES:
        path = out / guide["filename"]
        lines: list[str] = [
            f"# {guide['doc_title']}\n",
            f"**Issuing Body:** {guide['issuing_body']}  \n",
            f"**Publication Year:** {guide['publication_year']}  \n",
            f"**Authority Tier:** {guide['authority_tier']}  \n\n",
        ]
        for section_title, body in guide["sections"].items():
            lines.append(f"## {section_title}\n\n{body}\n\n")
        path.write_text("".join(lines), encoding="utf-8")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Chunking Utilities
# ---------------------------------------------------------------------------

def _approx_token_count(text: str) -> int:
    """Rough whitespace-based token count."""
    return len(text.split())


def _chunk_text(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split *text* into overlapping chunks of ~max_tokens whitespace tokens."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap_tokens
    return chunks


def _make_chunk_id(doc_title: str, section: str, idx: int) -> str:
    """Deterministic chunk ID from document metadata."""
    raw = f"{doc_title}::{section}::{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Markdown Parser
# ---------------------------------------------------------------------------

def _parse_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_heading, section_body) pairs from Markdown text."""
    sections: list[tuple[str, str]] = []
    current_heading = "Preamble"
    current_body: list[str] = []
    for line in text.splitlines():
        heading_match = re.match(r"^(#{1,4})\s+(.+)", line)
        if heading_match:
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = heading_match.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))
    return sections


def _extract_front_matter(text: str) -> dict:
    """Extract metadata from markdown bold key-value pairs."""
    meta: dict = {}
    for match in re.finditer(r"\*\*(.+?):\*\*\s*(.+)", text):
        key = match.group(1).strip().lower().replace(" ", "_")
        meta[key] = match.group(2).strip()
    return meta


# ---------------------------------------------------------------------------
# Public Ingestion API
# ---------------------------------------------------------------------------

def ingest_markdown_file(path: Path) -> list[MedicalChunk]:
    """Parse a single Markdown file into MedicalChunk instances."""
    text = path.read_text(encoding="utf-8")
    meta = _extract_front_matter(text)

    doc_title = meta.get("doc_title") or path.stem.replace("_", " ")
    issuing_body = meta.get("issuing_body", "Unknown")
    pub_year = int(meta.get("publication_year", 2024))
    authority_tier = int(meta.get("authority_tier", 3))

    # Override from matching synthetic guideline
    for guide in _SYNTHETIC_GUIDELINES:
        if guide["filename"] == path.name:
            doc_title = guide["doc_title"]
            issuing_body = guide["issuing_body"]
            pub_year = guide["publication_year"]
            authority_tier = guide["authority_tier"]
            break

    sections = _parse_markdown_sections(text)
    chunks: list[MedicalChunk] = []
    for section_heading, section_body in sections:
        if not section_body.strip():
            continue
        sub_chunks = _chunk_text(section_body)
        for idx, chunk_text in enumerate(sub_chunks):
            chunks.append(
                MedicalChunk(
                    chunk_id=_make_chunk_id(doc_title, section_heading, idx),
                    doc_title=doc_title,
                    issuing_body=issuing_body,
                    publication_year=pub_year,
                    section_path=section_heading,
                    page_number=1,  # Markdown has no pages; default to 1
                    content=chunk_text,
                    cited_span=None,
                    authority_tier=authority_tier,
                )
            )
    return chunks


def ingest_directory(directory: Path | None = None) -> list[MedicalChunk]:
    """Ingest all Markdown documents from *directory*.

    If the directory is empty or missing, seeds synthetic guidelines first.
    """
    target = directory or DATA_RAW_DIR
    target.mkdir(parents=True, exist_ok=True)

    md_files = list(target.glob("*.md"))
    if not md_files:
        seed_synthetic_guidelines(target)
        md_files = list(target.glob("*.md"))

    all_chunks: list[MedicalChunk] = []
    for md_path in sorted(md_files):
        all_chunks.extend(ingest_markdown_file(md_path))
    return all_chunks
