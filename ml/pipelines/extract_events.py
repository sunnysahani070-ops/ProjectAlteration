"""Stage 5 entry point: extract structured drilling events from Stage 4 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from ml.src.information_extraction import extract_events
except ModuleNotFoundError:
    from src.information_extraction import extract_events


def process_processed_document(document_dir: str | Path) -> Path:
    """Read Stage 4 JSON files and write extracted events.json beside them."""
    document_dir = Path(document_dir)
    text_path = document_dir / "text.json"
    tables_path = document_dir / "tables.json"
    metadata_path = document_dir / "metadata.json"

    for required_path in (text_path, tables_path, metadata_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required Stage 4 file not found: {required_path}")

    text_data = json.loads(text_path.read_text(encoding="utf-8"))
    tables_data = json.loads(tables_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    events = extract_events(
        text=text_data.get("cleaned_text", ""),
        tables=tables_data.get("tables", []),
        source_document=text_data.get("file_name", document_dir.name),
        page_count=metadata.get("page_count", 1),
        page_texts=text_data.get("page_texts") or None,
    )

    output_path = document_dir / "events.json"
    output_path.write_text(json.dumps({"events": events}, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured events from a processed document folder.")
    parser.add_argument("document_dir", help="Path to a Stage 4 processed document folder")
    args = parser.parse_args()

    output_path = process_processed_document(args.document_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print(f"Extracted {len(payload['events'])} event(s)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
