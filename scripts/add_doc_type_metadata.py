"""
Adds doc_type and company_name metadata to document_chunks.

Run AFTER restoring the database dump (pg_restore).
Sets metadata_json fields based on file_name patterns,
used by the hybrid search metadata filter.

Usage:
    cd scripts && python add_doc_type_metadata.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skillab-py" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from sqlalchemy import text
from database import transaction

# file_name prefix -> doc_type
DOC_TYPE_RULES = {
    "factura_": "factura",
    "client_": "client",
    "contract_": "contract",
    "raport_": "raport",
}

# file_name pattern (LIKE) -> company_name (lowercase)
COMPANY_RULES = {
    "%techsoft%": "techsoft",
    "%datapro%": "datapro",
    "%cloudnet%": "cloudnet",
    "%secureit%": "secureit",
    "%webdev%": "webdev",
}


def main():
    with transaction() as session:
        print("Setting doc_type...")
        for prefix, doc_type in DOC_TYPE_RULES.items():
            result = session.execute(
                text("""
                    UPDATE document_chunks
                    SET metadata_json = (COALESCE(metadata_json::jsonb, '{}'::jsonb) || :meta ::jsonb)::text
                    WHERE file_name LIKE :pattern
                """),
                {"pattern": f"{prefix}%", "meta": f'{{"doc_type": "{doc_type}"}}'},
            )
            print(f"  {doc_type}: {result.rowcount} chunks")

        print("Setting company_name...")
        for pattern, company in COMPANY_RULES.items():
            result = session.execute(
                text("""
                    UPDATE document_chunks
                    SET metadata_json = (COALESCE(metadata_json::jsonb, '{}'::jsonb) || :meta ::jsonb)::text
                    WHERE file_name LIKE :pattern
                """),
                {"pattern": pattern, "meta": f'{{"company_name": "{company}"}}'},
            )
            print(f"  {company}: {result.rowcount} chunks")

    print("Done!")


if __name__ == "__main__":
    main()
