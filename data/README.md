# Documents

Drop any PDF or .txt files in this folder, then run `python ingest.py` from the project root to index them.

## Getting the Premier League Handbook

The official Premier League Handbook is published publicly each season at <https://www.premierleague.com>. Search the site for "Handbook", download the season's PDF, and save it into this folder.

## Sample document

`sample_premier_league_info.txt` is a small, general-knowledge text file included so the project runs out of the box without needing to download anything. Replace it with the real Handbook (or any other documents) for a more substantive demo.

## Notes

- Scanned PDFs without a text layer will not work — they need OCR first (e.g. with `tesseract` or Adobe).
- The `cache/` subfolder is created automatically by `ingest.py` and is ignored by git.
