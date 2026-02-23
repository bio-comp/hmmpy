rabiner1_machine_readable package
================================

Files:
- rabiner1_machine_readable.json  (full structured extraction: pages, blocks, bboxes, figure index)
- rabiner1_machine_readable.md    (LLM/human-friendly markdown export)
- rabiner1_pages.jsonl            (one page per line, compact)
- rabiner1_blocks.jsonl           (one block per line with bbox + text)

Notes:
- OCR/symbol artifacts may remain from the source PDF.
- Two page text variants are included in JSON/JSONL: raw order and heuristic column order.
