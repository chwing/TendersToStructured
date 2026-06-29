#!/usr/bin/env python3
"""
Translate tender documents (FR/AR) to English using Helsinki-NLP MarianMT models.

Models used:
  French  → English : Helsinki-NLP/opus-mt-fr-en
  Arabic  → English : Helsinki-NLP/opus-mt-ar-en

Reads documents from input_dir, detects language, translates paragraph by
paragraph (preserving structure), and saves translated .txt to output_dir.

Usage:
    python run_translate.py tender_docs
    python run_translate.py tender_docs --force          # re-translate existing
"""
import argparse
import pathlib
import time

from src.staged_pipeline.ingestor import ingest
from src.extractor.document_reader import _detect_language

# MarianMT max safe input tokens; we stay well under the 512-token hard limit
_MAX_TOKENS = 400

LANG_TO_MODEL = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
}


def _load_translator(lang: str):
    """Load MarianMT model + tokenizer for the given language code."""
    model_name = LANG_TO_MODEL.get(lang)
    if model_name is None:
        raise ValueError(f"No MarianMT model configured for language '{lang}'. "
                         f"Supported: {list(LANG_TO_MODEL)}")
    from transformers import MarianMTModel, MarianTokenizer
    print(f"  [translate] Loading model '{model_name}' ...", flush=True)
    t0 = time.time()
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print(f"  [translate] Model ready ({time.time()-t0:.1f}s)", flush=True)
    return tokenizer, model


def _batch_paragraphs(paragraphs: list[str], tokenizer, max_tokens: int) -> list[list[str]]:
    """
    Group paragraphs into batches that each fit within max_tokens.
    Each paragraph that is individually too long is split at sentence boundaries.
    """
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        tok_len = len(tokenizer.encode(para, add_special_tokens=False))

        if tok_len > max_tokens:
            # Flush current batch first
            if current_batch:
                batches.append(current_batch)
                current_batch, current_len = [], 0
            # Split long paragraph at sentence boundaries
            sentences = _split_sentences(para)
            sub_batch: list[str] = []
            sub_len = 0
            for sent in sentences:
                s_len = len(tokenizer.encode(sent, add_special_tokens=False))
                if sub_len + s_len > max_tokens and sub_batch:
                    batches.append(sub_batch)
                    sub_batch, sub_len = [], 0
                sub_batch.append(sent)
                sub_len += s_len
            if sub_batch:
                batches.append(sub_batch)
        elif current_len + tok_len > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch, current_len = [para], tok_len
        else:
            current_batch.append(para)
            current_len += tok_len

    if current_batch:
        batches.append(current_batch)

    return batches


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on '.', '!', '?' followed by space or newline."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def translate_text(text: str, tokenizer, model) -> str:
    """Translate full document text, preserving paragraph structure."""
    paragraphs = text.split("\n\n")
    batches = _batch_paragraphs(paragraphs, tokenizer, _MAX_TOKENS)

    translated_paras: list[str] = []
    total = len(batches)

    for i, batch in enumerate(batches, 1):
        print(f"\r  translating batch {i}/{total} ...", end="", flush=True)
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs = model.generate(**inputs, num_beams=4)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_paras.append(" ".join(decoded))

    print()
    return "\n\n".join(translated_paras)


def main():
    parser = argparse.ArgumentParser(
        description="Translate tender documents to English via Helsinki-NLP MarianMT"
    )
    parser.add_argument("input_dir", help="Directory containing tender documents")
    parser.add_argument("--output-dir", default="translated_docs")
    parser.add_argument("--force", action="store_true",
                        help="Re-translate already translated files")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_dir)
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    files = [
        f for f in sorted(input_path.iterdir())
        if f.suffix.lower() in (".pdf", ".docx", ".doc")
    ]

    if not files:
        print("No PDF/DOCX files found.")
        return

    # Cache loaded models to avoid reloading for multiple docs in same language
    model_cache: dict[str, tuple] = {}

    for f in files:
        out_file = output_path / (f.stem + ".txt")
        if out_file.exists() and not args.force:
            print(f"[skip] {f.name} (already translated — use --force to redo)")
            continue

        print(f"\n[translate] {f.name}")
        t0 = time.time()
        try:
            doc = ingest(str(f))
            text = doc.text.strip()
            if not text:
                print("  WARNING: empty text, skipping.")
                continue

            lang = _detect_language(text)
            print(f"  detected language : {lang}")

            if lang == "en":
                print("  already English — copying as-is.")
                out_file.write_text(text, encoding="utf-8")
                continue

            if lang not in LANG_TO_MODEL:
                print(f"  WARNING: no model for '{lang}', skipping. "
                      f"Add it to LANG_TO_MODEL in run_translate.py.")
                continue

            if lang not in model_cache:
                model_cache[lang] = _load_translator(lang)
            tokenizer, model = model_cache[lang]

            translated = translate_text(text, tokenizer, model)
            out_file.write_text(translated, encoding="utf-8")
            print(f"  saved -> {out_file}  ({len(translated)} chars, {time.time()-t0:.1f}s)")

        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nAll done.")
    print(f"\nNext steps:")
    print(f"  python run_llm_extractor.py {args.output_dir} --model qwen2.5:7b --timeout 300 --num-ctx 8192")
    print(f"  python run_staged_extractor.py {args.output_dir} --model qwen2.5:7b --timeout 300 --num-ctx 8192 --no-embeddings --no-gliner")
    print(f"  python run_comparison.py --semantic-backend ollama")


if __name__ == "__main__":
    main()
