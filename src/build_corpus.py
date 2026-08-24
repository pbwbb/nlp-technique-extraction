#!/usr/bin/env python3
"""
build_corpus.py — Build a plain-text corpus of The Hacker News articles
under the "Cyber Attack" label, for NER annotation (Doccano) + CRF training.

Strategy
--------
1. Enumerate article URLs through the Blogger label feed (reliable pagination).
2. Fetch each article page and extract ONLY the article body text from
   #articlebody, dropping ads, image separators, video embeds and the
   "Found this article interesting?" footer.
3. Write:
     - corpus.jsonl   -> one {"text","title","url","date"} per line (Doccano-ready)
     - txt/<slug>.txt  -> one plain-text file per article (optional, for inspection)

Usage
-----
    python build_corpus.py --target 150 --outdir corpus_out
    python build_corpus.py --target 120 --no-txt --delay 1.5

Notes
-----
* The feed only supplies URLs, so feed truncation doesn't matter.
* Polite by default: identifiable User-Agent + delay between requests.
* Resumable: URLs already present in corpus.jsonl are skipped on re-run.
"""

import argparse
import json
import os
import re
import sys
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

BASE = "https://thehackernews.com"
LABEL = "Cyber Attack"
FEED_TMPL = BASE + "/feeds/posts/default/-/{label}?alt=json&start-index={start}&max-results={n}"

HEADERS = {
    # Identify the crawler honestly; be a good citizen.
    "User-Agent": "THN-corpus-builder/1.0 (academic NER research; contact: you@example.com)"
}

# Block-level tags whose text we keep, in document order.
KEEP_TAGS = ["p", "h2", "h3", "h4", "li", "blockquote", "pre"]

# Containers to delete before extraction (ads, images, embeds, footer).
DROP_SELECTORS = [
    ".dog_two",              # sponsored banner ads
    ".separator",            # image blocks
    ".tr-caption-container", # captioned images
    ".video-container",      # youtube embeds
    ".note-b",               # "Found this article interesting?" footer
    ".check_two_webinar",    # inline webinar promo boxes
    ".article-board",        # promo callout boxes
    "script",
    "style",
]


def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def enumerate_urls(session, target, feed_page=25, max_pages=40, delay=1.0):
    """Return a de-duplicated list of article URLs from the label feed."""
    urls = []
    seen = set()
    start = 1
    label_q = quote(LABEL)
    for _ in range(max_pages):
        url = FEED_TMPL.format(label=label_q, start=start, n=feed_page)
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[feed] stop: {e}", file=sys.stderr)
            break

        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            break

        for entry in entries:
            link = next(
                (l["href"] for l in entry.get("link", []) if l.get("rel") == "alternate"),
                None,
            )
            if link and link not in seen:
                seen.add(link)
                urls.append(link)

        print(f"[feed] start-index={start}: +{len(entries)} entries "
              f"({len(urls)} unique so far)")
        if len(urls) >= target:
            break
        start += feed_page
        time.sleep(delay)

    return urls[:target]


def clean_text(s):
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    # Inline tags (<strong>, <a>) can leave a space before punctuation.
    s = re.sub(r"\s+([,.;:!?%)])", r"\1", s)
    s = re.sub(r"([(])\s+", r"\1", s)
    return s.strip()


def slugify(url):
    m = re.search(r"/([^/]+)\.html$", url)
    base = m.group(1) if m else re.sub(r"\W+", "-", url)[-60:]
    return re.sub(r"[^a-z0-9\-]", "", base.lower())[:80] or "article"


def extract_article(html):
    """Return dict(title, date, text, paragraphs) or None if no body found."""
    soup = BeautifulSoup(html, "html.parser")

    body = soup.find(id="articlebody")
    if body is None:
        return None

    # Title
    title_el = soup.select_one("h1.story-title")
    title = clean_text(title_el.get_text(" ")) if title_el else ""

    # Publish date (metadata only)
    date_el = soup.find(attrs={"itemprop": "datePublished"})
    date = date_el.get("content", "") if date_el else ""

    # Everything from .stophere onward is footer/related — drop it.
    stop = body.find(class_="stophere")
    if stop is not None:
        for sib in list(stop.find_all_next()):
            sib.decompose()
        stop.decompose()

    # Remove ad / image / embed / promo containers.
    for sel in DROP_SELECTORS:
        for el in body.select(sel):
            el.decompose()

    # Collect block text in document order.
    paragraphs = []
    for el in body.find_all(KEEP_TAGS):
        txt = clean_text(el.get_text(" "))
        # Skip empties and stray one-word promo leftovers.
        if txt and len(txt) > 1:
            paragraphs.append(txt)

    text = "\n\n".join(paragraphs)
    if not text:
        return None

    return {"title": title, "date": date, "text": text, "paragraphs": paragraphs}


def load_done(jsonl_path):
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["url"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser(description="Build THN Cyber Attack NER corpus.")
    ap.add_argument("--target", type=int, default=150,
                    help="number of articles to collect (default 150)")
    ap.add_argument("--outdir", default="corpus_out", help="output directory")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between article requests (be polite)")
    ap.add_argument("--min-chars", type=int, default=400,
                    help="skip articles shorter than this many chars")
    ap.add_argument("--no-txt", action="store_true",
                    help="do not write per-article .txt files")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    txt_dir = os.path.join(args.outdir, "txt")
    if not args.no_txt:
        os.makedirs(txt_dir, exist_ok=True)
    jsonl_path = os.path.join(args.outdir, "corpus.jsonl")

    session = get_session()
    done = load_done(jsonl_path)
    if done:
        print(f"[resume] {len(done)} articles already in {jsonl_path}")

    # Over-fetch URLs a bit to absorb failures/short articles.
    urls = enumerate_urls(session, target=args.target + len(done) + 30)
    print(f"[feed] {len(urls)} candidate URLs collected")

    kept = 0
    with open(jsonl_path, "a", encoding="utf-8") as out:
        for i, url in enumerate(urls, 1):
            if kept >= args.target:
                break
            if url in done:
                continue
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
                art = extract_article(r.text)
            except Exception as e:
                print(f"[{i}] ERROR {url} -> {e}", file=sys.stderr)
                time.sleep(args.delay)
                continue

            if not art or len(art["text"]) < args.min_chars:
                print(f"[{i}] skip (too short/empty): {url}")
                time.sleep(args.delay)
                continue

            record = {
                "text": art["text"],
                "title": art["title"],
                "url": url,
                "date": art["date"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done.add(url)
            kept += 1

            if not args.no_txt:
                slug = slugify(url)
                with open(os.path.join(txt_dir, f"{slug}.txt"), "w",
                          encoding="utf-8") as tf:
                    tf.write(art["title"] + "\n\n" + art["text"])

            print(f"[{i}] kept ({kept}/{args.target}) "
                  f"{len(art['text'])} chars :: {art['title'][:70]}")
            time.sleep(args.delay)

    print(f"\nDone. {kept} new articles written to {jsonl_path}")
    print("Import into Doccano as JSONL (field: text). "
          "Each line is one document.")


if __name__ == "__main__":
    main()
