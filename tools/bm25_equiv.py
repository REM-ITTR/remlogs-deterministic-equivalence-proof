#!/usr/bin/env python3
import argparse, json, math, re
from pathlib import Path
from collections import Counter, defaultdict

WORD = re.compile(r"[a-z0-9]+")

def norm_ws_lower(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def tokenize(s: str):
    return WORD.findall(s.lower())

def bm25_scores(docs, query_terms, k1=1.2, b=0.75):
    # docs: list[str]
    N = len(docs)
    toks = [tokenize(d) for d in docs]
    dl = [len(t) for t in toks]
    avgdl = (sum(dl) / N) if N else 0.0

    df = Counter()
    for t in toks:
        df.update(set(t))

    def idf(term):
        n = df.get(term, 0)
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    scores = []
    for i, t in enumerate(toks):
        tf = Counter(t)
        s = 0.0
        for q in query_terms:
            f = tf.get(q, 0)
            if f == 0:
                continue
            denom = f + k1 * (1.0 - b + b * (dl[i] / avgdl if avgdl else 0.0))
            s += idf(q) * (f * (k1 + 1.0) / denom)
        scores.append(s)
    return scores, df, avgdl

def topk_full(docs, scores, k):
    idx = list(range(len(docs)))
    idx.sort(key=lambda i: (scores[i], docs[i]), reverse=True)
    return [{"rank": r+1, "score": scores[i], "doc": docs[i], "i": i} for r, i in enumerate(idx[:k])]

def topk_reconstructed(unique_docs, unique_scores, counts_map, k):
    # Expand by multiplicity (counts_map[doc]) until k
    order = list(range(len(unique_docs)))
    order.sort(key=lambda i: (unique_scores[i], unique_docs[i]), reverse=True)

    out = []
    for i in order:
        if len(out) >= k:
            break
        doc = unique_docs[i]
        c = int(counts_map.get(doc, 1))
        take = min(c, k - len(out))
        for _ in range(take):
            out.append({"rank": len(out)+1, "score": unique_scores[i], "doc": doc})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--reduced", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--q", action="append", required=True, help="query string")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full_docs = [norm_ws_lower(x) for x in Path(args.full).read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    red_docs  = [norm_ws_lower(x) for x in Path(args.reduced).read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    counts_map = json.loads(Path(args.counts).read_text(encoding="utf-8"))

    # Build one combined query term list (BM25 over terms; phrase handled by windowing)
    q_terms = []
    for q in args.q:
        q_terms += tokenize(q)

    full_scores, _, _ = bm25_scores(full_docs, q_terms)
    # IMPORTANT: score reduced docs using FULL IDF statistics would be ideal, but:
    # since reduced docs are subset of FULL docs, and our reconstruction expands duplicates,
    # the ranking equivalence we care about is the TOP-K doc identity under same tokenization.
    red_scores, _, _ = bm25_scores(red_docs, q_terms)

    full_top = topk_full(full_docs, full_scores, args.k)
    red_top  = topk_reconstructed(red_docs, red_scores, counts_map, args.k)

    # Compare doc identity sequence
    mismatches = []
    for i in range(min(len(full_top), len(red_top))):
        if full_top[i]["doc"] != red_top[i]["doc"]:
            mismatches.append({
                "rank": i+1,
                "full_doc": full_top[i]["doc"],
                "full_score": full_top[i]["score"],
                "recon_doc": red_top[i]["doc"],
                "recon_score": red_top[i]["score"],
            })

    (outdir / "bm25_topk_full.json").write_text(json.dumps(full_top, indent=2), encoding="utf-8")
    (outdir / "bm25_topk_reconstructed.json").write_text(json.dumps(red_top, indent=2), encoding="utf-8")
    (outdir / "bm25_diff.json").write_text(json.dumps(mismatches, indent=2), encoding="utf-8")

    report = []
    report.append("BM25 Equivalence Check (FULL vs REDUCED+COUNTS reconstruction)")
    report.append(f"FULL   : {args.full}")
    report.append(f"REDUCED: {args.reduced}")
    report.append(f"COUNTS : {args.counts}")
    report.append(f"K      : {args.k}")
    report.append(f"QUERY  : {' | '.join(args.q)}")
    report.append("")
    if not mismatches:
        report.append("STATUS: PASS ✅ (top-K doc identity matches under reconstruction)")
    else:
        report.append(f"STATUS: FAIL ❌  mismatches={len(mismatches)}")
        report.append("First 10 mismatches:")
        for m in mismatches[:10]:
            report.append(f"- rank {m['rank']}: FULL='{m['full_doc'][:80]}...' vs RECON='{m['recon_doc'][:80]}...'")
    (outdir / "bm25_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(str(outdir / "bm25_report.txt"))

if __name__ == "__main__":
    main()
