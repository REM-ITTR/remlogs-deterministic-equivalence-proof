#!/usr/bin/env python3
import argparse, json, math, re
from pathlib import Path
from collections import Counter

WORD = re.compile(r"[a-z0-9]+")

def norm_ws_lower(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def tokenize(s: str):
    return WORD.findall(s.lower())

def build_full_stats(full_docs):
    toks = [tokenize(d) for d in full_docs]
    dl = [len(t) for t in toks]
    N = len(toks)
    avgdl = (sum(dl) / N) if N else 0.0
    df = Counter()
    for t in toks:
        df.update(set(t))
    return toks, dl, N, avgdl, df

def idf(term, N, df):
    n = df.get(term, 0)
    return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

def bm25_score_doc(doc_tokens, q_terms, N, avgdl, df, k1=1.2, b=0.75):
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    s = 0.0
    for q in q_terms:
        f = tf.get(q, 0)
        if f == 0:
            continue
        denom = f + k1 * (1.0 - b + b * (dl / avgdl if avgdl else 0.0))
        s += idf(q, N, df) * (f * (k1 + 1.0) / denom)
    return s

def topk_docs(docs, scores, k):
    idx = list(range(len(docs)))
    # stable tiebreaker: doc text
    idx.sort(key=lambda i: (scores[i], docs[i]), reverse=True)
    return [{"rank": r+1, "score": scores[i], "doc": docs[i], "i": i} for r, i in enumerate(idx[:k])]

def topk_reconstructed(unique_docs, unique_scores, counts_map, k):
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
    ap.add_argument("--q", action="append", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full_docs = [norm_ws_lower(x) for x in Path(args.full).read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    red_docs  = [norm_ws_lower(x) for x in Path(args.reduced).read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    counts_map = json.loads(Path(args.counts).read_text(encoding="utf-8"))

    q_terms = []
    for q in args.q:
        q_terms += tokenize(q)

    full_toks, _, N, avgdl, df = build_full_stats(full_docs)

    full_scores = [bm25_score_doc(full_toks[i], q_terms, N, avgdl, df) for i in range(len(full_docs))]
    red_scores  = [bm25_score_doc(tokenize(d), q_terms, N, avgdl, df) for d in red_docs]

    full_top = topk_docs(full_docs, full_scores, args.k)
    red_top  = topk_reconstructed(red_docs, red_scores, counts_map, args.k)

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
    report.append("BM25 Strict Equivalence Check (FULL-IDF locked; FULL avgdl locked)")
    report.append(f"FULL   : {args.full}")
    report.append(f"REDUCED: {args.reduced}")
    report.append(f"COUNTS : {args.counts}")
    report.append(f"K      : {args.k}")
    report.append(f"QUERY  : {' | '.join(args.q)}")
    report.append(f"N(full) : {N}  avgdl(full): {avgdl:.6f}")
    report.append("")
    if not mismatches:
        report.append("STATUS: PASS ✅ (top-K doc identity matches under strict reconstruction)")
    else:
        report.append(f"STATUS: FAIL ❌  mismatches={len(mismatches)}")
        report.append("First 10 mismatches:")
        for m in mismatches[:10]:
            report.append(f"- rank {m['rank']}: FULL='{m['full_doc'][:80]}...' vs RECON='{m['recon_doc'][:80]}...'")
    (outdir / "bm25_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(str(outdir / "bm25_report.txt"))

if __name__ == "__main__":
    main()
