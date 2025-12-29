# REMLogs — Deterministic Reduction with Exact Search Equivalence

This repository contains a deterministic, reproducible proof that log search
(BM25) can be performed on a reduced corpus while producing identical results
to a full-corpus computation.

No embeddings.
No approximations.
No post-ranking filtering.

---

## Scope

This repository verifies **deterministic equivalence**, not performance claims.

It demonstrates that:
- A log corpus can be reduced deterministically
- BM25 statistics (IDF, avgdl, parameters) can be locked
- Top-K document identity and ordering are preserved exactly

---

## What Is Verified

- Deterministic corpus reduction prior to search
- Strict reconstruction of BM25 scoring inputs
- Exact equivalence between:
  - Full corpus BM25 results
  - Reduced corpus BM25 results

---

## Proof Result

STATUS: PASS

Top-K document identity and ordering match exactly under strict reconstruction.

---

## Repository Structure

inputs/
- Hash manifests or input corpus identifiers
- Query definitions
- Locked BM25 statistics

outputs/
- Full-corpus ranking output
- Reduced-corpus ranking output
- Strict equivalence report

tools/
- Deterministic reducer
- BM25 reconstruction and verifier

---

## Reproduce

1) Load the input corpus and query definition  
2) Apply deterministic reduction  
3) Lock BM25 statistics (IDF, avgdl, parameters)  
4) Compute rankings on full and reduced corpora  
5) Verify top-K identity equivalence  

All steps are deterministic and auditable.

---

## Interpretation

This proof does not claim improved relevance or quality.

It demonstrates that:
Identical search results can be obtained from a smaller search space,
provably and reproducibly.

## Datasets

The deterministic reduction and strict BM25 equivalence verification
were executed across multiple heterogeneous text corpora, including:

- The Enron Email Corpus
- SEC / EDGAR filing text from multiple issuers
- Additional long-form document corpora (four total corpora evaluated)

These datasets were selected to cover materially different
document structures, vocabulary distributions, and redundancy profiles.

Equivalence results were consistent across all evaluated corpora.
