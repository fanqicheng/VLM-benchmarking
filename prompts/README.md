# PathLang prompt corpus

Complete, pathologist-validated text inputs for all five datasets
(CAMELYON16, CAMELYON17, PANDA, TCGA-GBMLGG, UNITOPATHO). These are the exact
strings encoded by `prompt_encode/generate_text_embedding_*.py`; the same text
is used for all nine models.

| File | Task | Structure |
|---|---|---|
| `t1_zeroshot.json` | T1 zero-shot classification | `dataset → class → canonical prompt` |
| `t2_cross_modal_retrieval.json` | T2 cross-modal retrieval | `dataset → class → {positives: [4], hard_negatives: [64]}` |
| `t3_variant1_paraphrase.json` | T3 V1 semantic-equivalence paraphrases | `dataset → class → [5 paraphrases]` |
| `t3_variant2_length_style.json` | T3 V2 length / reporting-style variants | `dataset → class → {short, medium, long, clinical}` |
| `t4_open_vocab_retrieval.json` | T4 open-vocabulary diagnosis retrieval | `set → dataset → class → {ground_truth, candidates}` |

## Notes

- **Class keys.** PANDA classes `0`–`5` are ISUP grades (0 = benign).
  UNITOPATHO: `NORM`, `HP`, `TA.LG`, `TA.HG`, `TVA.LG`, `TVA.HG`.
  CAMELYON17: `NEGATIVE`, `ITC`, `MICRO`, `MACRO`.
- **TCGA-GBMLGG labels** are derived from IDH status in the clinical table
  (`idh mutation` = 0 → `GBM`, 1 → `LGG`), consistent with the WHO CNS5
  requirement that glioblastoma be IDH-wildtype.
- **T3 pool.** For each class, the 9-prompt pool used for ensembling is
  `variant1` (5) followed by `variant2` short, medium, long, clinical (4).
  The `clinical` variant corresponds to the T1 prompt; it is identical for
  CAMELYON17 and TCGA-GBMLGG, and differs in minor wording for some classes of
  PANDA, UNITOPATHO and CAMELYON16.
- **Key names.** The T3 files use the dataset keys `CAM16` / `CAM17` and
  descriptive class names (e.g., `Tubular Adenoma, Low-Grade Dysplasia (TA.LG)`);
  classes appear in the same order as in `t1_zeroshot.json`.
- All strings, including typos, are kept exactly as encoded.
- **T4 candidate lists are per class.** Each list is the exact candidate set
  ranked for slides of that class. Set 1 and Set 4 lists have 20 entries,
  Set 2 and Set 3 lists have 10. Set 4 is built from the 19 labels pooled across
  the five datasets; small per-class differences (e.g., a repeated entry or a
  class-specific substitute) are kept exactly as evaluated.
