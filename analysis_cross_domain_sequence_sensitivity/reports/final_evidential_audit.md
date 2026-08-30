# Final evidential audit for submission

## Audit conclusion

The requested evidence is complete. All 648 planned 2019 subgroup pairwise comparisons have finite gamma differences, with exactly 162 comparisons for each of the four outcomes. All 21 analyzed domain-year datasets have reconciled analytic and sequence counts. Score direction and sequence coding are validated for every included dataset. No validation failure was found, so no new analysis was added.

There is no unresolved coding ambiguity among included datasets. Four limitations or resolved provenance issues should remain visible in the submission record: Gender-Science 2019 lacks a supplied participant-score/Order file for an independent record-level check; President 2019 required participant-level `Order` to resolve multiple candidate-pair variants; the Age 2019 codebook prints identical text for the two `Order` values but the mapping is established by a 100% empirical cross-tab; and six early 2019 orientation CSVs omit the later-added `orientation_method` field even though their recorded correlations and the shared implementation establish the method. Religion remains excluded because its available materials do not permit orientation without inference.

## 1. Magnitude of subgroup heterogeneity

The audit quantity is `abs(gamma_difference)` among the 648 rows with `row_type = pairwise_test` in `cross_domain_subgroup_gamma_tests.csv`. Percentiles use the standard linear-interpolation empirical quantile. Values below are descriptive summaries of the already-completed comparisons, not additional inferential tests.

| Scope | Pairwise comparisons | Median | 90th percentile | 95th percentile | Maximum absolute Δγ |
|---|---:|---:|---:|---:|---:|
| Overall | 648 | 0.005633 | 0.028958 | 0.042883 | 0.078795 |
| Reconstructed public D | 162 | 0.011993 | 0.042178 | 0.049785 | 0.078431 |
| Modified D | 162 | 0.011650 | 0.042976 | 0.049107 | 0.078795 |
| Log-latency contrast | 162 | 0.003613 | 0.016710 | 0.020744 | 0.026630 |
| Error-rate contrast | 162 | 0.002607 | 0.012903 | 0.014872 | 0.018757 |

The maximum is in the **Weapons** domain, grouping variable **Approximate age**, groups **18–29 versus 60+**, outcome **Modified D**: gamma is 0.061014 for ages 18–29 and -0.017781 for ages 60+, so Δγ = 0.078795 (absolute Δγ = 0.078795; bootstrap interval 0.063787 to 0.093767).

This is a subgroup maximum and should not be confused with the previously reported maximum 2019 *between-domain* gamma difference of 0.147178 (Disability versus Skin Tone, Reconstructed public D), which answers a different question.

## 2. Domain-year orientation validation

`Analytic N` and the two sequence counts below use the base analytic frame represented by Modified D, log-latency contrast, and error-rate contrast. Reconstructed public D applies the additional documented fast-response screen and therefore has a smaller outcome-specific N in some datasets. `CF` means congruent-first and `ICF` incongruent-first.

| Domain | Year | Analytic N | CF | ICF | Score-direction validation | Sequence-coding validation | Status |
|---|---:|---:|---:|---:|---|---|---|
| Age | 2019 | 147,296 | 74,347 | 72,949 | Reconstructed vs supplied D: screened r=1.00000, RMSE=0.00004 (n=133,267) | Semantic critical-block pairing; Order agrees 146,599/146,599; 686 blank-Order records have unambiguous blocks | Resolved codebook-label issue |
| Age | 2021 | 320,403 | 161,223 | 159,180 | Archive-wide reconstructed vs supplied D, min abs(r)=0.999982 | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Arab-Muslim | 2019 | 57,463 | 28,743 | 28,720 | Archive-wide reconstructed vs supplied D, abs(r)=1.000000 | Four-block consistency; independent Order agreement=1.000000 | Method field omitted in early CSV; method recoverable |
| Arab-Muslim | 2021 | 82,478 | 41,432 | 41,046 | Archive-wide reconstructed vs supplied D, min abs(r)=1.000000 | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Asian American | 2019 | 70,496 | 35,239 | 35,257 | Archive-wide reconstructed vs supplied D, abs(r)=0.999996; sign/sequence flipped | Four-block consistency; independent Order agreement=1.000000 | Method field omitted in early CSV; method recoverable |
| Asian American | 2021 | 127,257 | 63,677 | 63,580 | Archive-wide reconstructed vs supplied D, min abs(r)=1.000000; both task variants flipped | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Disability | 2019 | 100,650 | 50,549 | 50,101 | Archive-wide reconstructed vs supplied D, abs(r)=1.000000; sign/sequence flipped | Four-block consistency; independent Order agreement=1.000000 | Method field omitted in early CSV; method recoverable |
| Disability | 2021 | 170,166 | 85,778 | 84,388 | Archive-wide reconstructed vs supplied D, min abs(r)=1.000000; both task variants flipped | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Gender-Career | 2019 | 251,946 | 126,535 | 125,411 | Archive-wide reconstructed vs supplied D, abs(r)=1.000000; sign/sequence flipped | Four-block consistency; independent Order agreement=1.000000 | Method field omitted in early CSV; method recoverable |
| Gender-Career | 2021 | 355,864 | 178,468 | 177,396 | Archive-wide reconstructed vs supplied D, min abs(r)=1.000000; both task variants flipped | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Gender-Science | 2019 | 133,460 | 67,274 | 66,186 | Explicit raw pairing semantics plus all nine N/beta/gamma reference benchmarks | Block-3 semantic pairing; both sequence-count benchmarks reproduced exactly | Independent record-level score/Order file unavailable |
| Gender-Science | 2021 | 185,266 | 92,951 | 92,315 | Archive-wide reconstructed vs supplied D, min abs(r)=1.000000 | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Native American | 2019 | 38,403 | 19,367 | 19,036 | Archive-wide reconstructed vs supplied D, abs(r)=1.000000; sign/sequence flipped | Four-block consistency; independent Order agreement=1.000000 | Method field omitted in early CSV; method recoverable |
| President | 2019 | 102,739 | 51,284 | 51,455 | Participant Order resolves session direction; supplied-D abs(r)=1.000000 validates it | Four-block consistency plus participant Order for multiple candidate-pair variants | Resolved raw-pairing ambiguity |
| Sexuality | 2019 | 195,486 | 98,548 | 96,938 | Archive-wide reconstructed vs supplied D, abs(r)=1.000000 | Four-block consistency; independent Order agreement=1.000000 | Method field omitted in early CSV; method recoverable |
| Skin Tone | 2019 | 185,625 | 93,331 | 92,294 | Archive-wide reconstructed vs supplied D, abs(r)=1.000000 | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Skin Tone | 2021 | 302,885 | 151,942 | 150,943 | Archive-wide reconstructed vs supplied D, min abs(r)=1.000000 | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Weapons | 2019 | 124,344 | 62,398 | 61,946 | Archive-wide reconstructed vs supplied D, abs(r)=1.000000; sign/sequence flipped | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Weapons | 2021 | 185,433 | 92,884 | 92,549 | Archive-wide reconstructed vs supplied D, min abs(r)=1.000000; both task variants flipped | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Weight | 2019 | 213,972 | 108,199 | 105,773 | Archive-wide reconstructed vs supplied D, abs(r)=1.000000 | Four-block consistency; independent Order agreement=1.000000 | Clear |
| Weight | 2021 | 333,330 | 167,861 | 165,469 | Archive-wide reconstructed vs supplied D, min abs(r)=1.000000 | Four-block consistency; independent Order agreement=1.000000 | Clear |

For non-reference datasets, the four-block rule requires blocks 3 and 4 to carry the same pairing, blocks 6 and 7 to carry the other pairing, and the two pairs to differ. A provisional score/sequence orientation is then checked against the supplied participant-level public D; a negative correlation triggers simultaneous score and sequence reversal. Except for President, participant `Order` is an independent cross-check and agrees in every usable case. For President, `Order` is the resolver and supplied D is the independent validation.

## 3. Unanalyzed inventory entries

These entries have no analytic or sequence counts and therefore cannot receive an orientation-validation method. They are not silently omitted from the evidential record.

| Domain | Year(s) | Analytic N | Orientation status / reason |
|---|---|---:|---|
| Asian Attitude | 2023–2025 | 0 | No 2019 or 2021 archive; collection begins in 2023 |
| Hispanic | 2023–2025 | 0 | No 2019 or 2021 archive; collection begins in 2023 |
| Jewish | 2023–2025 | 0 | No 2019 or 2021 archive; collection begins in 2023 |
| Multiple Skin Tone | 2023–2025 | 0 | No 2019 or 2021 archive; collection begins in 2023 |
| Native American | 2021 | 0 | Raw or participant CSV archive unavailable |
| President | 2021 | 0 | Raw or participant CSV archive unavailable |
| Race | 2019; 2021 | 0 | Participant archive exists but no raw trial-level component is exposed |
| Religion | 2019; 2021 | 0 | Raw trials exist, but no participant dataset/codebook permits congruent/incongruent orientation without inference; unresolved and excluded |
| Sexuality | 2021 | 0 | Raw or participant CSV archive unavailable |
| Transgender | 2020–2025 | 0 | No 2019 archive; collection begins in 2020 and was not added after fixing the common 2021 panel |

## Source ledger

- Pairwise gamma evidence: `results/cross_domain_subgroup_gamma_tests.csv`
- Domain-year analytic and sequence counts: `results/cross_domain_sequence_estimates.csv`, base-outcome rows
- Dataset inclusion, orientation correlation, and Order agreement: `results/dataset_inventory.csv`
- Task-level orientation evidence: `results/orientation_validation_<slug>_<year>.csv`
- Validated 2019 reference checks: `results/validation_status.csv` and `analysis_group_comparison_strengthening/reports/group_comparison_strengthening_audit.md`
- Machine-readable audit tables: `results/final_subgroup_heterogeneity_magnitude.csv` and `results/final_domain_year_orientation_validation.csv`
