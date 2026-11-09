# Submissions

Every strategy our team submitted to the arena, by round.
Files are in `strategies/roundN/` and run the same code the arena ran.

- **train**: local backtest of the file on the round's public tape (`data/roundN_train.csv`)
- **validation**: the arena's PUBLIC evaluation (`data/roundN_val.csv`)
- **test**: local backtest on the published private tape (`data/roundN_test.csv`).
  Only the final submission was officially scored on it

Order: the final submission first (bold), then by validation PnL.

## Round 1

In round 1 the arena's public evaluation ran on the training tape itself.

| file | train | validation | test |
|---|---:|---:|---:|
| **round1/adaptive_mm_plus.py** | 678,057 | - | **667,357** |
| round1/adaptive_mm.py | 667,889 | - | 659,691 |
| round1/adaptive_mm_1609.py | 662,359 | - | 658,409 |
| round1/adaptive_mm_1407.py | 650,611 | - | 644,824 |
| round1/imbalance_mm.py | 615,583 | - | 613,201 |

## Round 2

| file | train | validation | test |
|---|---:|---:|---:|
| **round2/hmm_mm_v2.py** | 515,547 | 1,291,224 | **1,195,019** |
| round2/hmm_mm_v2_twin.py | 432,669 | 1,293,574 | 1,128,562 |
| round2/hmm_mm_2010.py | 502,258 | 1,290,784 | 1,178,652 |
| round2/hmm_mm_xf.py | 526,287 | 1,288,980 | 1,185,099 |
| round2/hmm_mm_v2_xf.py | 532,026 | 1,286,575 | 1,199,008 |
| round2/hmm_mm.py | 526,924 | 1,286,221 | 1,207,837 |
| round2/lab_argon_predator.py | 513,518 | 1,081,343 | 1,027,204 |
| round2/regime_mm.py | 448,818 | 1,032,909 | 959,855 |
| round2/probe_depth_flow.py | −31,200 | 47,920 | −13,540 |
| round2/simple_probe_mm.py | −3,120 | 4,800 | −1,360 |
| round2/dumb.py | 1,190,923 | −49,500 | 13,496 |

## Round 3

| file | train | validation | test |
|---|---:|---:|---:|
| **round3/hmm_deep_clear_ks.py** | 713,244 | 1,249,097 | **1,134,420** |
| round3/hmm_imm_pon02.py | 710,629 | 1,225,970 | 1,124,203 |
| round3/hmm_imm_sv11.py | 721,624 | 1,223,266 | 1,118,795 |
| round3/hmm_deep_clear_r3.py | 693,522 | 1,203,891 | 1,154,199 |
| round3/r3_cand_nm6.py | 643,075 | 1,185,019 | 1,086,736 |
| round3/r3_deep_nm6.py | 648,566 | 1,162,363 | 1,093,968 |
| round3/hmm_617_fit.py | 625,338 | 1,091,395 | 1,045,777 |
| round3/hmm_617.py | 617,400 | 1,076,934 | 1,044,168 |
| round2/hmm_mm.py (resubmitted) | 568,382 | 1,028,431 | 961,407 |
| round2/probe_depth_flow.py (resubmitted) | −12,809 | 3,247 | 31,882 |
| round2/simple_probe_mm.py (resubmitted) | −1,289 | 328 | 3,200 |
| round3/probe_short_spread.py | 1,260 | −370 | −3,230 |
| round3/flow_probe_impact.py | −5,956 | −14,890 | −9,855 |
| round3/flow_probe_passive.py | −38,174 | −49,413 | −44,184 |
| round3/flow_probe_roundtrip.py | −46,943 | −98,910 | −101,348 |

## Round 4

| file | train | validation | test |
|---|---:|---:|---:|
| **round4/jteam_r4.py** | 1,216,562 | 2,055,997 | **1,885,793** |
| round4/pb_mldi_vol_r4.py | 1,207,470 | 2,036,482 | 1,911,323 |
| round4/r4m_mldi.py | 1,204,809 | 2,034,208 | 1,905,186 |
| round4/jteam_exec.py | 1,204,809 | 2,034,208 | 1,905,186 |
| round4/r4m_mldi_v2.py | 1,214,033 | 2,029,878 | 1,890,007 |
| round4/pb_mldi_wide_r4.py | 1,206,308 | 2,024,010 | 1,885,628 |
| round4/r4m_risk_a.py | 1,156,545 | 1,965,048 | 1,844,956 |
| round4/r4_best_snipe.py | 1,142,427 | 1,963,678 | 1,884,531 |
| round4/pb_best_r4.py | 1,190,497 | 1,948,166 | 1,873,939 |
| round4/pb_ge_r4.py | 1,141,593 | 1,932,065 | 1,820,399 |

