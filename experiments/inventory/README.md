# Can paying the inventory penalty increase net PnL?

> **Status: archived pending a new dataset.** These are historical round-1 experiments, not active recommendations for the current strategy. Re-run the relevant experiment on the new dataset before reconsidering inventory-cap relaxation or end-of-round liquidation.

This folder contains the analysis, experimental strategies, reporting helper and saved result summaries. It is not part of the submission.

**Short answer: mathematically yes, but only when the incremental gross trading gain pays for the exact breach time and all incremental execution costs. The existing evidence does not justify replacing cap 100.** At an illustrative 0.3 ticks/lot of gross edge, a small breach cannot survive even a reactive 5 ms round trip; sufficiently large batches can do so only in a gross-only calculation. Prior looser-cap experiments were worse even before their penalties, so this remains a hypothesis to test, not an established improvement.

## 1. Exchange accounting

For a run of strategy \(M\), define gross PnL \(G(M)\) as cash plus the final inventory mark-out, *before* the inventory penalty. The scored objective is

\[
J(M)=G(M)-10{,}000\,T_{\rm breach}(M),\qquad
T_{\rm breach}(M)=\int \mathbf 1\{|I_M(t)|>100\}\,dt,
\]

with time in seconds. Thus \(T_{\rm breach}\) is the total duration of the union of breach intervals, including fractional seconds. Exactly 100 lots is free; either sign above 100 starts the same meter. The CLI sets the soft line to 100 and the rate to $10,000/s (`simulator/main.py:17-20`, `simulator/main.py:108-120`). The engine accrues elapsed microseconds at that fixed rate whenever the inventory during the preceding interval is outside the line (`simulator/simulator.py:1273-1296`), and computes PnL as cash + mark-out − penalty (`simulator/simulator.py:1313-1315`). There is no whole-second rounding and no multiplication by the number of excess lots: one lot outside and 900 lots outside cost the same $10 per millisecond while the breach lasts.

The fixed charge has two opposing consequences:

- Very small or brief inventory additions usually cannot amortize the charge.
- Once a breach interval is already being paid for, adding lots during that same interval adds no penalty unless it lengthens the interval or creates another one. Batching therefore helps arithmetically, but also increases unwind, price-impact/mark-out, and adverse-selection exposure.

## 2. Marginal condition versus cap 100

Let \(M_{100}\) be the otherwise comparable cap-100 run. Then

\[
\Delta J
=\underbrace{G(M)-G(M_{100})}_{\text{gross uplift }\Delta G}
-10{,}000\,[T_{\rm breach}(M)-T_{\rm breach}(M_{100})].
\]

A deliberate breach is beneficial **if and only if**

\[
\Delta G>10{,}000\,\Delta T_{\rm breach}.
\]

For a cap-100 baseline with zero breach time, this is simply gross uplift \(>10{,}000T_{\rm breach}(M)\). Locally, the requirement is that incremental gross PnL per unit of released capacity exceed $10,000/s times the incremental breach duration; in practice finite paired differences are the right test because quantities are integers and the breach indicator and order policy are discontinuous.

For one simplified episode, suppose \(Q\) additional same-direction lots would remain beyond cap 100 for a common duration \(\tau\). Let \(e\) be their gross edge per lot relative to the cap-100 alternative, before incremental unwind cost \(C_u\), adverse-selection/mark-out loss \(C_a\), and displaced-fill or other opportunity cost \(C_o\). Then

\[
\Delta G=Qe-C,\qquad C=C_u+C_a+C_o,
\]
\[
\Delta J=Qe-C-10{,}000\tau.
\]

The required gross edge is therefore

\[
\boxed{e_{\min}=\frac{10{,}000\tau}{Q}+\frac{C_u+C_a+C_o}{Q}}.
\]

Equivalently, at edge \(e\), the gross-only affordable time is \(\tau<Qe/10{,}000\); positive costs make the affordable time strictly shorter. For staggered entries or overlapping episodes, use the integral and total \(\Delta G\), not a separate penalty for each lot. In the empirical accounting below, unwind and adverse selection are already embedded in \(G\); \(C\) is a diagnostic decomposition, not an additional deduction from `result.json`.

### Illustrative affordable times at \(e=0.3\) ticks/lot

One tick times one lot changes simulator cash by one cash unit (`simulator/simulator.py:1179-1185`). The table assumes all \(Q\) extra lots share one breach interval and includes **no** unwind, adverse-selection, or opportunity costs.

| Extra headroom \(Q\) (lots) | Gross uplift \(0.3Q\) | Gross-only break-even \(\tau=0.3Q/10{,}000\) (s) | Same time (ms) |
|---:|---:|---:|---:|
| 1 | $0.30 | 0.00003 | 0.03 |
| 10 | $3.00 | 0.00030 | 0.30 |
| 50 | $15.00 | 0.00150 | 1.50 |
| 100 | $30.00 | 0.00300 | 3.00 |
| 300 | $90.00 | 0.00900 | 9.00 |
| 900 | $270.00 | 0.02700 | 27.00 |

These are upper bounds, not expected holding times. A 5 ms breach costs $50 in total, or \(50/Q\) per extra lot. At 0.3 ticks/lot, the first four rows cannot break even over 5 ms even before costs; the 300- and 900-lot rows can do so only in this idealized gross arithmetic. The 0.3 figure is illustrative: the prior fill diagnostic found +0.29 ticks/lot after 5 s on average, not a guaranteed immediately realizable marginal edge for newly enabled fills (`docs/RESEARCH.md:384-392`).

## 3. Why 5 ms and the 1,000 rule matter

A send, cancel, or modify becomes active only after 5 ms (`docs/MARKET.md:52-56`; implementation in `simulator/simulator.py:822-855`, `simulator/simulator.py:858-904`). Consequently, an unwind initiated only after observing a breach cannot react before that delay; actual \(\tau\) also includes waiting for activation and fills, and can be extended by partial IOC fills or a moved book. Resting orders can fill while their cancels are in flight, so intended \(Q\) and exit time are not guaranteed. An anticipatory or passive unwind can be faster, but has its own fill and opportunity costs; 5 ms is not a universal lower bound on every possible breach.

Separately, the exchange checks **potential position**, not just current inventory. On activation it rejects an order exactly when

\[
|I+s(R+q)|>1000,
\]

where \(R\) is resting user quantity on the incoming order's side, \(q\) is its quantity, and \(s=+1\) for buy and \(-1\) for sell (`simulator/simulator.py:955-988`, `simulator/simulator.py:1261-1271`). The CLI sets this separate limit to 1,000 (`simulator/main.py:18-20`, `simulator/main.py:115-118`). Important distinctions:

- Crossing 100 costs money but does **not** itself cause rejection.
- Equality at potential position 1,000 passes; only \(>1{,}000\) is rejected.
- With \(I=+100\) and no resting buys, a new buy of 900 reaches exactly 1,000 and is admissible; 901 is rejected. Resting same-side quantity reduces that headroom.
- The table's “headroom” is therefore economic soft-penalty headroom only when the potential-position check also leaves enough room. There is no exemption from the $10,000/s charge between 100 and 1,000.

## 4. Prior evidence and why cap 100 remains the default

The current maker sets `MAX_POS = 100` and sizes each side using inventory plus its open-order quantity so that the all-open-orders-fill worst case stays within that cap (`strategies/adaptive_mm.py:20-23`, `strategies/adaptive_mm.py:86-98`, `strategies/adaptive_mm.py:195-207`). This avoids penalty leakage and emergency-unwind feedback while preserving room for cancel/fill races through fill-based accounting (`strategies/adaptive_mm.py:393-431`, `strategies/adaptive_mm.py:453-465`).

The direct prior evidence is negative. In the earlier round-1 research sequence, caps of 110–200 with an emergency IOC above 100 produced only 86k–207k after penalties of 166k–330k and were **worse even before the penalty**, so the idea was rejected (`docs/RESEARCH.md:294-310`, especially line 308). Those are historical results for that research variant, not new results for today's exact file, but they show that simply paying the fee does not automatically unlock profitable flow: extra capacity changes fill selection and unwind behavior too.

There is nevertheless a reason to retest carefully rather than treat the old sweep as a proof. Later diagnostics found inventory at 100 about one third of the time and 39% of volume filled at \(|I|\ge90\), so capacity really binds (`docs/RESEARCH.md:384-392`). However, attempts to free capacity through asymmetric edge, crossing, or inventory lean also reduced PnL (`docs/RESEARCH.md:395-403`). Cap 100 is the prior default because it is penalty-free and prior relaxations/clearing rules damaged gross economics, not because positive marginal uplift is theoretically impossible.

## 5. Empirical acceptance criteria

Use matched runs on the same tape, changing only the cap/unwind mechanism under investigation. For each result \(i\), reconstruct

\[
G_i=\texttt{pnl}_i+\texttt{breach\_penalty}_i,
\qquad
T_i=\frac{\texttt{breach\_penalty}_i}{10{,}000}.
\]

`result.json` contains both required fields (`docs/MARKET.md:109-120`). \(T_i\) is the exact engine-accrued penalty time up to floating-point output rounding; do not estimate it from one-second PnL samples or inventory quantiles.

For variant \(M\) versus \(M_{100}\), report:

1. **Gross uplift:** \(U=G_M-G_{100}\). This isolates whether the released capacity improved trading before the fee. It must be positive; a net comparison alone would hide the previously observed problem of worse gross economics.
2. **Exact incremental penalty seconds:** \(\Delta T=T_M-T_{100}\), and cost \(P=10{,}000\Delta T\). A properly capped baseline should have \(T_{100}=0\).
3. **Net acceptance:** require \(U>P\), equivalently \(\Delta J=U-P=\texttt{pnl}_M-\texttt{pnl}_{100}>0\) (apart from floating-point rounding). Higher gross PnL is not sufficient.
4. **Robust adoption:** require the net uplift to persist across time slices and relevant held-out/synthetic/control tapes, with no position-limit-reject-driven artifacts, rather than accepting one fortunate breach episode. Report incremental unwind and mark-out/adverse-selection losses alongside \(U\) so the mechanism is visible.

These criteria compare complete policies: they do not assume that an average historical per-fill mark-out is the marginal edge of the extra lots.

## 6. Results: persistent cap relaxation on round 1

We ran the exact current `adaptive_mm.py` on the full 12,000-second `round1` tape on Lambda, changing only top-level `MAX_POS`. The baseline and eight variants used the same unmodified simulator and locked environment. Gross PnL is `pnl + breach_penalty`; deltas are against cap 100. Raw output and the complete report are in `experiments/inventory/results/round1_cap_sweep.out` and `.csv` (local, git-ignored).

| `MAX_POS` | Net PnL | Gross PnL | Gross uplift | Breach penalty | Breach seconds |
|---:|---:|---:|---:|---:|---:|
| **100** | **662,359** | **662,359** | **0** | **0** | **0.00** |
| 101 | −30,367,880 | 663,795 | +1,436 | 31,031,675 | 3,103.17 |
| 105 | −32,853,594 | 677,547 | +15,188 | 33,531,141 | 3,353.11 |
| 110 | −34,134,340 | 689,710 | +27,351 | 34,824,050 | 3,482.40 |
| 120 | −34,833,324 | 723,831 | +61,472 | 35,557,155 | 3,555.72 |
| 150 | −36,017,345 | 803,430 | +141,071 | 36,820,775 | 3,682.08 |
| 200 | −38,994,330 | 888,636 | +226,277 | 39,882,966 | 3,988.30 |
| 400 | −44,808,357 | 1,160,061 | +497,702 | 45,968,418 | 4,596.84 |
| 1000 | −47,534,476 | 1,267,320 | +604,961 | 48,801,796 | 4,880.18 |

Every relaxed cap did increase gross PnL, so released capacity has real trading value in this policy. It nevertheless failed the net acceptance test by a wide margin. Even the closest cover ratio was at cap 1000: its +604,961 gross uplift paid only **1.24%** of the added penalty. Cap 101 is the clearest small-boundary example: +1,436 gross uplift against 3,103.17 seconds outside the band and a 31.03M penalty. The best persistent policy by far remains cap 100.

These runs reject **routine, persistent overfilling** for `adaptive_mm.py`; they do not rule out a selective policy that unwinds promptly enough to reduce breach duration while retaining useful gross uplift. That transient mechanism requires separate matched experiments, and its unwind costs are already part of its resulting gross PnL.

## 7. Repeated persistent-cap test on Adaptive MM Plus

After `git pull --ff-only` confirmed commit `fa90d61`, we repeated the complete sweep on the newer `adaptive_mm_plus.py`, which adds room-freeing by shrinking and has a cap-100 baseline of 678,057. Raw output and the full report are `experiments/inventory/results/round1_plus_static_sweep.out` and `.csv`.

| `MAX_POS` | Net PnL | Gross PnL | Gross uplift | Breach penalty | Breach seconds |
|---:|---:|---:|---:|---:|---:|
| **100** | **678,057** | **678,057** | **0** | **0** | **0.00** |
| 101 | −31,749,608 | 680,750 | +2,693 | 32,430,358 | 3,243.04 |
| 105 | −35,270,913 | 697,486 | +19,429 | 35,968,399 | 3,596.84 |
| 110 | −36,748,547 | 716,858 | +38,801 | 37,465,405 | 3,746.54 |
| 120 | −38,422,593 | 744,844 | +66,787 | 39,167,437 | 3,916.74 |
| 150 | −41,179,600 | 839,578 | +161,521 | 42,019,178 | 4,201.92 |
| 200 | −45,506,278 | 968,080 | +290,023 | 46,474,358 | 4,647.44 |
| 400 | −52,703,847 | 1,282,506 | +604,449 | 53,986,353 | 5,398.64 |
| 1000 | −54,159,859 | 1,349,730 | +671,673 | 55,509,589 | 5,550.96 |

The stronger baseline does not change the conclusion. All eight relaxed caps again gain gross PnL and lose heavily after the fee. Even cap 1000's +671,673 gross uplift covers only **1.21%** of its penalty. The cap-100 configuration is the only non-negative and best net result in this grid.

## 8. Reactive, milliseconds-scale overfill on Adaptive MM Plus

For the transient alternative, `experiments/inventory/strategies/overfill_mm.py` copies Plus and adds only a reactive unwinder: whenever actual inventory exceeds 100, one IOC at the touch targets exactly the excess back to 100. The order tracker handles partial fills and clears the unwind ID on filled/cancelled/rejected terminal events. `MAX_POS` remains the outer order-sizing cap; `LIMIT_POS` remains 100. At cap 100 the experimental file reproduced Plus exactly: **678,057 net, zero penalty, and identical fills, volume, cash, ending inventory and inventory quantiles**.

The sweep used the latest model and full round 1, with raw output/report in `experiments/inventory/results/round1_plus_reactive_sweep.out` and `.csv`.

| Outer cap | Net PnL | Gross PnL | Gross uplift | Breach penalty | Breach seconds |
|---:|---:|---:|---:|---:|---:|
| **100 (unwind inactive)** | **678,057** | **678,057** | **0** | **0** | **0.00** |
| 101 | 399,397 | 677,370 | −687 | 277,973 | 27.80 |
| 105 | 382,664 | 676,848 | −1,209 | 294,184 | 29.42 |
| 110 | 98,740 | 674,936 | −3,121 | 576,196 | 57.62 |
| 120 | 46,530 | 671,622 | −6,435 | 625,092 | 62.51 |
| 150 | −83,380 | 656,560 | −21,497 | 739,940 | 73.99 |
| 200 | −225,024 | 648,450 | −29,607 | 873,474 | 87.35 |
| 400 | −458,986 | 691,160 | +13,103 | 1,150,146 | 115.01 |
| 1000 | −482,260 | 701,547 | +23,490 | 1,183,807 | 118.38 |

Reactive liquidation reduces penalty duration by two orders of magnitude relative to persistent overfill, but still fails. Through cap 200, even **gross** PnL is worse than the compliant Plus baseline: emergency crossing and displaced quote economics destroy more value than the additional fills create. Caps 400 and 1000 recover a small positive gross uplift, but it funds only 1.14% and 1.98% of their respective penalties. The best deliberately-overfilling reactive variant, cap 101, still scores **278,660 below baseline**.

Every reactive row has q95 absolute inventory equal to 100, including the cap-1000 run with a 1.18M penalty. This is a concrete reason to use exact `breach_penalty / 10000` seconds rather than ordinary inventory quantiles for rare excursions.

**Decision:** keep `MAX_POS = 100` on Adaptive MM Plus. Neither persistent overfill nor immediate bounded overfill improved net PnL at any tested value. No robustness or held-out tuning is justified for these rejected variants. A future overfill idea would need a different selective entry mechanism with positive marginal gross economics, not more values of these two policies.

## 9. End-of-round inventory: keep the stock or sell it?

The exchange computes final PnL as cash plus the marked-out value of inventory, minus breach penalty (`simulator/simulator.py:1313-1315`). Long inventory is valued by walking market bids from best price outward, with the last swept price used beyond available depth; our resting orders are ignored (`simulator/simulator.py:1334-1357`). Consequently, final stock is already included at an IOC-like liquidation value: there is no additional reward for ending flat, and ending at exactly ±100 incurs no breach penalty.

The verified Adaptive MM Plus baseline ends with cash −324,843 and inventory +100 at an average markout price of 10,029:

\[
-324{,}843 + 100\times10{,}029 = 678{,}057.
\]

Negative cash therefore does **not** mean negative PnL. It is the net cash spent acquiring inventory whose final liquidation value is +1,002,900. On the same unchanged book, immediately selling those 100 lots for the same 1,002,900 proceeds would produce cash +678,057, inventory zero, and exactly the same PnL. More generally, if actual proceeds are \(X\), flat-minus-keep PnL is simply \(X-1{,}002{,}900\): liquidation helps only if it realizes more than the inventory's counterfactual markout. Selling beyond zero merely creates a short whose negative inventory valuation offsets the extra cash; it is not a separate source of profit.

We tested the implemented policy, not just this accounting identity. `experiments/inventory/strategies/endgame_mm.py` copies Plus and, during a configurable final window, stops both maker and taker activity, cancels passive orders, and sends one best-touch IOC at a time until inventory reaches zero. It then waits for terminal confirmation and recalculates any remaining quantity, allowing for 5 ms latency, cancellation races, moved books and partial fills. `FLATTEN_WINDOW_US = 0` bypasses the entire branch and reproduced the baseline exactly.

The full-round-1 results are in `experiments/inventory/results/round1_plus_endgame_sweep.out` and `.csv`:

| Final liquidation window | Final inventory | Final cash | Net PnL | Change from keeping stock | Breach penalty |
|---:|---:|---:|---:|---:|---:|
| **0 (keep stock)** | **+100** | **−324,843** | **678,057** | **0** | **0** |
| 10 ms | 0 | +678,057 | 678,057 | 0 | 0 |
| 100 ms | 0 | +678,057 | 678,057 | 0 | 0 |
| 1 s | 0 | +677,857 | 677,857 | −200 | 0 |
| 5 s | 0 | +677,742 | 677,742 | −315 | 0 |
| 30 s | 0 | +675,546 | 675,546 | −2,511 | 0 |
| 60 s | 0 | +674,343 | 674,343 | −3,714 | 0 |

Every nonzero window completed liquidation and remained penalty-free. The 10 ms and 100 ms runs demonstrate the accounting identity in the full simulator: they converted the final stock into positive cash with **no PnL gain or loss**. Starting earlier was not beneficial. The longer-window losses include forgone end-of-round market-making and changed fills after shutting down normal trading; they should not all be interpreted as pure liquidation slippage.

The simulator flushes pending actions after the final tape event before calculating markout (`simulator/simulator.py:773-802`), which is relevant to the very short windows: a final IOC can execute, but no later public book callback is available to schedule another one. These experiments hard-code round 1's known 12,000-second endpoint and are therefore research-only, not a proposed submission rule. They establish that early flattening has no economic advantage on this tape; they do not identify a portable optimal endgame for unknown rounds.

**Decision:** retain the baseline behavior and its final +100 inventory. Positive final cash is only an accounting presentation, not extra PnL. On round 1, selling in the final 10–100 ms is economically neutral, while stopping normal trading 1–60 seconds early reduces PnL.
