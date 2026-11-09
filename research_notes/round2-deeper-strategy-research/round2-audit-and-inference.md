# Round 2 audit and lightweight model-based inference

## 1. What has actually been tested, and which earlier conclusions still apply?

### Takeaway

Round 2 already rejects most of the obvious next steps: passive-entry/queue-front tweaks, faster flattening, a full Bayesian price grid, exact IOC-price oracles, and an ID-gap/clamp jump detector. The relevant baseline is **`hmm_mm_v2` at 515,547**, not `adaptive_mm_r2` at 235,976; the latter is useful experiment history but is no longer the strategy against which a new idea should be judged.

### Cited Findings

**Status vocabulary used below.** “Round-2 tested” means a backtest on `round2_public`; “combined” means several changes were present and the component has no clean isolated PnL; “partial” means a related mechanism, not the proposed one, was tested; “Round-1/sample only” is not evidence of a Round-2 result; “untried” means no test was found in the research record.

#### Current reference points

| Strategy or mechanism | Status | Result and interpretation |
|---|---|---|
| `hmm_mm_v2` | **Round-2 tested; strongest documented public result** | **515,547**, versus 502,258 for `hmm_mm`; faster offset learning plus taker-sweep bounds improved both halves by 2.6k and 10.6k, but the gain is within the approximately 26k standard error. [hmm_mm_v2.py:1–8](../../strategies/round2/hmm_mm_v2.py) |
| `hmm_mm` | **Round-2 tested** | **502,258**, split into 242,763 quiet and 259,495 volatile PnL under `regime_mm`’s labels. It beat `regime_mm` in both regime buckets, so a simple quiet/volatile hybrid of those two engines adds nothing. [RESEARCH.md:828–842](../../docs/RESEARCH.md#L828-L842) |
| `regime_mm_kf` | **Round-2 tested** | **469,844**, versus 448,818 for `regime_mm`; the Kalman filter helped in the volatile policy, while enabling it in the quiet policy scored only 297k–311k. Numerous take/exit/hold variants on this file lost. regime_mm_kf.py:1–12 |
| `regime_mm` | **Round-2 tested and separately validated** | **448,818** on public Round 2 and 1,032,909 over a separate 12,000-second validation tape. Quiet PnL was 228k and volatile PnL 221k on the public run. [RESEARCH.md:733–742](../../docs/RESEARCH.md#L733-L742) |
| `adaptive_mm_r2` | **Round-2 tested, but not the current best baseline** | **235,976** with its Kalman filter, versus 231,426 with running averages; the improvement was within approximately 19k standard error. Its split was only 43,214 quiet and 192,762 volatile. [RESEARCH.md:659–673](../../docs/RESEARCH.md#L659-L673); [RESEARCH.md:828–838](../../docs/RESEARCH.md#L828-L838) |
| `storm_mm` / `RegimeFilter` | **Round-2 combined test only** | The strategy combining `RegimeFilter`, a far-order trigger, momentum guard, and changed quoting scored approximately **172k**, with no penalty. This confirms that `tools/regime_filter.py` was integrated and backtested; it does **not** establish the filter’s isolated contribution. storm_mm.py:1–12; storm_mm.py:158–168; commit `0c7cbed9d7b9991b8e55cd10e5282ca9703a8c2d` |
| `jump_mm`, `lag_taker` | **Round-2 tested** | **220,583** and **−42,592**, respectively; both were far below `hmm_mm`. [RESEARCH.md:828–838](../../docs/RESEARCH.md#L828-L838) |

#### Price inference, regime inference, and jump detection

| Proposal | Status | Evidence / classification |
|---|---|---|
| Running-average hidden price (`s_hat`) and clamp-aware average (`S2`) | **Round-2 tested** | Faster averaging materially improved `adaptive_mm_r2` from 174,377 to 231,426 together with a lower crossing threshold. In the later estimator comparison, `S2` had the best volatile 100ms corr² (0.104), but the Kalman ran the strategy better; prediction rank and trading rank were not identical. [RESEARCH.md:639–649](../../docs/RESEARCH.md#L639-L649); [RESEARCH.md:767–775](../../docs/RESEARCH.md#L767-L775) |
| Scalar Kalman filter, learned offset/noise, IOC covariance inflation | **Round-2 tested; adopted in some strategies** | On `adaptive_mm_r2`, better 0.1–1s prediction produced only about 5k additional PnL; zero IOC-jump inflation scored 206k versus 230k–232k for useful nonzero values. On `regime_mm`, putting the KF in the volatile policy increased PnL from 448,818 to 469,844. [RESEARCH.md:659–673](../../docs/RESEARCH.md#L659-L673); regime_mm_kf.py:1–12 |
| Mean IOC push added directly to the price estimate | **Round-2 tested and rejected** | `KF_PUSH` 0.005 was roughly unchanged and 0.015 worse on `adaptive_mm_r2`; a permanent-push addition had already helped only the public Round-1 tape while losing on every synthetic twin. [RESEARCH.md:668–669](../../docs/RESEARCH.md#L668-L669); [RESEARCH.md:579–582](../../docs/RESEARCH.md#L579-L582) |
| Volatility-scaled edges/flow/memory (`RealizedVol` in `adaptive_mm_vol` / `adaptive_mm_r2`) | **Round-2 combined/partial test** | Volatility-responsive margins and flow were part of the move from the Round-1 design to the 131,725 `adaptive_mm_vol` baseline and then 235,976 `adaptive_mm_r2`, but the isolated contribution of the online volatility estimator is not reported. Several later edge sweeps were conditional on this scaling. adaptive_mm_r2.py:44–65; [RESEARCH.md:639–652](../../docs/RESEARCH.md#L639-L652) |
| Hidden-price predictability check and aggressive-flow predictability check | **Round-1/control tested for S check; Round-2 inherited but unablated** | The S check removed most Round-1 control-tape losses at no material Round-1 cost. `hmm_mm_v2` retains a flow check, but no Round-2 on/off result isolates either validation gate. [RESEARCH.md:368–382](../../docs/RESEARCH.md#L368-L382); [hmm_mm_v2.py:89–96](../../strategies/round2/hmm_mm_v2.py) |
| Censor an order one tick from the opposite best | **Round-2 tested; result depends on the observation rule** | Turning censoring off cost only about 4k in `adaptive_mm_r2`, but adding the analogous rule to `hmm_mm` collapsed PnL from 502,258 to 423,633 because at a one-tick spread almost every joining buy satisfies the censor condition. This is not a generally valid “exact bound” improvement. [RESEARCH.md:668–669](../../docs/RESEARCH.md#L668-L669); [RESEARCH.md:844–853](../../docs/RESEARCH.md#L844-L853) |
| Executed sweep prices as bounds on S | **Round-2 tested; weak/noise-level gain** | On `hmm_mm`, the bound variant scored 505,747 versus 502,258. `hmm_mm_v2` retains a half-weight, zero-slack version together with faster offset learning; there is no isolated `hmm_mm_v2` ablation for that exact combination. [RESEARCH.md:844–850](../../docs/RESEARCH.md#L844-L850); [hmm_mm_v2.py:1–8](../../strategies/round2/hmm_mm_v2.py) |
| Fitted OLS fair value using S gap, imbalance, flow, and trade sign | **Round-2 tested and rejected** | Slightly better one-second prediction (R² 0.034 versus 0.029) traded worse: fitted-fair-only variants scored 198,944–209,336 versus 235,976. The useful crossing flow weight was much larger than its average one-second forecasting coefficient. [RESEARCH.md:691–731](../../docs/RESEARCH.md#L691-L731) |
| Exact joint Bayesian grid over S and quiet/volatile regime | **Round-2 tested and rejected** | **262k** in both regimes and **324k** only in the volatile regime, versus approximately **470k** for the KF baseline. Grid volatile corr² was 0.033 versus KF 0.057 and S2 0.104; its P(volatile) averaged only about 0.10 during detector-labelled volatile spells, exposing a bad volatile fit. [RESEARCH.md:759–775](../../docs/RESEARCH.md#L759-L775); regime_mm_grid.py:1–15 |
| Joint MLE / properly fitted two-regime forward model | **Proposed, not completed** | The research record recommends joint estimation of switch rates, offset distributions, and diffusion after diagnosing the grid’s mislabelled volatile mixture. No resulting MLE or backtest is documented. [RESEARCH.md:816–824](../../docs/RESEARCH.md#L816-L824) |
| Round-2 synthetic twin with true S | **Attempted, then dropped** | One 6,000-second tape was judged insufficient to fit regimes and jumps: offline two-state EM implied volatile spells around 1s versus around 3s from `regime_mm`. Thus there is still no true-S Round-2 oracle measurement. [RESEARCH.md:816–819](../../docs/RESEARCH.md#L816-L819); [RESEARCH.md:840–842](../../docs/RESEARCH.md#L840-L842) |
| Exact IOC prices supplied to the filter, without lookahead | **Oracle-only Round-2 test; rejected even with unavailable information** | Exact `round(S ± 1)` observations scored **248k–377k**; push-corrected variants scored **155k–372k**, all below the KF baseline. Their mid-prediction corr² was only 0.02–0.04 versus 0.12–0.24 for the KF on matched samples. [RESEARCH.md:776–795](../../docs/RESEARCH.md#L776-L795) |
| Future-mid oracle as a price-estimation ceiling | **Diagnostic only, not an attainable filter** | Mid 20/50/100/300ms ahead scored 440k/750k/888k/1,021k, versus 79k for current mid and about 470k for the estimators. The correction in the research notes is crucial: most of this value comes from future IOCs and orders inside the lookahead window, not from a better estimate of present S. [RESEARCH.md:744–757](../../docs/RESEARCH.md#L744-L757); [RESEARCH.md:776–784](../../docs/RESEARCH.md#L776-L784) |
| ID gaps plus one-sided clamp bursts as an online jump detector | **Round-2 tested and rejected** | Bounds plus gap/clamp-triggered 15-tick jumps scored **415,135** versus 502,258. Of 2,162 online gaps, almost all were nontrading IOCs rather than the 80–82 far limit orders useful offline; clusters caught all 22 large moves but generated 192 alarms. [RESEARCH.md:786–795](../../docs/RESEARCH.md#L786-L795); [RESEARCH.md:844–857](../../docs/RESEARCH.md#L844-L857) |
| Standalone momentum jump guard / far-triggered volatile mode | **Round-2 combined or rejected** | `jump_guard_mm` scored approximately 104k; adding the regime filter and far trigger in `storm_mm` gave approximately 172k. `hmm_mm_plus` far triggers tied or lost against 502k and remain off. These are not clean tests of the per-second HMM alone. Commit `0c7cbed9d7b9991b8e55cd10e5282ca9703a8c2d`; hmm_mm_plus.py:1–18 |
| Generic particle filter / another full joint latent-state simulator | **Untried, but not justified by the completed grid result** | The research record identifies particle and interacting-multiple-model filters only as general forms of the grid/regime approach. It supplies no particle implementation or result, while the exact grid already demonstrates model-fit failure and adds much greater callback cost. [RESEARCH.md:798–805](../../docs/RESEARCH.md#L798-L805) |

#### Execution, queue, and inventory decisions

| Proposal | Status | Evidence / classification |
|---|---|---|
| Disable the generic OFI/IOC taker leg | **Round-2 tested and adopted in `adaptive_mm_r2`** | Its IOC fills marked out at −2.90 ticks/lot after 5s. This does not indict all aggressive trading: crossing LIMIT orders earned +0.87 at 5s in the same decomposition. [RESEARCH.md:600–616](../../docs/RESEARCH.md#L600-L616) |
| Passive entries at the touch, even in the signal direction | **Round-2 tested and rejected** | Passive opening fills lost at 1/5/30s in `adaptive_mm_vol`; resting-entry variants later scored 210,395–216,926 versus 235,976. Round 2 favoured paying the spread on a sufficiently strong signal. [RESEARCH.md:600–616](../../docs/RESEARCH.md#L600-L616); [RESEARCH.md:723–731](../../docs/RESEARCH.md#L723-L731) |
| Post-sweep passive guard | **Round-2 tested and rejected in this form** | Pulling passive quotes for 100–1000-lot sweeps had no benefit or lost PnL; for example, the 100-lot/1s guard scored 216k versus 231k off. [RESEARCH.md:650–652](../../docs/RESEARCH.md#L650-L652); adaptive_mm_r2.py:198–207 |
| Predict the next sweep from thin touch plus S lean before it occurs | **Proposed, not directly tested** | The research notes identify trade sign/size and best-quote liquidity as a possible jump predictor. The later guard reacts after a sweep, while the ID-gap/clamp detector is a different signal; neither closes every pre-sweep depth/S predictor. [RESEARCH.md:623–637](../../docs/RESEARCH.md#L623-L637); [RESEARCH.md:844–857](../../docs/RESEARCH.md#L844-L857) |
| “Buy the dip” / transient-impact reversion | **Rejected from Round-2 diagnostics; not a separate strategy result** | Round 2 trends rather than reverts, passive fills remain negative after 60s, and the fitted IOC impact on S is permanent. This rejects the economic premise, not merely one parameter setting. [RESEARCH.md:618–629](../../docs/RESEARCH.md#L618-L629) |
| Own-fill markout safety switch (`FillMarkout`) | **Present in a combined Round-2 baseline; no isolated ablation** | The switch shrinks passive size after adverse own-fill markouts and remains enabled in `adaptive_mm_r2`, but the documented tuning does not provide a clean on/off score separate from the other Round-2 changes. adaptive_mm_r2.py:60–65; adaptive_mm_r2.py:222–227 |
| Queue-front-only join/cancel rule | **Round-2 tested and rejected on `adaptive_mm_r2`** | Queue-front-only scored 226,301, or 231,556 with looser limits, versus 235,976; it mainly blocked useful passive unwinds. This is not a complete test of every queue-survival model. [RESEARCH.md:691–731](../../docs/RESEARCH.md#L691-L731) |
| Size crosses down by current signal strength (`CROSS_FULL`) | **Round-2 tested and rejected** | Values 1.5/2/3 scored 233k/224k/208k versus the 235,976 full-size baseline. This tested current-edge sizing, not a forward opportunity-cost price for inventory. [RESEARCH.md:675–689](../../docs/RESEARCH.md#L675-L689) |
| Flatten sooner, cheaper exit crosses, stronger passive unwind or crossing skew | **Round-2 tested and rejected across many settings** | Lower `EXIT_MULT` scored 193k–224k; stronger passive unwinding fell as low as 55k; stronger crossing skew also lost. Full positions retained +0.6 to +1.5 ticks/lot at 30s, so earlier/better entries—not simply more free capacity—were the remaining lead. [RESEARCH.md:675–689](../../docs/RESEARCH.md#L675-L689) |
| Passive or earlier exits in `regime_mm_kf` | **Round-2 tested and rejected** | Passive-exit variants scored 345k–399k; `V_KEEP` 0/1 scored 460k/310k; shorter maximum holds scored 421k versus about 469k. regime_mm_kf.py:1–6 |
| Deep calm sweeps, volatile-position recycling, probability-scaled quote edge, strong-signal side pull | **Round-2 tested; tied or lost** | All `hmm_mm_plus` variants remained off against the 502k baseline. These are distinct from a latency-aware probability-of-profitable-cross calculation. Commit `5544f28`; hmm_mm_plus.py:1–18 |
| Controlled inventory above ±100 on exceptional sweeps | **Round-2 tested and rejected** | The best calculated-overshoot unwind was about **487k**, below 502k before considering model risk, and the extra lots lost even before approximately 6k penalty. Commit `f7af8c51a1c095af67c96ac63b6ab4a1253d1659`; [MARKET.md:98–107](../../docs/MARKET.md#L98-L107) |
| Inventory thresholds from dynamic programming | **Untried** | Explicitly listed as a possible Round-1 idea; no learned per-inventory action table or Round-2 test appears later. [RESEARCH.md:478–489](../../docs/RESEARCH.md#L478-L489) |
| Inventory-conditioned thin-queue step-back | **Untried** | The hftbacktest-inspired rule was distinguished from the already-tested plain short-queue and imbalance guards, but the subsequent experiment list contains no result for it. [RESEARCH.md:498–505](../../docs/RESEARCH.md#L498-L505); [RESEARCH.md:531–537](../../docs/RESEARCH.md#L531-L537) |
| Keep/cancel value computed from queue shrinkage; full model-based quote table | **Untried** | Both were derived from the fitted Round-1 generator and listed as future experiments. Neither has a reported Round-2 implementation. [RESEARCH.md:520–537](../../docs/RESEARCH.md#L520-L537) |
| Directional-carry versus spread-capture accounting | **Missing diagnostic, not a tested strategy** | It was recommended as a useful PnL decomposition; no resulting decomposition was found in the Round-2 research. [RESEARCH.md:484–489](../../docs/RESEARCH.md#L484-L489) |
| Selection by `E[PnL] − 0.25 SD` over many generated sessions | **Untried as stated** | It was proposed before Round 2, but the later Round-2 twin was dropped; public-tape differences around one standard error therefore remain weak evidence. [RESEARCH.md:484–489](../../docs/RESEARCH.md#L484-L489); [RESEARCH.md:840–842](../../docs/RESEARCH.md#L840-L842) |

#### Round-1/sample ideas whose economics changed under Round-2 jumps

| Earlier idea | What was actually established | Round-2 interpretation |
|---|---|---|
| Passive market making and more opening size | Round 1 rewarded making: `adaptive_mm.py` earned 202,677 entirely from its maker, and later fair-value makers reached 650k–678k. Round-1 market maker markout at the touch was +0.03 ticks/lot after 1s. [RESEARCH.md:172–208](../../docs/RESEARCH.md#L172-L208); [RESEARCH.md:326–366](../../docs/RESEARCH.md#L326-L366) | **Do not transfer the unconditional opening-quote conclusion.** Round-2 maker markout was −0.40 at 1s and −0.49 at 60s, with the damage concentrated in jump sweeps; profitable quiet making still remains inside `hmm_mm`. [DATA_ANALYSIS.md:238–255](../../docs/DATA_ANALYSIS.md#L238-L255); [RESEARCH.md:832–836](../../docs/RESEARCH.md#L832-L836) |
| Sample momentum/OFI taker | On the trending `sample.csv`, PnL scaled with target position; on Round 1 the unchecked taker lost 157,341 and the momentum check kept it off. [RESEARCH.md:74–107](../../docs/RESEARCH.md#L74-L107); [RESEARCH.md:190–200](../../docs/RESEARCH.md#L190-L200) | **Sample trend results are irrelevant; generic OFI taking is also not the Round-2 edge.** Round-2 crosses based on S and flow pay, while the separate OFI IOC leg lost; a decayed executed-flow term nevertheless remains in the current HMM’s fair value. [RESEARCH.md:600–621](../../docs/RESEARCH.md#L600-L621) |
| Touch imbalance / microprice / deeper-book fair value | Touch imbalance was predictive on Round 1, and removing the imbalance term from the 615k Round-1 baseline cost about 15k. An explicit microprice weight on `sample_v2` was within noise; deeper-book and Boltzmann variants were literature suggestions without later strategy results. [RESEARCH.md:111–135](../../docs/RESEARCH.md#L111-L135); [RESEARCH.md:146–159](../../docs/RESEARCH.md#L146-L159); [RESEARCH.md:332–340](../../docs/RESEARCH.md#L332-L340) | **The small imbalance term survives in `hmm_mm_v2`, but has no isolated Round-2 ablation.** There is no evidence to prioritise another static microprice variant over posterior execution value. [hmm_mm_v2.py:58–65](../../strategies/round2/hmm_mm_v2.py) |
| Multi-level/participant-filtered OFI and Hawkes-style flow | Touch or decayed flow was tested in several forms; removing aggressive flow cost 110k on Round 1, while a Round-1 sweep-pull filter was approximately neutral. Participant clustering and multi-level PCA were proposals, not reported implementations. [RESEARCH.md:326–340](../../docs/RESEARCH.md#L326-L340); [RESEARCH.md:294–310](../../docs/RESEARCH.md#L294-L310); [RESEARCH.md:54–58](../../docs/RESEARCH.md#L54-L58) | **Round 2 retains decayed executed flow in fair value but rejects the separate generic taker and post-sweep guard.** A new participant/Hawkes model has no positive local result and is not a lightweight next step. [RESEARCH.md:600–616](../../docs/RESEARCH.md#L600-L616); [hmm_mm_v2.py:169–195](../../strategies/round2/hmm_mm_v2.py) |
| Queue ladder and older queue positions | The two-level Round-1 ladder increased PnL from roughly 270k to 354k; older Round-1 fills had better markouts. [RESEARCH.md:263–280](../../docs/RESEARCH.md#L263-L280); [RESEARCH.md:384–403](../../docs/RESEARCH.md#L384-L403) | **Queue age is not sufficient under jump adverse selection.** Round-2 passive entries and queue-front filters lost, although a forward fill/invalidation survival value remains untested. [RESEARCH.md:723–731](../../docs/RESEARCH.md#L723-L731) |
| Penny-jumping / improving inside wide spreads | It lost badly on Round 1 (179k directional, 50k both sides versus 270k baseline) and was also picked off on `sample_v2`. [RESEARCH.md:241–254](../../docs/RESEARCH.md#L241-L254) | **Already negative before Round 2; no reason to revive it as an inference experiment.** Round-2 resting entries reinforce the adverse-selection conclusion. [RESEARCH.md:723–727](../../docs/RESEARCH.md#L723-L727) |
| Imbalance/OFI pull filters | A tuned imbalance guard helped the Round-1 ladder, while aggressive-flow pull settings were mostly neutral; a different `adaptive_mm_r2` queue/fitted-fair combination lost. [RESEARCH.md:294–310](../../docs/RESEARCH.md#L294-L310); [RESEARCH.md:691–731](../../docs/RESEARCH.md#L691-L731) | **No clean Round-2 test of the original imbalance guard exists.** Do not label it either validated or definitively closed; it is lower priority than the S/jump mechanism. |
| Zero-edge clearing and freeing inventory | Zero-edge clearing lost on Round 1 (most variants below the 353,645 baseline), and attempts to get off the Round-1 cap also lost. [RESEARCH.md:282–292](../../docs/RESEARCH.md#L282-L292); [RESEARCH.md:395–403](../../docs/RESEARCH.md#L395-L403) | **The same economic warning is stronger on Round 2.** Persistent directional positions are valuable; forced early flattening repeatedly lost. A marginal opportunity-cost gate must not smuggle forced liquidation back in. [RESEARCH.md:675–689](../../docs/RESEARCH.md#L675-L689) |
| Queue hysteresis and shrink-in-place | Hysteresis helped on Round 1 and all three twins; shrink-plus-lower-inventory-weight improved all 30 validation sessions. [RESEARCH.md:405–439](../../docs/RESEARCH.md#L405-L439); adaptive_mm_r2.py:67–84 | **Inherited engineering, not a clean Round-2 causal result.** The current `hmm_mm_v2` does not use the same shrink/ladder machinery, so porting these back would be a new combined experiment rather than an established improvement. [hmm_mm_v2.py:306–343](../../strategies/round2/hmm_mm_v2.py) |
| Better present-S estimation | On generated Round-1 sessions, giving the strategy true S did not improve its score; the remaining constraints were inventory, queueing, and impactful fills. [DATA_ANALYSIS.md:164–167](../../docs/DATA_ANALYSIS.md#L164-L167) | **This does not prove Round-2 S estimation is worthless, but future-mid oracle numbers do not refute it either.** No Round-2 true-S test exists, and future order/IOC events drive much of the lookahead gain. [RESEARCH.md:807–814](../../docs/RESEARCH.md#L807-L814) |

### Inferences

- The next experiment should be built on `hmm_mm_v2`, with `hmm_mm` as the clean ablation anchor. Reporting a gain over 235,976 from `adaptive_mm_r2` would not show progress over the team’s current mechanism.
- The strong result is not “all making is bad” or “all taking is good.” `hmm_mm` makes money in both quiet and volatile buckets; the rejected objects are specific passive entries, generic IOC momentum, queue thresholds, and premature exits.
- The exact grid’s failure is primarily a **model-fit and decision-usefulness failure**, not proof that every non-Gaussian update is useless. A constant-memory robust update to the working KF is still a materially smaller and different hypothesis than fitting another full generative grid.
- The most defensible open decisions are no longer “can another oracle recover S exactly?” but “how should uncertain S and executable-book survival change a trade sent 5ms later?” and “when is marginal inventory capacity worth spending?”
- `RegimeFilter` has **not** received a filter-only PnL ablation. Its low combined `storm_mm` score cannot be assigned to the HMM without also changing the guard, quotes, cap, and far trigger.

### Gaps

- No isolated Round-2 result was found for `RegimeFilter` with the far trigger and storm-quoting changes removed; the approximately 172k `storm_mm` result is combined and rounded in the commit record.
- No Round-2 true-S oracle or accepted multi-session Round-2 generator exists, so the achievable value of better present-S estimation remains unknown.
- The exact isolated effects of taker-sweep bounds and faster `C_LR` inside final `hmm_mm_v2` are not both reported; only their combination and the earlier sweep-bound variant are documented.
- The original Round-1 imbalance guard, queue hysteresis, shrink-in-place, dynamic-programming inventory table, inventory-conditioned thin-queue rule, and carry/capture diagnostic have no clean Round-2 result.

## 2. Which discarded or invisible information channels can still support lightweight inference?

### Takeaway

The useful unused information is mainly **execution survival and conditional book evolution**, not hidden IOC prices. Nonexecuting IOC request prices are not exposed, ID gaps cannot identify them reliably, and even an exact-price oracle lost; another grid or particle estimator therefore attacks an already weakened channel rather than the current decision bottleneck.

### Cited Findings

| Channel | Observable online? | What the audit establishes |
|---|---|---|
| New visible limit price and side | **Yes** | This remains the strongest price channel. The generator uses buy `floor(S−X)` and sell `ceil(S+X)`, with clamping to prevent crossings. Existing strategies detect newly visible IDs and feed their prices to averages/KFs/HMMs. [DATA_ANALYSIS.md:40–55](../../docs/DATA_ANALYSIS.md#L40-L55); [hmm_mm_v2.py:239–288](../../strategies/round2/hmm_mm_v2.py) |
| Executed IOC sweep price and executed quantity | **Yes, through resulting trades** | A deepest swept price supplies a bound on S, while executed aggressive volume is useful for flow and impact uncertainty. `hmm_mm_v2` already uses the former and recent signed flow; `regime_mm_kf` also uses volume-driven KF variance inflation. [MARKET.md:66–73](../../docs/MARKET.md#L66-L73); [hmm_mm_v2.py:169–188](../../strategies/round2/hmm_mm_v2.py); regime_mm_kf.py:240–266 |
| Original IOC limit price | **No** | Callbacks expose resulting book snapshots and trades, not IOC request prices or unfilled residuals. Missing IDs do not reveal the request price, and exact IOC prices performed poorly even when supplied by an oracle. [MARKET.md:66–73](../../docs/MARKET.md#L66-L73); [RESEARCH.md:776–795](../../docs/RESEARCH.md#L776-L795) |
| Nontrading IOC event, side, and price | **No clean identification** | A gap can also be an order outside the 20-tick public window. Online gaps were dominated by nontrading IOCs, so their side and bound cannot be reconstructed; the offline 82 far limits are not identifiable from the other gaps. [MARKET.md:42–50](../../docs/MARKET.md#L42-L50); [RESEARCH.md:786–795](../../docs/RESEARCH.md#L786-L795); [RESEARCH.md:852–856](../../docs/RESEARCH.md#L852-L856) |
| Original IOC size when it trades | **Not generally; executed size is observed** | Trade quantities expose executed volume, not necessarily the original IOC request. The fitted generator makes IOC size independent of S but gives it a permanent impact on S, so executed size can inform impact uncertainty while not identifying the rounded request price exactly. [DATA_ANALYSIS.md:200–206](../../docs/DATA_ANALYSIS.md#L200-L206); [MARKET.md:66–73](../../docs/MARKET.md#L66-L73) |
| Limit size, cancel time, ordinary arrival time | **Yes, but not direct S readings** | The fitted model says sizes, sides, cancel times, and arrival times are independent of S. Their defensible use is queue/fill survival, not extra hidden-price measurements. [RESEARCH.md:786–796](../../docs/RESEARCH.md#L786-L796); [DATA_ANALYSIS.md:81–104](../../docs/DATA_ANALYSIS.md#L81-L104) |
| Spread, far-order rate, market-taker rate | **Yes** | `RegimeFilter` combines these once per second, but only a low-scoring combined strategy test exists. `hmm_mm_v2` instead infers regime per new order from order-to-mid offsets. [tools/regime_filter.py:1–17](../../tools/regime_filter.py#L1-L17); [hmm_mm_v2.py:19–26](../../strategies/round2/hmm_mm_v2.py) |
| Best-price depth, queue removals, and near-term book changes | **Yes; underused for action valuation** | Current `hmm_mm_v2` uses touch imbalance and a depth-normalised flow estimate, but its crossing tests compare point-estimate fair to the current touch. It does not value the probability that the same executable price survives the 5ms send delay. [hmm_mm_v2.py:189–237](../../strategies/round2/hmm_mm_v2.py); [MARKET.md:52–64](../../docs/MARKET.md#L52-L64) |
| Post-jump repetition / jump spacing | **Observable only after jumps; direction not predictable beforehand** | Public-tape analysis found large jumps unusually spaced, no reliable pre-jump direction signal, and an IOC surge after jumps. It did not establish a stable online timer suitable for trading, and the later ID-gap detector failed. [DATA_ANALYSIS.md:207–220](../../docs/DATA_ANALYSIS.md#L207-L220); [RESEARCH.md:844–857](../../docs/RESEARCH.md#L844-L857) |

### Inferences

#### Exact likelihoods already available—and why they do not rescue IOC reconstruction

These are derivations from the locally documented placement rules, not new empirical claims. Let `F` be the CDF of the nonnegative offset X and let `A`/`B` be the pre-insert best ask/bid.

For an uncapped new buy at integer price `p`:

`Pr(p | S) = F(S − p) − F(S − p − 1)`.

For an uncapped new sell:

`Pr(p | S) = F(p − S) − F(p − S − 1)`.

A buy clamped at the most aggressive permitted price supplies a one-sided likelihood proportional to `F(S−p)` rather than the uncapped mass; a sell is mirrored. These are the likelihoods the existing grid already implements with precomputed gamma tables, so reusing them in another full grid is not a new experiment. [tools/grid_price.py:1–21](../../tools/grid_price.py#L1-L21); [tools/grid_price.py:51–60](../../tools/grid_price.py#L51-L60)

If an *ideal oracle* said that a buy IOC did not execute against best ask `a`, then `round(S+1) < a` would imply the censored bound:

`S < a − 1.5`.

A nonexecuting sell IOC against best bid `b` would imply:

`S > b + 1.5`.

Those would be informative one-sided observations **if the event, side, and contemporaneous touch were known**. Online, none of that resolves an ID gap into a clean signed bound. Even granting the original price by oracle did not make the channel pay. Accordingly:

- **Reject IOC-price reconstruction and nonexecuting-IOC censored updates as the next experiment.** They add assumptions to recover a channel already shown weak with perfect information.
- **Reject a naïve second Bayesian grid and a generic particle filter.** They do not create new observations, and a particle approximation would be more expensive than the exact grid that already lost.
- **Retain executed-sweep bounds and executed-volume impact as auxiliary evidence**, but do not treat them as the original IOC price or original size.
- **Use sizes, cancels, and depth where they contain legitimate new information: execution survival.** That supports a crossing decision rather than pretending they reveal S directly.
- A robust innovation filter is still admissible because it changes how the proven visible-limit channel handles a small number of coherent large displacements; it is not another attempt to reconstruct invisible IOCs.

### Gaps

- The simulator callback does not expose IOC request prices, sides of invisible IOCs, or unfilled original quantities; there is no clean online measurement with which to validate a nonexecution likelihood. [MARKET.md:66–73](../../docs/MARKET.md#L66-L73)
- The relative value of queue survival conditioned on depth, spread, regime, and signed S gap has not been measured on Round 2. The existing missed-opportunity tool classifies hindsight bursts as taken/missed/blocked but does not fit a 5ms fill-survival model. [scripts/missed_opps.py:1–10](../../scripts/missed_opps.py#L1-L10)
- No clean result separates the information value of per-second rates from the other `storm_mm` mechanisms.

## 3. Which genuinely incremental experiments are worth running next?

### Takeaway

The best remaining lightweight path is to keep the successful strategy and change one mathematical decision at a time: **(A)** make the KF robust to sparse process jumps without an ID-gap detector, **(B)** value a cross at its 5ms arrival using posterior edge and shadow-measured execution survival, and only then **(C)** test a marginal inventory opportunity cost that gates entries without forcing exits. The first two are constant-memory and do not repeat the failed full grid, IOC oracle, or generic jump trigger.

### Cited Findings

- `hmm_mm_v2` currently updates one scalar S mean/variance from new-order prices and crosses when a point-estimate fair exceeds the current touch by a fixed edge. It does not propagate a posterior profitable-cross probability through the 5ms delay. [hmm_mm_v2.py:197–237](../../strategies/round2/hmm_mm_v2.py); [hmm_mm_v2.py:258–272](../../strategies/round2/hmm_mm_v2.py)
- Every send, cancel, and modify lands just before the first market event at or after `t+5ms`; a cancel can lose to a fill, and inventory must be tracked from private fills. [MARKET.md:52–64](../../docs/MARKET.md#L52-L64)
- Existing IOC-driven variance inflation matters in the older KF, but its gain is deterministic and occurs after aggressive volume. No tested filter performs a Bayesian normal/sparse-jump update from each new-limit innovation itself. [RESEARCH.md:659–669](../../docs/RESEARCH.md#L659-L669)
- The rejected jump detector used explicit clamp/ID-gap alarms and a fixed 15-tick S displacement; the rejected grid fitted full offset distributions and dynamic regime arrays. Neither is the same object as a few-scalar robust KF innovation update. [RESEARCH.md:759–775](../../docs/RESEARCH.md#L759-L775); [RESEARCH.md:844–857](../../docs/RESEARCH.md#L844-L857)
- Faster exits, current-edge position sizing, controlled overshoot, and passive-entry changes have already lost. A new inventory experiment must therefore leave the profitable hold logic intact and target only the marginal entry decision. [RESEARCH.md:675–689](../../docs/RESEARCH.md#L675-L689); commit `f7af8c51a1c095af67c96ac63b6ab4a1253d1659`
- No fees are charged, and an aggressive fill’s immediate cost is embedded in the price paid relative to value; the exchange uses integer ticks/lots and a 5ms action delay. [MARKET.md:31–40](../../docs/MARKET.md#L31-L40); [MARKET.md:52–56](../../docs/MARKET.md#L52-L56)

### Inferences

The formulas below are **derived proposals**, not measured results and not claims imported from an external paper. No external academic citation is needed for these derivations; all empirical premises are the local citations above.

#### Experiment A — sparse process-mixture innovations with robust nuisance adaptation

**Objective.** Let coherent large new-limit innovations move S quickly, while isolated heavy-X observations and noisy offset/noise estimates do not either freeze the filter or manufacture a 15-tick jump. This replaces only the observation update inside `hmm_mm_v2`; it does not force volatile mode, use ID gaps, or issue a detector-triggered trade.

**Scalar update.** Before an observation, the current prior is `N(m,P)`. For a new order, construct exactly the existing offset-corrected reading `y` and ordinary measurement variance `R`, with elapsed-time diffusion already included in `P`. Consider two process hypotheses:

- `j=0`: ordinary diffusion, innovation variance `V0=P+R`;
- `j=1`: a sparse additional process displacement with variance `J`, innovation variance `V1=P+J+R`.

For innovation `e=y−m` and prior jump mass `π`, evaluate:

`L0 = Normal(e; 0, V0)`, `L1 = Normal(e; 0, V1)`;

`w0 = (1−π)L0 / ((1−π)L0 + πL1)`, `w1 = 1−w0`.

Each hypothesis has its own Kalman update:

`K0=P/V0`, `K1=(P+J)/V1`;

`mj=m+Kj e`, `Pj=(1−Kj)(P+jJ)`.

Collapse back to constant memory:

`m' = w0m0 + w1m1`;

`P' = Σj wj [Pj + (mj−m')²]`.

The last term is essential: posterior disagreement between the ordinary and displaced explanations is real uncertainty and must not be discarded. This is an interacting-mixture approximation, not the failed dynamic S grid. It uses a handful of scalars and no regime arrays.

**Robust offset/noise adaptation.** Existing online updates of offset `c` and noise `R` can learn from the same extreme innovation that should instead update S. Update these nuisance parameters using posterior ordinary weight `w0`, and bound one observation’s contribution:

`c ← c + αc w0 clip(dc−c, −u, +u)`;

`R ← max(Rmin, R + αR w0 clip(e²−P−R, −v, +v))`.

Here `dc` is the same outward-distance reading used by the current code. The clipping bounds are numerical robustification settings, not jump-direction thresholds. Do **not** robustly reject the actual price observation merely because it is large: that would throw away the earliest evidence of a real permanent jump.

**Minimal ablation sequence.**

| Variant | Sole change from the same baseline |
|---|---|
| A0 | Identical `hmm_mm_v2` observations, policy, and order handling. |
| A1 | Two-component process update only; existing `c` and `R` adaptation retained. |
| A2 | Only `w0`-weighted/clipped nuisance adaptation; no process mixture. |
| A3 | Both A1 and A2, to measure interaction rather than assume additivity. |

Require a scalar-filter diagnostic before a long strategy run: S-error/corr² by regime, posterior mean squared-innovation calibration, fraction of time `w1` is material, and ordinary-period drift in `c`/`R`. On the eventual backtest, report public PnL, halves, volatile/quiet PnL, standard error, inventory quantiles, and penalty. A1 is the cleanest first test; A3 is not allowed to substitute for it.

**Cost and failure modes.** Two scalar normal likelihoods, two KF updates, and moment collapse per visible new order: **O(1) time and memory**, with roughly two exponential/log evaluations and no NumPy allocation. A single fixed `J` can still be misspecified; `π` can produce frequent overreaction or near-zero use; a moment-collapsed mixture can lag if it repeatedly chooses the ordinary component. This does not predict an exogenous jump before observations arrive.

#### Experiment B — probability and expected value of a profitable cross at `t+5ms`

**Objective.** Replace the binary point-estimate comparison `fair−ask ≥ threshold` with a decision for the book and uncertainty that will exist when the order arrives. Start only at the current best touch; deep calm sweeps were already rejected and would confound the test.

**Posterior value at arrival.** For buy direction `d=+1`, proposed integer execution price `p`, and latency `L=0.005s`, approximate:

`S(t+L) | data ~ N(mL, vL)`,

where `mL` is the filtered mean and `vL=P+qL` plus any process-mixture predictive variance if Experiment A is independently adopted. For a sell, mirror all signs below. Define required net edge `c0` (initially the existing take threshold), and:

`z = (mL − p − c0) / sqrt(vL)`.

Then the posterior probability that the price clears the required value is:

`P(edge > c0) = Φ(z)`.

The expected net edge per executed lot, relative to that required value, is simply:

`E[S−p−c0] = mL−p−c0`.

The probability and expected value answer different questions: a high win probability on tiny edges must not outrank a materially larger expected advantage, and a positive-part-only calculation would be invalid here because it would ignore adverse outcomes.

**Execution survival.** Let `s` be the measured probability that a market order at the current opposite best would still find executable market depth at `p` when a shadow IOC matures after 5ms. Under the explicitly testable approximation that arrival survival and posterior value are conditionally independent within each evidence cell:

`Pprof = s Φ(z)`;

`EVlot = s (mL−p−c0)`.

Trade only when both `Pprof` and `q·EVlot` clear the predetermined action thresholds, with actual quantity capped by risk room and currently visible depth. Mirror for sells using `d(p−S−c0)`.

**Measure `s` online without trading.** At a fixed low sampling rate (for example, one candidate per 50ms, not every event), register a virtual buy at the current ask and virtual sell at the current bid. On maturity at `t+5ms`, mark it filled only if visible market depth still crosses that price. Use decayed success/total counts in a deliberately tiny table: side × regime × three signed-standardised-edge bins. The edge bins allow `s` to depend on S moving toward or away from the stale book; assuming one global `s` would be especially suspect during jumps. Depth binning is a later refinement only if the cells have enough observations. This measures the same 5ms book race without affecting the tape.

The Gaussian calculation is a value surrogate, not proof of realised PnL. Pair each shadow candidate with a delayed mid markout matching the intended holding horizon and check whether `EVlot` is calibrated. If the fill event and S movement remain strongly dependent after the three edge bins, replace only `s` by those conditional cell estimates; do not add a large regression or a particle book simulator.

**Minimal ablation sequence.**

| Variant | Sole change from `hmm_mm_v2` |
|---|---|
| B0 | Current fixed-edge, current-touch crossing rule. |
| B1 | Posterior uncertainty/value gate only; set `s=1`. |
| B2 | Shadow-measured execution survival only; retain the point-estimate edge (`vL` not used to alter value). |
| B3 | Joint `Pprof`/`EVlot` gate. |

Do not enable Experiment A during B0–B3. Otherwise a result cannot distinguish better state inference from better execution valuation. First report shadow fill calibration by cell and the joint distribution of fill versus signed future move. Then measure cross count, lots, realised 1s/5s/30s markouts, missed positive-EV opportunities, full PnL and halves. Keep the existing order-acknowledgement and inventory accounting unchanged.

**Cost and failure modes.** A small maturity deque, 2×2×3 decayed cells, and one normal CDF calculation per actual candidate: **O(1) amortised time and small fixed memory**. Shadow candidates are only 20/s at the illustrative sampling rate. Gaussian tails may understate jumps; shadow fill is not a guarantee of full quantity; 5ms book survival does not determine 5–30s exit value; sampling only currently profitable candidates would censor the training distribution. The virtual sampling distribution must therefore be fixed independently of the final gate.

#### Experiment C — marginal inventory opportunity cost without faster liquidation

**Objective.** Test whether the *next same-direction cross* near the cap should have to beat the value of the opportunities it displaces. Do not flatten early, shrink all crosses by current edge, exceed ±100, or alter passive unwinds; those variants already lost.

Define the signed capacity room before a candidate as:

`room = 100 − dI`.

Let `U_d` be a decayed online estimate of the positive excess value of later validated shadow opportunities in direction `d`, using the same delayed outcomes as B:

`U_d ← U_d + α [max(y_d − c0, 0) − U_d]`.

A simple marginal shadow price for the last `R0` lots is:

`λd(I) = g(room) U_d`, with `g(room)=0` for `room≥R0`, otherwise `g(room)=1−room/R0` (clipped to [0,1]).

The proposed entry score is:

`Vtake = q EVlot − q λd(I)`.

This is a one-step approximation to the unknown continuation-value difference `V(I)−V(I+dq)`, not an optimal DP solution. The cap is treated as an option to wait for a better validated opportunity, not as a reason to realise a loss by unwinding a still-profitable position.

**Minimal ablation sequence.**

1. **Diagnostic first:** use B’s fixed shadow stream to measure the distribution of later positive `EVlot`, and join actual blocked opportunities with the existing taken/missed/blocked diagnostic. If blocked candidates do not have positive forward value after execution survival, stop.
2. **C1:** apply `λd` only to same-direction opening crosses with `room<R0`; leave opposite-direction crossings, existing exits, quote prices, and size unchanged.
3. **C2:** compare a fixed extra edge near the cap of the same average magnitude. This is necessary to show that the forward `U_d` estimate adds anything beyond a static threshold.
4. Never begin with a per-inventory DP table or multiple cap levels; that would be a different, much larger experiment.

Report blocked/missed/taken value, subsequent-opportunity value forgone, PnL by regime, inventory quantiles, holding time, and 30s cross markouts. C is lower priority than A/B because the existing evidence says full positions are usually worth keeping and current-edge capacity sizing already lost.

**Cost and failure modes.** Two decayed tail means and a delayed-candidate deque: **O(1) amortised updates**. `U_d` is sensitive to the chosen shadow sampling interval and holding horizon; multiplying a heuristic ramp by an average tail is not exact stochastic control; correlated opportunity bursts can overstate the value of waiting. A negative result should close this line rather than trigger a large DP build.

#### What not to rerun

- **Naïve S grid / full joint particle filter:** exact grid scored 262k/324k versus about 470k KF and had a wrong volatile posterior. [RESEARCH.md:759–775](../../docs/RESEARCH.md#L759-L775)
- **IOC-price or nonexecuting-IOC reconstruction:** prices are not exposed, gaps are ambiguous, and exact-price oracles scored only 248k–377k or worse after push correction. [RESEARCH.md:776–795](../../docs/RESEARCH.md#L776-L795)
- **Generic ID-gap/clamp jump detector:** 415,135 versus 502,258, with 192 alarms for 22 large moves; the research record closes it. [RESEARCH.md:844–857](../../docs/RESEARCH.md#L844-L857)
- **More forced flattening, overshoot beyond ±100, passive signal-side entries, deep calm sweeps, or current-edge position sizing:** all have direct negative Round-2 evidence above.
- **A large HJB, deep model, or particle queue simulator:** none adds an observation or resolves the identified model-fit problem, and all violate the objective of a small, auditable per-event experiment.

### Gaps

- The numerical values of `π`, `J`, clipping bounds, `c0`, shadow sampling interval, survival-cell shrinkage, and `R0` are deliberately not asserted. They require short diagnostics and minimal backtests; choosing them from the same single public score would repeat the overfitting problem.
- `vL` from a single Gaussian posterior will not fully capture sparse jump risk. Experiment B must therefore be evaluated both alone and, only after its isolated result, with A’s predictive variance.
- Shadow execution can faithfully check visible depth after 5ms on the fixed tape, but actual full-size fills, private acknowledgement timing, and inventory races still require the real simulator to validate.
- A posterior S advantage is not the same as realisable PnL at the eventual exit. The correct delayed markout horizon for B/C remains an explicit calibration gap.
- No claim is made that A, B, or C improves PnL. They are ranked, falsifiable mechanisms with isolated ablations; the existing evidence only rules out many easier alternatives.
