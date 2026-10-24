# Music Assassin — Research Roadmap

Written 2026-09-07. The first roadmap for this repo since the two-repo split
in `LIVE_APP_PLAN.md` (2026-07-02) made it a research lab rather than an app.
It covers what this repo should do next and why, and it is deliberately
scoped to *this* side of the split.

**Companion, not duplicate:** `music-assassin-live/docs/ROADMAP.md` is the
app's plan and is far more advanced. Where the two overlap, that document
wins on anything about routing, the engine, the UI or packaging; this one
wins on models, corpora, metrics and the promotion contract. Read its §2 and
§11 before starting anything here — §11 in particular is a list of measured
dead ends, and re-running any of them is pure waste.

For the master model/technique lookup table see
[COMPREHENSIVE_METHOD_COMPARISON.md](COMPREHENSIVE_METHOD_COMPARISON.md); for
the original split rationale see [LIVE_APP_PLAN.md](LIVE_APP_PLAN.md). This
document is what's next, not what was intended.

---

## 0. Status update — 2026-10-24

**This roadmap was written 2026-09-07 against a dormant repo, with one
correction (`c17266d`, "the OOM was our bug") added the next day. The repo is
no longer dormant, and several items below have been overtaken.** The
sections are kept as written, because the reasoning in them is still the
record of why each candidate was chosen; dated notes mark what has changed.
For the current state of the tree, read [`HANDOVER.md`](../HANDOVER.md)
first — it is the live document and this one is the direction behind it.

**A process note, not a model note.** `c17266d` is the commit above, and its
own PR (#4) shows as MERGED on GitHub — but into the wrong base branch
(`docs/download-links`, which PR #2 had already folded into `main`). The
commit never reached `main` by that route. It reached this branch by a direct
cherry-pick on 2026-10-24, and a second PR re-targeting `main` is needed to
close the loop GitHub's merge button did not. Mentioned here because the same
mistake — a merged-looking PR whose base branch was never `main` — is cheap
to make again and easy to miss, since every other signal says "done."

What changed between 2026-09-08 and 2026-10-24:

| | |
|---|---|
| **`main` moved** | PRs #2, #3, #5 merged (htdemucs crash fix, packaging fixes, crash logging). PR #4 did not, per the note above. `main` had been at `e9327e3` since 2026-07-25. |
| **A fast engine shipped** | `fast_separation.py`: Voc_FT on OpenVINO, BF16 where the CPU supports it. About **58 s per 10 min** of audio end to end on 16 cores, against 220–228 s for the old `audio-separator` path. It is now the app's default separator. |
| **Sound-effects preservation shipped** | "Keep sound effects": MRX (fast) or CDX23 (quality), separated in parallel and added back after the VAD gate. |
| **A quality harness was built here** | `quality_harness/` — true-stem corpora with no separator in the loop, and the metrics to score against them. |

Net effect on the plan below: **R2 is no longer a candidate to screen — it
shipped**, by ear and measured leakage rather than through R7/H1 as planned.
R7.3 (re-test `htdemucs_ft`) has a first real number, but not the §1.4
re-measure R7.1 still asks for. H1 and H2 are partly done, in a way that
contradicts what H2 advised; see the note there. R1, R3, R4, R5, R6 and the
rest of R7 are untouched and still stand.

The one thing this update does **not** change is §3. Keep reading it.

---

## 1. State of play

### 1.1 What this repo is now

> **Superseded 2026-10-24.** The paragraph below described a dormant repo and
> is kept for context. `main` has since taken PRs #2, #3, #5 (see §0 on PR
> #4's base-branch problem), and the active branch
> `feat/fast-engine-keep-effects` carries the fast engine, the keep-effects
> pipeline and the quality harness.

Last commit `e9327e3`, **2026-07-25** — clean, in sync with `origin/main`, and
quiet for six weeks. That is not neglect: the split moved app work out, and
what stayed behind is model research, which has not had a session since.

Shipped here: **v0.4.4** (Demucs + Spleeter + MDX-Net separation, VAD gate,
optional DeepFilterNet cleanup), a PyInstaller build for `.exe`/`.deb`, and a
Windows CI workflow. The last three commits were all packaging, not pipeline.

The one sign of life since: `music_assassin.log` was written 2026-08-13 — the
v0.4.4 app being *used*, not developed.

### 1.2 Where the work actually went

Both sibling repos have been active while this one stood still:

| Repo | State as of 2026-09-07 |
|---|---|
| `Music-Assassin` (this) | `e9327e3`, clean, dormant since 2026-07-25 |
| `music-assassin-live` | v0.1.3 released; **v0.1.4 built but untagged**; `main` 25 commits unpushed; active branch `feat/separator-spike`; uncommitted B4 mask-shaping probe last swept **2026-09-02** |
| `music-assassin-mobile` | Shizuku/`AudioPolicy` path; builds and runs on hardware, **audio sink has never run**; uncommitted work on `feat/audiopolicy-port` |

**The Phase 0 items this repo was assigned never happened here — they
happened over there.** `LIVE_APP_PLAN.md` §8 gave this repo two jobs: extract
the `StreamProcessor` interface, and grow `test_processors.py` into a
card-emitting benchmark harness. The interface now exists and is *implemented*
in `music-assassin-live/assassin_live/processors/base.py`, and the harness got
built as `music-assassin-live/tests/bench_quality.py` — 1,600+ lines with
manufactured ground truth, ~16 metrics, Pareto-frontier marking, regression
gates and `--dump-audio`. It is a better harness than the one this plan asked
for, and it is in the wrong repo for scoring offline separators.

`test_processors.py` here remains what it always was: a headless smoke and
RAM-guard suite (pass/skip/fail, `RAM_FLOOR_GB` before each separator). It has
no ground truth, no metrics and no sweep. **It is not going to become the
quality harness** — that already got built. See H2.

### 1.3 The assets this repo owns

Worth an inventory, because two of these are more valuable than anything in
the tree:

- **The ground-truth corpus** — 122 items × 2 mix ratios = **244 item-ratio
  pairs**, 5 held out, 7.6 GB, on the 4-core machine at
  `~/.local/state/music-assassin/bench/corpus/`. Each item carries
  `vocals.wav`, `music.wav`, `mix_a100/a200.wav` and `ceiling_a100/a200.wav`.
  Categories: `nasheed` 65, `anime_op_stereo` 35, `male_lead` 8,
  `sparse_acoustic` 4, `rap` 3, `orchestral_dialogue` 3, `dual_mono_control` 2,
  `dialogue_score` 2.
- **Its source material**, still intact and so the corpus is rebuildable:
  57 anime clips in `~/Music/MusicAssassin/Corpus/` and 72 nasheed originals
  in `~/Music/MusicAssassin/Corpus-Nasheed/` on the 4-core box, plus a 64-item
  shortlist on the workstation. (The scratchpad excerpt dirs the manifest
  points at are gone; the corpus items themselves are materialised and
  self-contained.)
- **`models/`** — sherpa-onnx UVR set, DTLN, SpeechDenoiser, MetricGAN+,
  gitignored and downloadable.
- **A working research venv**: `assassin_venv_v0.4.4_cpu` (Python 3.12) with
  `audio-separator 0.44.5`, `demucs 4.1.0`, `torch 2.13.0+cpu`,
  `sherpa-onnx 1.13.4`, `onnxruntime 1.28.0`, `librosa 0.11.0`, `soxr 1.1.0`,
  `numpy 2.4.6`. **Not** installed: `pytorch_lightning` (R2 needs it),
  `museval`/`mir_eval` (H2 should decide whether it wants them).

### 1.4 Two machines, and the discipline that requires

Every number in `COMPREHENSIVE_METHOD_COMPARISON.md` was measured on the
**4-core / 11 GB** box (`192.168.100.88`), which is where this repo and the
corpus live, and where htdemucs was OOM-killed at a 6 GB cap. The quality
sweeps of 2026-08-29 and 2026-09-02 ran somewhere else: a **32-core / 30 GB**
workstation.

This is good news and a trap at once.

**Good news:** the entries in §8 of the comparison table marked "high OOM
risk, do not run bare" — BS-RoFormer, MelBand-RoFormer, `htdemucs_ft`,
`mdx_extra`, and MRX (R2) — are screenable on the big machine. That reopens
candidates rejected for a reason that no longer binds everywhere.

**And one of them was never really OOM-risky at all.** On 2026-09-08 the
htdemucs OOM turned out to be an app bug (`segment=CHUNK_SEC` forcing a 60 s
transformer context); fixed, it peaks at 1.24 GB, well inside the 4-core box's
budget. See R7. The lesson generalises: before attributing a resource failure
to a model, check what the app is asking the model to do.

**The trap:** RTF measured on 32 cores says nothing about the 4-core target,
and mixing the two silently corrupts a comparison table whose whole value is
that its rows are comparable.

> **Rule for every benchmark from here on: the big machine screens quality and
> feasibility; the 4-core machine produces every speed and RAM number that
> goes into the comparison table or a model card. Record which machine
> produced each figure, in the row.**

---

## 2. The question this repo exists to answer

From `README.md`'s Known Limitation and the live roadmap's §2, both resting on
the same measurements: **speech enhancers do not remove music.** Every enhancer
tested attenuates noise by −29 to −63 dB and passes prominent music at −0.4 to
−1.7 dB. Only source separation removes music.

The live app has spent since June improving an enhancer chain — mid/side (B1),
stereo rebuild (B3), mask shaping (B4), top-octave recovery (B9). That work is
real and its numbers are sound, but its own roadmap states the ceiling
plainly: *"Everything else on this roadmap makes the app good; this is what
makes it work."*

So the question this repo owns, and the app cannot answer for itself:

> **Is there a separator that is both realtime-capable and actually good?**

Current state of that answer:

| Candidate | RTF | RAM | Status |
|---|---|---|---|
| sherpa-onnx Spleeter 2-stem int8 | **0.067** | ~600 MB | Only RT-capable separator ever found. **Quality never confirmed by ear** (R3). |
| **DEMUCS htdemucs (INT8)** | **0.177** ⚠️ | **~1.24 GB** | **Reinstated 2026-09-08** — previously recorded as OOM-killed and unusable; that was the `segment=60` app bug, not the model (R7). 2.5× faster than the current default in a third of the memory. Not yet A/B'd by ear. ⚠️ measured at 4 threads on the 32-core box. |
| MDX-Net Voc_FT | 0.45 | ~3.3 GB | v0.4.4's default. Good, not realtime. |
| Everything else in §1 of the comparison table | ≥1.4 | 2.8–7 GB | Not realtime. |

And the constraint the live repo's A1 spike measured: Spleeter's chunked
offline API costs **≥1.1 s of latency** regardless of its RTF headroom, which
is a product decision (their Q3), not a tuning detail.

---

## 3. The methodology blocker — the ground truth is demucs-derived ⭐

**This gates R1, R2 and R6, and it should be understood before any of them
start.**

`bench_quality.py --build-refs` manufactures ground truth by running demucs
`--two-stems` on real source audio to obtain a `vocals` and a `music` stem,
then remixing them at a known alpha. Because the mixture was built, the
ground truth is exact *in the sense that matters for the live chain*: we know
what was added and what should come back out.

For ranking realtime models that sit far below demucs, this is sound, and the
harness's own docstring is careful about it.

**It cannot evaluate a candidate that might beat demucs.** The reference
`vocals` stem *is* demucs's output, artifacts included. A model that is right
where demucs is wrong scores as error. The `ceiling` row is demucs on the
remix, so "% of ceiling" is bounded by the same model that defined truth.

Three of the candidates below — `2_HP-UVR`, MRX, the Roformer family — are
exactly the class this method cannot rank. **Fix it with H1 before drawing
conclusions from R2 or R6.** R1 is exempt: it is a *speed* screen, and speed
does not depend on the reference.

> **Update 2026-10-24 — partly addressed, deliberately.** `quality_harness/`
> now scores against stems it builds itself, with no separator in the loop, so
> the circularity is gone for anything it covers. Two caveats keep this section
> open rather than closed:
>
> - Its Tier 1 sources are **synthetic** (harmonic buzz, chord pad, transients).
>   Its own README is explicit that these are "good for regression and relative
>   A/B, not for absolute perceptual claims". H1's recorded-stem datasets are
>   still the thing that would let a candidate be scored absolutely.
> - MRX and CDX23 were adopted on measured leakage plus the user's ear, not on
>   a corpus score. That was the right call for shipping, but it means the
>   promotion contract in H3 still has not been exercised once.

---

## 4. Workstream H — harness, corpus, and the promotion contract

### H1. A true-stem corpus tier ⭐ unblocks §3

> **Partly done 2026-10-24.** `quality_harness/corpus.py` builds true-stem
> mixes with no separator in the loop — Tier 1 (5 synthetic clips) and Tier 2
> (the SFX-leak mix, a rebuild of the `fx_synth.py`/`fx_eval.py` scripts lost
> to a wiped scratchpad). What remains of H1 is the part synthesis cannot give:
> **recorded** stems. The licence check below is still unstarted.

Add a corpus tier built from datasets whose stems are ground truth **by
construction**, with no separator in the loop:

- **MUSDB18-HQ** — real isolated vocals/drums/bass/other.
- **DnR (Divide and Remaster)** — Dialogue / Music / Effects, which is
  precisely this project's target decomposition, and MRX's own test split.

With true stems the mixture is assembled from truth, any separator can be
scored absolutely, and the demucs ceiling becomes one more row rather than the
definition of correct.

This also repairs the corpus gap in §1.3: 100 of 122 items are nasheed or
anime OP, versus **5 items across `orchestral_dialogue` and `dialogue_score`
combined**. The existing corpus is a singing-over-music corpus. MRX's entire
claim is dialogue and SFX preservation, so scoring it on today's corpus would
under-test the thing you would adopt it for.

**Check the licences before committing to either dataset.** MUSDB18 is widely
described as non-commercial, and DnR as CC-licensed; if that holds it affects
both the promotion gate in `LIVE_APP_PLAN.md` §3 and R4's training data
choice. Verify rather than inherit this sentence.

### H2. Extract the metrics and corpus layer into a shared module

> **Overridden 2026-10-23, with a reason — read this before acting on the
> paragraph below.** This section said not to build a harness here. One was
> built anyway: `quality_harness/`. The justification is that the thing H2 told
> us to share is the thing §3 says is broken — `bench_quality.py` scores against
> demucs output as truth, so sharing its corpus layer would have propagated the
> circularity instead of fixing it. `quality_harness/` builds its own stems and
> avoids that.
>
> **The cost H2 warned about is real and is now owed.** There are two harnesses
> across the two repos, and their numbers are not comparable. What stays valid:
>
> - The **seam** described below is still the right one: streaming chain over
>   there, offline full-file separators here, metrics and corpus loader shared.
> - The sharing should now go the **other** direction — lift
>   `quality_harness/metrics.py` and `corpus.py` into the shared module and have
>   the live repo adopt them, rather than this repo adopting `bench_quality.py`.
> - The `museval`/`mir_eval` decision below is **still unmade**, and R2 shipping
>   without published-number comparability is the first bill for that.

Do **not** rebuild `bench_quality.py` here, and do not fork it. Two divergent
copies of a 1,600-line harness is how ground truth stops being comparable
between the repos, and cross-repo comparability is the entire point of the
promotion pipeline.

The seam that actually exists:

- The live repo's driver runs a **streaming chain** — 20 ms blocks, mirroring
  `AudioEngine._work()`, `StreamProcessor` in and out. That is its business.
- This repo needs a driver for **offline, full-file separators** with GB-scale
  torch dependencies. That is our business.
- **The metrics and the corpus loader are common to both** and are what should
  move into a shared, importable module.

Precedent already in the harness: it shells out to a separate interpreter via
`--demucs-python` specifically to keep torch out of the app environment. The
same discipline applies to anything shared — the live app must stay torch-free
and ~100 MB.

Decide while doing this whether to adopt `museval`/`mir_eval` for
field-standard SDR/SIR/SAR, or keep the harness's own implementations. Neither
is installed today. The argument for adopting them is comparability with
published numbers for MRX, Roformer and friends, which is exactly what R2 and
R6 will need to sanity-check their own measurements against.

### H3. Emit `model_card.json` — close the promotion contract

`LIVE_APP_PLAN.md` §3.2 specifies a model card, and the live repo's A3 says
the app hardcodes `_MODEL_FILES` and "the app never reads them."

**Neither repo has ever emitted one.** The actual hand-off is
`music-assassin-live/scripts/import_models.py`: six hardcoded filenames plus a
rename map, copying `.onnx` files out of this repo's `models/` by
`rglob`. The two-repo contract — "drop in a card and weights, no app code
changes" — is aspirational today.

Closing it is two halves, and this repo owns the first: **benchmark runs emit
a card** (name, file, runtime, sample rate, block size, licence, measured
`ms_per_block` / `ram_mb` / `rtf`, and which machine measured them per §1.4).
The live repo's A3 is the other half. Fold this into H2 rather than treating
it as separate work — the harness is what has the numbers.

The `license` field is the open-source gate. It is what already blocks Apollo
(CC-BY-SA) and Bandit/Banquet (non-commercial) from promotion, and it is why
H1's licence check matters.

---

## 5. Workstream R — separator candidates

### R1. `2_HP-UVR` / `Inst_HQ_4` / `kuielab_a` speed screen ⭐ do this first

The cheapest experiment available and the highest information per hour.
`audio-separator 0.44.5` is **already installed** and these models auto-
download, so setup cost is zero and no new infrastructure is needed.

The claim, from the `audio-separator` maintainer's model-selection guide
(Discussion #133), same 3-minute song on an Apple M3 Max: **19 s for
`2_HP-UVR.pth`** vs 36 s `UVR-MDX-NET-Inst_HQ_4.onnx`, 109 s BS-RoFormer,
157 s MDX23C. That is 2–8× faster than everything in §1 of the comparison
table.

**It is somebody else's hardware, so it is a hypothesis, not a measurement.**
Re-measure on the 4-core box per §1.4 or it does not go in the table.

Why it matters more than its "good enough quality" billing suggests: the
realtime gap today is between an RT-capable separator whose quality nobody has
verified (0.067 RTF) and a good one that is 7× too slow (0.45 RTF). A model
landing between them, at real quality, changes what the live app can ship.

Exempt from §3 — a speed screen does not depend on the reference. Quality
ranking of these three waits for H1.

### R2. MERL `cocktail-fork-separation` (MRX) — ✅ shipped 2026-10-23

> **This item is closed, and not by the route below.** MRX is in the app, as
> the "Fast" option of "Keep sound effects" and as a "MRX dialogue" separator
> choice. It went in as an int8 ONNX model, so none of the costs anticipated
> below were paid: `pytorch_lightning` was never needed, CPU feasibility is
> settled (≈16–18 s per 10 min on 4 threads), and the weights are a 31 MB
> download with a recorded sha256 rather than git-LFS.
>
> **Measured, against a ground-truth mix** (lower leak is better):
>
> | Model | Music leak into effects | Effects kept (SI-SDR) |
> |---|---|---|
> | MRX, raw | −16.4 dB | −0.3 dB |
> | MRX + music over-subtraction (β=1, now its default) | **−21.3 dB** | −1.3 dB |
> | CDX23, raw | **−24.9 dB** | **+3.6 dB** |
>
> So CDX23 leaks about 8.5 dB less and keeps effects better; MRX is the fast
> option and still leaks audibly, which is why its UI label says "some music
> bleed". Full method and results: `docs/LIVE_SFX_HANDOVER.md` §2.2.
>
> **It was adopted without H1 and without DnR**, which the text below says to
> sequence first. That was a shipping decision made on measured leakage and the
> user's ear — the same kind of real-measurement-over-table-verdict move R7
> made for htdemucs, just not run through this roadmap's own process. The DnR
> scoring it skipped is still the only thing that would tell us how good it
> actually is against the published numbers — fold it into H1 rather than
> treating the question as settled.

The best *architectural* fit found: trained directly on DnR's Dialogue /
Music / Effects split rather than being a repurposed 4-stem music model, and
**MIT-licensed**, which clears the promotion gate that blocks Bandit/Banquet
and Apollo.

Costs to plan for, none of them fatal but none of them zero:

- Needs `pytorch_lightning`, not in the venv.
- Upstream examples assume GPU; CPU feasibility is undocumented.
- Weights are git-LFS.
- Screen it on the 32-core box first (§1.4). If it is hopeless there, it is
  hopeless.

Sequence it **after H1**, so it is scored on DnR rather than on a nasheed
corpus that does not exercise its strength, and against true stems rather
than against demucs (§3).

Expect it to reach the v0.4.4 file processor long before it could reach the
live app. That is still worth having.

### R3. Confirm sherpa Spleeter int8 by ear

Open since the v0.4.4 documentation pass: *"files saved, comparison
pending."* This is the **only** RT-capable separator ever found on this
project, and nobody has confirmed its separation quality on singing content.

One listening session either kills the candidate or justifies making a
latency-budget decision (the live repo's Q3, still open). Cheap, and it has
been blocking a real decision for two months.

Note the project's own standing rule, earned twice: numbers rank candidates,
they do not settle quality. Synthetic tones scored −47 dB where real music
scores −0.4 dB, and aggressive mid/side exponents introduced musical noise no
energy ratio could see.

### R4. HS-TasNet — training, background track

The architecturally correct answer for the live app and the only item on this
list that could make the product work at low latency: **23 ms by design**,
genuinely streaming, versus the ≥1.1 s the A1 spike measured for chunked
Spleeter. lucidrains' MIT implementation is mature (training code plus a
realtime `sounddevice` example). Training on a free Colab/Kaggle GPU is the
realistic route.

Also the riskiest: no public weights exist partly *because* training it well
is hard, and there is no schedule under which this lands soon.

**Worth starting anyway**, precisely because it is long-running — it should be
accumulating progress in the background while R1–R3 run in the foreground.

Two things to settle early rather than after a training run: whether **DnR
rather than MUSDB** is the better training set (it may be both a better task
fit and a cleaner licence story — H1), and what the resulting weights' licence
status actually is, since the promotion gate applies to them.

RT-STT still has no code release. Check quarterly; do not wait on it.

### R5. Bleed suppressor and de-reverb post-passes — deferred, not rejected

`mel_band_roformer_bleed_suppressor_v1` is purpose-built to suppress residual
bleed in an already-separated vocal stem, which targets this project's
documented core complaint directly. The UVR de-echo/de-reverb family
(`UVR-DeEcho-DeReverb`, `Reverb_HQ_By_FoxJoy`,
`dereverb_mel_band_roformer_anvuew_sdr_19.1729`) addresses dialogue reverb,
which nothing in the pipeline touches today. All are loadable through
`audio-separator`, which is already a dependency.

Deferred for one reason: they are Roformer-class, therefore **offline only**.
They can improve v0.4.4 and can never reach the live app. Do them when you
want a v0.4.5 quality bump, not while the realtime blocker is open.

### R6. Roformer-class and `htdemucs_ft` — reopened by the big machine

Rejected in the comparison table as "high OOM risk on this machine" — a
judgement made on the 4-core box. On the 32-core workstation they are simply
runnable.

Screen for quality only, and read §1.4 before recording anything: their speed
numbers on that machine are meaningless for the target, and their value here
is as a **quality reference point** for H1's true-stem tier — a better ceiling
than demucs — rather than as shippable candidates.

### R7. htdemucs, reinstated — re-measure and A/B by ear ⭐ new 2026-09-08

Added after a crash report from a v0.4.4 release build turned out to be an app
bug with much wider consequences than the crash.

`separate_with_demucs` passed `segment=CHUNK_SEC` (60) to `apply_model`.
`CHUNK_SEC` is the app's *outer* chunk — a bound on the numpy array in RAM.
demucs's `segment` is the model's *internal context window*. htdemucs is a
transformer with a fixed 7.8 s `training_length`, so a 60 s segment made
`apply_model` pad the input to 60 s and `forward` reshape it back to 7.8 s.

Measured same-machine A/B on one 60 s chunk (4 threads, 32-core box):

| | `segment=60` (shipped) | fixed |
|---|---|---|
| result | RuntimeError | OK |
| peak RSS | 14,929 MB | **1,237 MB** |
| wall time | 28.3 s, then crashed | **10.5 s** (RTF 0.177) |

**Consequences beyond the crash.** Attention is quadratic in context length, so
this is also the whole of the ">6–7 GB, still climbing, OOM-killed" figure that
got Demucs written off. The `~1.2 GB` estimate that the benchmark doc dismissed
as "5-6× too low" was right. htdemucs is practical on the 4-core box, and at RTF
0.177 it is ~2.5× faster than MDX-Net Voc_FT, the app's current default.

What remains, in order:

1. **Re-measure on the 4-core box** per §1.4 — the RTF above does not belong in
   the comparison table until it is measured there. Cheapest high-value
   benchmark currently available.
2. **A/B htdemucs against Voc_FT by ear** on this project's own content. The
   quality numbers taken so far (−60.2 dB suppression, −0.02 dB retention) are
   against demucs-derived ground truth and therefore circular — see §3. This
   is the listening test that decides whether the app's default engine changes.
3. **Re-test `htdemucs_ft`**, whose only recorded reason for never being tried
   was the OOM that turned out to be this bug.

A methodological note worth keeping: the benchmark doc had already written down
the decisive fact — that the old estimate "appears to have never been measured
against this exact `apply_model` call shape (`segment=60, overlap=0.25`)" — and
read it as the estimate being wrong rather than the call shape. The evidence was
present and the inference went the wrong way.

> **Update 2026-10-18 — item 3 has a first number, not the measurement asked
> for.** A user ran `htdemucs_ft` in the packaged v0.4.4 app (INT8, CPU) and it
> completed: **VmHWM 3.15 GB**, against the 4-model ensemble's expected ~4×
> plain htdemucs (1.24 GB × 4 ≈ 5 GB — it came in under that, likely from
> staggered sub-model peaks). That is a real data point and it closes the
> "never tried, too risky" question. **It does not satisfy item 1**: the run
> was on the user's desktop with the job log showing 160 threads, not the
> 4-core reference box, so it still cannot enter the comparison table as a
> ranked row. Items 1 and 2 are both still open.

---

## 6. Sequenced plan

> **Rewritten 2026-10-24.** The 2026-09-08 sequence (immediately below the
> line) assumed R7.1–R7.3 and R2 would run through this roadmap's own process.
> Instead R2 shipped directly and R7.3 got a first, unranked number (§6 note
> above); everything else on that list is exactly as open as it was.

**Done, partially or fully, outside this sequence**

- **R2** — MRX and CDX23 shipped. Not scored on DnR; still owed (see R2).
- **R7.3** — `htdemucs_ft` confirmed working at 3.15 GB, not yet on the
  4-core box.
- **H1, synthetic half** — `quality_harness/` Tier 1 and Tier 2.
- Not on the 2026-09-08 plan at all: the fast engine, the keep-effects
  pipeline, crash logging, packaging fixes.

**Next, in order**

1. **Release what is already built.** PR #6 (the fast engine + keep-effects
   branch) is the largest body of unreleased work this repo has held, and the
   published v0.4.4 assets still carry the htdemucs crash R7 exists because of.
   Also re-land PR #4's content on `main` properly (§0) — it is a doc-only fix
   and should not wait on anything else. Run the Windows CI (unrun since the
   OpenVINO and model changes, and it now fails the build if the engine is not
   on OpenVINO), rebuild Linux, merge, tag. **Decide v0.4.4-in-place vs v0.4.5
   first** — the bundle went 640 MB → 890 MB and gained a user-facing feature,
   and identical version strings already caused `apt` to refuse an upgrade
   once.
2. **R7.1 + R1** — unchanged from the 2026-09-08 plan below: both are 4-core
   speed/RAM screens, cheap, and still unrun. Worth running alongside the new
   A2 audits (next item), same machine, same sitting.
3. **Use the harness on the questions it was built for** — the A2 audits:
   the −40 dB pre-gate suspected of crossover distortion, the Normal-mode
   10 kHz low-pass, whether DeepFilterNet and the VAD gate earn their cost,
   and overlap 0.25 vs 0.1. All four are wired and unrun.
4. **R7.2 + R3** — the listening session, as planned: htdemucs vs Voc_FT (now
   also vs the fast engine, which didn't exist when this was written), and
   sherpa Spleeter int8.
5. **H1, recorded half** — MUSDB18-HQ and DnR, licences checked first. This is
   what turns §3 from "partly addressed" into closed, and what would finally
   score MRX, CDX23 and htdemucs against published numbers and each other.
6. **H2 / H3** — shared metrics module (now sharing *outward*, see H2) and
   `model_card.json`. The promotion contract has still never been exercised.

**Background**

- **R4** — HS-TasNet training.
- **Track C**, new since this roadmap: distil a causal "music vs everything
  else" model with CDX23 as teacher. Plan in
  `docs/SIMILAR_PROJECTS_RESEARCH.md` §3.4.

**Deferred**

- **R5** — bleed suppressor and de-reverb post-passes.
- **R6, Roformer** — still unscreened (htdemucs_ft half answered, §6 note).

---

<details>
<summary>The 2026-09-08 sequence, superseded above</summary>

**Track 1 — this week, no new infrastructure**

1. **R7.1** — re-measure fixed htdemucs on the 4-core box. Smallest job on
   this list and it corrects a headline row in the comparison table.
2. **R1** — benchmark `2_HP-UVR`, `Inst_HQ_4`, `kuielab_a` for speed and RAM
   on the 4-core box. Zero setup; `audio-separator` is installed. Run it in the
   same sitting as R7.1 — same harness, same machine, comparable rows.
3. **R7.2 + R3** — one listening session covering both open ear tests:
   htdemucs vs Voc_FT, and sherpa Spleeter int8. Between them they decide the
   app's default engine and a latency-budget question open since July.

**Track 2 — the enabler, ~3–5 days**

4. **H1** — true-stem corpus tier (MUSDB18-HQ + DnR), licences verified first.
   R7.2 raises the priority of this: htdemucs now needs an unbiased score, and
   demucs-derived ground truth cannot give it one.
5. **H2** — extract shared metrics + corpus module; add this repo's offline
   separator driver.
6. **H3** — emit `model_card.json` from benchmark runs (folded into H2).

**Track 3 — after Track 2**

7. **R2** — MRX, screened on the big machine, scored on DnR.
8. **R6** — Roformer/`htdemucs_ft` (now also R7.3) as quality reference points.

**Track 4 — background, starts now**

9. **R4** — HS-TasNet training, running in parallel throughout.

**Deferred**

10. **R5** — bleed suppressor and de-reverb post-passes, for a v0.4.5 offline
   quality pass.

</details>

---

## 7. Open decisions

**Q1. Which task is this repo optimising for — "vocals only" or "dialogue +
SFX, no music"?** These are different products and they rank the candidates
differently, and the evidence is ambiguous.

`README.md` frames the goal as stripping music from anime and movies so
"only dialogue and sound effects remain," which is the DnR/MRX task and makes
SFX preservation the deciding property. But the corpus is 65/122 nasheed, and
`~/Music/Nasheed/` holds **78 `-vocals.m4a` files** — v0.4.4 outputs, i.e.
sustained real use on nasheeds, where the wanted result is voice only and SFX
preservation is irrelevant. For that task a vocals specialist like Voc_FT is
already the right architecture and MRX's advantage largely evaporates.

The honest reading is that both are real and the *dominant measured use* is
nasheeds. This decides how much R2 is worth. **Answer it before Track 3.**

**Q2. What latency ceiling is acceptable?** Inherited from the live repo's Q3
and unanswered there; R1's outcome may change what is achievable, and R4 is
the only route to the low end. ~50–70 ms (lip-sync safe) and ~500 ms+
(comfortable for a chunked non-causal separator) are different designs.

**Q3. Does this repo adopt `museval`/`mir_eval`?** Decided inside H2; the
argument for is comparability with published numbers for MRX and Roformer.

**Q4. Where do shared modules live?** H2 needs an answer: a third package,
a directory in one repo imported by the other, or vendored copies with a sync
script. Vendored copies are the option that has already failed elsewhere in
this project — two copies of `estimate_lag` are how the alignment bug in the
live repo's E2 survived as long as it did.

---

## 8. Dead ends and inherited findings — do not re-explore

**Read `music-assassin-live/docs/ROADMAP.md` §11 first.** It is the fuller
list and everything in it was measured. The entries that bind work planned
here:

- **Any additional speech enhancer.** The category is exhausted — all
  dominated by `dpdfnet2_48khz_hr`. Nothing in Workstream R should be an
  enhancer.
- **Self-quantising UVR models to int8.** 3.7× *slower* (Conv/STFT-bound,
  not MatMul-bound), no RAM saving, 12 % RMS quality loss. Applies directly
  to any temptation to shrink R1's or R2's candidates.
- **`UVR-MDX-NET_Crowd_HQ_1`** for subtraction: over-subtracts, mistakes
  dialogue for crowd noise.
- **Synthetic tones as a music fixture.** They suppress like noise (−47 dB)
  where real music gives −0.4 dB. Any new corpus tier in H1 must use real
  material.
- **Pure-DSP SFX add-back masks.** Plateau at ~8 % music leakage; only a
  semantic classifier (YAMNet, 0.5 %) works, and it is TF-based.
- **OpenVINO for the sherpa path.** ~16 % faster on raw ONNX, but sherpa-onnx
  statically links its own onnxruntime and would need a source rebuild.

From this repo's own record:

- ~~**Demucs `htdemucs` cannot be memory-bounded by shortening its
  segment.**~~ **Disproven 2026-09-08 — this entry was wrong and is kept only
  so it is not re-derived.** The reshape error was real, but its cause was the
  app passing `segment=CHUNK_SEC` (60), forcing a context *far above* the
  model's native 7.8 s rather than below it. Removing the kwarg: peak RSS
  14,929 MB → 1,237 MB, crash → success, RTF 0.177. See R7.
- **`dpdfnet8`** is bandwidth-capped at 8 kHz *and* slower than
  `dpdfnet2_48khz_hr`. Dominated on every axis.

---

## 9. Working notes

**`git commit -m` and `-F` fail over the gvfs/sftp mount** — git writes
`.git/COMMIT_EDITMSG` and cannot read it back. It is the mount, not the repo.
Workaround: `git commit-tree` + `git update-ref`, or commit from the machine
that owns the checkout.

**PyInstaller and `objcopy` also fail over the mount** — `--add-data` rejects
any source path containing a colon, which every gvfs mount point has. Build
from a colon-free path (a symlink is enough).

**Cross-repo loose end:** the live repo has 25 unpushed commits on `main` and
a built-but-untagged v0.1.4 in `dist/`. Not this repo's work, but the longer
it sits the more painful the merge, and H2/H3 both touch that tree.

---

*Written 2026-09-07 · companion to `music-assassin-live/docs/ROADMAP.md`
(2026-08-13) · supersedes §8 (Phases) of `LIVE_APP_PLAN.md` for this repo's
work*
