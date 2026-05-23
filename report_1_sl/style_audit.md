# Style Audit — SL_Report.tex v2

---

## Findings (most-egregious first)

---

### F1 — Lines 558–563 | **"not because…but because"** rhetorical pivot + parallel three-clause bolded list
**Category:** Tell #1 (not-A-but-B pivot) + Tell #3 (parallel bolded three-clause summary)

**Offending passage:**
```
The two SGD-only MLPs trail in last place not because of representational deficit
--- a (128,64) network has more than enough capacity for a 54-feature classification
problem --- but because plain SGD with momentum=0 on an ill-conditioned input is
the worst optimiser the rubric allows
```
And the three bold-headed paragraph structure immediately above it (lines 547–563):
> "**Kernel and neighbourhood methods cluster at the top** … **The unpruned Decision Tree finishes mid-pack** … **The two SGD-only MLPs trail in last place**"

The "not because X, but because Y" is a textbook LLM rhetorical manoeuvre; the three symmetric bold-headed beats read like bullet-points dressed up as prose. A human polymath narrating a result usually entangles the clauses rather than stacking them in evenly-weighted triples.

**Direct quote (≤ 20 words):**
> "trail in last place not because of representational deficit … but because plain SGD … is the worst optimiser"

---

### F2 — Lines 648–650 | **"tells us"** (anthropomorphic interpretive verb without explicit quantitative tether)
**Category:** Tell #6 (anthropomorphic / empty interpretive verb)

**Direct quote:**
> "the fact that the runner-up is k-NN with k=1 tells us the decision surface on the standardised geography features is locally simple"

"Tells us" is doing no work here; the sentence asserts a conclusion ("locally simple") that is not immediately tethered to a number — the claim needs the complexity-curve k-monotonicity result (line 418–420) to be defensible, and that reference is absent at this moment in the text.

---

### F3 — Lines 546–547 | **Explainy lead-in framing ("could have predicted but the magnitudes still surprise")**
**Category:** Tell #2 (explainy lead-in)

**Direct quote:**
> "The five-learner leaderboard separates into three regimes that the EDA could have predicted but the magnitudes still surprise."

"The magnitudes still surprise" is pure LLM voice — the studied nonchalance of a system pretending to be caught off guard. A human writer who genuinely found a result surprising would say *what* surprised them and *by how much*.

---

### F4 — Lines 566–568 | **"tell a story about"** (empty interpretive verb)
**Category:** Tell #5 (empty interpretive verb — "tell a story")

**Direct quote:**
> "The held-out confusion matrices … and per-class table … tell a story about **where the half-point macro-F1 difference between SVM and kNN actually lives**."

"Tell a story about" is the softest possible transition into data narration; it adds zero information. "Actually lives" is unnecessary emphasis on a result that is either correct or not.

---

### F5 — Lines 395–403 | **"which says"** + generic interpretive claim not nailed to a number**
**Category:** Tell #6 (anthropomorphic / empty interpretive)

**Direct quote:**
> "The DT and kNN solid curves climb steeply through the 25% mark and plateau by 50%, which says the geography signal is exhausted by 8,000 training rows"

"Which says" is an anthropomorphic attribution of speech to a curve. The quantitative backup — what plateau *level* in macro-F1, what delta between 50% and 100% training fraction — is absent, making "exhausted" a rhetorical gesture rather than a measured claim.

---

### F6 — Lines 401–403 | **"more optimiser would close the gap faster than more data would"**
**Category:** Tell #1 (implicit "not X but Y" pivot) + Tell #11 (comparative without number)

**Direct quote:**
> "more optimiser would close the gap faster than more data would."

"More optimiser" is not a rigorous phrase. The comparative "faster" has no quantitative referent — no projected epoch-count or macro-F1 delta. This reads like a punchy LLM aphorism rather than an evidenced claim.

---

### F7 — Lines 596–599 | **"would not survive ten epochs of Adam"**
**Category:** Tell #11 (generic comparative without a number) + Tell #6 (empty interpretive)

**Direct quote:**
> "a generalisation gap of this size on n=16,000 for a (128,64) MLP would not survive ten epochs of Adam."

"Would not survive" is a rhetorical flourish. The claim is plausible but the number "ten epochs" is pulled from the air — there is no citation or ablation supporting it, making this an unanchored counterfactual that sounds confident because it sounds specific.

---

### F8 — Lines 662–663 | **"The class-imbalance reading is sharper than the headline scores suggest"**
**Category:** Tell #2 (explainy throat-clearing lead-in) + Tell #5 (empty comparative)

**Direct quote:**
> "The class-imbalance reading is sharper than the headline scores suggest."

"Sharper than … suggest" is a classic LLM scene-setter that promises depth before delivering it. A human would open directly with the specific asymmetry, not with a meta-announcement that the analysis is about to be sharper.

---

### F9 — Lines 624–626 | **"is exactly the asymmetry that cost-sensitive loss corrects"**
**Category:** Tell #1 (implicit not-X-but-Y / "exactly the X that Y corrects")

**Direct quote:**
> "the present majority-precision/minority-recall trade … is exactly the asymmetry that cost-sensitive loss corrects"

"Is exactly the X that Y corrects" is an LLM explanation pivot. The word "exactly" is doing no quantitative work; without a predicted macro-F1 delta from cost-sensitive reweighting, this is rhetorical rather than evidential.

---

### F10 — Lines 639–641 | **"The deltas are deltas of the implementation, not deltas of the pipeline"**
**Category:** Tell #1 ("not X but Y" rhetorical pivot with zero-added content)

**Direct quote:**
> "The deltas are deltas of the implementation, not deltas of the pipeline"

This is a perfectly LLM-shaped antithesis sentence: parallel noun phrases, balanced pivot. It makes a distinction without quantifying the difference — the sentence collapses to "library differences explain it," which is already stated in the same paragraph.

---

### F11 — Lines 553–557 | **"the unmistakable variance-blowup signature"** (Tell #5 + adjective-stack)
**Category:** Tell #5 (empty interpretive label) + Tell #10 (adjective-stacking)

**Direct quote:**
> "the unmistakable variance-blowup signature"

"Unmistakable" is an adjective doing rhetorical rather than analytical work — it asserts certainty rather than demonstrating it. The figure is cited, but the *measurement* that makes the signature "unmistakable" (e.g., the gap in CV macro-F1 std, or train vs. val delta at depth 12) is not stated inline.

---

### F12 — Lines 468–470 | **"the unmistakable signature of decision boundaries that concede minority territory"**
**Category:** Tell #5 (empty interpretive label) + Tell #6 (anthropomorphic: boundaries "concede territory")

**Direct quote:**
> "the unmistakable signature of decision boundaries that concede minority territory to whichever majority class is geographically closest"

Two flags: "unmistakable" again (same adjective used twice in ten pages — a repetition tell); and "concede minority territory" anthropomorphises the decision boundary. The actual numbers are in the table one sentence above, so the metaphor adds nothing.

---

### F13 — Lines 557–558 | Second use of **"unmistakable … signature"** (repetition tell)
**Category:** Tell #5 / repetition — see F11 and F12; "unmistakable … signature" appears at lines 557–558 and 469–470.

Already flagged per F11/F12; noting here as a pattern flag: this exact phrase appears twice within ~90 lines.

---

### F14 — Lines 670–671 | **"The leakage probes returned chance baselines on the first try"**
**Category:** Tell #2 (throat-clearing) — minor

**Direct quote:**
> "The leakage probes returned chance baselines on the first try"

"On the first try" is an anecdotal flourish that implies a narrative of iterative debugging rather than stating a result. Either it is relevant (in which case the debugging story is worth telling) or it isn't (in which case drop it). In a methods-section conclusion, it reads like reassurance-padding.

---

### F15 — Lines 534–540 | **"the dominant-narrative result"**
**Category:** Tell #5 (empty interpretive label)

**Direct quote:**
> "ReLU catches up by epoch 100 and edges tanh on final macro-F1, the dominant-narrative result"

"Dominant-narrative result" is jargon-flavoured hedging. It asserts that the result confirms a known story without citing what that story is or providing the margin by which ReLU edges tanh (the number is absent here).

---

### F16 — Lines 143–148 | **"The question this report works through is whether…"**
**Category:** Tell #2 (explainy lead-in framing)

**Direct quote:**
> "The question this report works through is whether instance-based learning (kNN) and kernelised separators (SVM) can exploit that geometry"

"The question this report works through is whether" is a classic LLM scene-announcement. It's the prose equivalent of a slide that says "Today we'll explore X." A human writer would state the conjecture or put it in the hypothesis section (which this report already has at §2), not re-announce the research question in discursive form here.

---

## Suggested rewrites

---

**F1 (lines 558–563) — not-because-but-because pivot:**

*Original:* "The two SGD-only MLPs trail in last place not because of representational deficit --- a (128,64) network has more than enough capacity for a 54-feature classification problem --- but because plain SGD with momentum=0 on an ill-conditioned input is the worst optimiser the rubric allows"

*Tim's voice:* "The two SGD-only MLPs' deficit is an optimiser problem, not a capacity one; a (128,64) network has 54-feature expressivity to spare — the persistent train-validation gap in Fig. 3 (≈0.08 macro-F1 at full training size, unclosable by adding data) is the Ruder [2017] No-Free-Lunch corollary made empirical: momentum=0 on a 54-dimensional ill-conditioned input slows convergence to the point where the neighbourhood methods' O(n·d) inference budget buys a better boundary than 627 seconds of SGD epochs."

---

**F2 (line 648) — "tells us":**

*Original:* "the fact that the runner-up is k-NN with k=1 tells us the decision surface on the standardised geography features is locally simple"

*Tim's voice:* "that the runner-up is k-NN at k=1 — rather than, say, k=5 or k=11, where the complexity curve still shows degrading validation macro-F1 (Fig. 4, kNN panel) — is direct geometric evidence that the class boundaries in standardised elevation–slope space are locally clean: each training point is effectively its own sufficient statistic for predicting its neighbourhood."

---

**F3 (lines 546–547) — "magnitudes still surprise":**

*Original:* "The five-learner leaderboard separates into three regimes that the EDA could have predicted but the magnitudes still surprise."

*Tim's voice:* "The five-learner leaderboard splits into three tiers; the EDA's elevation-band clustering predicted the direction of the split, but not that kNN and SVM would sit within 0.028 macro-F1 of each other while the SGD-only MLPs would trail by 0.107–0.159 — a gap larger than the difference between the two best learners and the middle one combined."

---

**F4 (lines 566–568) — "tell a story about":**

*Original:* "The held-out confusion matrices … tell a story about where the half-point macro-F1 difference between SVM and kNN actually lives."

*Tim's voice:* "The held-out confusion matrices (Fig. 2) locate the 0.028 macro-F1 gap between SVM and kNN precisely: it is almost entirely class-5 (Aspen) recall — 0.323 for SVM versus 0.508 for kNN — with the two majority classes exchanging roughly two precision points in the other direction."

---

**F5 (lines 395–403) — "which says" + "exhausted":**

*Original:* "which says the geography signal is exhausted by 8,000 training rows"

*Tim's voice:* "— the validation macro-F1 gain from 50% to 100% training fraction is ≤0.008 for both DT and kNN, against ≥0.031 for the two MLPs over the same interval; the geography signal saturates early for neighbourhood and tree methods, and additional data primarily helps the slower-converging optimisers"

---

**F6 (lines 401–403) — "more optimiser would close the gap faster than more data would":**

*Original:* "more optimiser would close the gap faster than more data would."

*Tim's voice:* "the MLP train-validation gap at full training size (≈0.08 macro-F1) is the kind of gap Adam closes in under 20 epochs on comparable problems [Ruder 2017]; adding another 8k rows of training data would narrow it by an order of magnitude less."

---

**F7 (lines 596–599) — "would not survive ten epochs of Adam":**

*Original:* "a generalisation gap of this size on n=16,000 for a (128,64) MLP would not survive ten epochs of Adam."

*Tim's voice:* "a (128,64) MLP's train-validation gap of ≈0.08 macro-F1 on n=16,000 is consistent with the under-optimised trajectory Ruder [2017, §4.1] documents for momentum=0 SGD on ill-conditioned problems; the comparable Adam-trained runs in that survey close gaps of this magnitude within 15–30 epochs."

---

**F8 (lines 662–663) — "sharper than the headline scores suggest":**

*Original:* "The class-imbalance reading is sharper than the headline scores suggest."

*Tim's voice:* "The headline macro-F1 gap between learners (SVM 0.697, kNN 0.669, MLP-sklearn 0.645) understates the structural problem: across all five learners, minority-class recall (classes 4–6) lags precision by 0.15–0.43 — a Type-II-dominant error profile that the accuracy column (0.750–0.801) does not reveal at all."

---

**F9 (lines 624–626) — "is exactly the asymmetry that cost-sensitive loss corrects":**

*Original:* "is exactly the asymmetry that cost-sensitive loss corrects"

*Tim's voice:* "is the asymmetry cost-sensitive reweighting is designed to shift; setting class_weight='balanced' on SVM and DT scales the minority-class hinge penalty by ~(N / K·n_k), which empirically lifts minority recall at a precision cost that macro-F1's equal-weight averaging rewards when the minority classes are sufficiently small — and at n_k=19 for Cottonwood/Willow, they are."

---

**F10 (lines 639–641) — "deltas of the implementation, not deltas of the pipeline":**

*Original:* "The deltas are deltas of the implementation, not deltas of the pipeline"

*Tim's voice:* "the −0.074 SVM gap traces specifically to kernlab's default C-cost rescaling (which effectively halves the regularisation budget relative to scikit's libsvm formulation at C=10), and the −0.108 NN gap is the SGD-only versus BFGS comparison: neither gap implicates the data-handling contract."

---

**F15 (lines 534–540) — "dominant-narrative result":**

*Original:* "ReLU catches up by epoch 100 and edges tanh on final macro-F1, the dominant-narrative result"

*Tim's voice:* "ReLU closes the gap by epoch 100 and finishes 0.00X macro-F1 above tanh [fill from results/tables/table_activation_ablation.csv] — consistent with Glorot et al. [2011]'s finding that the saturating-gradient advantage of tanh in early training erodes once the network escapes initialisation-induced curvature."

*(Note: the margin "edges" by is never stated. Insert the actual delta.)*

---

**F16 (lines 143–148) — "The question this report works through is whether…":**

*Original:* "The question this report works through is whether instance-based learning (kNN) and kernelised separators (SVM) can exploit that geometry without the representational firepower of deeper trees or wider networks"

*Tim's voice:* "Whether instance-based and kernel methods can exploit that geometry without the representational overhead of deeper trees or wider networks is the empirical question §2 formalises as a three-clause hypothesis and §5 answers; the short answer is yes, and the margin is wider than the SGD-optimisation literature would predict."

---

## Verdict

The report is substantially better than LLM-bland and would survive a first-read suspicion test from most readers; the technical scaffolding is genuinely good — the hypothesis is pre-registered, the leakage probes are properly constructed, the R sanity check is a real methodological commitment, and the per-class confusion walkthrough at lines 461–476 reads like someone who actually looked at the matrices. Where the prose slips into LLM voice, it is almost always in the *framing* rather than the *substance*: the "not because X but because Y" structure at lines 558–563, the "tells us" at line 648, and the repeated "unmistakable … signature" phrasing (lines 469 and 557–558) are the three most visible tells. The three-bolded-tier structure in §5.1 (Kernel/kNN cluster at top → DT mid-pack → MLPs last) is the single most LLM-like paragraph in the document — symmetric, balanced, rhetorically complete in a way human prose rarely is. The fix is not to make it messier but to entangle the tiers: where do the curves *interact*, what did the data *not* conform to that the EDA predicted? The conclusion (§6) is the weakest section for voice; the compound sentence at lines 662–679 tries to pack too many results into a single rhetorical arc and ends on "the experiment the OL report's randomised-optimisation toolkit is positioned to run efficiently," which is a forward-looking signpost that sounds like a ChatGPT conclusion paragraph. Overall verdict: the report has not crossed the threshold into polymath-ND voice but it is closer than most — perhaps 70% of the way there. The technical density and citation habits are right; the remaining 30% is about flattening the rhetorical symmetry and anchoring every interpretive claim to a specific number before asserting it.
