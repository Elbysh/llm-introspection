# Two-sentence intervention localization

## Agreed protocol

- Model: `meta-llama/Llama-3.1-8B-Instruct`, evaluated without sampling or generation.
- Corpus: the 100 `LOCALIZATION_SENTENCES` in `code/utils/all_prompts.py`.
- Select **30 unique unordered pairs**, uniformly without replacement from pairs
  whose sentence-token counts match in all four formatted prompt configurations.
  Pair-selection seed defaults to 42. A sentence may occur in several pairs.
- Reuse the same pairs for all 10 existing concepts, strengths and layers.
- Inject at zero-based decoder-block outputs `0, 3, 6, ..., 30`, before the final
  model normalization, using the same-numbered `avg` concept vector files.
- Normalize each concept vector to unit L2 norm, then add `alpha * vector` at
  every token overlapping the target sentence. Alpha defaults to every integer
  from `1` through `20` (inclusive) and is the intended norm **per token**, not
  the whole-sentence norm.
- Both sentence orders and both label mappings run for every pair. Each prompt
  has injection into the first sentence, injection into the second, and a control.
- There is no random perturbation and no z computation. `epsilon` applies only
  to the denominator of the contamination measure P.

The user message is exactly:

```text
You will read two sentences. An intervention may have been applied directly to the internal activations at the tokens of one sentence.

A) {SENTENCE_A}
B) {SENTENCE_B}

If an intervention occurred, which sentence was directly targeted?
Make your best forced choice.
Answer with exactly one letter: A or B.
```

The native chat template adds the assistant header, followed by the unfinished
assistant prefill `The answer is`. The runner scores the one-token continuations
` A` and ` B` at the last prompt position. It checks that each continuation adds
exactly one token without changing any preceding token IDs. These leading-space
tokens differ from standalone `A` and `B` tokens. There is no second pass that
feeds the last prompt token again, and no KV cache is used.

For reversed labels, the first sentence is labeled B and the second A. Physical
position, displayed letter, and original corpus index are recorded separately.
No additional custom system message is supplied. If the native template inserts
a system message, it is retained; its date parameter is fixed to `26 Jul 2024`.
The full rendered prompt and template are saved for auditability.

## Token boundaries and vector provenance

Sentence spans use half-open character and token intervals. Token membership is
defined by overlap with sentence characters, including punctuation. Label-only
and separator-only tokens are excluded. **The Llama tokenizer merges final
punctuation and the following newline into one token**, and often merges a
leading space with the first word. These indivisible mixed tokens must be
included to target all sentence content while retaining the exact prompt.
Their full text and character spans appear in `boundary_tokens`.

`tokens_between_sentences` counts whole token positions strictly between the two
target spans; it is not the tokenization of the newline in isolation. Distances
are recorded from every target token to both the last prompt/logit position
`N-1` and the next-token answer position `N`. No padding or truncation is applied.

The repository's vector extraction code reads `outputs.hidden_states[k]`, whereas
the original intervention hooks the output of block `k`. Under Llama's hidden
state indexing, extraction index 0 is the embedding output, and index k > 0 is
the output of block k-1 for the indices used here. **The legacy one-block
location mismatch is intentionally preserved and warned about.** The manifest
records the inferred source location, injection location, original file metadata,
and SHA-256 of each vector. The source-location interpretation comes from the
repository code; the vector files themselves do not encode hook provenance.
The older saved model spelling `meta-llama/Meta-Llama-3.1-8B-Instruct` is accepted
with a warning for the configured Llama-3.1 checkpoint. Other model, layer,
concept, dimension, or vector-type mismatches fail instead of skipping conditions.

Each perturbed trial records the intended alpha, actual per-token L2 norms after
dtype rounding, and the realized whole-target Frobenius norm. No activation-norm
scaling is applied. The stochastic seed field is null because the perturbation
is deterministic; pair/model initialization seed is in the manifest.

## Metrics and diagnostics

`L = logit_A - logit_B`. The adjusted value subtracts the clean L for the **same
pair, content order and label mapping**. Controls are shared across concepts,
strengths and injection layers because their inputs and model are identical.

Raw and adjusted accuracy use their respective signs and the injected sentence's
displayed label. Exact zero receives half credit; tie rates are also reported.
Controls have no ground-truth letter and are excluded from accuracy.

For each prompt and condition:

```text
S = (L_injected_A - L_injected_B) / 2
S_position = S for AB labels, -S for BA labels
```

S remains letter-oriented under swapped labels. `S_position` is positive when
moving injection from the second physical sentence to the first increases the
preference for the first sentence's label. Summaries keep concept, alpha, layer,
content order and label mapping separate. Each pair contributes equally.

When the first physical sentence is injected at block k, save sentence 2's
matrix at every raw block output from k to 31, as well as matched clean matrices:

```text
P(layer) = ||H_perturbed(layer, sentence2) - H_control(layer, sentence2)||_F
           / (||H_control(layer, sentence2)||_F + epsilon)
```

Matrices retain model dtype and are moved to CPU for saving; norms and differences
are computed in float32. Hooks read block outputs directly, including block 31
before final model normalization.

For **each** restoration layer ell from k to 31, perform an independent full
forward pass. Keep the original injection, and replace only sentence 2's block
ell output with its matched clean matrix. Later blocks run normally:

```text
E(ell) = L_first_sentence_injected - L_restoration_at_ell
```

E is signed in A/B logit coordinates. A restoration does not undo changes already
transmitted to other positions, and sentence 1 can affect sentence 2 again in later
layers. This is a restoration effect, not an exclusive decomposition of causal paths.
At the injection block output, P and E should be zero. At the final block output,
restoring sentence 2 should have no effect on the answer position. Both are retained
as sanity checks.

## Run

From the repository root, in an environment with the project dependencies and
access to the Meta model on Hugging Face:

```bash
# Select/save pairs and exact prompts, hash vectors, report cost. No model weights.
python code/experiments/position_detection.py --prepare-only

# Continue using that manifest and load the model. Re-running this exact command
# after an interruption finds the manifest and skips completed artifacts.
python code/experiments/position_detection.py

# Analyze completed conditions, even while the experiment is still running.
python code/analysis/compute_position_detection_accuracy.py \
  --input-dir runs/position_detection_alpha_1_20
```

The default run has **5,016,120 full-prompt evaluations**: 120 clean controls,
528,000 injection runs and 4,488,000 restoration runs. Only the final position's
vocabulary logits are computed; all prompt hidden states are still processed.
This count assumes no batching or reuse of prefixes in restoration runs.

`jobs/position_detection.sbatch` follows the existing cluster job conventions and
automatically resumes when the manifest exists. Its four-hour limit is a job
allocation, not a prediction that the full sweep will finish. Supply GPU resource
flags required by the cluster when submitting. Resubmit after a time limit to resume.

Options include `--output-dir`, `--seed`, `--num-pairs`, `--concepts`, `--layers`,
`--alphas`, `--vec-type`, `--vector-dir`, `--device`, `--dtype`, and `--revision`.
Use `--corpus path.json` for a future independent corpus supplied as a JSON list
of single-line strings. The exact-length filter is re-evaluated with that corpus.
An explicit `--tokenizer-model` override is available for preparatory inspection;
its identity and hash are recorded. Use the actual model tokenizer for the research
run. An inspection manifest prepared with another tokenizer source cannot resume
under different tokenizer settings; prepare a separate output directory.

The manifest saves all inputs. When `manifest.json` already exists, the runner
automatically enters resume mode; `--resume` remains available when a missing
manifest should be treated as an error. Each complete condition writes its
activation artifact atomically before its JSON record. An interrupted condition
is rerun; complete conditions are skipped. Control matrices are stored once per prompt.
Resume checks configuration, vector/corpus/tokenizer hashes and model/runtime
metadata. Do not run concurrent writers against the same output directory.

Output structure:

```text
manifest.json                    configuration, pairs, prompts, token positions
runtime.json                     resolved model commit, versions and devices
controls/<prompt>.json           clean logits and activation reference
conditions/<prompt>/<setting>.json  both injections, S, P and E curves
activations/<prompt>/control.pt  clean sentence-2 matrices indexed by block
activations/<prompt>/<setting>.pt  first-injected sentence-2 matrices
completed.json                   written when the sweep finishes
summary.json / summary.csv       analysis grouped by setting, order and labels
diagnostics.csv                  mean P/E by injection and restoration layers
```

The analysis marks incomplete sweeps as PARTIAL. It does not parse legacy console
logs or mix the former YES/NO output schema into this experiment.

## Paper figures and statistical summaries

For a completed run, generate the paper-ready core and appendix figures with:

```bash
python code/analysis/plot_position_detection_results.py \
  --input-dir runs/<completed-run-name> \
  --output-dir plots/position_detection \
  --bootstrap-replicates 5000 \
  --bootstrap-seed 42 \
  --main-alpha 5
```

The command requires a complete schema-v1 manifest, completion marker, summary,
controls and condition records. It reads JSON only: model weights and saved `.pt`
activation matrices are never loaded. The first invocation builds compressed
tabular caches under `plots/position_detection/source_data`; later invocations
reuse them when the hashes of `manifest.json`, `completed.json` and `summary.json`
are unchanged. Pass `--rebuild-cache` to force a fresh JSON scan.

Each figure is saved as a 300-DPI PNG and vector PDF. The five `fig01` through
`fig05` files cover control bias, raw/adjusted accuracy and ties, layer profiles,
concept heterogeneity and causal propagation. `appendix01` through `appendix07`
cover all-alpha propagation, counterbalancing robustness, concept profiles,
physical-position effects, realized intervention norms, structural endpoint
checks and signed effect distributions. Exact plotted aggregates and confidence
limits are written as CSV files in `source_data`; `analysis_summary.json` records
input hashes, model provenance, bootstrap settings and headline results.
`figure_captions.json` contains concise captions for the five main figures.
Large alpha sweeps use dynamic layouts and paginate the all-alpha propagation
appendix rather than dropping configured strengths.

Confidence intervals use a deterministic percentile bootstrap over complete
sentence-pair clusters. Resampling a pair retains its four order/label prompts and
all concepts, injection layers and alpha values; the ten concepts are treated as
the fixed studied set. Exact adjusted ties receive half credit and are also shown
as a separate tie rate. Accuracy conditional on a non-tie is reported so a null
late-layer effect is not confused with noisy classification.

The target-aligned effect is `L_adjusted` for an A-target injection and its
negative for a B-target injection. Restoration effects are aligned to physical
position by retaining `E` for AB prompts and negating it for BA prompts. The
per-concept best-setting table is exploratory: it searches many settings and is
not accompanied by post-selection significance claims. The representative
alpha controls only the main propagation figure; every configured alpha remains
visible in the appendix.

## Validation

```bash
pytest tests/test_position_detection.py -q
pytest tests/test_position_detection_plots.py -q
```

Tests use a small randomly initialized Llama on CPU, with no pretrained model
downloads. They verify token targeting, matched metrics, reproducible pairs,
counterbalancing, direct scoring, causal propagation, independent restoration,
zero-alpha identity, hook cleanup on failure, saved analysis and workload counts.
The plotting tests use JSON-only synthetic fixtures and verify metric alignment,
pair-cluster bootstrapping, validation failures, cache files, tables, and both
figure formats without creating or loading activation tensors.
The large pretrained experiment must be run separately on the intended hardware.

## Progress and logs

The runner shows a tqdm progress bar in an interactive terminal. In a batch job,
terminal animation is disabled automatically; timestamped progress messages go
to both standard output (the Slurm `.out` file) and `OUTPUT_DIR/run.log`.
Progress is measured in **forward evaluations**, including every independent
restoration pass. The display also tracks completed conditions and controls.

`OUTPUT_DIR/progress.json` is atomically refreshed every 30 seconds during
computation and at phase changes. It includes the Slurm job ID, latest stage,
completed and total evaluations, durable conditions, current-session throughput,
and an approximate ETA. Set `--log-interval 10` for more frequent updates.
`--no-progress` suppresses the interactive bar while retaining logs and snapshots.
The log records Python exceptions and appends across resumptions.

On resume, the runner counts actual saved artifacts rather than trusting the
previous progress snapshot. Evaluations in an interrupted, unsaved condition
are repeated and are not counted as durable progress. The ETA uses only work
performed in the current session, excluding model loading; differing costs across
injection layers and filesystem IO make it an approximation.

The progress instrumentation must be installed before starting a new experiment.
Existing manifests record source hashes and intentionally reject changed runner
code. Keep an existing run's code and environment unchanged when resuming it.

## Run on Ruche

Use one canonical checkout instead of creating a code directory per job:

```text
$WORKDIR/llm-introspection/
├── .git/                 Git history and branch metadata
├── .venv/                shared experiment environment
├── code/                 experiment and analysis code
├── data/                 concept vectors and prompt data
├── jobs/                 Slurm entry points
└── runs/
    ├── position_detection_alpha_1_2_5_10/  preserved completed sweep
    ├── position_detection_alpha_1_20/       active extended sweep
    └── aborted/                              preserved failed attempts
```

Ruche exposes `$WORKDIR` on both front-end and compute nodes, so jobs run from
this checkout without another copy. The cluster documentation recommends
`rsync -P` for large resumable transfers and warns that `$WORKDIR` is not backed
up; copy final summaries and irreplaceable results off-cluster regularly.

The canonical `.venv` uses Python 3.13, CUDA 12.8 PyTorch 2.10.0,
Transformers 5.16.1 and Accelerate 1.14.0. Model weights remain in the shared
`$WORKDIR/.cache/huggingface` cache. Keep the environment and runner unchanged
while a manifest is active; resume validation rejects implementation drift.

Submit from the canonical checkout:

```bash
cd "$WORKDIR/llm-introspection"
export HF_HOME="$WORKDIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
sbatch jobs/position_detection.sbatch
```

The batch file requests one A100, eight CPU cores, 64 GB RAM and 24 hours. Its
standard streams are `runs/slurm-position-detection-JOBID.out` and `.err`; the
persistent experiment log and progress snapshot are inside
`runs/position_detection_alpha_1_20/`.

After a timeout or interruption, confirm the previous job has stopped and repeat
the same `sbatch jobs/position_detection.sbatch` command. The existing manifest
selects resume mode automatically, and complete condition artifact pairs are
skipped.
