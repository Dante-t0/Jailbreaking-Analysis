"""Build generate_scores.ipynb and analysis.ipynb from Python sources.

Keeping the cell text here (rather than hand-editing JSON) makes the notebooks
easy to regenerate and review. Run once: `python _build_notebooks.py`.
"""

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def md(src):
    return new_markdown_cell(src)


def code(src):
    return new_code_cell(src)


# ======================================================================== #
# analysis.ipynb
# ======================================================================== #

analysis_cells = [
    md(
        "# Jailbreak scale: does attack success change with model size?\n"
        "\n"
        "This notebook answers the question from the DSC 261 project proposal:\n"
        "\n"
        "> If we fix the model family and change only its size, does attack success go up, down, or\n"
        "> stay flat, and does the answer differ by attack type?\n"
        "\n"
        "We hold the family fixed (Qwen2.5-Instruct, 0.5B to 32B) and read a table of StrongREJECT\n"
        "scores produced by the generation stage (`generate_scores.ipynb`). This notebook is only the\n"
        "analysis; it never runs an attack or a model.\n"
        "\n"
        "**Lift.** For a behavior, size, and attack, lift is the StrongREJECT score with the attack\n"
        "minus the score for the same behavior and size with no attack. A size trend in lift reflects\n"
        "the attack itself, not larger models simply being more capable.\n"
        "\n"
        "**Hypotheses** (each about the slope of lift on `log(parameters)`):\n"
        "\n"
        "- **H0 (null):** the slope is zero for every attack.\n"
        "- **H1:** the persuasion (PAP) and multi-turn (Crescendo, or PAIR) slopes are positive.\n"
        "  Failing to reject H0 for PAP is evidence against the Liu and Lin persuasion claim.\n"
        "- **H2:** the DAN slope is negative, because larger models better recognize a fixed public\n"
        "  template.\n"
        "\n"
        "**Decision rule.** We report every slope with a 95% behavior-bootstrap interval. We call an\n"
        "attack *up* / *down* / *flat* by whether that interval is entirely above zero, entirely below\n"
        "zero, or straddles it. H1/H2 also require a Holm-corrected two-sided test (family-wise 5%)\n"
        "across the three pre-registered slopes. A flat-but-high arm is distinguished from a\n"
        "flat-near-zero arm (the proposal's \"attacks fail at every size\" case) using mean lift."
    ),
    code(
        "from jbscale import analysis as A, config\n"
        "import pandas as pd, numpy as np, matplotlib.pyplot as plt\n"
        "import os\n"
        "\n"
        "pd.set_option('display.width', 120)\n"
        "pd.set_option('display.max_columns', 20)\n"
        "\n"
        "# Analysis settings.\n"
        "N_BOOT = 1000          # behavior cluster-bootstrap draws\n"
        "SEED = config.SEED\n"
        "SIZES = [(m.key, m.n_params) for m in config.TARGET_MODELS]\n"
        "print('Qwen2.5-Instruct sizes:', SIZES)\n"
        "print('Attacks:', config.ATTACKS, '| adaptive arm:', config.ADAPTIVE_ARM)\n"
        "os.makedirs(config.PATHS.figures, exist_ok=True)"
    ),
    md(
        "## Load scores, or fall back to a labeled synthetic table\n"
        "\n"
        "If `results/scores.csv` exists we analyze the real Qwen runs. Otherwise we synthesize a table\n"
        "with a *known* planted pattern (PAP rising, DAN falling, the adaptive arm flat) so the whole\n"
        "pipeline runs and can be self-checked. On synthetic data every output is a pipeline check, not\n"
        "a finding about Qwen2.5, and we never write the synthetic table to `scores.csv`."
    ),
    code(
        "scores = A.load_scores()\n"
        "USING_SYNTHETIC = scores is None\n"
        "if USING_SYNTHETIC:\n"
        "    scores = A.synthetic_scores(seed=SEED)\n"
        "    banner = ('SYNTHETIC DATA - pipeline check only. '\n"
        "              'No results/scores.csv found, so these numbers do NOT describe Qwen2.5.')\n"
        "else:\n"
        "    banner = f'REAL DATA from {config.PATHS.scores_csv}'\n"
        "print(banner)\n"
        "\n"
        "val = A.validate(scores)\n"
        "print('Validation ok:', val.ok)\n"
        "for m in val.messages:\n"
        "    print('  -', m)\n"
        "scores.head()"
    ),
    md(
        "## Lift and level summaries\n"
        "\n"
        "Lift is taken against each behavior's own no-attack baseline at the same size. We also report\n"
        "mean StrongREJECT, refusal rate, benign over-refusal, and (for real data) attack-failure rate,\n"
        "so a flat-high arm can be told apart from a flat-near-zero one."
    ),
    code(
        "lift = A.compute_lift(scores)\n"
        "lift_levels = A.lift_by_size(lift, n_boot=N_BOOT, seed=SEED)\n"
        "print('Mean lift by attack and size:')\n"
        "lift_levels.pivot(index='n_params', columns='attack', values='mean_lift').round(3)"
    ),
    code(
        "refusal = A.refusal_by_size(scores)\n"
        "over_refusal = A.over_refusal_by_size(scores)\n"
        "failures = A.attack_failure_by_size(scores)\n"
        "print('Refusal rate by attack and size:')\n"
        "display(refusal.pivot(index='n_params', columns='attack', values='refusal_rate').round(3))\n"
        "print('\\nBenign over-refusal by size:')\n"
        "display(over_refusal.round(3))\n"
        "if not failures.empty and failures['attack_failure_rate'].sum() > 0:\n"
        "    print('\\nAttack-failure rate by attack and size:')\n"
        "    display(failures.pivot(index='n_params', columns='attack', values='attack_failure_rate').round(3))"
    ),
    md(
        "## Size slopes, intervals, and the Holm-corrected tests\n"
        "\n"
        "For each attack we fit lift on `log(parameters)` with a random intercept per behavior (point\n"
        "estimate), and take a 95% interval and a two-sided p-value from a behavior cluster bootstrap.\n"
        "Holm correction is applied across the three slopes. `holm_reject` is the corrected decision;\n"
        "`ci_excludes_zero` is the raw per-attack call used for the up/down/flat direction."
    ),
    code(
        "slopes = A.slope_table(lift, n_boot=N_BOOT, seed=SEED)\n"
        "slopes.round(4)"
    ),
    md(
        "## Figures\n"
        "\n"
        "The main figure is lift against `log(parameters)`, one line per attack, with per-size\n"
        "bootstrap intervals. We also plot refusal and benign over-refusal by size."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "for attack, g in lift_levels.groupby('attack'):\n"
        "    g = g.sort_values('n_params')\n"
        "    x = np.log(g['n_params'])\n"
        "    ax.plot(x, g['mean_lift'], marker='o', label=attack)\n"
        "    ax.fill_between(x, g['ci_low'], g['ci_high'], alpha=0.15)\n"
        "ax.axhline(0, color='gray', lw=0.8, ls='--')\n"
        "ax.set_xlabel('log(parameters)')\n"
        "ax.set_ylabel('StrongREJECT lift over no-attack baseline')\n"
        "title = 'Lift vs model size by attack'\n"
        "if USING_SYNTHETIC:\n"
        "    title += '  [SYNTHETIC - pipeline check]'\n"
        "ax.set_title(title)\n"
        "ax.legend(title='attack')\n"
        "fig.tight_layout()\n"
        "fig.savefig(os.path.join(config.PATHS.figures, 'lift_vs_size.png'), dpi=150)\n"
        "plt.show()"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "for attack, g in refusal.groupby('attack'):\n"
        "    g = g.sort_values('n_params')\n"
        "    ax.plot(np.log(g['n_params']), g['refusal_rate'], marker='o', label=attack)\n"
        "ax.set_xlabel('log(parameters)')\n"
        "ax.set_ylabel('refusal rate')\n"
        "t = 'Refusal rate vs size by attack' + ('  [SYNTHETIC]' if USING_SYNTHETIC else '')\n"
        "ax.set_title(t)\n"
        "ax.legend(title='attack')\n"
        "fig.tight_layout()\n"
        "fig.savefig(os.path.join(config.PATHS.figures, 'refusal_vs_size.png'), dpi=150)\n"
        "plt.show()"
    ),
    code(
        "if not over_refusal.empty:\n"
        "    fig, ax = plt.subplots(figsize=(7, 5))\n"
        "    g = over_refusal.sort_values('n_params')\n"
        "    ax.plot(np.log(g['n_params']), g['over_refusal_rate'], marker='o', color='firebrick')\n"
        "    ax.set_xlabel('log(parameters)')\n"
        "    ax.set_ylabel('benign over-refusal rate')\n"
        "    t = 'Over-refusal vs size' + ('  [SYNTHETIC]' if USING_SYNTHETIC else '')\n"
        "    ax.set_title(t)\n"
        "    fig.tight_layout()\n"
        "    fig.savefig(os.path.join(config.PATHS.figures, 'over_refusal_vs_size.png'), dpi=150)\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No benign rows; skipping over-refusal figure.')"
    ),
    md(
        "## Conclusion\n"
        "\n"
        "The cell below states, per attack, whether success goes up, down, or stays flat as size grows,\n"
        "and whether H1 and H2 are supported. On synthetic data it only confirms the pipeline recovers\n"
        "the planted pattern; it makes no claim about Qwen2.5."
    ),
    code(
        "v = A.verdict(slopes, lift_levels)\n"
        "print(banner)\n"
        "print('=' * 72)\n"
        "for attack, info in v['per_attack'].items():\n"
        "    lo, hi = info['ci']\n"
        "    extra = f\" ({info['note']})\" if info['note'] else ''\n"
        "    print(f\"{attack:>10}: {info['direction']:<4} slope={info['slope']:+.4f} \"\n"
        "          f\"[{lo:+.4f}, {hi:+.4f}] Holm-reject={info['holm_reject']} \"\n"
        "          f\"mean_lift={info['mean_lift']:+.3f}{extra}\")\n"
        "print('=' * 72)\n"
        "print('H1 (PAP and adaptive grow stronger with size):', v['H1_supported'])\n"
        "print('H2 (DAN weakens with size):', v['H2_supported'])\n"
        "\n"
        "if USING_SYNTHETIC:\n"
        "    print('\\nThis is synthetic. The question about Qwen2.5 is not answered until '\n"
        "          'generate_scores.ipynb produces results/scores.csv.')\n"
        "else:\n"
        "    dirs = {a: i['direction'] for a, i in v['per_attack'].items()}\n"
        "    print('\\nQwen2.5 answer -- attack success vs size by attack type:')\n"
        "    for a, d in dirs.items():\n"
        "        print(f'  {a}: {d}')"
    ),
    md(
        "### Synthetic self-check\n"
        "\n"
        "When running on synthetic data, assert that the analysis recovers the planted pattern: PAP up\n"
        "(and Holm-significant), DAN down (and Holm-significant), and the adaptive arm's interval\n"
        "covering zero. This guards the pipeline; it is skipped on real data."
    ),
    code(
        "if USING_SYNTHETIC:\n"
        "    s = slopes.set_index('attack')\n"
        "    ad = config.ADAPTIVE_ARM\n"
        "    assert s.loc['pap', 'ci_low'] > 0 and bool(s.loc['pap', 'holm_reject']), 'PAP should be up'\n"
        "    assert s.loc['dan', 'ci_high'] < 0 and bool(s.loc['dan', 'holm_reject']), 'DAN should be down'\n"
        "    assert s.loc[ad, 'ci_low'] <= 0 <= s.loc[ad, 'ci_high'], 'adaptive arm should be flat'\n"
        "    print('Synthetic self-check passed: planted PAP-up / DAN-down / adaptive-flat recovered.')\n"
        "else:\n"
        "    print('Real data: synthetic self-check skipped.')"
    ),
]


# ======================================================================== #
# generate_scores.ipynb
# ======================================================================== #

generate_cells = [
    md(
        "# Generate StrongREJECT scores for the six Qwen2.5 sizes\n"
        "\n"
        "This notebook produces `results/scores.csv`, the input to `analysis.ipynb`. For every\n"
        "harmful behavior it runs four arms against each Qwen2.5-Instruct size and grades the reply\n"
        "with the StrongREJECT rubric judge:\n"
        "\n"
        "- `none` - the behavior as written (baseline for lift)\n"
        "- `dan` - a fixed in-the-wild DAN template (Shen et al.), selected by ID\n"
        "- `pap` - persuasion rewrite from the `strong_reject` package\n"
        "- `crescendo` - PyRIT's multi-turn Crescendo (or PAIR fallback; see config)\n"
        "\n"
        "**Attack text is never authored here.** DAN, PAP, Crescendo, and PAIR all come from published\n"
        "research code. This notebook only wires them to the target models and the judge.\n"
        "\n"
        "**Safety / terms.** This assumes the academic-review approval the user stated. Running these\n"
        "published attacks against models you host on Together for safety evaluation is consistent\n"
        "with Together's documented red-teaming use; the judge call is ordinary OpenAI API usage.\n"
        "Confirm both providers' current terms before a full run. Raw prompts and replies are written\n"
        "under `results/raw/` (git-ignored); only `scores.csv` and figures are shared.\n"
        "\n"
        "**Cost.** The six sizes run on Together *dedicated* endpoints that bill by the hour (serverless\n"
        "has no Qwen2.5). Deploy each endpoint out of band, set its model string in the environment\n"
        "(see the config cell), run smallest-first, and tear endpoints down afterwards."
    ),
    code(
        "import os, sys, json, time, platform\n"
        "import jbscale\n"
        "from jbscale import config\n"
        "\n"
        "# Required credentials. Targets + attacker use Together; the judge uses OpenAI;\n"
        "# the gated DAN dataset needs a Hugging Face token.\n"
        "need = [config.TOGETHER_API_KEY_ENV, config.OPENAI_API_KEY_ENV, config.HF_TOKEN_ENV]\n"
        "missing = [k for k in need if not os.environ.get(k)]\n"
        "print('Python:', sys.version.split()[0], '|', platform.platform())\n"
        "for k in need:\n"
        "    print(f'  {k}: ' + ('set' if os.environ.get(k) else 'MISSING'))\n"
        "if missing:\n"
        "    print('\\nSet the missing variables before running the smoke test or full run.')\n"
        "HAVE_KEYS = not missing"
    ),
    md(
        "## Configuration\n"
        "\n"
        "Everything that defines the run lives in `jbscale/config.py`. The key thing to set per\n"
        "environment is the deployed Together model string for each size, via `JBSCALE_TARGET_<KEY>`\n"
        "environment variables (e.g. `JBSCALE_TARGET_32B`) or `TARGET_ENDPOINT_OVERRIDES`. The 32B\n"
        "model is deployable from Together's dedicated catalog; the smaller sizes are uploaded once\n"
        "from their Hugging Face repos (see `jbscale/config.py` for the repo IDs)."
    ),
    code(
        "for m in config.TARGET_MODELS:\n"
        "    print(f'{m.key:>5}  n_params={m.n_params:>14,}  -> {config.target_model_string(m)}')\n"
        "print('\\nAttacker model:', config.ATTACKER_MODEL)\n"
        "print('Judge:', config.JUDGE_EVALUATOR, config.JUDGE_MODELS)\n"
        "print('PAP techniques:', config.PAP_TECHNIQUES)\n"
        "print('Adaptive arm:', config.ADAPTIVE_ARM,\n"
        "      '(Crescendo max_turns=%d, max_backtracks=%d)' % (config.CRESCENDO_MAX_TURNS, config.CRESCENDO_MAX_BACKTRACKS))"
    ),
    md(
        "## Smoke test\n"
        "\n"
        "Before the expensive full run, exercise the whole path on a few behaviors and the smallest\n"
        "model only: load data, build each attack, query the target, and grade. This confirms the\n"
        "endpoint responds and the judge returns scores, and prints a rough cost estimate. It needs\n"
        "the 0.5B endpoint deployed and the keys set."
    ),
    code(
        "if HAVE_KEYS:\n"
        "    from jbscale.data import load_behaviors\n"
        "    from jbscale.runner import prepare_pap_rewrites, run_model\n"
        "    from jbscale.store import ResultStore\n"
        "\n"
        "    smoke_target = next(m for m in config.TARGET_MODELS if m.key == config.SMOKE_TARGET_KEY)\n"
        "    harmful = load_behaviors('harmful', limit=config.SMOKE_N_BEHAVIORS)\n"
        "    benign = load_behaviors('benign', limit=config.SMOKE_N_BEHAVIORS)\n"
        "    print(f'Smoke: {len(harmful)} harmful + {len(benign)} benign on {smoke_target.key}')\n"
        "\n"
        "    smoke_store = ResultStore(os.path.join(config.PATHS.raw, 'smoke_records.jsonl'))\n"
        "    pap_cache = prepare_pap_rewrites(harmful,\n"
        "                                     store_path=os.path.join(config.PATHS.raw, 'smoke_pap.jsonl'))\n"
        "    t0 = time.time()\n"
        "    run_model(smoke_target, harmful, benign, pap_cache, smoke_store)\n"
        "    print('Smoke run finished in %.1fs' % (time.time() - t0))\n"
        "    import pandas as pd\n"
        "    df = pd.DataFrame(smoke_store.records())\n"
        "    display(df.groupby('attack')['strongreject'].mean())\n"
        "else:\n"
        "    print('Keys missing: skipping smoke test.')"
    ),
    code(
        "# Rough cost estimate for the full run (very approximate; refine with smoke timings).\n"
        "n_sizes = len(config.TARGET_MODELS)\n"
        "n_harmful = config.N_BEHAVIORS\n"
        "arms = len(config.ATTACKS)\n"
        "target_calls = n_sizes * n_harmful * arms + n_sizes * n_harmful  # +benign baseline\n"
        "print('Target completions (excluding multi-turn/attacker overhead):', target_calls)\n"
        "print('Judge calls (harmful arms + benign):', n_sizes * n_harmful * (arms) + n_sizes * n_harmful)\n"
        "print('Note: Crescendo/PAIR add several attacker+target calls per behavior.')\n"
        "print('Note: dedicated endpoints also bill per hour of uptime, not just per token.')"
    ),
    md(
        "## Prepare PAP rewrites once\n"
        "\n"
        "Persuasion rewrites are target-independent, so they are generated a single time with the\n"
        "fixed attacker model and reused at every size. Rewrites are cached in `results/raw/` and the\n"
        "attacker's refusal rate is reported."
    ),
    code(
        "if HAVE_KEYS:\n"
        "    from jbscale.data import load_behaviors\n"
        "    from jbscale.runner import prepare_pap_rewrites, attacker_refusal_rate\n"
        "    harmful_full = load_behaviors('harmful', limit=config.N_BEHAVIORS)\n"
        "    pap_cache_full = prepare_pap_rewrites(harmful_full)\n"
        "    print('PAP rewrites cached:', len(pap_cache_full))\n"
        "    print('Attacker refusal rate: %.3f' % attacker_refusal_rate(pap_cache_full))\n"
        "else:\n"
        "    print('Keys missing: skipping PAP preparation.')"
    ),
    md(
        "## Full run across the six sizes\n"
        "\n"
        "Runs every arm for all 100 harmful behaviors plus the benign baseline, smallest size first.\n"
        "The run is resumable: finished (behavior, size, attack) cells are skipped, so you can deploy\n"
        "one endpoint, run its size, tear it down, and move to the next. **This spends money on\n"
        "Together and OpenAI** and is intended to be started deliberately by the team, not re-run on\n"
        "every notebook execution."
    ),
    code(
        "RUN_FULL = False  # set True once endpoints are deployed and you intend to spend\n"
        "\n"
        "if RUN_FULL and HAVE_KEYS:\n"
        "    from jbscale.data import load_behaviors\n"
        "    from jbscale.runner import run_model\n"
        "    from jbscale.store import ResultStore\n"
        "    store = ResultStore()\n"
        "    benign_full = load_behaviors('benign', limit=config.N_BEHAVIORS)\n"
        "    for target in config.TARGET_MODELS:  # already smallest-first\n"
        "        print('=== running', target.key, '===')\n"
        "        t0 = time.time()\n"
        "        run_model(target, harmful_full, benign_full, pap_cache_full, store)\n"
        "        print('  done in %.1fs' % (time.time() - t0))\n"
        "    print('All sizes complete.')\n"
        "else:\n"
        "    print('RUN_FULL is False (or keys missing). Not running the paid full sweep.')"
    ),
    md(
        "## Export scores.csv and a run manifest\n"
        "\n"
        "Flatten the stored records into the long `results/scores.csv` the analysis notebook reads, and\n"
        "write a manifest recording package versions, the attacker model, and the attacker refusal rate\n"
        "for reproducibility."
    ),
    code(
        "if RUN_FULL and HAVE_KEYS:\n"
        "    path = store.export_scores()\n"
        "    import importlib.metadata as im\n"
        "    def ver(p):\n"
        "        try:\n"
        "            return im.version(p)\n"
        "        except Exception:\n"
        "            return 'unknown'\n"
        "    manifest = {\n"
        "        'attacker_model': config.ATTACKER_MODEL,\n"
        "        'judge_models': list(config.JUDGE_MODELS),\n"
        "        'adaptive_arm': config.ADAPTIVE_ARM,\n"
        "        'pap_techniques': list(config.PAP_TECHNIQUES),\n"
        "        'attacker_refusal_rate': attacker_refusal_rate(pap_cache_full),\n"
        "        'target_model_strings': {m.key: config.target_model_string(m) for m in config.TARGET_MODELS},\n"
        "        'versions': {p: ver(p) for p in ['strong_reject', 'pyrit', 'openai', 'datasets']},\n"
        "    }\n"
        "    with open(config.PATHS.manifest, 'w', encoding='utf-8') as f:\n"
        "        json.dump(manifest, f, indent=2)\n"
        "    print('Wrote', path)\n"
        "    print('Wrote', config.PATHS.manifest)\n"
        "else:\n"
        "    print('Nothing to export (full run not executed this session).')"
    ),
]


def build(path, cells):
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (jbscale venv)",
        "language": "python",
        "name": "jbscale",
    }
    nb.metadata["language_info"] = {"name": "python"}
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


build("analysis.ipynb", analysis_cells)
build("generate_scores.ipynb", generate_cells)
