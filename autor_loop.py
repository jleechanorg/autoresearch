#!/usr/bin/env python3
"""
Karpathy-style autonomous research loop for GPT training.

Runs for up to 12 hours, using MiniMax (Claude API) to propose improvements
to train.py. Each experiment is git-committed, run, and scored against val_bpb.
Lower val_bpb is better. Crashes are reverted. Improvements are kept.

Usage:
    python autor_loop.py
    # or with uv:
    uv run python autor_loop.py
"""
from __future__ import annotations

import dataclasses
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Data classes (defined before use)
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    val_bpb: float          # 0.0 means crash
    training_seconds: float
    peak_vram_mb: float
    mfu_percent: float
    total_tokens_M: float
    num_steps: int
    num_params_M: float
    depth: int
    status: str            # "ok", "crash", "timeout"
    description: str


@dataclass
class ExperimentRecord:
    commit: str
    val_bpb: float
    memory_gb: float
    status: str   # "keep", "discard", "crash"
    description: str


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_DIR = Path("/home/jleechan/projects_other/autoresearch")
TRAIN_FILE = REPO_DIR / "train.py"
CACHE_DIR = Path.home() / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
RESULTS_FILE = REPO_DIR / "results.tsv"
LOG_FILE = REPO_DIR / "autor_loop.log"
RUN_LOG = REPO_DIR / "run.log"
BRANCH_NAME = "autoresearch/rtx4090"

# Time budgets
MAX_RUN_SECONDS = 600          # 10 minutes per run
MAX_LOOP_SECONDS = 12 * 3600   # 12 hours

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str, print_it: bool = True) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if print_it:
        print(line)
    LOG_FILE.write_text(LOG_FILE.read_text() + line + "\n")


def log_section(title: str) -> None:
    sep = "=" * 60
    log(sep)
    log(title)
    log(sep)


# ---------------------------------------------------------------------------
# MiniMax API helper
# ---------------------------------------------------------------------------

def claudem(
    prompt: str,
    model: str = "MiniMax-M2.5",
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> str:
    """
    Call MiniMax (Claude API) with proper env var setup.
    Sets ANTHROPIC_BASE_URL and ANTHROPIC_MODEL before calling Anthropic SDK.
    Returns the assistant's text response.

    Handles both text blocks and thinking blocks (anthropic SDK 0.96+).
    """
    env_copy = dict(os.environ)
    env_copy["ANTHROPIC_BASE_URL"] = os.environ.get(
        "ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic"
    )
    env_copy["ANTHROPIC_MODEL"] = model or os.environ.get(
        "MINIMAX_MODEL", "MiniMax-M2.5"
    )

    import anthropic
    client = anthropic.Anthropic(
        api_key=env_copy.get("MINIMAX_API_KEY", ""),
        base_url=env_copy["ANTHROPIC_BASE_URL"],
    )

    response = client.messages.create(
        model=env_copy["ANTHROPIC_MODEL"],
        max_tokens=max_tokens,
        temperature=temperature,
        system=(
            "You are an autonomous AI researcher improving GPT training code. "
            "IMPORTANT HARDWARE CONSTRAINTS (RTX 4090 24GB, no torch.compile): "
            "- DEPTH must be 4-8 (8 causes OOM with batch 16+ during compile) "
            "- DEVICE_BATCH_SIZE must be 16 or smaller "
            "- MUST comment out or remove torch.compile (causes OOM on this GPU) "
            "- TOTAL_BATCH_SIZE must be reduced proportionally "
            "- Flash Attention 3 kernels are REQUIRED — do not remove the kernels import "
            "- Any change that increases model params beyond ~15M or batch >16 will OOM "
            "- CRITICAL: keep PYTORCH_ALLOC_CONF=expandable_segments:True and torch.compile commented out "
            "CRASH PATTERNS to avoid (these caused 80% of failures): "
            "- DEPTH=8 or higher → OOM during forward pass compilation "
            "- torch.compile enabled → OOM during torch.compile forward call "
            "- DEVICE_BATCH_SIZE > 16 → memory allocation fails "
            "- TOTAL_BATCH_SIZE > 2**18 → OOM "
            "- Removing the kernels/flash attention → training is 10x slower "
            "Propose ONE specific, focused change with clear reasoning. "
            "Output ONLY the complete modified train.py file content. "
            "Do NOT change the data loading, tokenizer, or evaluation code."
        ),
        messages=[{"role": "user", "content": prompt}],
    )

    # Collect text from all content blocks (handles ThinkingBlock in SDK 0.96+)
    texts = []
    for block in response.content:
        if hasattr(block, "text"):
            texts.append(block.text)
        elif hasattr(block, "thinking"):
            # Skip thinking blocks — we only want the output text
            pass
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args: str, cwd: Path = REPO_DIR) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True
    ).strip()


def git_branch_exists(name: str) -> bool:
    try:
        git("rev-parse", "--verify", f"refs/heads/{name}", cwd=REPO_DIR)
        return True
    except subprocess.CalledProcessError:
        return False


# ---------------------------------------------------------------------------
# Data check
# ---------------------------------------------------------------------------

def check_data() -> bool:
    """Return True if cache data exists and is usable."""
    tokenizer_pkl = TOKENIZER_DIR / "tokenizer.pkl"
    token_bytes = TOKENIZER_DIR / "token_bytes.pt"
    shard_files = list(DATA_DIR.glob("shard_*.parquet"))
    return (
        tokenizer_pkl.exists()
        and token_bytes.exists()
        and len(shard_files) >= 2
    )


# ---------------------------------------------------------------------------
# Parse run output
# ---------------------------------------------------------------------------

def parse_run_log(log_text: str) -> RunResult:
    """Extract metrics from a completed run.log."""
    defaults = RunResult(
        val_bpb=0.0,
        training_seconds=0.0,
        peak_vram_mb=0.0,
        mfu_percent=0.0,
        total_tokens_M=0.0,
        num_steps=0,
        num_params_M=0.0,
        depth=0,
        status="crash",
        description="",
    )

    val_bpb_match = re.search(r"^val_bpb:\s+([\d.]+)", log_text, re.MULTILINE)
    if not val_bpb_match:
        defaults.description = "no val_bpb in output (crash or early failure)"
        return defaults

    val_bpb = float(val_bpb_match.group(1))

    def get(key: str, default: float = 0.0) -> float:
        m = re.search(rf"^{key}:\s+([\d.]+)", log_text, re.MULTILINE)
        return float(m.group(1)) if m else default

    return RunResult(
        val_bpb=val_bpb,
        training_seconds=get("training_seconds"),
        peak_vram_mb=get("peak_vram_mb"),
        mfu_percent=get("mfu_percent"),
        total_tokens_M=get("total_tokens_M"),
        num_steps=int(get("num_steps")),
        num_params_M=get("num_params_M"),
        depth=int(get("depth")),
        status="ok",
        description="",
    )


# ---------------------------------------------------------------------------
# Run train.py with timeout
# ---------------------------------------------------------------------------

class RunTimeout(Exception):
    pass


def run_with_timeout(
    cmd: list[str],
    timeout_s: int,
    log_path: Path,
    cwd: Path = REPO_DIR,
) -> tuple[int, str]:
    """Run cmd, writing output to log_path. Returns (returncode, log_text)."""

    def alarm_handler(signum, frame):
        raise RunTimeout()

    old_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout_s)

    try:
        with open(log_path, "w", buffering=1) as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                env={k: v for k, v in os.environ.items() if k in (
                    "PATH", "HOME", "USER", "LANG", "LC_ALL",
                    "PYTORCH_ALLOC_CONF", "HF_HUB_DISABLE_PROGRESS_BARS",
                    "MINIMAX_API_KEY", "MINIMAX_MODEL", "ANTHROPIC_BASE_URL",
                )},
            )
            stdout, _ = proc.communicate()
        signal.alarm(0)
        return proc.returncode, log_path.read_text()
    except RunTimeout:
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass
        with open(log_path, "a") as f:
            f.write(f"\n[TIMEOUT] run exceeded {timeout_s}s and was killed\n")
        signal.alarm(0)
        return -1, log_path.read_text()
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Build MiniMax improvement prompt
# ---------------------------------------------------------------------------

def build_prompt(
    current_train_py: str,
    best_val_bpb: float,
    recent_experiments: list[ExperimentRecord],
) -> str:
    recent_lines = []
    for exp in recent_experiments[-3:]:
        recent_lines.append(
            f"  - commit={exp.commit} val_bpb={exp.val_bpb:.6f} "
            f"status={exp.status} desc={exp.description!r}"
        )
    recent_context = "\n".join(recent_lines) if recent_lines else "(none yet)"

    return f"""You are improving GPT training code. RTX 4090 24GB constraints apply.

Current train.py (full source):
---
{current_train_py}
---

Best achieved val_bpb so far: {best_val_bpb:.6f} (lower is better).

Recent experiments (last 3):
{recent_context}

Your task: propose ONE specific change to train.py that should improve val_bpb.

CRITICAL CONSTRAINTS:
- DEPTH: 4-8 only (8 risky, use 6 max)
- DEVICE_BATCH_SIZE: 16 or smaller only
- torch.compile MUST remain commented out
- TOTAL_BATCH_SIZE: 2**18 or smaller
- Flash Attention must remain (do not remove kernels import)
- PYTORCH_ALLOC_CONF=expandable_segments:True must be set

Focus areas (in priority order):
1. DEPTH changes within 4-8 range (DEPTH=6 gave best result)
2. ASPECT_RATIO (controls model width) — try 48, 56, 64
3. Learning rate tuning: EMBEDDING_LR, MATRIX_LR, SCALAR_LR
4. HEAD_DIM: try 96, 128 (affects attention compute)
5. WINDOW_PATTERN changes within SSSL/SSSL range
6. Weight decay: try 0.1, 0.15, 0.2
7. Warmup/warmdown ratios

DO NOT:
- Change data loading or evaluation code
- Enable torch.compile
- Increase DEPTH above 8
- Increase batch sizes
- Remove flash attention kernels

Output: ONLY the complete modified train.py file content. No explanations.
"""


# ---------------------------------------------------------------------------
# Write train.py (safeguard against empty/wrong content)
# ---------------------------------------------------------------------------

def write_train_py(content: str) -> bool:
    """Write content to train.py only if it looks like valid Python and passes constraint checks."""
    # Strip markdown fences first
    content = re.sub(r'^\s*```[\w]*\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\s*```\s*$', '', content, flags=re.MULTILINE)
    content = content.strip()
    if not content or len(content) < 500:
        log("ERROR: proposed train.py too short — rejecting")
        return False
    if "def " not in content or "import " not in content:
        log("ERROR: proposed train.py missing Python fundamentals — rejecting")
        return False
    if content.count("\n") < 50:
        log("ERROR: proposed train.py has very few lines — rejecting")
        return False
    # Constraint checks — reject proposals that violate hardware limits
    if re.search(r"DEPTH\s*=\s*([0-9]+)", content):
        depth_match = re.search(r"DEPTH\s*=\s*([0-9]+)", content)
        depth_val = int(depth_match.group(1))
        if depth_val > 8:
            log(f"ERROR: DEPTH={depth_val} > 8 — would OOM — rejecting")
            return False
        if depth_val < 4:
            log(f"ERROR: DEPTH={depth_val} < 4 — too small — rejecting")
            return False
    batch_match = re.search(r"DEVICE_BATCH_SIZE\s*=\s*([0-9]+)", content)
    if batch_match:
        batch_val = int(batch_match.group(1))
        if batch_val > 16:
            log(f"ERROR: DEVICE_BATCH_SIZE={batch_val} > 16 — would OOM — rejecting")
            return False
    if "torch.compile(" in content and "# model = torch.compile" not in content:
        log("ERROR: torch.compile not commented out — would OOM — rejecting")
        return False
    # Crash pattern detection
    crash_patterns = [
        (r"TOTAL_BATCH_SIZE\s*=\s*2\*\*19", "TOTAL_BATCH_SIZE=2**19 would OOM"),
        (r"TOTAL_BATCH_SIZE\s*=\s*[5-9][0-9]{5,}", "TOTAL_BATCH_SIZE too large"),
        (r"\bHEAD_DIM\s*=\s*2[5-9][6-9]\b|\bHEAD_DIM\s*=\s*[3-9][0-9]{2,}\b", "HEAD_DIM > 256 too large"),
        (r"ASPECT_RATIO\s*=\s*(9[6-9]|[1-9][0-9]{2,})", "ASPECT_RATIO > 96 too large"),
        (r"vocab_size\s*=\s*6[5-9]", "vocab_size > 64k may cause memory issues"),
        (r"n_layer\s*=\s*(1[3-9]|[2-9][0-9])", "n_layer > 12 too deep for this GPU"),
        (r"n_head\s*=\s*([2-9][0-9]|[1-9][0-9]{2,})", "n_head > 20 may OOM"),
        (r"fa3\s*=\s*None", "fa3=None will cause AttributeError crash"),
        (r'get_kernel\s*\(\s*["\'][^"\']+["\']\s*\)', "get_kernel with non-const arg may fail"),
        (r"torch\.set_float32_matmul_precision\([^)]+,\s*['\"]width['\"]", "width-level matmul precision on wide models OOMs"),
    ]
    for pattern, reason in crash_patterns:
        if re.search(pattern, content):
            log(f"ERROR: crash pattern detected ({reason}) — rejecting")
            return False
    try:
        compile(content, "train.py", "exec")
    except SyntaxError as e:
        log(f"ERROR: proposed train.py has syntax error {e.msg} at line {e.lineno} — rejecting")
        return False
    TRAIN_FILE.write_text(content)
    return True


# ---------------------------------------------------------------------------
# Results TSV
# ---------------------------------------------------------------------------

RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"


def init_results_tsv() -> None:
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER + "\n")


def append_result(record: ExperimentRecord) -> None:
    line = (
        f"{record.commit}\t{record.val_bpb:.6f}\t"
        f"{record.memory_gb:.1f}\t{record.status}\t{record.description}"
    )
    RESULTS_FILE.write_text(RESULTS_FILE.read_text() + line + "\n")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    train_py_content: str,
    description: str,
    experiment_count: int,
    best_val_bpb: float,
) -> tuple[RunResult, str, bool]:
    """
    Commit, run train.py, parse results.
    Returns (result, commit_hash, kept_improvement).
    """
    # Commit the change
    try:
        git("add", str(TRAIN_FILE))
        git("commit", "-m", f"autor exp {experiment_count}: {description}")
        commit_hash = git("rev-parse", "--short", "HEAD")
    except Exception as e:
        log(f"ERROR: git commit failed: {e}")
        return RunResult(status="crash", description=f"git commit failed: {e}"), "", False

    log(f"Running experiment {experiment_count} (commit={commit_hash}): {description}")

    # First attempt
    rc, log_text = run_with_timeout(
        ["uv", "run", "train.py"],
        MAX_RUN_SECONDS,
        RUN_LOG,
    )

    double_crash = False
    if rc != 0:
        log(f"Experiment {experiment_count} failed (rc={rc}) — retrying once...")
        rc2, log_text2 = run_with_timeout(
            ["uv", "run", "train.py"],
            MAX_RUN_SECONDS,
            RUN_LOG,
        )
        if rc2 != 0:
            double_crash = True
        else:
            rc, log_text = rc2, log_text2
    elif log_text and "val_bpb" not in log_text:
        # rc=0 but no val_bpb — treat as crash
        log(f"Experiment {experiment_count} produced no val_bpb — retrying once...")
        rc2, log_text2 = run_with_timeout(
            ["uv", "run", "train.py"],
            MAX_RUN_SECONDS,
            RUN_LOG,
        )
        if rc2 != 0 or "val_bpb" not in log_text2:
            double_crash = True
        else:
            rc, log_text = rc2, log_text2

    if double_crash:
        log(f"Experiment {experiment_count} crashed TWICE — reverting")
        try:
            git("reset", "--hard", "HEAD~1")
        except Exception:
            pass
        return RunResult(
            status="crash",
            description=f"double crash (reverted {commit_hash})",
            val_bpb=0.0,
            training_seconds=0.0,
            peak_vram_mb=0.0,
            mfu_percent=0.0,
            total_tokens_M=0.0,
            num_steps=0,
            num_params_M=0.0,
            depth=0,
        ), commit_hash, False

    result = parse_run_log(log_text)

    if result.status != "ok":
        log(f"Experiment {experiment_count} produced no val_bpb — reverting")
        try:
            git("reset", "--hard", "HEAD~1")
        except Exception:
            pass
        return result, commit_hash, False

    # Evaluate improvement
    if result.val_bpb < best_val_bpb:
        kept = True
        log(
            f"IMPROVEMENT: val_bpb {result.val_bpb:.6f} < {best_val_bpb:.6f}"
        )
    else:
        kept = False
        try:
            git("reset", "--hard", "HEAD~1")
        except Exception:
            pass
        log(
            f"No improvement: val_bpb={result.val_bpb:.6f} >= {best_val_bpb:.6f}"
        )

    return result, commit_hash, kept


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup() -> bool:
    """Setup: branch, data, results.tsv. Returns False on failure."""
    log_section("SETUP")

    # Branch
    try:
        if not git_branch_exists(BRANCH_NAME):
            git("checkout", "-b", BRANCH_NAME)
            log(f"Created branch {BRANCH_NAME}")
        else:
            current = git("rev-parse", "--abbrev-ref", "HEAD")
            if current != BRANCH_NAME:
                git("checkout", BRANCH_NAME)
            log(f"Using existing branch {BRANCH_NAME}")
    except Exception as e:
        log(f"ERROR: could not setup branch: {e}")
        return False

    # Data
    if not check_data():
        log("Data not found — running prepare.py (first-time setup)...")
        try:
            subprocess.run(
                ["uv", "run", "prepare.py", "--num-shards", "10"],
                cwd=REPO_DIR,
                timeout=600,
                check=True,
            )
        except Exception as e:
            log(f"ERROR: prepare.py failed: {e}")
            return False
        if not check_data():
            log("ERROR: data still missing after prepare.py")
            return False

    log("Data check: OK")
    init_results_tsv()
    log("results.tsv initialized")
    return True


# ---------------------------------------------------------------------------
# Baseline run
# ---------------------------------------------------------------------------

def run_baseline() -> tuple[bool, RunResult]:
    """Run baseline once. Returns (success, result)."""
    log_section("BASELINE RUN")
    log("Running baseline: uv run train.py")

    rc, log_text = run_with_timeout(
        ["uv", "run", "train.py"],
        MAX_RUN_SECONDS,
        RUN_LOG,
    )

    if rc != 0 or "val_bpb" not in log_text:
        log("Baseline failed — retrying once...")
        rc2, log_text2 = run_with_timeout(
            ["uv", "run", "train.py"],
            MAX_RUN_SECONDS,
            RUN_LOG,
        )
        if rc2 != 0 or "val_bpb" not in log_text2:
            log("BASELINE FAILED TWICE — cannot continue")
            return False, RunResult(
                status="crash", description="baseline crash",
                val_bpb=0.0, training_seconds=0.0, peak_vram_mb=0.0,
                mfu_percent=0.0, total_tokens_M=0.0, num_steps=0,
                num_params_M=0.0, depth=0,
            )
        rc, log_text = rc2, log_text2

    result = parse_run_log(log_text)
    log(
        f"Baseline: val_bpb={result.val_bpb:.6f}  "
        f"train_s={result.training_seconds:.1f}  "
        f"mfu={result.mfu_percent:.1f}%  vram={result.peak_vram_mb:.0f}MB"
    )
    return True, result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Init log file
    LOG_FILE.write_text("")
    log_section("AUTOR LOOP START")
    log(f"Repository: {REPO_DIR}")
    log(f"Branch: {BRANCH_NAME}")
    log(f"Max loop: {MAX_LOOP_SECONDS/3600:.0f}h | Max run: {MAX_RUN_SECONDS}s")

    # Verify MiniMax API connectivity
    try:
        test_resp = claudem(
            "Reply with exactly one word: test",
            max_tokens=50,
            temperature=0.0,
        )
        log(f"MiniMax API check: OK (response length={len(test_resp)})")
    except Exception as e:
        log(f"ERROR: MiniMax API call failed: {e}")
        log("Cannot proceed without MiniMax API access.")
        sys.exit(1)

    # Setup
    if not setup():
        sys.exit(1)

    # Baseline
    ok, baseline_result = run_baseline()
    if not ok:
        sys.exit(1)

    baseline_commit = git("rev-parse", "--short", "HEAD")
    baseline_record = ExperimentRecord(
        commit=baseline_commit,
        val_bpb=baseline_result.val_bpb,
        memory_gb=baseline_result.peak_vram_mb / 1024,
        status="keep",
        description="baseline",
    )
    append_result(baseline_record)
    log(f"Baseline recorded: val_bpb={baseline_result.val_bpb:.6f}")

    # Loop state
    best_val_bpb = baseline_result.val_bpb
    best_train_py = TRAIN_FILE.read_text()
    recent_experiments: list[ExperimentRecord] = [baseline_record]
    experiment_count = 1
    loop_start = time.time()

    log_section("EXPERIMENT LOOP START")
    log(f"Best val_bpb at start: {best_val_bpb:.6f}")

    while True:
        elapsed = time.time() - loop_start
        if elapsed >= MAX_LOOP_SECONDS:
            log(f"TIME LIMIT REACHED after {elapsed/3600:.1f}h")
            break

        remaining = MAX_LOOP_SECONDS - elapsed
        if remaining < 600:
            log(f"Only {remaining:.0f}s remaining — stopping cleanly")
            break

        # Progress report every 10 experiments
        if experiment_count % 10 == 0:
            log_section("PROGRESS REPORT")
            log(f"Experiments: {experiment_count}")
            log(f"Best val_bpb: {best_val_bpb:.6f}")
            log(f"Elapsed: {elapsed/3600:.1f}h / {MAX_LOOP_SECONDS/3600:.0f}h")

        # Build prompt
        current_train_py = TRAIN_FILE.read_text()
        prompt = build_prompt(
            current_train_py,
            best_val_bpb,
            recent_experiments,
        )

        # Ask MiniMax
        log(f"Asking MiniMax for experiment {experiment_count}...")
        try:
            proposed = claudem(prompt, max_tokens=16384, temperature=1.0)
        except Exception as e:
            log(f"MiniMax call failed: {e} — backing off 30s")
            time.sleep(30)
            try:
                proposed = claudem(prompt, max_tokens=16384, temperature=1.0)
            except Exception as e2:
                log(f"MiniMax call failed again: {e2} — skipping iteration")
                time.sleep(60)
                continue

        if not write_train_py(proposed):
            time.sleep(10)
            continue

        # Run experiment
        result, commit_hash, kept = run_experiment(
            proposed,
            f"experiment_{experiment_count}",
            experiment_count,
            best_val_bpb,
        )

        memory_gb = result.peak_vram_mb / 1024

        if kept:
            status = "keep"
            best_val_bpb = result.val_bpb
            best_train_py = proposed
        else:
            status = result.status if result.status == "crash" else "discard"

        record = ExperimentRecord(
            commit=commit_hash or "none",
            val_bpb=result.val_bpb,
            memory_gb=memory_gb,
            status=status,
            description=f"experiment_{experiment_count}",
        )
        append_result(record)
        recent_experiments.append(record)
        experiment_count += 1
        time.sleep(2)

    # Final summary
    log_section("FINAL SUMMARY")
    log(f"Total experiments: {experiment_count}")
    log(f"Best val_bpb: {best_val_bpb:.6f}")
    log(f"Improvement: {baseline_result.val_bpb - best_val_bpb:.6f}")
    log(f"Total elapsed: {(time.time() - loop_start)/3600:.1f}h")
    if RESULTS_FILE.exists():
        log("Results table:")
        for line in RESULTS_FILE.read_text().splitlines():
            log("  " + line)
    log_section("AUTOR LOOP COMPLETE")
    print(f"\nAutor loop complete. Results in {RESULTS_FILE}")
    print(f"Log in {LOG_FILE}")


if __name__ == "__main__":
    main()
