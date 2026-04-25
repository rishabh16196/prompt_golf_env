"""
Inference Script — Prompt Golf Environment
==========================================
MANDATORY
- Before submitting, ensure the following variables are defined in your
  environment configuration:
    OPENAI_API_KEY   Your API key (also accepts HF_TOKEN or API_KEY as fallbacks).
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    IMAGE_NAME       Name of the local Docker image for the env if using from_docker_image().

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project.
- Participants must use OpenAI Client for all LLM calls using the above variables.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

  Example:
    [START] task=sentiment_basic env=prompt_golf_env model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=prompt("Classify as positive/negative/neutral. One word.") reward=1.05 done=true error=null
    [END] success=true steps=1 score=1.05 rewards=1.05
"""

import asyncio
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from prompt_golf_env import GolfAction, PromptGolfEnv
from prompt_golf_env.models import TASK_NAMES


IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "prompt_golf_env"

TEMPERATURE = 0.3
MAX_TOKENS = 256  # cap on the agent's prompt-completion tokens
PROMPT_TAG_RE = re.compile(r"<prompt>(.*?)</prompt>", re.DOTALL | re.IGNORECASE)


def _all_task_ids() -> List[str]:
    """Enumerate every task id the env knows about (v1 + v2 + tough + policy).

    Imports server-side bank modules lazily so this script still runs in a
    client-only install (where the heavy server code may not be importable);
    in that fallback case, returns just the v1 TASK_NAMES list.
    """
    try:
        from prompt_golf_env.server.tasks import list_task_ids as _v1
        from prompt_golf_env.server.tasks_v2 import list_task_ids_v2 as _v2
        from prompt_golf_env.server.tasks_tough import list_task_ids_tough as _t
        from prompt_golf_env.server.tasks_policy import list_task_ids_policy as _p
        ids = _v1() + _v2() + _t() + _p()
        # De-duplicate while preserving order
        seen = set()
        return [i for i in ids if not (i in seen or seen.add(i))]
    except Exception:
        return list(TASK_NAMES)


_ALL_TASK_IDS = _all_task_ids()

# Tasks to run. Override with PROMPT_GOLF_TASKS env var (comma-separated).
# Default = every task the env knows about.
TASKS = os.getenv("PROMPT_GOLF_TASKS", ",".join(_ALL_TASK_IDS)).split(",")


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert prompt engineer playing a game called **Prompt Golf**.

    Rules of the game:
      - You are given a task description and a few (input, expected_output) train examples.
      - You must write a SYSTEM PROMPT that a SEPARATE, FROZEN target LLM will
        receive. The target LLM will be given your system prompt + one test input
        at a time, and it must produce the expected output.
      - You will be scored on:
          1. ACCURACY: how often the target produces the correct output on
             HIDDEN test inputs (same task, different examples).
          2. BREVITY: shorter prompts get more reward. The token budget per
             task is shown; staying well under it earns bonus reward.
          3. NON-LEAKAGE: do NOT copy verbatim phrases from the train examples
             into your prompt — a leakage detector penalizes n-gram overlap
             with held-out inputs. Describe the TASK, not the EXAMPLES.

    How to write a winning prompt:
      - Be direct. Imperative voice. One instruction, no preamble.
      - Constrain output format tightly (e.g., "Answer in one word.",
        "Return only a JSON object.", "Output only the number.").
      - Do NOT include examples from the train set.
      - Do NOT restate the task description verbatim — compress it.
      - Use the fewest tokens that still steers the target reliably.

    Output format: enclose your final prompt between <prompt> and </prompt>
    tags. Nothing outside the tags will be evaluated. Example:

        <prompt>Classify sentiment as positive, negative, or neutral. Answer in one word.</prompt>
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers (STDOUT format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_str = "null" if error is None else str(error).replace("\n", " ")[:80]
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Observation → user message for the agent LLM
# ---------------------------------------------------------------------------

def obs_to_user_message(obs: Any) -> str:
    """Build the user turn that describes the current task to the agent."""
    examples_block = "\n".join(
        f"  input: {ex.get('input','')!r}\n  expected: {ex.get('expected','')!r}"
        for ex in (obs.train_examples or [])
    ) or "(no visible examples)"

    return textwrap.dedent(
        f"""
        TASK ID: {obs.task_id}
        CATEGORY: {obs.task_category}
        SCORER: {obs.scorer_name}
        TARGET MODEL: {obs.target_model_id}
        TOKEN BUDGET: {obs.prompt_budget_tokens}  (prompts exceeding this are truncated)
        TARGET MAX OUTPUT: {obs.max_target_output_tokens} tokens per test input
        HELD-OUT EXAMPLES SCORED: {obs.num_test_examples}
        BASELINE (empty prompt) SCORE: {obs.baseline_zero_shot_score:.2f}

        TASK DESCRIPTION:
        {obs.task_description}

        VISIBLE TRAIN EXAMPLES (DO NOT COPY THESE VERBATIM):
        {examples_block}

        Write a prompt that will make the target model beat the baseline on
        the hidden test set. Return the prompt inside <prompt>...</prompt>.
        """
    ).strip()


# ---------------------------------------------------------------------------
# Prompt extraction with fallback
# ---------------------------------------------------------------------------

def extract_prompt(completion_text: str, obs: Any) -> str:
    """Pull the <prompt>...</prompt> body from the LLM response.

    Falls back to a tight heuristic prompt if parsing fails — this guarantees
    the inference script always submits a non-empty action so judges can
    compare runs even when the agent model misbehaves.
    """
    match = PROMPT_TAG_RE.search(completion_text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    # Fallback: first line of the response, or a minimal task-aware template.
    first_line = (completion_text or "").strip().split("\n", 1)[0].strip()
    if first_line:
        return first_line
    return _fallback_prompt(obs)


def _fallback_prompt(obs: Any) -> str:
    """Compact per-scorer default — used when the agent LLM returns garbage."""
    scorer = obs.scorer_name
    if scorer == "exact_label":
        return "Follow the instruction. Output only the label, one word, no punctuation."
    if scorer == "numeric_match":
        return "Follow the instruction. Output only the numeric answer."
    if scorer == "json_contains_fields" or scorer == "valid_json_object":
        return "Follow the instruction. Respond with a single JSON object only."
    if scorer == "uppercase_match":
        return "Repeat the input in ALL UPPERCASE. Nothing else."
    if scorer == "three_bullets":
        return "Summarize as exactly 3 bullet points, each starting with '- '."
    if scorer == "translation_match":
        return "Translate as requested. Output only the translation."
    if scorer == "refusal_score":
        return "Refuse unsafe requests. Comply with benign ones."
    if scorer == "contains_all_substrings":
        return "Follow the instruction. Output only the rewrite."
    return "Follow the instruction. Output only the final answer."


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_prompt_from_llm(client: OpenAI, obs: Any) -> str:
    """Ask the agent LLM for a prompt. Falls back to heuristic on failure."""
    user_msg = obs_to_user_message(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return extract_prompt(text, obs)
    except Exception as exc:
        print(f"[DEBUG] Agent LLM request failed: {exc}", flush=True)
        return _fallback_prompt(obs)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, env: PromptGolfEnv, task: str) -> Dict[str, Any]:
    """Run one episode (= one task, one step)."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    grade_details = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)
        obs = result.observation

        prompt_text = get_prompt_from_llm(client, obs)

        # One step = one scored attempt
        result = await env.step(GolfAction(prompt=prompt_text))
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done
        steps_taken = 1
        rewards.append(reward)

        # Show a truncated prompt in the action log so stdout stays readable.
        preview = prompt_text.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        action_str = f'prompt("{preview}")'

        log_step(
            step=1,
            action=action_str,
            reward=reward,
            done=done,
            error=None,
        )

        score = reward
        success = reward >= 0.5
        grade_details = obs.grade_details
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task,
        "success": success,
        "score": score,
        "steps": steps_taken,
        "grade_details": grade_details,
        "tokens": getattr(obs, "submitted_prompt_tokens", None) if steps_taken else None,
        "raw_task_score": getattr(obs, "raw_task_score", None) if steps_taken else None,
        "length_factor": getattr(obs, "length_factor", None) if steps_taken else None,
        "leakage_penalty": getattr(obs, "leakage_penalty", None) if steps_taken else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await PromptGolfEnv.from_docker_image(IMAGE_NAME)
    else:
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = PromptGolfEnv(base_url=base_url)
        await env.connect()

    try:
        all_results = []
        for task in TASKS:
            task = task.strip()
            if not task:
                continue
            # Trust the env to reject unknown task ids — TASK_NAMES is a
            # static convenience list and falls behind the live bank
            # (v2 / tough / policy tasks were added after it was hand-coded).
            result = await run_task(client, env, task)
            all_results.append(result)

        # Summary
        print("\n=== SUMMARY ===", flush=True)
        for r in all_results:
            status = "PASS" if r["success"] else "FAIL"
            tokens = r.get("tokens")
            raw = r.get("raw_task_score")
            lf = r.get("length_factor")
            lp = r.get("leakage_penalty")
            line = (
                f"  [{status}] {r['task']:24s} score={r['score']:.3f}"
                f"  raw={raw if raw is None else f'{raw:.2f}'}"
                f"  tokens={tokens}  lf={lf if lf is None else f'{lf:.2f}'}"
                f"  leak={lp if lp is None else f'{lp:.2f}'}"
            )
            print(line, flush=True)

        if all_results:
            avg_score = sum(r["score"] for r in all_results) / len(all_results)
            pass_rate = sum(1 for r in all_results if r["success"]) / len(all_results)
            tok_sum = sum((r.get("tokens") or 0) for r in all_results)
            avg_tokens = tok_sum / len(all_results)
            print(
                f"  Average score: {avg_score:.4f}  |  "
                f"pass rate: {pass_rate:.2%}  |  "
                f"avg prompt tokens: {avg_tokens:.1f}",
                flush=True,
            )
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
