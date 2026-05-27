"""
Prompt self-optimization loop v2.

Pipeline:
  Agent2 (Question Generator) -> Agent3 (Answerer) -> Agent4 (Evaluator)
       -> Agent1 (Prompt Optimizer) -> Gatekeeper -> [Agent5 Re-ingester]
       -> Experiment Logger

Usage:
  1. Start API server: cd apps/api && python -m uvicorn main:app --reload --port 8000
  2. Run: python scripts/eval_loop.py [--rounds 5] [--reingest]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"
ADMIN_USER = "admin"
ADMIN_PASS = "admin123"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def discover_method_files():
    """Scan all method_* folders and return list of (filepath, filename, mime) tuples."""
    import glob as _glob
    method_dirs = sorted(_glob.glob(os.path.join(PROJECT_ROOT, "method_*")))
    files = []
    for d in method_dirs:
        for fname in sorted(os.listdir(d)):
            fpath = os.path.join(d, fname)
            if not os.path.isfile(fpath):
                continue
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext == "pdf":
                mime = "application/pdf"
            elif ext in ("py", "r", "jl", "do", "sas", "txt", "md", "csv"):
                mime = "text/plain"
            else:
                continue
            files.append((fpath, fname, mime))
    return files


PARSE_PROMPT_FILE = os.path.join(PROJECT_ROOT, "apps", "api", "routers", "methods.py")
CHAT_SERVICE_FILE = os.path.join(PROJECT_ROOT, "packages", "chat", "service.py")

LOG_DIR = os.path.join(PROJECT_ROOT, "scripts", "eval_logs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_token():
    r = requests.post(f"{API_BASE}/api/auth/login", json={
        "username": ADMIN_USER, "password": ADMIN_PASS,
    })
    r.raise_for_status()
    return r.json()["access_token"]


def auth_h(token):
    return {"Authorization": f"Bearer {token}"}


def llm_call(api_key, system, user_msg, temperature=0.5, max_tokens=2000):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def strip_fences(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def get_openai_key():
    # Check both project root and apps/api for .env
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(PROJECT_ROOT, "apps", "api", ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: No OPENAI_API_KEY found")
        sys.exit(1)
    return key


def fetch_knowledge_units(token):
    r = requests.get(f"{API_BASE}/api/knowledge", headers=auth_h(token))
    if r.status_code != 200:
        return []
    data = r.json()
    return data if isinstance(data, list) else []


def read_current_prompts():
    with open(PARSE_PROMPT_FILE, "r", encoding="utf-8") as f:
        methods_code = f.read()
    with open(CHAT_SERVICE_FILE, "r", encoding="utf-8") as f:
        service_code = f.read()

    marker = 'PARSE_PROMPT = """'
    start = methods_code.find(marker)
    end = methods_code.find('"""', start + len(marker)) + 3
    parse_prompt = methods_code[start:end]

    prompts = {}
    for name in ["DIRECT_ANSWER_PROMPT", "COMPARISON_PROMPT", "LLM_ONLY_PROMPT"]:
        start = service_code.find(f'{name} = """\\')
        if start == -1:
            start = service_code.find(f'{name} = """')
        end = service_code.find('"""', start + len(name) + 8) + 3
        prompts[name] = service_code[start:end]

    return {"PARSE_PROMPT": parse_prompt, **prompts}


# ---------------------------------------------------------------------------
# Experiment Logger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(LOG_DIR, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

    def log_round(self, round_num, data):
        path = os.path.join(self.run_dir, f"round_{round_num:02d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def log_summary(self, summary):
        path = os.path.join(self.run_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Logs saved to {self.run_dir}")


# ---------------------------------------------------------------------------
# Agent 2: Question Generator (knowledge-aware)
# ---------------------------------------------------------------------------

QUESTION_GEN_SYSTEM = (
    "You are a statistics professor testing a Q&A system built for researchers. "
    "You will receive a summary of knowledge units currently stored in the system, "
    "which may cover MULTIPLE different statistical methods from different fields.\n\n"
    "Generate test questions in FOUR categories:\n\n"
    "## Theory Coverage (2 questions)\n"
    "Questions about the theoretical content stored in the knowledge base: "
    "definitions, assumptions, convergence rates, formulas, proofs. "
    "IMPORTANT: Pick questions from AT LEAST 2 DIFFERENT methods in the knowledge base.\n\n"
    "## Practical Coverage (2 questions)\n"
    "Questions a working researcher would ask when trying to USE the stored methods. "
    "Write these as if the researcher has real data and needs actionable advice. "
    "Pick from DIFFERENT methods than the theory questions if possible.\n\n"
    "## Cross-Method (1 question)\n"
    "A question that requires comparing or choosing between 2+ methods in the knowledge base. "
    "Example: 'I have high-dimensional data with potential treatment effects - "
    "should I use OGA for variable selection first, or go directly with DML?'\n\n"
    "## Boundary (1 question)\n"
    "A question that pushes BEYOND what the stored knowledge covers. "
    "Tests whether the system honestly admits gaps vs. fabricates answers.\n\n"
    "IMPORTANT: \n"
    "- Spread questions across ALL available methods, not just one.\n"
    "- Practical questions should sound like a real person with real data.\n"
    "- They can be in English or Chinese.\n\n"
    "Return JSON with this exact structure:\n"
    '{\n'
    '  "coverage": [\n'
    '    {"question": "...", "expected_knowledge": "which stored unit should be used"}\n'
    '  ],\n'
    '  "practical": [\n'
    '    {"question": "...", "expected_knowledge": "which stored unit should help answer this"}\n'
    '  ],\n'
    '  "cross_method": [\n'
    '    {"question": "...", "expected_knowledge": "which methods should be compared"}\n'
    '  ],\n'
    '  "boundary": [\n'
    '    {"question": "...", "why_boundary": "why this goes beyond stored knowledge"}\n'
    '  ]\n'
    '}\n'
)


def agent_question_generator(api_key, knowledge_units):
    knowledge_summary = []
    for u in knowledge_units:
        summary = f"- [{u.get('knowledge_type', '?')}] {u.get('title', '?')}: {u.get('content', '')[:150]}"
        tags = u.get("topic_tags", [])
        if tags:
            summary += f" (tags: {', '.join(tags[:3])})"
        knowledge_summary.append(summary)

    user_msg = (
        f"## Stored Knowledge Units ({len(knowledge_units)} total)\n\n"
        + "\n".join(knowledge_summary)
        + "\n\nGenerate test questions based on this knowledge."
    )

    raw = llm_call(api_key, QUESTION_GEN_SYSTEM, user_msg, temperature=0.7)
    result = json.loads(strip_fences(raw))
    # Tag question types for evaluation
    if "practical" in result:
        for q in result["practical"]:
            q["_type"] = "practical"
    for q in result.get("coverage", []):
        q["_type"] = "theory"
    for q in result.get("cross_method", []):
        q["_type"] = "cross_method"
    return result


# ---------------------------------------------------------------------------
# Agent 3: Answerer (uses live pipeline)
# ---------------------------------------------------------------------------

def agent_answerer(question, token, session_id):
    r = requests.post(
        f"{API_BASE}/api/chat",
        json={"message": question},
        headers={**auth_h(token), "X-Session-Id": session_id},
    )
    if r.status_code != 200:
        return f"[ERROR {r.status_code}]: {r.text}"
    return r.json()["response"]


# ---------------------------------------------------------------------------
# Agent 4: Evaluator (multi-dimensional)
# ---------------------------------------------------------------------------

EVALUATOR_SYSTEM = (
    "You are a strict evaluator for a statistical research Q&A system.\n\n"
    "You will receive:\n"
    "- The question and its type (coverage or boundary)\n"
    "- The system's answer\n"
    "- For coverage questions: which knowledge unit was expected to be used\n"
    "- For boundary questions: why this question goes beyond stored knowledge\n\n"
    "## Scoring Dimensions (each 1-5)\n\n"
    "### 1. Knowledge Retrieval\n"
    "Did the system find and use the right knowledge units?\n"
    "- 5: Retrieved the most relevant unit(s) and used them accurately\n"
    "- 3: Retrieved somewhat relevant units but missed the best match\n"
    "- 1: Failed to retrieve relevant knowledge or used wrong units\n\n"
    "### 2. Answer Accuracy\n"
    "Is the answer factually correct and precise?\n"
    "- 5: All claims are accurate, no hallucination\n"
    "- 3: Mostly correct but some inaccuracies or vague claims\n"
    "- 1: Contains significant errors or hallucinated content\n\n"
    "### 3. Core Question Addressed\n"
    "Does the answer actually address what was asked?\n"
    "- 5: Directly and completely answers the question\n"
    "- 3: Partially addresses it, missing key aspects\n"
    "- 1: Does not address the question\n\n"
    "### 4. Honesty About Gaps\n"
    "Does the system acknowledge when it does not have relevant knowledge?\n"
    "- 5: Clearly distinguishes known facts from LLM general knowledge\n"
    "- 3: Sometimes blurs the line between stored knowledge and general knowledge\n"
    "- 1: Presents general knowledge as if it came from the knowledge base\n\n"
    "### 5. Practicality (for practical questions only, otherwise set to 0)\n"
    "Does the answer give actionable advice a researcher can directly use?\n"
    "- 5: Gives specific, actionable steps or recommendations with context\n"
    "- 3: Gives general advice but lacks specific steps or conditions\n"
    "- 1: Only gives abstract theory without connecting to the practical scenario\n"
    "- 0: Not a practical question (set to 0)\n\n"
    "## Error Types\n"
    "Classify the PRIMARY error type (pick one or none):\n"
    "- retrieval_miss: relevant knowledge exists but was not retrieved\n"
    "- retrieval_wrong: wrong knowledge unit was retrieved\n"
    "- hallucination: answer contains fabricated facts\n"
    "- incomplete: answer is correct but misses important parts\n"
    "- off_topic: answer does not address the question\n"
    "- false_confidence: claims certainty about things not in the knowledge base\n"
    "- none: no significant errors\n\n"
    "Check the debug info at the bottom of the answer for strategy used.\n\n"
    "Return JSON:\n"
    '{\n'
    '  "retrieval_score": <1-5>,\n'
    '  "accuracy_score": <1-5>,\n'
    '  "core_answer_score": <1-5>,\n'
    '  "honesty_score": <1-5>,\n'
    '  "practicality_score": <0-5>,\n'
    '  "error_type": "<primary error type or none>",\n'
    '  "error_detail": "<specific description of what went wrong>",\n'
    '  "strategy_used": "<from debug info>",\n'
    '  "strategy_appropriate": true or false,\n'
    '  "improvement_suggestions": "<specific actionable suggestions for prompt changes>"\n'
    '}'
)


def agent_evaluator(api_key, question_data, answer, question_type):
    is_practical = question_data.get("_type") == "practical"
    if question_type == "coverage":
        context = (
            f"Question type: {("PRACTICAL" if is_practical else "THEORY")} COVERAGE (should be answerable from stored knowledge)\n"
            f"Expected knowledge: {question_data.get('expected_knowledge', 'N/A')}"
        )
    else:
        context = (
            f"Question type: BOUNDARY (goes beyond stored knowledge)\n"
            f"Why boundary: {question_data.get('why_boundary', 'N/A')}"
        )

    user_msg = (
        f"## Context\n{context}\n\n"
        f"## Question\n{question_data['question']}\n\n"
        f"## System Answer\n{answer}"
    )
    raw = llm_call(api_key, EVALUATOR_SYSTEM, user_msg, temperature=0.2)
    return json.loads(strip_fences(raw))


# ---------------------------------------------------------------------------
# Agent 1: Prompt Optimizer (patch-based)
# ---------------------------------------------------------------------------

PROMPT_OPTIMIZER_SYSTEM = (
    "You are a prompt optimization specialist. You improve system prompts by proposing "
    "targeted patches, NOT full rewrites.\n\n"
    "You will receive:\n"
    "1. Current system prompts\n"
    "2. Multi-dimensional evaluation results with error types\n"
    "3. Previous round scores (if available) for comparison\n\n"
    "## Rules\n"
    "- Only propose changes that directly address identified error patterns\n"
    "- Each patch should be small and testable\n"
    "- Explain the causal link: error X -> patch Y -> expected improvement Z\n"
    "- Do NOT change things that are working well\n"
    "- Prioritize: fix the most common/severe error type first\n\n"
    "## Error type -> likely prompt fix mapping\n"
    "- retrieval_miss -> improve PARSE_PROMPT to extract more granular units, "
    "or improve how content/tags are generated for better embedding match\n"
    "- retrieval_wrong -> improve topic_tags / reusable_for_questions in PARSE_PROMPT\n"
    "- hallucination -> add stronger grounding instructions in chat prompts\n"
    "- incomplete -> expand chat prompt to require covering all relevant aspects\n"
    "- off_topic -> improve chat prompt to restate and directly address the question\n"
    "- false_confidence -> add instructions to distinguish knowledge-base info from general knowledge\n\n"
    "Return JSON:\n"
    '{\n'
    '  "patches": [\n'
    '    {\n'
    '      "target_prompt": "<PARSE_PROMPT|DIRECT_ANSWER_PROMPT|COMPARISON_PROMPT|LLM_ONLY_PROMPT>",\n'
    '      "action": "add|modify|remove",\n'
    '      "location": "<where in the prompt: beginning/end/after section X>",\n'
    '      "content": "<the text to add or the replacement text>",\n'
    '      "original": "<if action=modify, the text being replaced; null for add/remove>",\n'
    '      "reasoning": "<error_type -> this patch -> expected improvement>"\n'
    '    }\n'
    '  ],\n'
    '  "error_pattern_summary": "<what error patterns were most common>",\n'
    '  "expected_improvement": "<what scores should improve and why>"\n'
    '}'
)


def agent_prompt_optimizer(api_key, current_prompts, eval_results, prev_scores=None):
    error_counts = {}
    for er in eval_results:
        etype = er["eval"].get("error_type", "none")
        error_counts[etype] = error_counts.get(etype, 0) + 1

    user_msg = "## Current Prompts\n\n"
    for name, code in current_prompts.items():
        user_msg += f"### {name}\n```\n{code}\n```\n\n"

    user_msg += "## Error Pattern Summary\n"
    for etype, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        user_msg += f"- {etype}: {count} occurrences\n"

    user_msg += "\n## Evaluation Results\n\n"
    for i, er in enumerate(eval_results):
        q = er["question"]
        ev = er["eval"]
        user_msg += (
            f"### Q{i+1} ({er['type']}): {q[:100]}\n"
            f"- Retrieval: {ev['retrieval_score']}/5 | "
            f"Accuracy: {ev['accuracy_score']}/5 | "
            f"Core: {ev['core_answer_score']}/5 | "
            f"Honesty: {ev['honesty_score']}/5\n"
            f"- Error: {ev['error_type']} - {ev.get('error_detail', '')}\n"
            f"- Suggestion: {ev.get('improvement_suggestions', '')}\n\n"
        )

    if prev_scores:
        user_msg += "## Previous Round Scores\n"
        for k, v in prev_scores.items():
            user_msg += f"- {k}: {v:.1f}/5\n"

    raw = llm_call(api_key, PROMPT_OPTIMIZER_SYSTEM, user_msg, temperature=0.3, max_tokens=4000)
    return json.loads(strip_fences(raw))


# ---------------------------------------------------------------------------
# Gatekeeper
# ---------------------------------------------------------------------------

def gatekeeper_decide(current_scores, prev_scores, patches):
    if prev_scores is None:
        print("  [Gatekeeper] First round - accepting patches")
        return True

    improved = False
    degraded = False
    for dim in current_scores:
        if dim in prev_scores:
            diff = current_scores[dim] - prev_scores[dim]
            if diff > 0.2:
                improved = True
            if diff < -0.5:
                degraded = True

    if degraded:
        print("  [Gatekeeper] REJECTED - scores degraded significantly")
        return False
    if not improved and len(patches) > 0:
        print("  [Gatekeeper] REJECTED - no improvement detected")
        return False

    print("  [Gatekeeper] ACCEPTED - scores improved or stable")
    return True


# ---------------------------------------------------------------------------
# Agent 5: Re-ingester (on-demand)
# ---------------------------------------------------------------------------

def agent_reingest(token):
    """Delete all existing units, then parse & upload all method_* folder files."""
    headers = auth_h(token)

    r = requests.get(f"{API_BASE}/api/knowledge", headers=headers)
    if r.status_code == 200:
        units = r.json()
        for unit in units:
            requests.delete(f"{API_BASE}/api/knowledge/{unit['id']}", headers=headers)
        print(f"    Deleted {len(units)} existing units")

    method_files = discover_method_files()
    if not method_files:
        print("    ERROR: No files found in method_* folders")
        return False

    print(f"    Found {len(method_files)} files across method folders:")
    for fpath, fname, mime in method_files:
        folder = os.path.basename(os.path.dirname(fpath))
        print(f"      [{folder}] {fname}")

    # Parse each method folder separately to avoid hitting token limits
    folders_seen = {}
    for fpath, fname, mime in method_files:
        folder = os.path.basename(os.path.dirname(fpath))
        if folder not in folders_seen:
            folders_seen[folder] = []
        folders_seen[folder].append((fpath, fname, mime))

    total_units = []
    for folder, folder_files in folders_seen.items():
        files_payload = []
        for fpath, fname, mime in folder_files:
            with open(fpath, "rb") as f:
                files_payload.append(("files", (fname, f.read(), mime)))

        print(f"    Parsing {folder} ({len(folder_files)} files)...")
        r = requests.post(f"{API_BASE}/api/knowledge/parse", files=files_payload, headers=headers)
        if r.status_code != 200:
            print(f"    Parse failed for {folder}: {r.status_code} {r.text[:200]}")
            continue

        units = r.json().get("units", [])
        print(f"    Parsed {len(units)} units from {folder}")
        total_units.extend(units)

    if not total_units:
        print("    ERROR: No units parsed from any folder")
        return False

    r = requests.post(
        f"{API_BASE}/api/knowledge/upload",
        json={"units": total_units},
        headers={**headers, "Content-Type": "application/json"},
    )
    if r.status_code != 201:
        print(f"    Upload failed: {r.status_code} {r.text[:200]}")
        return False

    print(f"    Uploaded {len(r.json())} units total from {len(folders_seen)} methods")
    return True


def apply_patches(patches):
    parse_changed = False
    chat_changed = False

    for patch in patches:
        target = patch["target_prompt"]
        action = patch["action"]
        content = patch["content"]
        original = patch.get("original")
        location = patch.get("location", "end")

        filepath = PARSE_PROMPT_FILE if target == "PARSE_PROMPT" else CHAT_SERVICE_FILE

        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        if action == "modify" and original:
            if original in code:
                code = code.replace(original, content, 1)
                print(f"    Patched {target}: modified text")
            else:
                print(f"    WARNING: Could not find original text in {target}, skipping")
                continue
        elif action == "add":
            if target == "PARSE_PROMPT":
                marker = 'PARSE_PROMPT = """'
            else:
                marker = f'{target} = """'
            start = code.find(marker)
            if start == -1:
                print(f"    WARNING: Could not find {target}, skipping")
                continue
            end_marker_pos = code.find('"""', start + len(marker))
            if "end" in location.lower():
                code = code[:end_marker_pos] + "\n" + content + "\n" + code[end_marker_pos:]
            else:
                insert_pos = start + len(marker)
                code = code[:insert_pos] + "\n" + content + code[insert_pos:]
            print(f"    Patched {target}: added text at {location}")
        elif action == "remove" and original:
            if original in code:
                code = code.replace(original, "", 1)
                print(f"    Patched {target}: removed text")
            else:
                print(f"    WARNING: Could not find text to remove in {target}, skipping")
                continue

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        if target == "PARSE_PROMPT":
            parse_changed = True
        else:
            chat_changed = True

    return parse_changed, chat_changed


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_round_scores(eval_results):
    dims = ["retrieval_score", "accuracy_score", "core_answer_score", "honesty_score"]
    scores = {}
    for dim in dims:
        values = [r["eval"].get(dim, 0) for r in eval_results]
        scores[dim] = sum(values) / len(values) if values else 0
    # Practicality: only average over practical questions (where score > 0)
    prac_values = [r["eval"].get("practicality_score", 0) for r in eval_results
                   if r["eval"].get("practicality_score", 0) > 0]
    if prac_values:
        scores["practicality_score"] = sum(prac_values) / len(prac_values)
    scores["overall"] = sum(v for k, v in scores.items() if k not in ("overall", "practicality_score")) / 4
    return scores


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_eval_loop(max_rounds=5, do_reingest=False):
    api_key = get_openai_key()
    token = get_token()
    logger = ExperimentLogger()

    print("=" * 60)
    print("PROMPT OPTIMIZATION LOOP v2")
    print(f"Run ID: {logger.run_id}")
    print("=" * 60)

    knowledge = fetch_knowledge_units(token)
    if len(knowledge) == 0 or do_reingest:
        print("\n[INIT] Running initial ingest...")
        agent_reingest(token)
        knowledge = fetch_knowledge_units(token)

    if len(knowledge) == 0:
        print("ERROR: No knowledge units after ingest. Check API server.")
        sys.exit(1)

    print(f"\n  Knowledge base: {len(knowledge)} units")

    prev_scores = None
    prompt_version = 0
    current_scores = {}
    round_num = 0

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_num}/{max_rounds} (prompt v{prompt_version})")
        print(f"{'=' * 60}")

        round_data = {
            "round": round_num,
            "prompt_version": prompt_version,
            "timestamp": datetime.now().isoformat(),
        }

        # --- Agent 2: Generate questions from knowledge ---
        print("\n[Agent 2] Generating questions from knowledge base...")
        questions = agent_question_generator(api_key, knowledge)

        coverage_qs = questions.get("coverage", []) + questions.get("practical", []) + questions.get("cross_method", [])
        boundary_qs = questions.get("boundary", [])

        theory_qs = [q for q in coverage_qs if q.get("_type") == "theory"]
        practical_qs = [q for q in coverage_qs if q.get("_type") == "practical"]
        cross_qs = [q for q in coverage_qs if q.get("_type") == "cross_method"]
        print(f"  Theory questions ({len(theory_qs)}):")
        for i, q in enumerate(theory_qs):
            print(f"    T{i+1}: {q['question'][:80]}")
        print(f"  Practical questions ({len(practical_qs)}):")
        for i, q in enumerate(practical_qs):
            print(f"    P{i+1}: {q['question'][:80]}")
        print(f"  Cross-method questions ({len(cross_qs)}):")
        for i, q in enumerate(cross_qs):
            print(f"    X{i+1}: {q['question'][:80]}")
        print(f"  Boundary questions ({len(boundary_qs)}):")
        for i, q in enumerate(boundary_qs):
            print(f"    B{i+1}: {q['question'][:80]}")

        round_data["questions"] = questions

        # --- Agent 3 + 4: Answer and evaluate ---
        print("\n[Agent 3+4] Answering and evaluating...")
        session_id = f"eval-r{round_num}-{int(time.time())}"
        eval_results = []

        all_qs = [(q, "coverage") for q in coverage_qs] + [(q, "boundary") for q in boundary_qs]

        for i, (q_data, q_type) in enumerate(all_qs):
            if q_type == "coverage":
                label = f"{'P' if q_data.get('_type') == 'practical' else 'T'}{i+1}"
            else:
                label = f"B{i+1}"
            question = q_data["question"]
            print(f"\n  {label}: {question[:70]}...")

            answer = agent_answerer(question, token, session_id)
            evaluation = agent_evaluator(api_key, q_data, answer, q_type)

            eval_results.append({
                "question": question,
                "type": q_type,
                "answer": answer,
                "eval": evaluation,
            })

            ret = evaluation.get("retrieval_score", 0)
            acc = evaluation.get("accuracy_score", 0)
            core = evaluation.get("core_answer_score", 0)
            hon = evaluation.get("honesty_score", 0)
            prac = evaluation.get("practicality_score", 0)
            err = evaluation.get("error_type", "none")
            prac_str = f" Prac:{prac}" if prac > 0 else ""
            print(f"    Ret:{ret} Acc:{acc} Core:{core} Hon:{hon}{prac_str} | Error: {err}")

        round_data["eval_results"] = [
            {**r, "answer": r["answer"][:500]} for r in eval_results
        ]

        # --- Compute scores ---
        current_scores = compute_round_scores(eval_results)
        round_data["scores"] = current_scores

        print(f"\n  Scores: " + " | ".join(f"{k}={v:.1f}" for k, v in current_scores.items()))

        # --- Check if good enough ---
        if current_scores["overall"] >= 4.2 and all(v >= 3.5 for v in current_scores.values()):
            print(f"\n  All scores above threshold. Stopping early at round {round_num}.")
            round_data["decision"] = "early_stop_pass"
            logger.log_round(round_num, round_data)
            break

        # --- Agent 1: Generate patches ---
        print("\n[Agent 1] Analyzing errors and proposing patches...")
        current_prompts = read_current_prompts()
        optimizer_result = agent_prompt_optimizer(api_key, current_prompts, eval_results, prev_scores)

        patches = optimizer_result.get("patches", [])
        round_data["optimizer_result"] = optimizer_result

        if not patches:
            print("    No patches proposed.")
            round_data["decision"] = "no_patches"
            logger.log_round(round_num, round_data)
            prev_scores = current_scores
            continue

        print(f"    {len(patches)} patches proposed:")
        for p in patches:
            print(f"      - [{p['action']}] {p['target_prompt']}: {p['reasoning'][:80]}")

        # --- Gatekeeper ---
        print("\n[Gatekeeper] Evaluating patches...")
        accepted = gatekeeper_decide(current_scores, prev_scores, patches)
        round_data["gatekeeper_accepted"] = accepted

        if accepted:
            parse_changed, chat_changed = apply_patches(patches)
            prompt_version += 1
            round_data["prompt_version_after"] = prompt_version

            if parse_changed:
                print("\n[Agent 5] PARSE_PROMPT changed, re-ingesting...")
                time.sleep(3)
                agent_reingest(token)
                knowledge = fetch_knowledge_units(token)
            elif chat_changed:
                print("    Chat prompts updated. Server reloading...")
                time.sleep(3)

            round_data["decision"] = "patches_applied"
        else:
            round_data["decision"] = "patches_rejected"

        prev_scores = current_scores
        logger.log_round(round_num, round_data)

    # --- Final report ---
    print(f"\n{'=' * 60}")
    print("FINAL REPORT")
    print(f"{'=' * 60}")
    print(f"  Rounds completed: {round_num}")
    print(f"  Prompt versions: {prompt_version}")
    for k, v in current_scores.items():
        print(f"  {k}: {v:.1f}/5")

    logger.log_summary({
        "run_id": logger.run_id,
        "rounds_completed": round_num,
        "prompt_versions": prompt_version,
        "final_scores": current_scores,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt optimization eval loop v2")
    parser.add_argument("--rounds", type=int, default=5, help="Max optimization rounds")
    parser.add_argument("--reingest", action="store_true", help="Force re-ingest before starting")
    args = parser.parse_args()
    run_eval_loop(max_rounds=args.rounds, do_reingest=args.reingest)
