#!/usr/bin/env python3
"""GPT-based PDF Paper Parser. Each paper saved immediately after parsing.

Usage:
    python scripts/parse_papers_gpt.py --domain missing_data
    python scripts/parse_papers_gpt.py --domain all --max-papers 10
    python scripts/parse_papers_gpt.py --list
    python scripts/parse_papers_gpt.py --domain missing_data --dry-run
"""
import argparse, json, os, sys, time, traceback, io
from pathlib import Path

# Force UTF-8 output on Windows to handle non-ASCII filenames
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    print("Error: openai not installed"); sys.exit(1)
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "papers" / "download_config.json"
METADATA_DIR = PROJECT_ROOT / "papers" / "metadata"
PARSED_DIR = PROJECT_ROOT / "papers" / "parsed"
DEFAULT_MODEL = "gpt-4o-mini"
DELAY_BETWEEN_PAPERS = 5
MAX_RETRIES = 5
RETRY_DELAY = 60


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_metadata(domain):
    meta_path = METADATA_DIR / f"{domain}.json"
    if not meta_path.exists():
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_existing_outputs(domain):
    output_dir = PARSED_DIR / domain / "output"
    if not output_dir.exists():
        return set()
    return {f.stem + ".pdf" for f in output_dir.iterdir() if f.suffix == ".json"}

import re as _re
# Match short section title lines that signal proof/appendix/supplement sections to skip
_SKIP_SECTION_RE = _re.compile(
    r'^(?:[A-Z0-9]+[.):]\s+|[A-Z]\s+)?'
    r'(?:appendix|supplementary(?:\s+material)?|supplemental(?:\s+material)?|supplement|proofs?)'
    r'(?:\s+[A-Z0-9][.):]?)?\s*$',
    _re.IGNORECASE,
)

def _is_skip_section_start(page_text):
    head = page_text.strip()[:300]
    first_line = head.split('\n')[0].strip()
    return bool(_SKIP_SECTION_RE.match(first_line))

def extract_pdf_text(pdf_path, max_pages=80):
    try:
        reader = PdfReader(str(pdf_path))
        text_parts = []
        for i, page in enumerate(reader.pages[:max_pages]):
            t = page.extract_text()
            if not t:
                continue
            if _is_skip_section_start(t):
                print(f"    Skipping proof/appendix section from page {i+1}")
                break
            text_parts.append(f"--- Page {i+1} ---\n{t}")
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[ERROR extracting PDF: {e}]"

def get_pdf_path(config, domain, paper):
    filename = paper.get("filename", "")
    if not filename:
        return None
    pdf_parent = config[domain]["pdf_parent"]
    cluster = paper.get("cluster", "")
    for base in [
        PROJECT_ROOT / "papers" / "pdf" / pdf_parent / cluster / filename,
        PROJECT_ROOT / "papers" / "pdf" / "all" / filename,
    ]:
        if base.exists():
            return base
    return None


def load_system_prompt():
    prompt_path = PROJECT_ROOT / "scripts" / "parse_prompt.txt"
    if prompt_path.exists():
        raw = prompt_path.read_text(encoding="utf-8")
        lines = raw.split("\n")
        filtered = []
        skip = False
        for line in lines:
            if "## Papers to Process" in line or "## Output Directory" in line or "## Workflow" in line:
                skip = True
                continue
            if line.startswith("## ") and skip:
                skip = False
            if not skip:
                line = line.replace("{domain}", "").replace("{papers_list}", "").replace("{output_dir}", "")
                filtered.append(line)
        return "\n".join(filtered).strip()
    return "You are a knowledge extraction engine. Extract structured knowledge from academic papers. Return valid JSON."

_SYSTEM_PROMPT = None

def get_system_prompt():
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = load_system_prompt()
    return _SYSTEM_PROMPT

def build_user_prompt(paper, pdf_text, domain):
    filename = paper.get("filename", "")
    meta_lines = [
        "## Paper Metadata (may be inaccurate — verify from PDF)",
        "- Title: " + str(paper.get("title", "Unknown")),
        "- Authors: " + str(paper.get("authors", "Unknown")),
        "- Year: " + str(paper.get("year", "Unknown")),
        "- DOI: " + str(paper.get("doi", "")),
        "- Domain: " + domain,
        "- Cluster: " + str(paper.get("cluster", "")),
        "- Filename: " + filename,
    ]
    fmt_block = (
        "\n## Output Format\n\n"
        "Return a single JSON object with paper_metadata and knowledge_units.\n"
        "Return ONLY valid JSON, no markdown fences.\n"
        "IMPORTANT: Extract the ACTUAL title, authors, year from the PDF text itself.\n\n"

        "Use the JSON schema specified in the system instructions for each unit.\n\n"

        "## MANDATORY EXTRACTION CHECKLIST\n\n"
        "You MUST extract ALL of the following (each as a separate unit):\n"
        "1. Core method definition and purpose (1-2 units)\n"
        "2. Every Theorem, Proposition, Corollary, Lemma in the paper "
        "(each as a SEPARATE unit with full statement in evidence_span)\n"
        "3. Key model equations and optimization objectives (each as a SEPARATE unit)\n"
        "4. Every named assumption (e.g., Assumption 1, Condition A) as a separate unit\n"
        "5. Use case: when/why to use this method, what problem it solves\n"
        "6. Comparison results: how this method compares to specific alternatives "
        "(name the baselines, give numbers)\n"
        "7. Algorithm steps or computational procedure\n"
        "8. Convergence rate or theoretical guarantee (with the actual rate expression)\n"
        "9. Limitations, failure modes, or conditions where it does NOT work\n"
        "10. Key simulation/empirical findings with specific numbers\n\n"

        "## UNIT COUNT REQUIREMENT\n\n"
        "MINIMUM: 12 units. TARGET: 15-20 units.\n"
        "If you produce fewer than 12, you MUST go back and extract more.\n"
        "A typical paper has 3-5 theorems, 2-3 model equations, 2-3 assumptions, "
        "1-2 algorithms, 1-2 comparisons, and 1-2 use cases = 12-17 units.\n"
        "If the paper has fewer theorems, extract more from experiments and comparisons.\n\n"

        "## FIELD QUALITY RULES\n\n"
        "- knowledge_type: MUST be one of: definition, assumption, algorithm, "
        "parameter, result, limitation, implementation, failure_mode, comparison, theorem\n"
        "  (use underscores, never spaces: 'failure_mode' not 'failure mode')\n"
        "- content: 2-3 sentences with SPECIFIC details (numbers, conditions, model names)\n"
        "- evidence_span: copy the EXACT formula or key sentence from the paper\n"
        "- method_name: the PRIMARY method name/acronym from the paper (never a section name)\n"
        "- related_methods: ALWAYS list 1-3 related or compared methods. "
        "Every paper compares to something — name those methods.\n"
        "- confidence: use 'high' for theorems/proofs, 'medium' for empirical results, "
        "'low' for heuristic/conjecture claims\n"
        "- typical_questions: 2-4 realistic questions a researcher would ask\n"
        "- reusable_for_questions: 2-4 questions this unit helps answer\n"
        "- problem_it_solves: one specific sentence, not 'solves regression' "
        "but 'selects relevant variables when p >> n under sparsity'\n"
        "- NEVER write vague descriptions like 'improves convergence' — say HOW and by HOW MUCH\n"
        "- NEVER write 'outperforms baselines' — name the baselines and give numbers\n\n"

        "## BAD vs GOOD examples\n\n"
        "BAD content: 'The method improves convergence of MCMC algorithms.'\n"
        "GOOD content: 'BSR uses block-specific optimal working parameters w_t to reparametrize "
        "latent states differently for each parameter block in the Gibbs sampler. "
        "For SV models with phi=0.95 and sigma2_eta=0.5, BSR achieves inefficiency factor 13 "
        "for sigma2_eta vs ASIS at 24.'\n\n"
        "BAD evidence_span: 'The convergence rate is optimized.'\n"
        "GOOD evidence_span: 'Rate of convergence = rho(w)^T V_0 rho(w) / tau(w), "
        "where tau(w) = sigma^{-2}_eps w^T w + sigma^{-2}_eta w_bar^T Lambda w_bar'\n\n"
        "BAD related_methods: []\n"
        "GOOD related_methods: ['LASSO', 'Ridge Regression', 'Elastic Net']\n"
    )
    return "\n".join(meta_lines) + "\n\n## Paper Text\n\n" + pdf_text + fmt_block


def _call_gpt(client, system_prompt, user_prompt, model):
    """Single GPT API call with retry logic. Returns parsed JSON or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=16384,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)

            usage = response.usage
            if usage:
                print(f"    Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")
            return parsed

        except RateLimitError:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"    Rate limited ({attempt}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Rate limit exceeded after {MAX_RETRIES} attempts")
                return None
        except APIError as e:
            print(f"    API error ({attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(10)
            else:
                return None
        except json.JSONDecodeError as e:
            print(f"    JSON parse error ({attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5)
            else:
                return None
        except Exception as e:
            print(f"    Unexpected error: {e}")
            traceback.print_exc()
            return None
    return None


MIN_UNITS = 12


def _request_more_units(client, paper, pdf_text, domain, model, existing_parsed):
    """Follow-up request when initial parse produced too few units."""
    existing_units = existing_parsed.get("knowledge_units", [])
    existing_titles = [u.get("title", "") for u in existing_units]

    follow_up = (
        "You previously extracted these units from this paper:\n"
        + "\n".join(f"- {t}" for t in existing_titles)
        + f"\n\nThat is only {len(existing_units)} units. The MINIMUM is {MIN_UNITS}.\n"
        "Go back through the paper and extract MORE units you missed. Focus on:\n"
        "- Theorems/Propositions/Corollaries you skipped\n"
        "- Assumptions listed in the paper\n"
        "- Comparison results with specific baselines and numbers\n"
        "- Algorithm steps or computational procedures\n"
        "- Simulation findings with specific metrics\n\n"
        "Return ONLY a JSON object with a single key 'knowledge_units' "
        "containing the ADDITIONAL units (not duplicates of existing ones).\n"
        "Return ONLY valid JSON, no markdown fences.\n"
        "Use the same JSON schema as before for each unit."
    )
    user_prompt = build_user_prompt(paper, pdf_text, domain) + "\n\n" + follow_up
    return _call_gpt(client, get_system_prompt(), user_prompt, model)


def parse_paper_with_gpt(client, paper, pdf_path, domain, model):
    pdf_text = extract_pdf_text(pdf_path)
    if pdf_text.startswith("[ERROR"):
        print(f"    PDF extraction failed: {pdf_text}")
        return None
    if len(pdf_text) > 100000:
        pdf_text = pdf_text[:100000] + "\n\n[... truncated ...]"

    user_prompt = build_user_prompt(paper, pdf_text, domain)
    parsed = _call_gpt(client, get_system_prompt(), user_prompt, model)

    if not parsed or not parsed.get("knowledge_units"):
        print("    Warning: No/empty knowledge_units")
        return None

    # If too few units, request more
    n = len(parsed["knowledge_units"])
    if n < MIN_UNITS:
        print(f"    Only {n} units (min {MIN_UNITS}), requesting more...")
        extra = _request_more_units(client, paper, pdf_text, domain, model, parsed)
        if extra and extra.get("knowledge_units"):
            # Deduplicate by title (case-insensitive)
            existing_titles = {
                u.get("title", "").strip().lower()
                for u in parsed["knowledge_units"]
            }
            new_units = [
                u for u in extra["knowledge_units"]
                if u.get("title", "").strip().lower() not in existing_titles
            ]
            parsed["knowledge_units"].extend(new_units)
            skipped = len(extra["knowledge_units"]) - len(new_units)
            msg = f"    Added {len(new_units)} more -> {len(parsed['knowledge_units'])} total"
            if skipped:
                msg += f" (skipped {skipped} duplicates)"
            print(msg)

    return parsed


def list_domains(config):
    header = f"{'Domain':<45} {'Meta':>6} {'Parsed':>7} {'Remaining':>10}"
    print(header)
    print("-" * 75)
    for domain in sorted(config.keys()):
        meta_path = METADATA_DIR / f"{domain}.json"
        mc = 0
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                mc = len(json.load(f))
        existing = get_existing_outputs(domain)
        print(f"{domain:<45} {mc:>6} {len(existing):>7} {mc - len(existing):>10}")


def process_domain(domain, config, client, model, max_papers=None, dry_run=False):
    stats = {"domain": domain, "processed": 0, "skipped": 0, "failed": 0, "total_units": 0}
    metadata = load_metadata(domain)
    if not metadata:
        print(f"  No metadata for this domain. Run fetch_metadata.py first.")
        return stats

    existing = get_existing_outputs(domain)
    pending = [p for p in metadata if p.get("filename") and p["filename"] not in existing]
    if not pending:
        print(f"  All {len(metadata)} papers already parsed.")
        return stats
    if max_papers:
        pending = pending[:max_papers]

    output_dir = PARSED_DIR / domain / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Total: {len(metadata)}, Done: {len(existing)}, To process: {len(pending)}")

    for i, paper in enumerate(pending, 1):
        filename = paper["filename"]
        stem = Path(filename).stem
        out = output_dir / f"{stem}.json"

        # Re-check at process time: skip if already done (handles duplicate filenames in metadata)
        if out.exists():
            stats["skipped"] += 1
            continue

        print(f"\n  [{i}/{len(pending)}] {filename}")

        pdf_path = get_pdf_path(config, domain, paper)
        if not pdf_path:
            print("    PDF not found, skipping")
            stats["skipped"] += 1
            continue

        if dry_run:
            print("    DRY RUN: would parse")
            stats["processed"] += 1
            continue

        parsed = parse_paper_with_gpt(client, paper, pdf_path, domain, model)
        if parsed and parsed.get("knowledge_units"):
            with open(out, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            n = len(parsed["knowledge_units"])
            stats["processed"] += 1
            stats["total_units"] += n
            print(f"    Saved: {n} units -> {out.name}")
        else:
            stats["failed"] += 1
            print("    FAILED")

        if i < len(pending):
            time.sleep(DELAY_BETWEEN_PAPERS)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Parse PDFs into knowledge units using GPT API")
    ap.add_argument("--domain", type=str, help="Domain name or all")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    config = load_config()

    if args.list:
        list_domains(config)
        return
    if not args.domain:
        ap.error("--domain required (or use --list)")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    client = None if args.dry_run else OpenAI(api_key=api_key)

    if args.domain == "all":
        domains = sorted(config.keys())
    elif args.domain in config:
        domains = [args.domain]
    else:
        print(f"Error: Unknown domain")
        sys.exit(1)

    domains = [d for d in domains if (METADATA_DIR / f"{d}.json").exists()]
    print(f"Model: {args.model}")
    if args.dry_run:
        print("=== DRY RUN ===")

    all_stats = []
    for domain in domains:
        sep = "=" * 60
        print(f"\n{sep}\nDomain: {domain}\n{sep}")
        stats = process_domain(domain, config, client, args.model,
                               max_papers=args.max_papers, dry_run=args.dry_run)
        all_stats.append(stats)

    sep = "=" * 60
    print(f"\n{sep}\nSummary\n{sep}")
    tp = sum(s["processed"] for s in all_stats)
    tf = sum(s["failed"] for s in all_stats)
    ts = sum(s["skipped"] for s in all_stats)
    tu = sum(s["total_units"] for s in all_stats)
    print(f"  Processed: {tp}")
    print(f"  Failed:    {tf}")
    print(f"  Skipped:   {ts}")
    print(f"  KU total:  {tu}")


if __name__ == "__main__":
    main()
