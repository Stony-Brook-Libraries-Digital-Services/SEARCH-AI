from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from time import perf_counter
import re
import urllib.parse
from typing import Dict, Optional, Tuple, List
from requests.adapters import HTTPAdapter, Retry
import requests
import openai
from datetime import datetime
from hidden.varlist import online_flagg, local_flag, peer_flagg, OPENAI_KEY, OPENAI_KEY_BOOLEAN, Prompt1, Prompt2, Prompt3, Prompt4, prefix, MaterialTypeMap, rtypes, pattern_time, pattern_type, school_held, held_by_school


    
REQUEST_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60")) 
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.5"))
OPENAI_API_KEY = OPENAI_KEY
OPENAI_API_KEY_BOOLEAN = OPENAI_KEY_BOOLEAN
OPENAI_MODEL = "gpt-4.1-mini"
openai.api_key = OPENAI_API_KEY

DEBUG = os.getenv("DEBUG", "False").upper() == "TRUE"

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Warn if API key missing
if not OPENAI_API_KEY:
    logger.warning("Environment variable OPENAI_API_KEY is not set. OpenAI calls will fail.")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Streaming LLM Request with Retry Logic
# ─────────────────────────────────────────────────────────────────────────────
def send_chat_prompt(
    system_prompt: str,
    user_prompt: str,
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    api_key: Optional[str] = None,   # <-- NEW
) -> str:
    """
    Sends a chat request to OpenAI and returns the response text.
    Implements exponential backoff on failures.
    """
    key_to_use = api_key or OPENAI_API_KEY
    if not key_to_use:
        raise RuntimeError("OpenAI API key is not set")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            # IMPORTANT: pass per-request key so we don't mutate global openai.api_key
            resp = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                api_key=key_to_use,   # <-- NEW (supported by modern openai python)
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait_time = RETRY_BACKOFF ** attempt
                logger.info("OpenAI call failed (attempt %d): %s — retrying in %.1fs", attempt, e, wait_time)
                time.sleep(wait_time)
            else:
                logger.critical("OpenAI call failed after %d attempts", attempt)
                raise RuntimeError(f"OpenAI request failed after maximum retries: {e}") from e

# ─────────────────────────────────────────────────────────────────────────────
# Agent 1: Query Splitter with Flags
# ─────────────────────────────────────────────────────────────────────────────
def split_query(user_query: str, verbose: bool = False) -> Dict[str, any]:
    """
    Splits the user's query into:
      - peer_flag: True if 'peer reviewed' in query
      - sbu_flag: True if 'sbu held' or 'held by sbu' in query
      - online_flag: True if 'available online' in query
      - topic_input: subject/filter terms (excluding words used in other flags and inputs)
      - time_input: exact time phrase (if present)
      - type_input: material-type word normalized (if present)

    Returns a dict with keys:
      'peer_flag', 'sbu_flag', 'online_flag',
      'topic_input', 'time_input', 'type_input'.

    It tries to use the LLM to parse topic/time/type out of the query,
    but first strips any of the three flags so they don’t pollute the topic.
    """
    if verbose:
        logger.info("Splitting query with flags: %s", user_query)

    lower_q = user_query.lower()
    peer_flag = bool(re.search(peer_flagg, lower_q))
    sbu_flag = bool(re.search(local_flag, lower_q))
    online_flag = bool(re.search(online_flagg, lower_q))

    cleaned_query = user_query
    for phrase in ["peer reviewed", "available online", school_held , held_by_school]:
        cleaned_query = re.sub(re.escape(phrase), "", cleaned_query, flags=re.IGNORECASE)
    cleaned_query = re.sub(r"\s{2,}", " ", cleaned_query).strip()
    system_prompt = Prompt1[0]["content"]

    user_prompt = f"User Query: \"{cleaned_query}\""
    raw = ""

    try:
        raw = send_chat_prompt(system_prompt, user_prompt)
        json_start = raw.find("{")
        json_end = raw.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            raw_json = raw[json_start : json_end + 1]
            parsed = json.loads(raw_json)
            topic = parsed.get("topic_input", "").strip()
            time_phrase = parsed.get("time_input", "").strip()
            raw_type = parsed.get("type_input", "")
            if isinstance(raw_type, str):
                type_word = [raw_type.strip()] if raw_type.strip() else []
            elif isinstance(raw_type, list):
                type_word = [t.strip().lower() for t in raw_type if isinstance(t, str)]
            else:
                type_word = []
            if verbose:
                logger.info(
                    "Parsed output: peer=%s, sbu=%s, online=%s, topic='%s', time='%s', type='%s'",
                    peer_flag,
                    sbu_flag,
                    online_flag,
                    topic,
                    time_phrase,
                    type_word,
                )
            return {
                "peer_flag": peer_flag,
                "sbu_flag": sbu_flag,
                "online_flag": online_flag,
                "topic_input": topic,
                "time_input": time_phrase,
                "type_input": type_word,
            }
        else:
            raise ValueError("No valid JSON object found in LLM response.")
    except Exception as e:
        logger.warning("Failed to parse JSON from split_query (error: %s). Falling back to regex.", e)

        time_pattern = (pattern_time)
        type_pattern = (pattern_type)
        time_match = re.search(time_pattern, cleaned_query, re.IGNORECASE)
        type_match = re.search(type_pattern, cleaned_query, re.IGNORECASE)

        time_phrase = time_match.group("time").strip() if time_match else ""
        type_word = type_match.group(0).strip().lower() if type_match else ""
        if type_word.endswith("s"):
            type_word = type_word[:-1]

        temp = cleaned_query
        if time_phrase:
            temp = temp.replace(time_phrase, "")
        if type_match:
            temp = temp.replace(type_match.group(0), "")
        topic = re.sub(r"\s{2,}", " ", temp).strip()

        if verbose:
            logger.info(
                "Fallback parsing: peer=%s, sbu=%s, online=%s, topic='%s', time='%s', type='%s'",
                peer_flag,
                sbu_flag,
                online_flag,
                topic,
                time_phrase,
                type_word,
            )
        return {
            "peer_flag": peer_flag,
            "sbu_flag": sbu_flag,
            "online_flag": online_flag,
            "topic_input": topic,
            "time_input": time_phrase,
            "type_input": type_word,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2: Year Range Extractor
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache
from typing import Tuple, Optional, Dict

DT_FILE = "deterministicttimes.txt"

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # Normalize different dash / punctuation to spaces
    s = re.sub(r"[-–—/_.:,;()\"'`\\\[\]]+", " ", s)
    # Normalize common non-breaking spaces and other unicode spaces to a regular space
    s = re.sub(r"\s+", " ", s)
    # Normalize ordinal suffixes (1st, 2nd, 3rd, 19th -> 1,2,3,19)
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", s)
    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _here_path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)

@lru_cache(maxsize=1)
def _load_dt_txt_cached(path: str) -> Dict[str, Tuple[int, int]]:
    mapping: Dict[str, Tuple[int, int]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, years = line.split("=", 1)
                key = _normalize(key)
                try:
                    a_str, b_str = years.split(",", 1)
                    a, b = int(a_str), int(b_str)
                    if a > b:
                        a, b = b, a
                    mapping[key] = (a, b)
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return mapping

def refresh_dt_cache() -> None:
    """Call this if you edit dt.txt while the app is running."""
    _load_dt_txt_cached.cache_clear()

def extract_year_range(time_input: str, verbose: bool = False) -> Tuple[Optional[int], Optional[int]]:
    """
    Converts a time phrase into a standardized year range tuple (start_year, end_year).
    1. First checks dt.txt for an exact match.
    2. Falls back to LLM only if not found.
    """
    if not time_input:
        if verbose:
            logger.info("No time input provided, returning (None, None)")
        return (None, None)

    # --- Step 1: dt.txt lookup ---
    mapping = _load_dt_txt_cached(_here_path(DT_FILE))
    s = _normalize(time_input)
    variants = {
        s,
        re.sub(r"^(the|a|an)\s+", "", s),  # allow leading-article-free form
        re.sub(r"\s+", " ", s).strip(),    # collapse spaces
    }
    for v in variants:
        if v in mapping:
            if verbose:
                logger.info("Resolved via dt.txt: %s -> %s", time_input, mapping[v])
            return mapping[v]

    # --- Step 2: LLM fallback ---
    if verbose:
        logger.info("Extracting year range via LLM from: %s", time_input)

    try:
        system_prompt =  Prompt2[0]["content"]
        
        user_prompt = f'Normalize this time phrase to a year range: "{time_input}"\nReturn only format: YYYY,YYYY or NONE,NONE'

        raw = send_chat_prompt(system_prompt, user_prompt).strip()

        if verbose:
            logger.info("LLM normalized time phrase output: %s", raw)

        if raw.upper() == "NONE,NONE":
            return (None, None)

        # --- Parse the LLM output into a (start, end) tuple ---
        m = re.match(r"^\s*(\d{4})\s*,\s*(\d{4})\s*$", raw)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return (min(a, b), max(a, b))

        years = re.findall(r"\b(\d{4})\b", raw)
        if len(years) >= 2:
            a, b = int(years[0]), int(years[1])
            return (min(a, b), max(a, b))
        elif len(years) == 1:
            y = int(years[0])
            return (y, y)
        
        return (None, None)

    except Exception as e:
        logger.warning("Failed to normalize time phrase '%s'. Error: %s", time_input, str(e))
        return (None, None)

# ─────────────────────────────────────────────────────────────────────────────
# Agent 3: Boolean Expression Builder (Fixed)
# ─────────────────────────────────────────────────────────────────────────────
def build_boolean_string(topic_input: str, verbose: bool = False) -> str:
    if not topic_input:
        return ""
    if verbose:
        logger.info("Building boolean string from: %s", topic_input)

    system_prompt = Prompt3[0]["content"]
    user_prompt = f"Topic Input: \"{topic_input}\"\nBuild a Comprehensive Boolean search string"

    try:
        raw = send_chat_prompt(
            system_prompt,
            user_prompt,
            api_key=OPENAI_API_KEY_BOOLEAN,  # <-- uses second key
        )
        boolean_expr = raw.strip()

        # Clean up quotes in each Boolean segment
        segments = re.split(r'(\s+(?:AND|OR|NOT)\s+)', boolean_expr)
        cleaned_segments = []

        for segment in segments:
            if segment.strip() in ['AND', 'OR', 'NOT'] or re.match(r'^\s+(?:AND|OR|NOT)\s+$', segment):
                cleaned_segments.append(segment)
                continue

            if '*' in segment:
                segment = segment.replace('"', '')
            else:
                quote_count = segment.count('"')
                if quote_count % 2 != 0:
                    segment = segment.replace('"', '')

            cleaned_segments.append(segment)

        boolean_expr = ''.join(cleaned_segments)

    except Exception as e:
        logger.error("OpenAI API error in build_boolean_string: %s", e)
        boolean_expr = ""

    if verbose:
        logger.info("Boolean expression: %s", boolean_expr)

    return boolean_expr




# ─────────────────────────────────────────────────────────────────────────────
# Agent 4: Material-Type Normalizer
# ─────────────────────────────────────────────────────────────────────────────

def extract_material_types(type_input: str, use_llm_fallback: bool = True, verbose: bool = False) -> List[str]:

    """
    Normalizes material type phrases to their Discovery-compatible resource types.
    Returns a list (deduplicated). Uses LLM fallback if enabled.
    """
    if not type_input:
        return []

    # Tokenize by common conjunctions and separators
    if isinstance(type_input, list):
        tokens = [t.lower().strip() for t in type_input if isinstance(t, str)]
    elif isinstance(type_input, str):
        tokens = re.split(r"[,\s]*\b(?:and|or|,)\b[\s]*", type_input.lower())
        tokens = [t.strip() for t in tokens if t.strip()]
    else:
        tokens = []

    normalized_types = []

    # Controlled vocabulary map
    MATERIAL_TYPE_MAP = MaterialTypeMap

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Try map lookup
        if token in MATERIAL_TYPE_MAP:
            normalized_types.append(MATERIAL_TYPE_MAP[token])
        elif use_llm_fallback and len(token) > 2:
            # Try LLM-based normalization if unknown
            system_prompt = Prompt4[0]["content"]
            user_prompt = f"Material type: \"{token}\""
            try:
                raw = send_chat_prompt(system_prompt, user_prompt).strip().lower()
                mapped = MATERIAL_TYPE_MAP.get(raw, raw)
                if mapped in MATERIAL_TYPE_MAP.values():
                    normalized_types.append(mapped)
                elif verbose:
                    logger.warning("LLM returned unrecognized material type: %s", raw)
            except Exception as e:
                if verbose:
                    logger.warning("LLM fallback failed on '%s': %s", token, e)

        elif verbose:
            logger.warning("Unrecognized material type (no LLM fallback): %s", token)

    return list(set(normalized_types))


# ─────────────────────────────────────────────────────────────────────────────
# URL Construction: Discovery Search
# ─────────────────────────────────────────────────────────────────────────────
def construct_url(
    boolean: str,
    rtype: Optional[List[str]] = None,
    peer_flag: bool = False,
    sbu_flag: bool = False,
    online_flag: bool = False,
    creationdate: Optional[str] = None,
) -> str:
    """
    Build a Library Discovery search URL using the boolean query
    and optional filters (resource types, peer reviewed, SBU-held, online).
    Properly URL-encodes the boolean expression.
    """
    _prefix = prefix
    _query_prefix = "&query="
    _query_field = "any,"
    _filter = "contains,"

    encoded_boolean = urllib.parse.quote_plus(boolean or "*")

    url = f"{_prefix}{_query_prefix}{_query_field}{_filter}{encoded_boolean}"

    if peer_flag:
        url += "&mfacet=tlevel,include,peer_reviewed,1"
    if sbu_flag:
        url += "&mfacet=tlevel,include,available_p,1"
    if online_flag:
        url += "&mfacet=tlevel,include,online_resources,1"
    if creationdate:
        url += f"&mfacet=searchcreationdate,include,{creationdate},1"

    valid_rtypes = rtypes
    
    for r in rtype or []:
        rtype_match = r.strip().lstrip("$")
        if rtype_match in valid_rtypes:
            url += f"&mfacet=rtype,include,{rtype_match},1"
        elif rtype_match.lower() != "none" and DEBUG:
            print(f"Warning: Unexpected resource type '{rtype_match}'")

    url += "&search_scope=EverythingNZBooks"
    return url


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Process Query for URL Mode (Revised)
# ─────────────────────────────────────────────────────────────────────────────
def process_query(user_query: str, verbose: bool = False) -> Tuple[bool, bool, bool, List[str], str, Optional[str]]:
    """
    Process the user_query to extract flags (handled at splitter), boolean query, and creation date:
      - peer_reviewed, sbuheld, online_resources now provided by split_query
      - rtype_list: list containing plural material type if present
      - boolean_query: constructed boolean expression if topic_input present
      - creationdate: encoded date range if time_input present and valid

    Returns a tuple: (peer_reviewed, sbuheld, online_resources, rtype_list, boolean_query, creationdate)
    """
    # Step 1: Split query via Agent 1 (now includes flags)
    parts = split_query(user_query, verbose=verbose)
    peer_flag = parts.get("peer_flag", False)
    sbu_flag = parts.get("sbu_flag", False)
    online_flag = parts.get("online_flag", False)
    time_input = parts.get("time_input", "")
    type_input = parts.get("type_input", "")
    topic_input = parts.get("topic_input", "")

    # Step 2: If time_input is present, call Agent 2 to get year range and format creation date
    creationdate = None
    if time_input:
        start_year, end_year = extract_year_range(time_input, verbose=verbose)
        if start_year is not None and end_year is not None:
            creationdate = format_creation_date(start_year, end_year)

    # Step 3: If type_input is present, call Agent 4 to normalize material type
    rtype_list: List[str] = []
    if type_input:
        # Extract material types (plural form, normalized)
        rtype_list = extract_material_types(type_input, verbose=verbose)
        # Ensure all material types are pluralized properly
        rtype_list = [rtype + "s" if not rtype.endswith("s") else rtype for rtype in rtype_list]

    # Step 4: If topic_input is present, call Agent 3 to build boolean expression
    boolean_expr = ""
    if topic_input:
        boolean_expr = build_boolean_string(topic_input, verbose=verbose)

    return peer_flag, sbu_flag, online_flag, rtype_list, boolean_expr, creationdate

def format_creation_date(start: int, end: int) -> str:
    """
    Format a creation date range for the Discovery URL.
    Encoded as: 1560%7C,%7C2025
    """
    return f"{start}%7C,%7C{end}"

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: Process Full Query (JSON) and URL Mode
# ─────────────────────────────────────────────────────────────────────────────
def split_full_query(user_query: str, verbose: bool = False) -> Dict[str, Optional[str]]:
    """
    End-to-end pipeline for JSON output:
      1. split_query → peer_flag, sbu_flag, online_flag, topic_input, time_input, type_input
      2. extract_year_range(time_input) → year_range tuple
      3. build_boolean_string(topic_input) → boolean_query
      4. extract_material_type(type_input) → mtype

    Returns a dict with keys:
      - 'original_query': original user query
      - 'boolean_query': constructed boolean expression
      - 'year_range': normalized year range as string (YYYY-YYYY) or empty string
      - 'type': normalized material type
      - 'split_parts': dict of intermediate outputs (including flags)
    """
    if verbose:
        logger.info("Starting full query split for: %s", user_query)

    parts = split_query(user_query, verbose=verbose)
    topic_input = parts.get("topic_input", "")
    time_input = parts.get("time_input", "")
    type_input = parts.get("type_input", "")

    start_year, end_year = extract_year_range(time_input, verbose=verbose)

    # Format year range as string for backward compatibility
    if start_year is not None and end_year is not None:
        if start_year == end_year:
            year_range = str(start_year)
        else:
            year_range = f"{start_year}-{end_year}"
    else:
        year_range = ""

    boolean_query = build_boolean_string(topic_input, verbose=verbose)
    mtype = extract_material_types(type_input, verbose=verbose)

    result = {
        "original_query": user_query.strip(),
        "boolean_query": boolean_query,
        "year_range": year_range,
        "type": mtype,
        "split_parts": parts,
    }
    if verbose:
        logger.info("Full split result: %s", json.dumps(result, ensure_ascii=False, indent=2))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# JSON Output Helper
# ─────────────────────────────────────────────────────────────────────────────
def print_json(data: Dict) -> None:
    """
    Pretty-print a dictionary as compact JSON to standard output.
    """
    sys.stdout.write(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Command-Line Interface
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """
    Set up and parse CLI arguments.

    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Enhanced LLM-based query agents: split|extract_year|build_bool|extract_type|split_full|url"
    )
    parser.add_argument(
        "mode",
        choices=["split", "extract_year", "build_bool", "extract_type", "split_full", "url"],
        help="Which agent or function to run",
    )
    parser.add_argument(
        "query",
        nargs=argparse.REMAINDER,
        help="The user query to process (enclose in quotes if multi-word)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print a timing breakdown for split_full stages"
    )
    parser.add_argument(
        "--profile-json",
        action="store_true",
        help="Emit JSON with result and timing breakdown when profiling"
    )
    return parser.parse_args()

def get_retrying_session(
    total: int = MAX_RETRIES,
    backoff_factor: float = RETRY_BACKOFF,
    status_forcelist: List[int] = [429, 500, 502, 503, 504],
) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["POST", "GET"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def run_split_full_profile(raw_query: str, verbose: bool = False, json_out: bool = False):
    # Stage timings
    t0 = perf_counter()
    t1 = perf_counter()
    parts = split_query(raw_query, verbose=verbose)
    t2 = perf_counter()
    t3 = perf_counter()
    start_year, end_year = extract_year_range(parts.get("time_input", ""), verbose=verbose)
    t4 = perf_counter()
    t5 = perf_counter()
    boolean_query = build_boolean_string(parts.get("topic_input", ""), verbose=verbose)
    t6 = perf_counter()
    t7 = perf_counter()
    mtype = extract_material_types(parts.get("type_input", ""), verbose=verbose)
    t8 = perf_counter()
    # Outputs comparable to split_full_query
    if start_year is not None and end_year is not None:
        year_range = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"
    else:
        year_range = ""
    result = {
        "original_query": raw_query.strip(),
        "boolean_query": boolean_query,
        "year_range": year_range,
        "type": mtype,
        "split_parts": parts,
    }
    breakdown = {
        "split_query": t2 - t1,
        "extract_year_range": t4 - t3,
        "build_boolean_string": t6 - t5,
        "extract_material_types": t8 - t7,
        "total": t8 - t0,
    }
    if json_out:
        print(json.dumps({"result": result, "timings": breakdown}, indent=2))
    else:
        print("Timing breakdown:")
        for k, v in breakdown.items():
            print(f"  {k:24s} {v*1000:9.2f} ms")
        print()
        print("Boolean Query:", boolean_query)
        if year_range:
            print("Year Range:", year_range)
        if mtype:
            print("Type:", mtype)
    return result, breakdown

def main() -> None:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    if not args.query:
        logger.error("No query provided. Specify the query after the mode.")
        sys.exit(1)

    raw_query = " ".join(args.query).strip()
    if not raw_query:
        logger.error("Empty query string.")
        sys.exit(1)

    try:
        if args.mode == "split":
            result = split_query(raw_query, verbose=args.verbose)
            print_json(result)
        elif args.mode == "extract_year":
            start_year, end_year = extract_year_range(raw_query, verbose=args.verbose)
            if start_year is not None and end_year is not None:
                if start_year == end_year:
                    year_range = str(start_year)
                else:
                    year_range = f"{start_year}-{end_year}"
            else:
                year_range = ""
            print_json({"year_range": year_range})
        elif args.mode == "build_bool":
            boolean_str = build_boolean_string(raw_query, verbose=args.verbose)
            print_json({"boolean_query": boolean_str})
        elif args.mode == "extract_type":
            material = extract_material_types(raw_query, verbose=args.verbose)
            print_json({"type": material})
        elif args.mode == "split_full":
            if getattr(args, "profile", False):
                run_split_full_profile(
                    raw_query,
                    verbose=args.verbose,
                    json_out=getattr(args, "profile_json", False),
                )
                return
            result = split_full_query(raw_query, verbose=args.verbose)
            print_json(result)
        elif args.mode == "url":
            # process_query to follow agentic workflow
            peer_flag, sbu_flag, online_flag, rtype_list, boolean_expr, creationdate = process_query(
                raw_query, verbose=args.verbose
            )
            redirect_url = construct_url(
                boolean_expr,
                rtype_list,
                peer_flag,
                sbu_flag,
                online_flag,
                creationdate=creationdate
            )
            if DEBUG:
                print("Debug Mode Enabled")
                print("Processed Query Values:")
                print(f"Peer Reviewed: {peer_flag}")
                print(f"SBU Held: {sbu_flag}")
                print(f"Online Resources: {online_flag}")
                print(f"Resource Type(s): {rtype_list}")
                print(f"Boolean Query: {boolean_expr}")
                print(f"Creation Date: {creationdate}")
                print(f"Constructed URL: {redirect_url}")
            else:
                print(redirect_url.strip())
        else:
            logger.error("Unknown mode: %s", args.mode)
            sys.exit(1)
    except Exception as e:
        logger.exception("Error occurred during processing: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")

        sys.exit(1)

