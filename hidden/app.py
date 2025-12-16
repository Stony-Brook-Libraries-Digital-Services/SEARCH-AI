# -----------------------------------------------------------------------------
# SearchAI Log Dashboard (Flask app)
# -----------------------------------------------------------------------------
# PURPOSE 
# This file powers a small web application that turns the raw SearchAI log file
# (`search.jsonl`) into a human-readable dashboard for librarians, analysts,
# and developers.
#
# The dashboard lets you:
#   - Watch new searches appear in real time.
#   - See how long searches take (latency) and which ones are slowest.
#   - Track how many searches are happening per day and when errors occur.
#   - Estimate how complex users' searches are (simple vs. advanced queries).
#   - Tag searches by "material type" (books, articles, videos, etc.) and
#     "subject area" (AI, psychology, medicine, etc.) using fuzzy matching.
#   - Attach human comments/annotations to individual log lines.
#
# HOW IT FITS INTO SEARCHAI
# This app does not run the search itself. Instead, it reads a log file that is
# written by the SearchAI proxy. Each line in `search.jsonl` is one search
# event, stored as a JSON object (for example, with fields like `query`,
# `elapsed_ms`, `timestamp`, `status`, etc.).
#
# CUSTOMIZING FOR YOUR OWN LIBRARY OR PROJECT
# You can adapt this file to your own environment by changing:
#   - JSONL_PATH: where your log file lives and what it is called.
#   - MATERIAL_TYPES / SUBJECT_AREAS: how you classify searches.
#   - The metrics in the `/metrics` route: what you want to measure.
#   - The comment system: keep it, remove it, or point it to another database.
#
# NON-TECHNICAL SUMMARY
# In short, this is a "window" into how people use your library search:
# it reads a plain-text log file, organizes it into meaningful statistics
# and tags, and presents it through a web interface so humans can understand
# and improve the search experience.

# app.py (updated with slowest list, histogram, CSV export, row modal, heatmap)
import time
import os
import json
from datetime import datetime, timezone, timedelta
# time zone support: prefer stdlib zoneinfo (Py 3.9+), otherwise use dateutil.tz
try:
    from zoneinfo import ZoneInfo
    LOCAL_ZONE = ZoneInfo("America/New_York")
except Exception:
    # fallback to python-dateutil which is pure-python and doesn't require MSVC
    from dateutil import tz
    def ZoneInfo(name):
        # return an object that's compatible with tzinfo
        return tz.gettz(name)
    LOCAL_ZONE = ZoneInfo("America/New_York")
from collections import Counter, defaultdict
from flask import Flask, Response, request, jsonify, render_template_string, render_template
import sqlite3
import hashlib
from datetime import datetime
import re
from rapidfuzz import fuzz, process

COMMENTS_DB = "comments.db"

def init_db():
    # create DB and table if missing
    con = sqlite3.connect(COMMENTS_DB, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS comments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      line_hash TEXT NOT NULL,
      author TEXT,
      text TEXT NOT NULL,
      created_at TEXT NOT NULL
    );""")
    con.commit()
    return con

# global DB connection (simple for a small single-process app)
DB_CON = init_db()


APP = Flask(__name__, static_folder='static', static_url_path='/static')
JSONL_PATH = "search.jsonl"
POLL_INTERVAL = 1.0

# -----------------------------------------------------------------------------
# CONTROLLED VOCABULARIES FOR TAGGING
# -----------------------------------------------------------------------------
# MATERIAL_TYPES
# --------------
# `MATERIAL_TYPES` is a simple "dictionary of lists" that describes what
# kinds of library resources people might be looking for: books, articles,
# videos, images, maps, etc.
#
# Each key (e.g., "books", "articles") is a label that will appear in the
# dashboard. Each value is a list of words and phrases that, if they appear
# in the user's query, suggest that the user is looking for that material type.
#
# For example:
#   "books": ["book", "books", "ebook", "ebooks", ...]
#
# You can freely edit these keywords to match the language used in your own
# institution. If your users say "texts" instead of "books", you can add
# "text" / "texts" to the list; if you do not track "maps", you can remove
# that entry entirely.
#
# SUBJECT_AREAS
# -------------
# `SUBJECT_AREAS` works the same way but for topics instead of formats.
# Each key is a broad subject (like "AI & Computer Science", "Psychology",
# or "Environmental Science"), and the list under it contains words/phrases
# that should trigger that subject.
#
# CUSTOMIZATION NOTES
# -------------------
# - These lists do not have to be perfect. The fuzzy matching logic below
#   (see `extract_tags_fuzzy`) can handle typos and partial matches.
# - You can add more subjects (e.g., "Local History", "Law", "Business") or
#   remove ones that don't make sense for your library.
# - Think of this section as your “tagging policy”: non-technical staff can
#   maintain it directly as long as they keep the structure:
#       SUBJECT_AREAS = { "Label": ["keyword1", "keyword2", ...], ... }

# Controlled vocabulary map
MATERIAL_TYPES = {
    "books": [
        "book", "books",
        "ebook", "ebooks",
        "monograph", "monographs",
        "book chapter", "book chapters"
    ],
    "articles": [
        "article", "articles",
        "paper", "papers",
        "research paper"
    ],
    "journals": [
        "journal", "journals"
    ],
    "videos": [
        "video", "videos",
        "film", "films",
        "movie", "movies"
    ],
    "audios": [
        "audio", "audios",
        "audiobook", "audiobooks"
    ],
    "images": [
        "image", "images"
    ],
    "maps": [
        "map", "maps"
    ],
    "microform": [
        "microform"
    ],
    "dissertations": [
        "dissertation", "dissertations",
        "thesis", "theses"
    ],
    "government_documents": [
        "government document",
        "gov doc", "gov docs",
        "government_documents"
    ],
    "reports": [
        "report", "reports"
    ],
    "scores": [
        "score", "scores"
    ],
    "archival_material_manuscripts": [
        "archival material",
        "manuscript", "manuscripts"
    ],
    "market_researchs": [
        "market research"
    ]
}

SUBJECT_AREAS = {
    'AI & Computer Science': [
        'artificial intelligence', 'machine learning', 'neural network', 'deep learning', 
        'AI', 'computer science', 'computing', 'algorithm', 'software', 'programming',
        'data science', 'computational', 'informatics', 'cybersecurity'
    ],
    'Medicine & Health': [
        'medical', 'medicine', 'health', 'healthcare', 'disease', 'clinical', 'patient',
        'treatment', 'therapy', 'diagnosis', 'hospital', 'doctor', 'nursing', 'pharmacology',
        'epidemiology', 'pathology', 'surgery', 'pharmaceutical', 'wellness'
    ],
    'Psychology': [
        'psychology', 'psychological', 'mental', 'cognitive', 'behavior', 'behavioral',
        'psychiatric', 'psychiatry', 'emotion', 'emotional', 'consciousness', 'perception',
        'personality', 'developmental', 'social psychology', 'neuroscience', 'brain',
        'mind', 'thinking', 'memory', 'attention', 'persuasion', 'persuade'
    ],
    'Biology & Life Sciences': [
        'biology', 'biological', 'genetics', 'genetic', 'cell', 'cellular', 'molecular',
        'organism', 'evolution', 'ecology', 'biodiversity', 'anatomy', 'physiology',
        'biochemistry', 'microbiology', 'zoology', 'botany', 'marine biology'
    ],
    'Chemistry': [
        'chemistry', 'chemical', 'molecule', 'molecular', 'compound', 'reaction',
        'organic chemistry', 'inorganic', 'analytical', 'synthesis', 'catalysis',
        'polymer', 'spectroscopy', 'laboratory', 'lab'
    ],
    'Physics': [
        'physics', 'physical', 'quantum', 'particle', 'energy', 'force', 'motion',
        'mechanics', 'thermodynamics', 'electromagnetism', 'optics', 'relativity',
        'astrophysics', 'cosmology', 'nuclear'
    ],
    'Environmental Science': [
        'environment', 'environmental', 'climate', 'climate change', 'sustainability',
        'sustainable', 'ecology', 'ecological', 'conservation', 'pollution', 'ecosystem',
        'carbon', 'greenhouse', 'renewable', 'biodiversity', 'green', 'nature'
    ],
    'Sociology & Anthropology': [
        'sociology', 'sociological', 'social', 'society', 'culture', 'cultural',
        'anthropology', 'ethnography', 'community', 'inequality', 'race', 'class',
        'gender', 'identity', 'materialistic', 'consumerism', 'poverty', 'demographics'
    ],
    'Education': [
        'education', 'educational', 'learning', 'teaching', 'pedagogy', 'student',
        'students', 'undergraduate', 'undergraduates', 'graduate', 'school', 'university',
        'academic', 'curriculum', 'instruction', 'literacy', 'classroom'
    ],
    'Business & Economics': [
        'business', 'economics', 'economic', 'finance', 'financial', 'management',
        'marketing', 'entrepreneurship', 'commerce', 'trade', 'market', 'accounting',
        'investment', 'corporate', 'startup', 'company', 'industry'
    ],    
    'Engineering': [
        'engineering', 'engineer', 'mechanical', 'electrical', 'civil', 'structural',
        'aerospace', 'industrial', 'manufacturing', 'robotics', 'automation',
        'design', 'construction', 'architecture', 'infrastructure'
    ],
    'History': [
        'history', 'historical', 'ancient', 'medieval', 'modern', 'century',
        'war', 'revolution', 'empire', 'civilization', 'archaeology', 'heritage',
        'timeline', 'era', 'period', 'past'
    ],
    'Literature & Languages': [
        'literature', 'literary', 'novel', 'poetry', 'poem', 'fiction', 'narrative',
        'author', 'writer', 'writing', 'language', 'linguistics', 'translation',
        'rhetoric', 'criticism', 'genre', 'text'
    ],
    'Philosophy': [
        'philosophy', 'philosophical', 'ethics', 'ethical', 'moral', 'metaphysics',
        'epistemology', 'logic', 'reasoning', 'existential', 'phenomenology',
        'ontology', 'kant', 'plato', 'aristotle'
    ],
    'Political Science & Law': [
        'political', 'politics', 'government', 'policy', 'law', 'legal', 'legislation',
        'democracy', 'election', 'voting', 'constitution', 'justice', 'court',
        'rights', 'governance', 'diplomacy', 'international relations'
    ],
    'Sports & Kinesiology': [
        'sports', 'sport', 'athletic', 'athletics', 'fitness', 'exercise', 'physical activity',
        'training', 'performance', 'kinesiology', 'biomechanics', 'steroids', 'doping',
        'nutrition', 'coaching', 'team', 'competition', 'athlete'
    ],
    'Mathematics & Statistics': [
        'mathematics', 'math', 'mathematical', 'statistics', 'statistical', 'algebra',
        'calculus', 'geometry', 'probability', 'analysis', 'theorem', 'equation',
        'numerical', 'quantitative', 'computation'
    ],
    'Music & Performing Arts': [
        'music', 'musical', 'musician', 'composition', 'performance', 'theater',
        'theatre', 'drama', 'dance', 'opera', 'concert', 'symphony', 'instrument',
        'melody', 'rhythm', 'acoustics'
    ],
    'Visual Arts & Design': [
        'art', 'arts', 'visual', 'painting', 'sculpture', 'drawing', 'graphic design',
        'illustration', 'photography', 'aesthetics', 'gallery', 'museum', 'artist',
        'creative', 'contemporary', 'modern art'
    ],
    'Communication & Media': [
        'communication', 'media', 'journalism', 'broadcasting', 'public relations',
        'advertising', 'digital media', 'social media', 'film studies', 'television',
        'radio', 'news', 'press', 'propaganda'
    ],
}

# ===================== ROUTES & ENDPOINTS =====================

# Make home (/) the landing page with tiles (includes "All Searches")
@APP.route("/")
def index():
    return render_template('home.html', file=JSONL_PATH)

# Overview (KPI + daily/errors)
@APP.route("/overview")
def page_overview():
    return render_template('overview.html', file=JSONL_PATH)

# Query Types (SCI pie + histogram)
@APP.route("/query-types")
def page_query_types():
    return render_template('query_types.html', file=JSONL_PATH)

# Performance (latency + errors)
@APP.route("/performance")
def page_performance():
    return render_template('performance.html', file=JSONL_PATH)

# Heatmap (uses /metrics; shows meta if present)
@APP.route("/heatmap")
def page_heatmap():
    return render_template('heatmap.html', file=JSONL_PATH)

# Slowest (richer item details)
@APP.route("/slowest")
def page_slowest():
    return render_template('slowest.html', file=JSONL_PATH)

# All Searches (filter + pagination)
@APP.route("/all-searches")
def page_all_searches():
    return render_template('all_searches.html', file=JSONL_PATH)

# ---------- server helpers ----------
def compute_line_hash(line):
    return hashlib.sha1(line.encode("utf-8")).hexdigest()

def read_tail(n=1000):
    out = []
    if not os.path.exists(JSONL_PATH):
        return out
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    tail = lines[-n:]
    for ln in tail:
        line_hash = compute_line_hash(ln)
        try:
            obj = json.loads(ln)
        except Exception:
            obj = {"_raw": ln, "_error": "invalid_json"}

        # === new: attach local-time display fields ===
        ts_val = obj.get("ts") or obj.get("timestamp") or None
        if ts_val:
            dt = parse_iso(ts_val)   # parse_iso already returns a timezone-aware UTC dt if possible
            if dt:
                try:
                    local_dt = dt.astimezone(LOCAL_ZONE)
                    obj["_ts_local"] = local_dt.isoformat()          # machine-readable
                    obj["_ts_local_pretty"] = local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")  # human
                except Exception:
                    # if something weird, skip adding local
                    pass
        obj["_line_hash"] = line_hash

        # === Extract tags ===
        q = obj.get("query") or obj.get("q") or ""
        obj["_tags"] = extract_tags_fuzzy(q)

        out.append(obj)
    return out

def parse_iso(ts):
    if not ts:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        s = ts
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        fmts = ["%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%S","%Y-%m-%d %H:%M:%S"]
        for fmt in fmts:
            try:
                dt = datetime.strptime(ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue
    return None

def get_comment_counts(line_hashes):
    if not line_hashes: return {}
    qmarks = ",".join("?" for _ in line_hashes)
    cur = DB_CON.cursor()
    cur.execute(f"SELECT line_hash, COUNT(*) FROM comments WHERE line_hash IN ({qmarks}) GROUP BY line_hash", line_hashes)
    return {h: c for h, c in cur.fetchall()}
def get_comment_counts_for_hashes(hashes):
    """Return dict {hash: count} for the given iterable of hashes."""
    if not hashes:
        return {}
    # ensure unique list (preserves small size)
    uniq = list(dict.fromkeys(hashes))
    # prepare placeholders
    placeholders = ",".join("?" for _ in uniq)
    cur = DB_CON.cursor()
    query = f"SELECT line_hash, COUNT(*) as c FROM comments WHERE line_hash IN ({placeholders}) GROUP BY line_hash"
    cur.execute(query, uniq)
    rows = cur.fetchall()
    result = {r[0]: r[1] for r in rows}
    # fill zeros for missing
    for h in uniq:
        result.setdefault(h, 0)
    return result

# -----------------------------------------------------------------------------
# SEARCH COMPLEXITY INDEX (SCI)
# -----------------------------------------------------------------------------
# WHY WE COMPUTE SCI
# -------------------
# Not all searches are equal. Some are very simple ("cats"), others are
# complex Boolean queries ("(climate change OR global warming) AND policy").
# We define a simple numeric "Search Complexity Index" (SCI) to estimate how
# advanced a query is.
#
# HOW SCI IS CALCULATED
# ----------------------
# For each query we:
#   - Count how many Boolean operators appear: AND, OR, NOT.
#   - Count the total number of word-like tokens.
#   - Count how many unique non-trivial words are used (ignoring common
#     stopwords like "the", "and", "of", "in", etc.).
#
# The SCI is roughly:
#   SCI = (#Boolean operators) + (#tokens) + (#unique non-stopword terms)
#
# HOW THIS IS USED
# ----------------
# - SCI is stored per log line as `_sci`.
# - In the `/metrics` route we use SCI to:
#     * Build a histogram of query complexity.
#     * Classify queries into Informational / Exploratory / Creative based
#       on adaptive percentiles.
#
# HOW TO TUNE OR DISABLE THIS
# ---------------------------
# - If you feel this metric over- or under-values Boolean logic, you can
#   adjust the formula in `compute_search_complexity`.
# - You can switch it off entirely by always returning 0, or by removing
#   the pieces in `/metrics` that use `_sci`.

def percentile_from_sorted(sorted_arr, p):
    """Compute percentile p (0..100) from sorted array using linear interpolation."""
    if not sorted_arr:
        return None
    n = len(sorted_arr)
    if n == 1: return sorted_arr[0]
    # position in [0..n-1]
    pos = (p / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(n - 1, lo + 1)
    if lo == hi:
        return sorted_arr[lo]
    frac = pos - lo
    return sorted_arr[lo] * (1 - frac) + sorted_arr[hi] * frac

_STOPWORDS = {
    "the","and","or","of","a","an","in","on","for","to","is","are","as","at","by",
    "with","from","that","this","these","those","be","been","being","was","were",
    "it","its","into","about","over","under","than","then","but","if","so","not"
}

_BOOL_RE  = re.compile(r"\b(AND|OR|NOT)\b", flags=re.IGNORECASE)
_TOKEN_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

def compute_search_complexity(query: str) -> int:
    """SCI = #Boolean(AND/OR/NOT) + token_count + unique_terms(excl. stopwords)."""
    if not isinstance(query, str) or not query.strip():
        return 0
    # count boolean operators
    bool_ops = len(_BOOL_RE.findall(query))
    # tokens
    tokens = _TOKEN_RE.findall(query.lower())
    tok_len = len(tokens)
    # unique terms excluding stopwords
    uniq_terms = len({t for t in tokens if t not in _STOPWORDS})
    return int(bool_ops + tok_len + uniq_terms)

def classify_query_type_from_sci(sci: int, p65: float, p90: float) -> str:
    """
    Adaptive tertiles: <= p65 -> Informational; <= p90 -> Exploratory; else -> Creative.
    Keeps classes balanced as traffic changes without hard-coded thresholds.
    """
    if sci <= (p65 or 0):
        return "Informational"
    elif sci <= (p90 or 0):
        return "Exploratory"
    return "Creative"

# -----------------------------------------------------------------------------
# FUZZY TAGGING OF QUERIES (MATERIAL TYPES + SUBJECT AREAS)
# -----------------------------------------------------------------------------
# PURPOSE
# -------
# `extract_tags_fuzzy` looks at a single search query string and tries to guess:
#   - What material types the user might want (books, articles, videos, etc.).
#   - What subject areas the query belongs to (AI, medicine, psychology, etc.).
#
# HOW IT WORKS IN PLAIN LANGUAGE
# ------------------------------
# For each controlled vocabulary entry:
#   1. We lowercase the query and the keyword.
#   2. We try several match strategies:
#        - Simple substring (keyword appears exactly in the query).
#        - Fuzzy similarity scores using RapidFuzz:
#            * ratio         (overall similarity)
#            * partial_ratio (handles short words or phrases)
#            * token_set_ratio (ignores word order)
#   3. We keep the best score per material type or subject, and give a small
#      bonus when several keywords for the same category match.
#   4. We then:
#        - Keep the top few material types.
#        - Keep the top few subjects.
#        - Compute an overall "confidence" label based on how strong the
#          matches were ("high", "medium", "low", or "none").
#
# OUTPUT
# ------
# This function returns a dictionary that looks like:
#   {
#     "materials": ["books", "articles"],
#     "subjects": ["AI & Computer Science"],
#     "confidence": "high",
#     "scores": {
#       "materials": {"books": 92, "articles": 81, ...},
#       "subjects": {"AI & Computer Science": 95, ...}
#     }
#   }
#
# HOW NON-TECHNICAL USERS CAN INFLUENCE THIS
# ------------------------------------------
# - You do not have to touch this function to customize tagging.
#   Most customization happens by editing the keyword lists in
#   MATERIAL_TYPES and SUBJECT_AREAS above.
# - If you want the system to be stricter or more relaxed, you can discuss
#   with a developer about:
#     * Thresholds (e.g., minimum score needed to count as a match).
#     * How many top categories to keep.
#     * What "confidence" values mean in your reporting.


def extract_tags_fuzzy(query: str) -> dict:
    """
    Extract material types and subject areas using multi-strategy fuzzy matching.
    Returns dict with materials, subjects, and confidence scores.
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "materials": [],
            "subjects": [],
            "confidence": "none",
            "scores": {}
        }
    
    q_lower = query.lower()
    
    # ---- MATERIAL TYPE MATCHING ----
    material_matches = {}
    for mat_type, keywords in MATERIAL_TYPES.items():
        max_score = 0
        for kw in keywords:
            # Strategy 1: Exact substring match (highest confidence)
            if kw in q_lower:
                max_score = max(max_score, 100)
                break
            
            # Strategy 2: Partial ratio (handles plurals, variations)
            partial = fuzz.partial_ratio(kw, q_lower)
            if partial > 85:  # High threshold for materials
                max_score = max(max_score, partial)
        
        if max_score > 0:
            material_matches[mat_type] = max_score
    
    # Take top 2 materials by score
    materials = sorted(material_matches.items(), key=lambda x: x[1], reverse=True)[:2]
    materials = [m[0] for m in materials if m[1] >= 85]
    
    # ---- SUBJECT AREA MATCHING ----
    subject_matches = {}
    query_words = set(q_lower.split())
    
    for subject, keywords in SUBJECT_AREAS.items():
        scores = []
        
        for kw in keywords:
            kw_lower = kw.lower()
            
            # Strategy 1: Exact phrase match (highest confidence)
            if kw_lower in q_lower:
                scores.append(100)
                continue
            
            # Strategy 2: Word overlap (multi-word keywords)
            if ' ' in kw_lower:
                kw_words = set(kw_lower.split())
                overlap = len(kw_words & query_words)
                if overlap > 0:
                    overlap_score = (overlap / len(kw_words)) * 90
                    scores.append(overlap_score)
            
            # Strategy 3: Partial ratio (single words with typos/variations)
            partial = fuzz.partial_ratio(kw_lower, q_lower)
            if partial > 75:  # Lower threshold for subjects
                scores.append(partial * 0.8)  # Downweight fuzzy matches
            
            # Strategy 4: Token set ratio (word order independent)
            token_set = fuzz.token_set_ratio(kw_lower, q_lower)
            if token_set > 75:
                scores.append(token_set * 0.7)
        
        if scores:
            # Use highest score, with bonus for multiple keyword matches
            max_score = max(scores)
            match_bonus = min(len(scores) * 5, 20)  # Up to +20 for many matches
            subject_matches[subject] = min(max_score + match_bonus, 100)
    
    # Take top 3 subjects by score
    subjects = sorted(subject_matches.items(), key=lambda x: x[1], reverse=True)[:3]
    subjects = [s[0] for s in subjects if s[1] >= 60]  # Minimum confidence threshold
    
    # ---- CONFIDENCE ASSESSMENT ----
    if materials or subjects:
        avg_score = sum([m[1] for m in material_matches.items()] + 
                       [s[1] for s in subject_matches.items()])
        count = len(material_matches) + len(subject_matches)
        avg_score = avg_score / count if count > 0 else 0
        
        if avg_score >= 90:
            confidence = "high"
        elif avg_score >= 75:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "none"
    
    return {
        "materials": materials,
        "subjects": subjects,
        "confidence": confidence,
        "scores": {
            "materials": {k: v for k, v in material_matches.items()},
            "subjects": {k: v for k, v in subject_matches.items()}
        }
    }


@APP.route("/snapshot")
def snapshot():
    n = int(request.args.get("n", "800"))
    rows = read_tail(n)

    # collect line_hashes (only for rows that have them; compute if necessary)
    hashes = []
    for r in rows:
        h = r.get("_line_hash")
        if not h:
            # if read_tail didn't compute it for some reason, try to compute from _raw or by re-serialization
            raw = None
            if "_raw" in r:
                raw = r["_raw"]
            else:
                try:
                    raw = json.dumps(r, separators=(',',':'))
                except Exception:
                    raw = None
            if raw:
                import hashlib
                h = hashlib.sha1(raw.encode("utf-8")).hexdigest()
                r["_line_hash"] = h
        if h:
            hashes.append(h)

    # fetch counts in one query
    counts = get_comment_counts_for_hashes(hashes)

    # attach _comment_count for each row
    for r in rows:
        h = r.get("_line_hash")
        if h:
            r["_comment_count"] = counts.get(h, 0)
        else:
            r["_comment_count"] = 0
    snapshot_scis = []
    for r in rows:
        q = r.get("query") or r.get("q") or ""
        s = compute_search_complexity(q)  # uses AND/OR/NOT + token count + unique terms (excl. stopwords)
        r["_sci"] = s
        snapshot_scis.append(s)

    # 2) Derive adaptive thresholds from current snapshot (tertiles ~33% and ~66%)
    scis_sorted = sorted(snapshot_scis)
    p65 = percentile_from_sorted(scis_sorted, 65) if scis_sorted else 0
    p90 = percentile_from_sorted(scis_sorted, 90) if scis_sorted else 0

    # 3) Classify each row into Informational / Exploratory / Creative based on its _sci
    for r in rows:
        r["_query_type"] = classify_query_type_from_sci(r.get("_sci", 0), p65, p90)

    # --- return the enriched rows ---
    return jsonify(rows)

# -----------------------------------------------------------------------------
# METRICS ENDPOINT (/metrics)
# -----------------------------------------------------------------------------
# PURPOSE
# -------
# The `/metrics` route reads the entire `search.jsonl` log file and computes
# summary statistics that the front-end can display as charts, tables, and
# indicators.
#
# WHAT IT MEASURES
# ----------------
# For the current contents of the log file, we compute:
#   - `total_rows`: Total number of log entries (total searches).
#   - `rows_last_24h`: How many searches happened in the last 24 hours
#     (using local time in the configured time zone).
#   - `count_400`: Number of HTTP 400 errors (bad requests) if present.
#   - `daily`: For each local calendar day, how many searches occurred.
#   - `errors`: For each local calendar day, how many errors occurred.
#   - `latency`: A histogram of response times plus 50th, 90th, and 95th
#     percentiles (p50, p90, p95).
#   - `slowest`: A list of the slowest individual requests, for debugging.
#   - `heatmap`: A matrix of counts by (day, hour) to show when searches
#     are most active over time.
#   - `sci`: Aggregate statistics for the Search Complexity Index, including
#     a histogram and counts per query type (Informational / Exploratory /
#     Creative).
#   - `tags`: Summary counts of material types and subject areas, plus a
#     simple co-occurrence table to see which subjects appear together.
#
# HOW TO CUSTOMIZE
# ----------------
# - You can add new metrics (for example, breakdown by HTTP status, average
#   latency per subject area, etc.) by extending this function.
# - You can remove metrics that you do not care about by deleting them from
#   the returned JSON.
# - Because this endpoint reads the whole log file each time, very large logs
#   may need extra optimization (e.g., pre-aggregated data or time windows).
#
# NON-TECHNICAL TAKEAWAY
# ----------------------
# Think of `/metrics` as the "summary data feed" for the dashboard. The web
# pages call this endpoint to draw charts, and you can pull this data into
# other tools (like Excel, R, or a BI tool) if you want to analyze it further.

@APP.route("/metrics")
def metrics():
    """Compute metrics using LOCAL_ZONE (America/New_York) for bucketing:
       - daily counts (per local date)
       - 400 error counts per local date
       - heatmap uses local hour (0..23)
       - rows_last_24h computed using local now
       - percentiles/histogram use elapsed values unchanged
    """
    days = int(request.args.get("days", "120"))
    lines = read_tail(20000)  # window size; adjust if you want more history
    total_rows = len(lines)

    # Use 'now' in LOCAL_ZONE so 24h/local-day comparisons are consistent
    now_utc = datetime.now(timezone.utc)
    try:
        now_local = now_utc.astimezone(LOCAL_ZONE)
    except Exception:
        # fallback: treat local as UTC if zone mishap
        now_local = now_utc

    rows_last_24h = 0
    count_400 = 0
    daily = Counter()
    errors_daily = Counter()
    elapsed_values = []
    sci_values=[]

    # heatmap matrix days Mon..Sun
    dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    heatmap = {d: {h: 0 for h in range(24)} for d in dow_names}

    for ln in lines:
        ts = ln.get("ts") or ln.get("timestamp") or None
        dt = parse_iso(ts)  # dt is timezone-aware UTC if parse_iso works
        if dt:
            # convert to LOCAL_ZONE for bucketing/display
            try:
                local_dt = dt.astimezone(LOCAL_ZONE)
            except Exception:
                local_dt = dt
            date_key = local_dt.date().isoformat()
            # heatmap uses local hour
            try:
                dow = local_dt.weekday()  # 0 = Monday
                heatmap[dow_names[dow]][local_dt.hour] += 1
            except Exception:
                pass
            # last 24 hours relative to local now
            if (now_local - local_dt).total_seconds() <= 86400:
                rows_last_24h += 1
        else:
            date_key = "unknown"
        daily[date_key] += 1

        # status 400 counts (bucketed by local date_key)
        status = ln.get("status")
        try:
            if isinstance(status, int) and 400 <= status < 500:
                # specifically count 400-range; adjust if you want only 400
                if status == 400:
                    count_400 += 1
                errors_daily[date_key] += 1
        except Exception:
            pass

        # elapsed values for percentiles/histogram
        try:
            e = ln.get("elapsed_ms")
            if e is None:
                e = ln.get("elapsed")
            if e is not None:
                elapsed_values.append(float(e))
        except Exception:
            pass
         # ---- NEW: SCI per line ----
        try:
            q = ln.get("query") or ln.get("q") or ""
            sci_values.append(compute_search_complexity(q))
        except Exception:
            pass
        
    # Build last `days` local-date series (from LOCAL_ZONE now)
    dates = [(now_local.date() - timedelta(days=i)).isoformat() for i in range(days-1, -1, -1)]
    labels = dates
    counts = [daily.get(d, 0) for d in dates]
    err_counts = [errors_daily.get(d, 0) for d in dates]

    # percentiles (p50/p90/p95) computed from elapsed_values
    p50 = p90 = p95 = None
    if elapsed_values:
        arr = sorted(elapsed_values)
        p50 = percentile_from_sorted(arr, 50)
        p90 = percentile_from_sorted(arr, 90)
        p95 = percentile_from_sorted(arr, 95)

    # histogram buckets (same as before)
    bucket_bounds = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 10000, 30000]
    bucket_labels = [f"{bucket_bounds[i]}–{bucket_bounds[i+1]-1}" for i in range(len(bucket_bounds)-1)]
    bucket_counts = [0] * (len(bucket_bounds)-1)
    for v in elapsed_values:
        for i in range(len(bucket_bounds)-1):
            if bucket_bounds[i] <= v < bucket_bounds[i+1]:
                bucket_counts[i] += 1
                break

    # slowest top N (unchanged)
    slow_items = []
    for ln in lines:
        try:
            e = ln.get("elapsed_ms")
            if e is None:
                e = ln.get("elapsed")
            if e is not None:
                slow_items.append((float(e), ln))
        except Exception:
            continue
    slow_items.sort(key=lambda x: x[0], reverse=True)
    topN = [item[1] for item in slow_items[:20]]

    # heatmap max
    max_count = 0
    for d in heatmap:
        for h in range(24):
            max_count = max(max_count, heatmap[d][h])
    sci_mean = sci_median = sci_p90 = None
    p65_sci = p90_sci = 0
    sci_hist_bounds = list(range(0, 51, 5)) + [75, 100, 150]  # tune bins as desired
    sci_hist_labels = []
    sci_hist_counts = []
    type_counts = Counter()

    if sci_values:
        scis_sorted = sorted(sci_values)
        sci_mean   = sum(scis_sorted) / len(scis_sorted)
        sci_median = percentile_from_sorted(scis_sorted, 50)
        sci_p90    = percentile_from_sorted(scis_sorted, 90)
        p65_sci    = percentile_from_sorted(scis_sorted, 65)
        p90_sci    = percentile_from_sorted(scis_sorted, 90)

        # histogram across the chosen bounds
        for i in range(len(sci_hist_bounds) - 1):
            lo, hi = sci_hist_bounds[i], sci_hist_bounds[i+1]
            sci_hist_labels.append(f"{lo}–{hi-1}")
            sci_hist_counts.append(sum(1 for v in sci_values if lo <= v < hi))
        # open-ended final bin
        last_lo = sci_hist_bounds[-1]
        sci_hist_labels.append(f"{last_lo}+")
        sci_hist_counts.append(sum(1 for v in sci_values if v >= last_lo))

        # adaptive classification counts
        for v in sci_values:
            t = classify_query_type_from_sci(v, p65_sci, p90_sci)
            type_counts[t] += 1
        meta = {
        "local_zone": str(LOCAL_ZONE),
        "now_local": now_local.isoformat(),
        "date_start": labels[0] if labels else None,
        "date_end": labels[-1] if labels else None
    }
        # ---- TAG STATISTICS ----
    tag_stats = {
        "materials": Counter(),
        "subjects": Counter(),
        "confidence_distribution": Counter(),
        "co_occurrence": defaultdict(Counter)
    }
    
    for ln in lines:
        tags = ln.get("_tags", {})
        
        # Count materials
        for mat in tags.get("materials", []):
            tag_stats["materials"][mat] += 1
        
        # Count subjects
        subjects = tags.get("subjects", [])
        for subj in subjects:
            tag_stats["subjects"][subj] += 1
        
        # Track confidence levels
        conf = tags.get("confidence", "none")
        tag_stats["confidence_distribution"][conf] += 1
        
        # Track which subjects appear together
        for i, s1 in enumerate(subjects):
            for s2 in subjects[i+1:]:
                pair = tuple(sorted([s1, s2]))
                tag_stats["co_occurrence"][pair[0]][pair[1]] += 1

    return jsonify({
        "total_rows": total_rows,
        "rows_last_24h": rows_last_24h,
        "count_400": count_400,
        "p50_elapsed_ms": p50,
        "p90_elapsed_ms": p90,
        "p95_elapsed_ms": p95,
        "daily": {"labels": labels, "counts": counts},
        "errors": {"labels": labels, "counts": err_counts},
        "latency": {"buckets": bucket_labels, "counts": bucket_counts, "p50": p50, "p90": p90, "p95": p95},
        "slowest": topN,
        "heatmap": {"matrix": heatmap, "max": max_count},
        "sci": {
            "mean": sci_mean,
            "median": sci_median,
            "p90": sci_p90,
            "p65": p65_sci,
            "p90": p90_sci,
            "histogram": {"buckets": sci_hist_labels, "counts": sci_hist_counts},
            "type_counts": dict(type_counts)  # {"Informational": x, "Exploratory": y, "Creative": z}
        },
        "tags": {
            "materials": dict(tag_stats["materials"]),
            "subjects": dict(tag_stats["subjects"]),
            "confidence": dict(tag_stats["confidence_distribution"]),
            "co_occurrence": {k: dict(v) for k, v in tag_stats["co_occurrence"].items()}
      }})

@APP.route("/comments")
def get_comments():
    """GET /comments?hash=<linehash>  -> returns list of comments for that linehash"""
    line_hash = request.args.get("hash", "")
    if not line_hash:
        return jsonify([])   # no hash -> empty
    cur = DB_CON.cursor()
    cur.execute("SELECT id, author, text, created_at FROM comments WHERE line_hash=? ORDER BY id ASC", (line_hash,))
    rows = cur.fetchall()
    comments = [{"id": r[0], "author": r[1], "text": r[2], "created_at": r[3]} for r in rows]
    return jsonify(comments)

@APP.route("/comment", methods=["POST"])
def post_comment():
    """POST /comment  with JSON {hash, author, text} -> stores comment and returns saved row"""
    try:
        payload = request.get_json(force=True)
        line_hash = payload.get("hash")
        author = payload.get("author") or "anon"
        text = payload.get("text") or ""
        if not line_hash or not text.strip():
            return jsonify({"error":"hash and text required"}), 400
        created_at = datetime.now(timezone.utc).isoformat()
        cur = DB_CON.cursor()
        cur.execute("INSERT INTO comments (line_hash, author, text, created_at) VALUES (?, ?, ?, ?)",
                    (line_hash, author, text, created_at))
        DB_CON.commit()
        cid = cur.lastrowid
        return jsonify({"id": cid, "line_hash": line_hash, "author": author, "text": text, "created_at": created_at})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# SSE generator - streams appended lines
def gen_sse():
    last_size = 0
    if os.path.exists(JSONL_PATH):
        last_size = os.path.getsize(JSONL_PATH)
    while True:
        try:
            if not os.path.exists(JSONL_PATH):
                time.sleep(POLL_INTERVAL)
                continue
            cur_size = os.path.getsize(JSONL_PATH)
            if cur_size > last_size:
                with open(JSONL_PATH, "r", encoding="utf-8") as f:
                    f.seek(last_size)
                    new_data = f.read()
                last_size = cur_size
                lines = [ln for ln in new_data.splitlines() if ln.strip()]
                for ln in lines:
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        obj = {"_raw": ln, "_error": "invalid_json"}
                    yield f"data: {json.dumps(obj)}\n\n"
            else:
                # keepalive (no-op) to allow loop to continue
                yield ""
            time.sleep(POLL_INTERVAL)
        except GeneratorExit:
            break
        except Exception as e:
            try:
                yield f": error {str(e)}\n\n"
            except:
                pass
            time.sleep(POLL_INTERVAL)

@APP.route("/stream")
def stream():
    return Response(gen_sse(), mimetype="text/event-stream")

if __name__ == "__main__":
    if not os.path.exists(JSONL_PATH):
        open(JSONL_PATH, "a").close()
    APP.run(debug=True, port=5000, threaded=True)