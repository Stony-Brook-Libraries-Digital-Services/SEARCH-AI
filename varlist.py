# Change based on institution
# Institution-specific configuration for SearchAI
# Each setting below plugs into the pipeline in open.py:
#   • split_query(): reads the regex flags, prompt text, and held-by-school phrases
#   • extract_year_range(): uses Prompt2 for date normalization
#   • build_boolean_string(): uses Prompt3 to create a Primo/Discovery Boolean
#   • extract_material_types(): uses MaterialTypeMap/Prompt4 for rtypes
#   • construct_url(): uses `prefix` + rtypes to build the redirect URL
# Change these to match your institution’s discovery system, metadata, and wording.
# ─────────────────────────────────────────────────────────────────────────────

# OPENAI_KEY
# Purpose: API key the agents use for Prompt1–Prompt4 calls.
# Where used: Plumbed into OPENAI_API_KEY in open.py at import time.
# Prefer: export OPENAI_KEY in env and read via os.getenv in open.py.

OPENAI_KEY = (" your_openai_api_key_here ")

# OPENAI_KEY
# Purpose: API key the agents use for the boolean agent.
# This can be the same as the first key is using an untuned model. 
# Where used: Plumbed into OPENAI_API_KEY in open.py at import time.

OPENAI_KEY_BOOLEAN = (" your_trained_api_key_here ")

# prefix
# Purpose: Base Discovery search URL (everything before &query=...).
# Where used: construct_url() in open.py.
# Institution notes: Change to your Primo/EDS/Summon base with correct vid/tab/scope.
#   Example (Primo VE): "https://your.univ.edu/discovery/search?vid=YOUR_VID"
#   If your scope or tab changes, update here so links land in the right UI.

prefix = (" your wehsite base url here ")

# online_flagg
# Purpose: Regex that detects “online”/full-text requests in the natural-language query.
# Where used: split_query() to set online_flag and to strip these words from the topic
#   so the Boolean builder doesn’t include them.
# Institution notes: Localize to the phrases your users actually type (“online access”,
#   “full text”, “e-resources”). Keep it conservative to avoid false positives.

online_flagg = (r"\b(available online|online only|full text online|access online)\b")


# local_flag
# Purpose: Regex that detects “institution-held/locally available” requests.
# Where used: split_query() to set sbu_flag (your “held by library” facet).
# Institution notes: Replace with your library’s wording. DO NOT reuse the
#   same pattern as online_flagg. Examples: r"\b(held by .*university|in our library)\b".

local_flag = (r" held by langauage here ")


# peer_flagg
# Purpose: Regex for “peer-reviewed/scholarly” filter requests.
# Where used: split_query() to set peer_flag and to strip those words from topic.
# Institution notes: Add local synonyms (“refereed”) if your community uses them.

peer_flagg = (r"\b(peer[- ]?reviewed|scholarly|academic (sources|research)|research articles?)\b")


# OpenAI Prompts# ─────────────────────────────────────────────────────────────────────────────

# Prompt1
# Purpose: “Query Splitter” system message. LLM parses the user query into:
#   topic_input (subject), time_input (publication time constraint), type_input (material types).
# Where used: split_query() → Prompt1[0]["content"] sent to the model.
# Institution notes: If you maintain a different controlled vocabulary or want different
#   splitting behavior, edit the instructions here (e.g., add local period labels or type rules).

Prompt1 = [ { "role": "system", "content": "You are a 'Query Splitter' agent. Your ONLY job is to parse a natural-language search request "
       "into exactly three fields. Follow these rules strictly:\n"
       "\n"
       "1. 'topic_input':\n"
        "- Capture ALL subject terms, keywords, names, entities, named temporal coverage of the subject, or topical phrases from the complete query.\n"
        "- If author names are listed, include them.\n"
        "- Always include well-known acronyms, historical names, standard terminology, and questions asked.\n"
        "- EXCLUDE publication time references and material types. (they belong in other fields).\n"
        "- Must be a non-empty string if the user provides any subject matter.\n"
        "\n"
        "2. 'time_input':"
        "- Include only constraints on the production/issuance/publication/recording date of results, including explicit operators (before/after/between/since) and explicit recency requests."
        "- Treat general recency adjectives/adverbs (terms indicating up-to-date/current/latest) as production-recency, unless they function as part of a canonical subject period/movement label or taxonomy term; in that case they belong in 'topic_input'."
        "- If no production-time constraint or recency request is present, return an empty string."
        "\n"
        "3. 'type_input':\n"
        "- Identify material types ONLY if explicitly named by the user.\n"
        "- Normalize to this controlled vocabulary (plural, lowercase):\n"
        "[\"books\", \"journals\", \"articles\", \"images\", \"microform\", \"audios\", \"maps\", "
        "\"videos\", \"dissertations\", \"government_documents\", \"reports\", "
        "\"book_chapters\", \"scores\", \"archival_material_manuscripts\", \"market_researchs\"]\n"
        "- If mentioned, output in a JSON array. If none, output [] (empty array).\n"
        "- NEVER infer based on tone or likely intent.\n"
        "\n"
        "OUTPUT FORMAT RULES:\n"
        "- Return ONLY a single valid JSON object.\n"
        "- Use the keys: 'topic_input', 'time_input', 'type_input'.\n"
        "- Always return:\n"
        "{\n"
        "  \"topic_input\": string,\n"
        "  \"time_input\": string,\n"
        "  \"type_input\": array\n"
        "}\n"
        "\n"
        "If the user query is empty or irrelevant, return empty values for all fields in strict JSON format."}]

# Prompt2
# Purpose: “Date Range Normalizer” system message. Converts time phrases to YYYY,YYYY.
# Where used: extract_year_range() when not found in deterministic list.
# Institution notes: Adjust the “relative date” rule if your definition of “recent” differs
#   (e.g., last 3 years instead of 5), or if you want a different reference year.

Prompt2 = [ { "role": "system", "content":"You are a strict date-range normalizer.\n"
            "\n"
            "RULES:\n"
            "1. Input: A user-provided time phrase.\n"
            "2. Output: Exactly one value — a comma-separated year range in the format YYYY,YYYY (earlier year first).\n"
            "\n"
            "INTERPRETATION RULES:\n"
            "- Relative phrases (e.g. 'last 5 years', 'past decade', 'previous 6 months') must be interpreted relative to the reference date 2025. Input like 'recently' or 'current' or 'modern' should be the last five years.\n"
            "- If the user provides a specific year range (e.g. '2021–2025', 'between 1950 and 1955') normalize it to YYYY,YYYY.\n"
            "- If the phrase contains only one year, repeat it (e.g. '1999' → '1999,1999').\n"
            "- Named historical events, cultural eras, and ambiguous periods (e.g. 'World War II', 'Cold War', 'Renaissance', 'late 18th century') MUST be normalized into concrete year ranges.\n"
            "- If interpretation is impossible, return 'NONE,NONE'.\n"
            "\n"
            "STRICT OUTPUT FORMAT:\n"
            "- Return ONLY the normalized year range.\n"
            "- Do NOT add text, explanations, or repeat the input phrase.\n" }]

# Prompt3
# Purpose: Boolean string generator tuned for your Discovery syntax (Primo).
# Where used: build_boolean_string() to produce the final query string.
# Institution notes: If your platform differs (EDS/Summon), adjust syntax guidelines:
#   phrase quotes, wildcards, field scoping, and boolean operators vary by system.

Prompt3 = [{  "role": "system", "content": """ You are an assistant that creates Boolean search strings for academic library catalogs (Ex Libris PRIMO syntax).
        These catalogs search titles, abstracts, subject headings, and keywords—NOT full-text.

        Your goal: Create focused, high-precision searches that retrieve directly relevant materials.

        Strategy:
        1. Parse the topic literally—identify the core concepts explicitly stated
        2. For each concept, provide a set of synonyms/variations that would appear in titles and abstracts
        3. Join concepts with AND; join synonyms within a concept with OR
        4. Use parentheses to group OR terms: (term1 OR term2) AND (term3 OR term4)

        Term selection guidelines:
        - Think: "How would scholars title/abstract this work?" 
        - For word families with common variations, use wildcards WITHOUT quotes: adolescen*, cultur*, technolog*
        - For exact multi-word phrases (established terms), use quotes WITHOUT wildcards: "machine learning", "cognitive behavioral therapy"
        - When a multi-word phrase needs a wildcard for variations, DO NOT use quotes: anthropomorphic character* NOT "anthropomorphic character*"
        - Avoid vague, tangential, or overly broad terms
        - Don't invent concepts not stated in the topic
        - Use NOT sparingly—only to exclude a dominant irrelevant meaning

        Output only the Boolean string with no explanation. """ }]

# Prompt4
# Purpose: Material-type extractor/normalizer instructions for LLM fallback.
# Where used: extract_material_types() when a token isn’t found in MaterialTypeMap.
# Institution notes: If your Discovery facet labels differ (e.g., “audio_visual” instead
#   of “videos”), change the enumerated list here and mirror it in MaterialTypeMap/rtypes.

Prompt4 = [ { "role": "system", "content":"""You are a Material-Type Extractor agent. Your task is to identify and normalize explicit mentions of academic material types from user queries.
                Only return a value if the user clearly and directly refers to a material type. Match to one of the following controlled vocabulary types (singular or plural):
                books, journals, articles, images, microform, audios, maps, videos, dissertations, government_documents, reports, book_chapters, scores, archival_material_manuscripts, market_researchs
                If the user asks about or mentions a specific material type (e.g., "Do you have books on...", "Show me peer-reviewed articles about..."), return the best matching label.
                If no material type is explicitly mentioned or implied with high confidence, return an empty string.
                Do not infer material types based on topic or intent alone. Be conservative.
                Return only one label from the list above, or an empty string.""" 
}]


# rtypes
# Purpose: Canonical set of **facet values the URL builder will accept**.
# Where used: construct_url() – only rtypes found here are appended to &mfacet=rtype,...
# Institution notes: These MUST match your Discovery instance’s facet codes exactly.
#   Add/remove values to mirror your local Primo (or EDS/Summon) rtype codes.

rtypes = frozenset({"journals", "books", "articles", "images", "microform", "reviews", "reports", 
        "book_chapters", "scores", "archival_material_manuscripts", "market_researchs",
        "audios", "maps", "videos", "dissertations", "government_documents"})


# MaterialTypeMap
# Purpose: Normalizes raw user words to your facet codes (plural labels above).
# Where used: extract_material_types(); LLM fallback also maps back into this dict.
# Institution notes: Keep keys as user-language variants (“movie”, “film”, “gov doc”),
#   and values as the exact facet codes used in your Discovery. If your codes differ,
#   change ONLY the dict values to your local rtype tokens.

MaterialTypeMap = {
        "book": "books", "books": "books", "ebook": "books", "ebooks": "books",
        "monograph": "books", "monographs": "books",
        "article": "articles", "articles": "articles",
        "journal": "journals", "journals": "journals",
        "paper": "articles", "papers": "articles", "research paper": "articles",
        "video": "videos", "videos": "videos", "film": "videos", "films": "videos", "movie": "videos", "movies": "videos",
        "audio": "audios", "audios": "audios", "audiobook": "audios", "audiobooks": "audios",
        "image": "images", "images": "images",
        "map": "maps", "maps": "maps",
        "microform": "microform",
        "dissertation": "dissertations", "dissertations": "dissertations", "thesis": "dissertations", "theses": "dissertations",
        "government document": "government_documents", "gov doc": "government_documents", "gov docs": "government_documents",
        "government_documents": "government_documents",
        "report": "reports", "reports": "reports",
        "book chapter": "book_chapters", "book chapters": "book_chapters",
        "score": "scores", "scores": "scores",
        "archival material": "archival_material_manuscripts",
        "manuscript": "archival_material_manuscripts", "manuscripts": "archival_material_manuscripts",
        "market research": "market_researchs"
    }


# pattern_time
# Purpose: Regex fallback for extracting a time phrase when the LLM parse fails.
# Where used: split_query() → “regex fallback” path.
# Institution notes: Extend if your users often write other date styles (e.g., “’90s”,
#   “early 20th century”). Anything matched here is passed to Prompt2 or the dt.txt map.

pattern_time = (r"(?P<time>(?:\d{1,2}(?:st|nd|rd|th)?\s)?\d{4}|late\s\d{1,2}(?:st|nd|rd|th)?\scentury|between\s\d{4}\sand\s\d{4})")


# pattern_type
# Purpose: Regex fallback to detect an explicit material type mention.
# Where used: split_query() → “regex fallback” path.
# Institution notes: If your patrons say “films” but your facet is “videos”, add “film|films”
#   here as keys (they will later map through MaterialTypeMap). 
pattern_type = (r"\b(book|books|article|articles|video|videos|audio|audiobook|audiobooks|ebook|ebooks|thesis|theses|report|reports|journal|journals)\b")

# school_held / held_by_school
# Purpose: Literal phrases we strip from the query to detect “held by the institution”
#   without polluting the topic. The splitter removes these exact strings BEFORE
#   sending topic text to the LLM. (open.py removes both of these when cleaning text.)
# Where used: split_query(): they’re in the list of phrases to remove from the user query.
# Institution notes: Set these to your local phrasings, e.g., ("held by yale") / ("yale held").
#   You may also update `local_flag` to a robust regex that matches your variants.
# line 110
school_held = ("sbu held")

held_by_school = ("held by sbu")
