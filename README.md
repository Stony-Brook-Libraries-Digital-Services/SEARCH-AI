# SEARCH AI  

**Advancing Academic Library Discovery with Artificial Intelligence**  

Stony Brook University Libraries has developed **SEARCH AI**, an AI-powered search tool designed to transform how users engage with the library catalog. Users can now interact with a full library catalog using **natural language queries** instead of relying solely on traditional keyword searches, allowing them to receive more relevant and precise results and navigate the catalog more efficiently and intuitively.  

---

## How it works

SEARCH AI is built on an **agentic AI framework** that parses natural language input (phrases and questions closer to how we speak) and dynamically translates them into **refined Boolean queries with relevant filters**.  

### **Agentic Pipeline Overview**
1. **Splitter Agent**  
   - Identifies the core topic  
   - Detects peer-reviewed requests, online availability, local holdings, and physical vs. digital filters  
   - Extracts date expressions (e.g., “early 1700s”, “recent years”, “mid-19th century”)  
   - Determines intended material type (books, articles, theses, etc.)

2. **Time Agent**  
   - Converts natural-language dates into numeric year ranges  
   - Uses a deterministic lookup table (`deterministicttimes.txt`)  
   - Handles expressions such as:  
     - “early 1700s” → 1700–1724  
     - “late 20th century” → 1975–1999  
     - “recent years” → 2020–2025  
     - “17th century” → 1600–1699  

3. **Boolean Agent**  
   - Constructs refined Boolean search strings  
   - Groups terms into concept buckets  
   - Expands with controlled vocabulary where appropriate  
   - Produces catalog-ready Boolean expressions

4. **Material Type Agent**  
   - Maps intent to Primo facets (books, articles, journals, etc.)

5. **URL Builder**  
   - Produces the final Primo/Alma discovery URL  
   - Includes Boolean query, facets, filters, and date ranges

This creates a refined, production-ready discovery URL such as:
**https://search.library.stonybrook.edu/discovery/search?vid=01SUNY_STB:01SUNY_STB&query=any,contains,%28sample+AND+search%29&search_scope=EverythingNZBooks**



## 📦 Prerequisites

1. **Python 3.10+**  
2. **PHP 8.0+** 
3. **OpenAI API Key**  
   - If you're new to the OpenAI API, [sign up for an account](https://platform.openai.com/signup).
   - Follow the [Quickstart](https://platform.openai.com/docs/quickstart) to retrieve your API key.

## How to use

- **Clone the Repository**
```bash
git clone https://github.com/Stony-Brook-Libraries-Digital-Services/SEARCH-AI.git
``` 


- Install dependencies
  
   - Open terminal of file requirements.txt and run the following command: 
   ```bash
   pip install -r requirements.txt
   ```

- Edit varlist.py
   - **Set your API key:** in **varlist.py**
   ```bash
   OPENAI_KEY = (" your_openai_api_key_here ")
   ```

  - You will want to update these fields to use your local institution’s nomenclature, including library name references and material type options.
  
   Required: 
   ```bash
   prefix = ("https://search.library.stonybrook.edu/discovery/search?vid=01SUNY_STB:01SUNY_STB")
   local_flag = (r"\b(held (by|at)? sbu|(by|at)? our library|in our library|at sbu|stony brook|stony brook university|the university?|library      holdings?)\b")
   MaterialTypeMap = {
        "book": "books", "books": "books", "ebook": "books", "ebooks": "books",
        "monograph": "books", "monographs": "books",
        "article": "articles", "articles": "articles",
        "journal": "journals", "journals": "journals",...
   ```
   These MUST match your Discovery instance’s facet codes exactly.
  ```bash
   rtypes = frozenset({"journals", "books", "articles", "images", "microform", "reviews", "reports", ....
  ```
  Match with your institution
  ```bash
  school_held = ("__ held")
  held_by_school = ("held by __")
  ```

- Important: Allow origins in proxy.php to include any page where this function is accessible.

  Example:
  ```bash
  $allowed_origins = [
  'https://search.library.stonybrook.edu',
  'https://library.stonybrook.edu'
  ];
  ```

### Optional Additions
- Add custom time-range entries (changes must be made in (`deterministicttimes.txt`)
- Adjust splitter heuristics
```bash
Prompt1 = [ { "role": "system", "content": "You are a 'Query Splitter' agent. . .
``` 
- Add your own agents (e.g., subject-area classifier)
   - Edits must be made in open.py as well

SEARCH AI is intentionally modular so universities can adapt it easily.

# 🧪 Testing Modes

Use `open.py` to run different testing modes. In open.py we have created modes for testing and seeing more specific details.  The modes are split, extract_year, build_bool, extract_type We have highlighted two key modes: url and split_full which can be ran in the terminal.


### **1. Full Agent Breakdown**
```bash
python open.py split_full "your query here"
```
Returns:

- Splitter agent output  
- Time agent output  
- Boolean agent output  
- Material type agent output  
- Final assembled components  

### **2. URL-Only Mode**
```bash
python open.py url "your query here"
```
Returns only the **final generated Primo URL**.

---

## Fine-tuning is used in SEARCH AI to improve the creation of Boolean strings. This step isn't strictly necessary, but can improve the output of the Boolean agent.
- How to fine-tune using the provided files
- This repository includes a JSONL file called Boolean_finetune_top100.jsonl
- This is an example training file that was used in the training of SEARCH AI at Stony Brook University at basic OpenAI fine-tuning presets.
- We recommend adjusting fine-tuning data as needed, especially in the testing of different models. We fine-tuned GPT4.1 nano.
- The exact JSONL format should be kept consistent (e.g., OpenAI chat-style training format). Each line is a separate training example.

⚠️ Note: We found that over-tuning the model can result in the overuse of Boolean parameters—especially OR cases and wildcards (*)—which can make the strings difficult for humans to read.

### 1. Run a Training Job  
Example (from `fine_tunetesting.py`):

```bash
python3 fine_tunetesting.py --train boolean_finetune_top100.jsonl
```

After training, OpenAI will output a model ID like:

```
ft:gpt-4.1-nano-2025-02-xx:your-org:model-searchai
```

### 2. Update SEARCH AI to Use Your Model  
Place the model ID inside:

```
model_id.txt
```

The system will automatically load this model for Boolean generation.

## 📊 SearchAI Dashboard Module (Search Logs & Analytics)

The **SearchAI Dashboard** is an optional analytics module that visualizes real user query behavior, system performance. It helps library teams understand how users interact with SearchAI and identify patterns.

Make sure to run in the integrated terminal of the logs folder as the search.jsonl is a relative path and would not be accessible from outside of the integrated terminal.

---

### 🗂 Included Files

#### **`app.py` — Dashboard Backend (Flask Application)**
The primary backend service for the SearchAI Dashboard.

It provides:
- A **Flask-based API** for dashboard routes  
- Endpoints that read and visualize search logs (`search.jsonl`)  
- Endpoints for storing and retrieving user/librarian comments (`comments.db`)  
- Optional charting/analytics exports  
- CORS support for embedding into admin portals  

Run the dashboard locally:

```bash
python app.py
```

By default, the dashboard loads at a pointer to the local host

---

#### **`search.jsonl` — Search Log File (JSONL Format)**
A production-style log of every search processed by SearchAI.

Each line typically includes:
- Timestamp  
- Natural-language query  
- Agentic outputs (splitter, time, boolean, material type)  
- Final generated \Primo URL  
- Latency and performance metadata  

Used to generate:
- Query frequency charts  
- Keyword analysis  
- Error pattern detection  
- Agent behavior insights  

You may archive or clear this file at any time.

---

## 🚀 Running the Dashboard

### 1. Install all dependencies
(Already included in `requirements.txt`)

```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python app.py
```

### 3. View the dashboard
Open:

```
By default, the dashboard loads at a pointer to the local host
```

You will see visual tools for:
- Search log review  
- Boolean accuracy assessment  
- Time and frequency analytics  
- Feedback/comment curation  

---


## 🤝 Get Involved  

We welcome collaboration with institutions and researchers interested in **next-generation discovery tools**.  

Ways to contribute:  
- Open an issue in this repository  
- Submit a pull request  
- Contact the project team at Stony Brook University Libraries  

---

## 📜 License  


This project is licensed under the MIT License. See the MIT LICENSE file for details.



MIT License

Copyright (c) 2025 Stony Brook University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
