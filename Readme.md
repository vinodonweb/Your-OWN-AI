# VectorDB — Build a Vector Database from Scratch in Python

A fully working **Vector Database** built from scratch in Python with a web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood — including the approximate nearest neighbor algorithms, embedding geometry, and retrieval-augmented generation pipelines that power modern AI applications.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

**Why cosine similarity for text embeddings?** Dense vector models like `nomic-embed-text` produce unit-normalizable vectors where the angle between vectors captures semantic relatedness better than raw magnitude. Cosine distance measures exactly this — making it the right choice for 768D embedding spaces.

---

## Prerequisites

You need **3 things** installed:

1. **Python 3.10+**
2. **Git**
3. **Ollama** (runs the local AI models)

---

## Step-by-Step Setup

### Step 1 — Install Python

1. Go to **https://www.python.org/downloads/** and download Python 3.10 or newer
2. Run the installer — **check "Add Python to PATH"** during setup
3. Verify in PowerShell:
```powershell
python --version
```
You should see `Python 3.10.x` or newer.

---

### Step 2 — Install Git

1. Go to **https://git-scm.com/download/win** and download Git for Windows
2. Run the installer with default settings
3. Verify in PowerShell:
```powershell
git --version
```

---

### Step 3 — Install Ollama (Local AI Models)

1. Go to **https://ollama.com** and click **Download for Windows**
2. Run the installer
3. Ollama starts automatically in the system tray
4. Open **PowerShell** and pull the two required models:

```powershell
ollama pull nomic-embed-text
```
*(~274 MB — the embedding model that maps text → 768D vectors)*

```powershell
ollama pull llama3.2
```
*(~2 GB — the language model for answer generation)*

5. Verify Ollama is running:
```powershell
ollama list
```
You should see both models listed.

> **Minimum specs for Ollama:** 8GB RAM recommended. The models will use ~3GB total.

---

### Step 4 — Clone the Repository

Open **PowerShell** and run:

```powershell
git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB
```

*(Replace `YOUR_USERNAME` with the actual GitHub username)*

---

### Step 5 — Install Python Dependencies

```powershell
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, and HTTPX. Takes about 10 seconds.

---

### Step 6 — Run Everything

**Terminal 1** — Start Ollama (if not already running):
```powershell
ollama serve
```
*(If Ollama is already in the system tray, skip this)*

**Terminal 2** — Start the VectorDB server:
```powershell
python main.py
```

You should see:
```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

**Open your browser** and go to:
```
http://localhost:8080
```

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

**The scatter plot** shows all 20 vectors projected to 2D using PCA. Notice how the 4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is what "semantic similarity" looks like visually.

### Tab 2: Documents (Real Embeddings)

This uses Ollama to generate **real 768-dimensional embeddings** from any text.

1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)

1. Make sure you have inserted some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:
```
1. Your question → embedded with nomic-embed-text (768D vector)
2. HNSW search → finds 3 most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2 → generates an answer based only on your documents
```

The answer streams in with a typewriter effect. Click the **context chips** to see exactly which chunks the AI used.

---

## REST API Reference

The server exposes a full REST API at `http://localhost:8080`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

### Example: Search via curl

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl

```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Project Structure

```
VectorDB/
├── main.py          ← Python backend (HNSW, KD-Tree, BruteForce, REST API, RAG)
├── index.html       ← Frontend (PCA scatter plot, chat UI, benchmark)
├── requirements.txt ← Python dependencies (FastAPI, Uvicorn, HTTPX)
└── README.md        ← This file
```

### Architecture (main.py)

```
BruteForce          O(N·d)      Exact, baseline
KDTree              O(log N)    Exact, axis-aligned space partitioning
HNSW                O(log N)    Approximate, multilayer navigable small-world graph

VectorDB            Unified thread-safe interface over all 3 (16D demo vectors)
DocumentDB          HNSW-only index for real Ollama embeddings (768D)
OllamaClient        HTTP client → /api/embeddings + /api/generate
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node randomly gets assigned a maximum layer (probability decays exponentially with `mL = 1/ln(M)`). Layer 0 has all nodes with dense connections; higher layers have exponentially fewer nodes acting as long-range "highways."

**Insert:** Greedily descend from the top layer to find the best entry point, then at each layer from the node's assigned max down to 0, run a beam search (`ef_construction=200`) and connect bidirectionally to the M nearest neighbors. Prune neighbor lists back to M when they grow too large.

**Search:** Same greedy top-down descent. At layer 0, run a full beam search with a dynamic candidate set — expand neighbors, keep the ef closest found so far, terminate when no candidate can improve the result.

**Why it's fast:** The upper layers act as a skip-list — you cover large distances quickly, then zoom into the exact neighborhood at layer 0. Empirically achieves O(log N) with recall >95% at M=16, ef=50.

### KD-Tree (K-Dimensional Tree)

Binary space partitioning. Each node splits the embedding space along one axis (cycling through all dimensions). During search, entire subtrees are pruned when the minimum possible distance to that partition can't beat the current worst result — the "ball-within-hyperslab" condition.

**Weakness:** Degrades severely with high dimensions (curse of dimensionality). Works well for ≤20D, approaches brute force at 768D because the hyperslab pruning becomes useless — in high-D space, nearly all points are equidistant.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high-dimensional spaces, almost all volume concentrates near the surface of the hypersphere, and no axis-aligned subspaces are prunable. HNSW's graph-based navigation doesn't depend on coordinate geometry — it exploits the navigable small-world property regardless of dimensionality.

### RAG Pipeline Design

The retrieve-then-generate approach used here mirrors production LLM systems:

1. **Chunking with overlap** (250 words, 30-word overlap) prevents semantic context from being cut at chunk boundaries — overlapping windows ensure no concept is split across two non-adjacent chunks.
2. **Cosine similarity retrieval** finds chunks whose embedding directions are closest to the query, not necessarily the chunks with the most word overlap.
3. **Context injection** keeps the LLM grounded — it answers from retrieved evidence rather than hallucinating, at the cost of the retrieval quality ceiling.

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal |
| Embedding takes forever | Ollama is downloading the model on first use, wait 2 min |
| `ModuleNotFoundError: fastapi` | Run `pip install -r requirements.txt` |
| Port 8080 already in use | Kill the process: `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| LLM answer is slow | Normal — llama3.2 takes 10–30s on a laptop CPU. Use llama3.2:1b for faster answers |

### Use a Smaller/Faster LLM

If llama3.2 is too slow on your laptop, switch to the 1B model:

```powershell
ollama pull llama3.2:1b
```

Then edit `main.py` — find the `OllamaClient.__init__` method and change:
```python
self.gen_model = "llama3.2:1b"   # change this
```
Restart the server.

---

## License

MIT — use this however you want.
