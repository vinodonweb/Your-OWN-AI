from __future__ import annotations

import heapq
import math
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # demo vector dimensionality

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id:       int
    metadata: str
    category: str
    emb:      list[float]

@dataclass
class DocItem:
    id:    int
    title: str
    text:  str
    emb:   list[float]

DistFn = Callable[[list[float], list[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a)
    nb  = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))

def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self) -> None:
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: list[float], k: int,
            dist_fn: DistFn) -> list[tuple[float, int]]:
        r = [(dist_fn(q, v.emb), v.id) for v in self.items]
        r.sort()
        return r[:k]

    def remove(self, id_: int) -> None:
        self.items = [v for v in self.items if v.id != id_]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item  = item
        self.left:  Optional[KDNode] = None
        self.right: Optional[KDNode] = None

class KDTree:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _ins(self, node: Optional[KDNode],
             v: VectorItem, d: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left  = self._ins(node.left,  v, d + 1)
        else:
            node.right = self._ins(node.right, v, d + 1)
        return node

    def insert(self, v: VectorItem) -> None:
        self.root = self._ins(self.root, v, 0)

    def _knn(self, node: Optional[KDNode], q: list[float], k: int,
             depth: int, dist_fn: DistFn, heap: list) -> None:
        if node is None:
            return
        dn = dist_fn(q, node.item.emb)
        # heap is a max-heap via negated distances: (-dist, id)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax   = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer,  q, k, depth + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)

    def knn(self, q: list[float], k: int,
            dist_fn: DistFn) -> list[tuple[float, int]]:
        heap: list = []
        self._knn(self.root, q, k, 0, dist_fn, heap)
        result = [(-neg_d, id_) for neg_d, id_ in heap]
        result.sort()
        return result

    def rebuild(self, items: list[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.root = self._ins(self.root, v, 0)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

@dataclass
class HNSWNode:
    item:    VectorItem
    max_lyr: int
    nbrs:    list[list[int]]  # nbrs[layer] = neighbor IDs at that layer

class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200) -> None:
        self.M         = m
        self.M0        = 2 * m
        self.ef_build  = ef_build
        self.mL        = 1.0 / math.log(m)
        self.G:        dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt  = -1
        self._rng      = random.Random(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int,
                      lyr: int, dist_fn: DistFn) -> list[tuple[float, int]]:
        visited: set[int] = {ep}
        d0    = dist_fn(q, self.G[ep].item.emb)
        cands = [(d0, ep)]    # min-heap: (dist, id)
        found = [(-d0, ep)]   # max-heap: (-dist, id)

        while cands:
            cd, cid = heapq.heappop(cands)
            worst   = -found[0][0]
            if len(found) >= ef and cd > worst:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd    = dist_fn(q, self.G[nid].item.emb)
                worst = -found[0][0]
                if len(found) < ef or nd < worst:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        res = [(-neg_d, id_) for neg_d, id_ in found]
        res.sort()
        return res

    def _select_nbrs(self, cands: list[tuple[float, int]],
                     max_m: int) -> list[int]:
        return [id_ for _, id_ in cands[:max_m]]

    def insert(self, item: VectorItem, dist_fn: DistFn) -> None:
        id_  = item.id
        lvl  = self._rand_level()
        self.G[id_] = HNSWNode(
            item=item, max_lyr=lvl,
            nbrs=[[] for _ in range(lvl + 1)]
        )

        if self.entry_pt == -1:
            self.entry_pt  = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            ep_node = self.G.get(ep)
            if ep_node and lc < len(ep_node.nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W     = self._search_layer(item.emb, ep, self.ef_build, lc, dist_fn)
            max_M = self.M0 if lc == 0 else self.M
            sel   = self._select_nbrs(W, max_M)
            self.G[id_].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                nnode = self.G[nid]
                while len(nnode.nbrs) <= lc:
                    nnode.nbrs.append([])
                conn = nnode.nbrs[lc]
                conn.append(id_)
                if len(conn) > max_M:
                    ds = [
                        (dist_fn(nnode.item.emb, self.G[c].item.emb), c)
                        for c in conn if c in self.G
                    ]
                    ds.sort()
                    nnode.nbrs[lc] = [c for _, c in ds[:max_M]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt  = id_

    def knn(self, q: list[float], k: int, ef: int,
            dist_fn: DistFn) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            ep_node = self.G.get(ep)
            if ep_node and lc < len(ep_node.nbrs):
                W = self._search_layer(q, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, id_: int) -> None:
        if id_ not in self.G:
            return
        for nd in self.G.values():
            for i, layer in enumerate(nd.nbrs):
                nd.nbrs[i] = [x for x in layer if x != id_]
        if self.entry_pt == id_:
            self.entry_pt = -1
            for nid in self.G:
                if nid != id_:
                    self.entry_pt = nid
                    break
        del self.G[id_]

    def get_info(self) -> dict:
        max_l           = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes: list[dict] = []
        edges: list[dict] = []
        for id_, nd in self.G.items():
            nodes.append({
                "id": id_, "metadata": nd.item.metadata,
                "category": nd.item.category, "maxLyr": nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})
        return {
            "topLayer":      self.top_layer,
            "nodeCount":     len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes":         nodes,
            "edges":         edges,
        }

    def size(self) -> int:
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

@dataclass
class Hit:
    id:       int
    meta:     str
    cat:      str
    emb:      list[float]
    distance: float

@dataclass
class SearchOut:
    hits:   list[Hit]
    us:     int
    algo:   str
    metric: str

class VectorDB:
    def __init__(self, dims: int) -> None:
        self.dims     = dims
        self.store:   dict[int, VectorItem] = {}
        self.bf       = BruteForce()
        self.kdt      = KDTree(dims)
        self.hnsw     = HNSW(16, 200)
        self.lock     = threading.Lock()
        self._next_id = 1

    def insert(self, meta: str, cat: str, emb: list[float],
               dist_fn: DistFn) -> int:
        with self.lock:
            v = VectorItem(id=self._next_id, metadata=meta,
                           category=cat, emb=emb)
            self._next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist_fn)
            return v.id

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: list[float], k: int,
               metric: str, algo: str) -> SearchOut:
        with self.lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dist_fn)
            else:
                raw = self.hnsw.knn(q, k, 50, dist_fn)
            us = int((time.perf_counter() - t0) * 1_000_000)
            hits = [
                Hit(id=id_, meta=self.store[id_].metadata,
                    cat=self.store[id_].category,
                    emb=self.store[id_].emb, distance=d)
                for d, id_ in raw if id_ in self.store
            ]
            return SearchOut(hits=hits, us=us, algo=algo, metric=metric)

    def benchmark(self, q: list[float], k: int, metric: str) -> dict:
        with self.lock:
            dist_fn = get_dist_fn(metric)

            def timed(fn) -> int:
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return {
                "bruteforceUs": timed(lambda: self.bf.knn(q, k, dist_fn)),
                "kdtreeUs":     timed(lambda: self.kdt.knn(q, k, dist_fn)),
                "hnswUs":       timed(lambda: self.hnsw.knn(q, k, 50, dist_fn)),
                "itemCount":    len(self.store),
            }

    def all(self) -> list[VectorItem]:
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.lock:
            return len(self.store)

# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
# =====================================================================

class DocumentDB:
    def __init__(self) -> None:
        self.store:   dict[int, DocItem] = {}
        self.hnsw     = HNSW(16, 200)
        self.bf       = BruteForce()
        self.lock     = threading.Lock()
        self._next_id = 1
        self._dims    = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self.lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(id=self._next_id, title=title, text=text, emb=emb)
            self._next_id += 1
            self.store[item.id] = item
            vi = VectorItem(id=item.id, metadata=title, category="doc", emb=emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int,
               max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self.lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            return [
                (d, self.store[id_])
                for d, id_ in raw
                if id_ in self.store and d <= max_dist
            ]

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> list[DocItem]:
        with self.lock:
            return list(self.store.values())

    def size(self) -> int:
        with self.lock:
            return len(self.store)

    def get_dims(self) -> int:
        return self._dims

# =====================================================================
#  OLLAMA CLIENT  — wraps local Ollama REST API
#  Install:  https://ollama.com
#  Models:   ollama pull nomic-embed-text
#            ollama pull llama3.2
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434) -> None:
        self.base_url    = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        try:
            r = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30.0,
            )
            if r.status_code == 200:
                return r.json().get("embedding", [])
        except Exception:
            pass
        return []

    def generate(self, prompt: str) -> str:
        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180.0,
            )
            if r.status_code == 200:
                return r.json().get("response", "")
        except Exception:
            pass
        return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250,
               overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks: list[str] = []
    step = chunk_words - overlap_words
    i    = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    db.insert("Linked List: nodes connected by pointers", "cs",
        [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06], dist)
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
        [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10], dist)
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
        [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07], dist)
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
        [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07], dist)
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
        [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09], dist)
    db.insert("Calculus: derivatives integrals and limits", "math",
        [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10], dist)
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09], dist)
    db.insert("Probability: distributions random variables Bayes theorem", "math",
        [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08], dist)
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
        [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06], dist)
    db.insert("Combinatorics: permutations combinations generating functions", "math",
        [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07], dist)
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07], dist)
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
        [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08], dist)
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06], dist)
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09], dist)
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
        [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06], dist)
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72], dist)
    db.insert("Football: tackles touchdowns field goals and strategy", "sports",
        [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76], dist)
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82], dist)
    db.insert("Chess: openings endgames tactics strategic board game", "sports",
        [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90], dist)
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80], dist)

# =====================================================================
#  APP + GLOBALS
# =====================================================================

db     = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)

app = FastAPI(title="VectorDB", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

def parse_vec(s: str) -> list[float]:
    result: list[float] = []
    for part in s.split(","):
        try:
            result.append(float(part.strip()))
        except ValueError:
            pass
    return result

# =====================================================================
#  DEMO VECTOR ENDPOINTS
# =====================================================================

@app.get("/search")
def search(v: str = "", k: int = 5,
           metric: str = "cosine", algo: str = "hnsw"):
    q = parse_vec(v)
    if len(q) != DIMS:
        return JSONResponse({"error": f"need {DIMS}D vector"})
    out = db.search(q, k, metric, algo)
    return JSONResponse({
        "results": [
            {
                "id":        h.id,
                "metadata":  h.meta,
                "category":  h.cat,
                "distance":  round(h.distance, 6),
                "embedding": h.emb,
            }
            for h in out.hits
        ],
        "latencyUs": out.us,
        "algo":      out.algo,
        "metric":    out.metric,
    })

@app.post("/insert")
async def insert_item(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"})
    meta = body.get("metadata", "")
    cat  = body.get("category", "")
    emb  = body.get("embedding", [])
    if not meta or not emb or len(emb) != DIMS:
        return JSONResponse({"error": "invalid body"})
    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return JSONResponse({"id": id_})

@app.delete("/delete/{id}")
def delete_item(id: int):
    ok = db.remove(id)
    return JSONResponse({"ok": ok})

@app.get("/items")
def get_items():
    return JSONResponse([
        {"id": v.id, "metadata": v.metadata,
         "category": v.category, "embedding": v.emb}
        for v in db.all()
    ])

@app.get("/benchmark")
def benchmark(v: str = "", k: int = 5, metric: str = "cosine"):
    q = parse_vec(v)
    if len(q) != DIMS:
        return JSONResponse({"error": f"need {DIMS}D vector"})
    return JSONResponse(db.benchmark(q, k, metric))

@app.get("/hnsw-info")
def hnsw_info():
    return JSONResponse(db.hnsw_info())

@app.get("/stats")
def stats():
    return JSONResponse({
        "count":      db.size(),
        "dims":       DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics":    ["euclidean", "cosine", "manhattan"],
    })

# =====================================================================
#  DOCUMENT + RAG ENDPOINTS
# =====================================================================

@app.post("/doc/insert")
async def doc_insert(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"})
    title = body.get("title", "")
    text  = body.get("text",  "")
    if not title or not text:
        return JSONResponse({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids: list[int] = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return JSONResponse({
                "error": (
                    "Ollama unavailable. "
                    "Install from https://ollama.com then run: "
                    "ollama pull nomic-embed-text && ollama pull llama3.2"
                )
            })
        chunk_title = (
            f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
        )
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return JSONResponse({
        "ids":    ids,
        "chunks": len(chunks),
        "dims":   doc_db.get_dims(),
    })

@app.delete("/doc/delete/{id}")
def doc_delete(id: int):
    ok = doc_db.remove(id)
    return JSONResponse({"ok": ok})

@app.get("/doc/list")
def doc_list():
    return JSONResponse([
        {
            "id":      doc.id,
            "title":   doc.title,
            "preview": doc.text[:120] + ("…" if len(doc.text) > 120 else ""),
            "words":   len(doc.text.split()),
        }
        for doc in doc_db.all()
    ])

@app.post("/doc/search")
async def doc_search(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"})
    question = body.get("question", "")
    k        = body.get("k", 3)
    if not question:
        return JSONResponse({"error": "need question"})
    q_emb = ollama.embed(question)
    if not q_emb:
        return JSONResponse({"error": "Ollama unavailable"})
    hits = doc_db.search(q_emb, k)
    return JSONResponse({
        "contexts": [
            {"id": doc.id, "title": doc.title, "distance": round(d, 4)}
            for d, doc in hits
        ]
    })

@app.post("/doc/ask")
async def doc_ask(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"})
    question = body.get("question", "")
    k        = body.get("k", 3)
    if not question:
        return JSONResponse({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return JSONResponse({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)

    ctx = "".join(
        f"[{i+1}] {doc.title}:\n{doc.text}\n\n"
        for i, (_, doc) in enumerate(hits)
    )
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}"
        f"Question: {question}\n\n"
        "Answer:"
    )
    answer = ollama.generate(prompt)

    return JSONResponse({
        "answer":   answer,
        "model":    ollama.gen_model,
        "contexts": [
            {
                "id":       doc.id,
                "title":    doc.title,
                "text":     doc.text,
                "distance": round(d, 4),
            }
            for d, doc in hits
        ],
        "docCount": doc_db.size(),
    })

@app.get("/status")
def status():
    up = ollama.is_available()
    return JSONResponse({
        "ollamaAvailable": up,
        "embedModel":      ollama.embed_model,
        "genModel":        ollama.gen_model,
        "docCount":        doc_db.size(),
        "docDims":         doc_db.get_dims(),
        "demoDims":        DIMS,
        "demoCount":       db.size(),
    })

@app.get("/")
def index():
    return FileResponse("index.html")

# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    uvicorn.run(app, host="0.0.0.0", port=8080)
