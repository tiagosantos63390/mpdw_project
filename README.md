# mpdw_project

## ✅ Phase 1 Progress Summary

### 2.1 Video Data and Metadata
- ✅ Parsed **ActivityNet Captions** dataset (`train.json`)
- ✅ Selected 10 videos with the most moments
- ✅ Extracted metadata:
  - `video_id`
  - `timestamps`
  - `captions`
  - `duration`
  - `YouTube URL`

---

### 2.2 Text-Based Search
- ✅ Set up **OpenSearch** and created custom index mappings
- ✅ Defined `caption_bow` field with BM25
- ✅ Indexed all selected video moments (189 total)
- ✅ Tested search with `match` queries (BM25)
- ✅ Verified results from text-based search

---

### 2.3 Embeddings Neighborhood
- ✅ Added `caption_vec` field as `knn_vector` (384-dim)
- ✅ Computed embeddings using `all-MiniLM-L6-v2` (Sentence-BERT)
- ✅ Indexed embeddings in OpenSearch
- ✅ Performed semantic search using KNN vector queries
- ❌ Compared BM25 vs. semantic search results
- ❌ *Embeddings not persisted to file (optional step skipped)*

---

### 2.4 Constrained Embedding Search – **Next Steps**
- ✅ Text based search, e.g., for keyword-based search;
- ✅Embeddings based search; e.g., for semantic search;
- ✅ Add support for **filterable fields** (e.g. duration, tags)
- ✅ Implement search with **boolean filters** (e.g., `"woman" AND duration < 180s`)
- ⬜ Propose and evaluate optimal index mappings

---

### 2.5 Contextual Embeddings & Self-Attention – **Pending**
- ⬜ Visualize contextual word embeddings (layer-wise)
- ⬜ Analyze positional embeddings (repetition behavior)
- ⬜ Examine self-attention in dual vs. cross encoders
- ⬜ Visualize token attention across layers

---

### 2.6 Persistent Storage
- ⬛ *Optional* – not implemented
- 💡 Considered unnecessary for the current small dataset

---

**✔️ Phase 1 Completion: ~75%**
- All core indexing and search functionality is complete.
- Filtering and embedding visualization to be finalized in upcoming steps.