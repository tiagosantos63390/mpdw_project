# mpdw_project


71802 André Santos
71921 Tiago Santos
70663 Maximo Volynets


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
- ✅ Compared BM25 vs. semantic search results
- ❌ *Embeddings not persisted to file (optional step skipped)*

---

### 2.4 Constrained Embedding Search – **Next Steps**
- ✅ Text based search, e.g., for keyword-based search;
- ✅Embeddings based search; e.g., for semantic search;
- ✅ Add support for **filterable fields** (e.g. duration, tags)
- ✅ Implement search with **boolean filters** (e.g., `"woman" AND duration < 180s`)
- ✅ Propose and evaluate optimal index mappings  

---

### 2.5 Contextual Embeddings & Self-Attention – **Pending**
- ✅ Visualize contextual word embeddings (layer-wise)
- ✅ Analyze positional embeddings (repetition behavior)
- ✅ Examine self-attention in dual vs. cross encoders
- ✅ Visualize token attention across layers

---

### 2.6 Persistent Storage
- ⬛ *Optional* – not implemented
- 💡 Considered unnecessary for the current small dataset

---

**✔️ Phase 1 Completion: ~100%**
- All core indexing and search functionality is complete.
- Filtering and embedding visualization to be finalized in upcoming steps.



## ⬜ Phase 2 Progress Summary

### 🔹 1. Cross-Modal Retrieval with CLIP

- ⬜ **Extract Keyframes**
  - Sample one frame every 2 seconds.
  - Save as JPEGs in `data/frames/`.

- ⬜ **Compute CLIP Embeddings**
  - Use CLIP to compute:
    - `image_vec` for keyframes.
    - `text_vec` for captions.

- ⬜ **Extend OpenSearch Index**
  - Add a new field:
    ```json
    "image_vec": { "type": "knn_vector", "dimension": 512 }
    ```

- ⬜ **Index Keyframes**
  - Index each frame with:
    - `video_id`, `timestamp`, `image_path`, `image_vec`.

- ⬜ **Implement Search Queries**
  - ⬜ Text → Image
  - ⬜ Image → Image
  - ⬜ (Optional) Text + Image → Image

- ⬜ **Evaluate Retrieval**
  - Compare cross-modal vs. unimodal.
  - Log top-5 results for each query type.

---

### 🔹 2. Visual Question Answering with LLaVA

- ⬜ **Set Up LLaVA**
  - Use the API or run locally (GPU ≥ 12 GB).

- ⬜ **Retrieval-Augmented VQA**
  - ⬜ Encode the visual question (text).
  - ⬜ Use CLIP to retrieve top-1 frame.
  - ⬜ Pass frame + question to LLaVA.
  - ⬜ Collect and log the answer.

- ⬜ **Evaluate VQA**
  - Prepare 10–20 questions per video.
  - Manual or automatic assessment.

---

### 🔹 3. Interpretability of LVLMs

- ⬜ **Attention Maps**
  - Visualize attention weights from CLIP and/or LLaVA.

- ⬜ **Relevancy Maps**
  - Apply Grad-CAM or similar over image inputs.

- ⬜ **Causal Graphs (Advanced)**
  - Explore masking-based influence on outputs.

- ⬜ **Analysis**
  - Discuss differences in focus between questions, images, and answers.
  - Identify hallucination or bias cases.

---

### 💾 Optional: Persistent Storage

- ⬜ Use `pickle`, `HDF5`, or `parquet` to store:
  - CLIP embeddings
  - VQA answers

---

### 📝 Reporting Guidelines

Include this phase as a 5-page section in your report:
- CLIP + cross-modal indexing
- Llava + QA pipeline
- Retrieval/VQA results
- Interpretability insights

Report latex link: https://www.overleaf.com/2958122766xhmchvsvpwyj#6c7348