from .config import TOP_K


def retrieve(
    db,
    bm25_retriever,
    question
):

    # 1. Dense search (FAISS)
    dense_results = db.similarity_search_with_score(
        question,
        k=10
    )

    filtered_dense = []
    for doc, score in dense_results:
        if score < 1.5:
            filtered_dense.append((doc, score))

    # 2. Sparse search (BM25)
    if bm25_retriever is not None:
        try:
            sparse_results = bm25_retriever.invoke(question)
        except Exception as e:
            print(f"Warning: BM25 retrieval failed: {e}")
            sparse_results = []
    else:
        sparse_results = []

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL FOR:", question)
    print("=" * 80)
    print(f"Dense (FAISS) retrieved {len(dense_results)} docs (filtered to {len(filtered_dense)} with score < 1.5)")
    print(f"Sparse (BM25) retrieved {len(sparse_results)} docs")

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_k = 60
    rrf_scores = {}
    doc_map = {}

    # Rank in dense
    for rank, (doc, score) in enumerate(filtered_dense):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    # Rank in sparse
    for rank, doc in enumerate(sparse_results):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    combined_results = []
    for key in sorted_keys:
        combined_results.append((doc_map[key], rrf_scores[key]))

    print("\nCOMBINED HYBRID RESULTS (RRF):")
    for i, (doc, score) in enumerate(combined_results[:TOP_K]):
        print(f"\nRESULT {i + 1} (RRF Score: {score:.4f})")
        print("FILE:", doc.metadata.get("source_file"))
        print("PAGE:", doc.metadata.get("page"))
        print(doc.page_content[:300])

    print("\nFILTERED RESULTS:", len(combined_results))
    print("=" * 80)

    return combined_results[:TOP_K]
