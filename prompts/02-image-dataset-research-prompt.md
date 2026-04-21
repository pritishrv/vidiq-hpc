## Image Dataset Research Prompt

Use this prompt when asking an LLM to help survey candidate image datasets for the next stage of the embedding-validation project.

---

You are helping me choose an image dataset for the next stage in a multi-modal embedding-validation pipeline. We already validated sentence embeddings on SST-2 and dair-ai/emotion by extracting raw, L2-normalized, and mean-centered embeddings, then computing centroid distances, silhouette/Davies-Bouldin, k-NN and logistic regression macro F1, and generating PCA/UMAP diagnostics. The next stage must keep those diagnostics but run them on image embeddings, so the dataset must have clear emotion or sentiment labels.

Please provide the following:
1. **Dataset shortlist (2–3 options)**: name, citation/link, licensing, total size, label set (number of emotion/sentiment classes), modality (image, GIF, etc.), typical resolution, preprocessing notes, and any known annotation quality issues.
2. **Embedding recommendations**: for each dataset, note which image encoder people typically use (CLIP, ResNet+contrastive head, ViT, etc.) and whether published work reports centroid gaps, intra/inter distance behavior, or other geometry observations.
3. **Pipeline fit assessment**: for each dataset give concise pros/cons relative to our validation objective—class cleanliness, contextual richness, data quality, ease of integration, and whether the labels align with emotion clusters.
4. **Follow-up questions**: suggest 2–3 clarifications I should ask in a second prompt (e.g., available metadata, train/val split sizes, required annotations for context, licensing caveats).

Frame your response in bullets or tables, emphasize that we will reuse the same embedding diagnostics from the text experiments, and keep the answer within ~250 words.
