## 🚀 Run the Demo on Google Colab

You can try the full toy example directly in your browser using Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Minxing-Zheng/CMU_10701_GroupProject/blob/main/10701_ConformalPrediction_demo.ipynb)

The demo notebook walks you through:

- 📦 Loading a pretrained recommendation ranking model  
- 🔍 Predicting on a small sample dataset  
- 🎯 Constructing conformal FDR-controlled selection sets  
- 📈 Running repeated experiments with λ̂-search and empirical FDR  

This provides a lightweight end-to-end introduction to the pipeline.

---

### 📘 Usage with Your Own Model or Dataset

To reuse this conformal FDR pipeline with custom data, you only need to provide a `pandas.DataFrame` like the structure of `df_summary` with the following columns:

| Column       | Description                                      |
| ------------ | ------------------------------------------------ |
| `qid`        | Query / group ID (any hashable type)             |
| `doc_id`     | Unique identifier for each item within the query |
| `label`      | True relevance label (e.g., 0–4 in Yahoo LTR)    |
| `pred_score` | Model prediction score (float)                   |

Example structure:

```python
df_summary = pd.DataFrame({
    "qid": my_query_ids,
    "doc_id": np.arange(len(my_query_ids)),
    "label": my_labels,
    "pred_score": my_scores,
})
```

Then we simply follow the following code in the notebook. 