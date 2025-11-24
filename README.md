### 🚀 Toy Example on Colab

Try the **`10701_ConformalPrediction_demo.ipynb`** notebook for a quick end-to-end demonstration on Colab.

This example shows how to:

- 📦 Load a pretrained recommendation ranking model  
- 🔍 Run predictions on a small sample dataset  
- 🎯 Apply conformal FDR control step-by-step  

Perfect for getting started with the pipeline in a lightweight setting.

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