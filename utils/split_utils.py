import numpy as np
from sklearn.model_selection import train_test_split

def make_query_dict(df):
    return (
        df.groupby("qid")[["doc_id","label","pred_score"]]
        .apply(lambda g: list(zip(g["doc_id"], g["label"], g["pred_score"])))
        .to_dict()
    )


def split_query_dict(query_dict, min_docs=5, test_ratio=0.5, random_state=42):
    valid = [qid for qid, docs in query_dict.items() if len(docs) > min_docs]
    qcal, qtest = train_test_split(valid, test_size=test_ratio,
                                   random_state=random_state)
    return (
        {q: query_dict[q] for q in qcal},
        {q: query_dict[q] for q in qtest},
    )
