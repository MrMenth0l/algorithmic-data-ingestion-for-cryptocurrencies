import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Compute embeddings for a list of texts using a pre-loaded SentenceTransformer model.

    Parameters:
        texts (List[str]): List of text documents.
        batch_size (int): Batch size for embedding computation.

    Returns:
        np.ndarray: 2D array of shape (len(texts), embedding_dim).
    """
    return MODEL.encode(texts, batch_size=batch_size, show_progress_bar=False)

def embed_series(df: pd.DataFrame, text_col: str = 'text', batch_size: int = 32) -> pd.DataFrame:
    """
    Embed a DataFrame column of texts into numeric features.

    Parameters:
        df (pd.DataFrame): DataFrame containing a column of texts.
        text_col (str): Name of the column with text data.
        batch_size (int): Batch size for embedding computation.

    Returns:
        pd.DataFrame: DataFrame of embeddings with same index as df and columns emb_0...emb_n.
    """
    texts = df[text_col].astype(str).tolist()
    embeddings = embed_texts(texts, batch_size=batch_size)
    cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, index=df.index, columns=cols)
