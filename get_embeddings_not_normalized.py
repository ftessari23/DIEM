from transformers import AutoTokenizer, AutoModel
import torch

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

from datasets import load_dataset
ds = load_dataset("sentence-transformers/stsb")

def get_non_normalized_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"]

        # Mean pooling (manual)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / count

    return mean_pooled.squeeze(0)  # shape: (hidden_dim,)

def map_func(example):
    emb = get_non_normalized_embedding(example['sentence1'])
    example['embedding'] = emb.numpy()
    return example

import numpy as np
import matplotlib.pyplot as plt
class TextEmbeddingSimilarity:
    def __init__(self, embedding_1, embedding_2, is_file = False):
        if is_file:
            self.sent1 = np.loadtxt(embedding_1, delimiter=',', skiprows=1)
            self.sent2 = np.loadtxt(embedding_2, delimiter=',', skiprows=1)
        else:
            self.sent1 = embedding_1
            self.sent2 = embedding_2

    def compute_cosine_similarity(self):
        A = self.sent1
        B = self.sent2
        return (A @ B.T) / (np.linalg.norm(A, axis=1, keepdims=True) @ np.linalg.norm(B, axis=1, keepdims=True).T)

    def compute_diem_similarity(self, maxV, minV, exp_center, vard):
        DIEM, _ = getDIEM(self.sent1.T, self.sent2.T, maxV, minV, exp_center, vard)
        return DIEM

    def plot_comparison(self, cosine_sim, diem_sim, min_DIEM, max_DIEM):
        diag_cos = np.diag(cosine_sim)
        diag_diem = np.diag(diem_sim)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        plt.hist(diag_cos, bins=50, alpha=0.7)
        plt.title("Cosine Similarity (Rated)")

        plt.subplot(2, 2, 2)
        plt.hist(diag_diem, bins=50, alpha=0.7)
        plt.title("DIEM Similarity (Rated)")
        plt.axvline(min_DIEM, color='k', linestyle='--')
        plt.axvline(max_DIEM, color='k', linestyle='--')

        plt.subplot(2, 2, 3)
        plt.hist(cosine_sim.flatten(), bins=50, alpha=0.7)
        plt.title("Cosine Similarity (All)")

        plt.subplot(2, 2, 4)
        plt.hist(diem_sim.flatten(), bins=50, alpha=0.7)
        plt.title("DIEM Similarity (All)")
        plt.axvline(min_DIEM, color='k', linestyle='--')
        plt.axvline(max_DIEM, color='k', linestyle='--')

        plt.tight_layout()
        plt.show()



emb1= get_non_normalized_embedding(ds['train']['sentence1'] + ds['validation']['sentence1']  + ds['test']['sentence1'])
emb2= get_non_normalized_embedding(ds['train']['sentence2'] + ds['validation']['sentence2']  + ds['test']['sentence2'])
gold_scores = ds['train']['score'] + ds['validation']['score']  + ds['test']['score']
gold_scores = np.array(gold_scores)
from diem_functions import DIEM_Stat, getDIEM
import torch
maxV1 = 1
minV1 = 0
exp_center, vard, std_one, orth_med, min_DIEM, max_DIEM = DIEM_Stat(768, maxV1, minV1, fig_flag=0)
emb_sim = TextEmbeddingSimilarity(emb1.numpy(), emb2.numpy())
diem_sim = emb_sim.compute_diem_similarity(maxV1, minV1, exp_center, vard)
cosine_sim = emb_sim.compute_cosine_similarity()
emb_sim.plot_comparison(cosine_sim, diem_sim, min_DIEM, max_DIEM)
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

print("=== DIEM ===")
print("Spearman:", spearmanr(np.diag(diem_sim), gold_scores).correlation)
print("Pearson:", pearsonr(np.diag(diem_sim), gold_scores).statistic)
print("MSE:", mean_squared_error(gold_scores, np.diag(diem_sim)))

print("\n=== Cosine ===")
print("Spearman:", spearmanr(np.diag(cosine_sim), gold_scores).correlation)
print("Pearson:", pearsonr(np.diag(cosine_sim), gold_scores).statistic)
print("MSE:", mean_squared_error(gold_scores, np.diag(cosine_sim)))
# ds_with_emb = ds['train'].map(map_func)
