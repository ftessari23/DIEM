import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.spatial.distance import cdist
from scipy.linalg import norm
import os
from diem_functions import DIEM_Stat, getDIEM

# --- Utility Functions ---
def cos_sim(a, b):
    return (a.T @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def randu_sphere(n, N, max_val=1.0, min_val=-1.0):
    x = np.random.randn(n, N)
    x /= np.linalg.norm(x, axis=0)
    radius = np.random.uniform(min_val, max_val, size=(1, N))
    return x * radius

# --- Class for Vector Distance Simulations ---
class DistanceAnalysis:
    def __init__(self, N_range, vmax=1, vmin=0, dist_type=1):
        self.N = N_range
        self.vmax = vmax
        self.vmin = vmin
        self.dist_type = dist_type
        self.results = {}

    def simulate(self):
        for i, n_dim in enumerate(self.N):
            print(f"Simulating for N = {n_dim}...")
            self.results[n_dim] = {
                'cos_p': [], 'cos_n': [], 'cos_t': [],
                'd_p': [], 'd_n': [], 'd_t': [],
                'dnorm_p': [], 'dnorm_n': [], 'dnorm_t': [],
                'man_p': [], 'man_n': [], 'man_t': []
            }
            for _ in range(10000):
                a_p, a_n, a_t = self._generate_vectors(n_dim)
                b_p, b_n, b_t = self._generate_vectors(n_dim)

                self._append_metrics(n_dim, a_p, b_p, 'p')
                self._append_metrics(n_dim, a_n, b_n, 'n')
                self._append_metrics(n_dim, a_t, b_t, 't')

    def _generate_vectors(self, dim):
        if self.dist_type == 1:  # Uniform
            return (
                self.vmax * np.random.rand(dim, 1),
                -self.vmax * np.random.rand(dim, 1),
                (2 * self.vmax) * np.random.rand(dim, 1) - self.vmax
            )
        elif self.dist_type == 2:  # Gaussian
            return (
                0.3 * np.random.randn(dim, 1) + self.vmax / 2,
                0.3 * np.random.randn(dim, 1) - self.vmax / 2,
                0.6 * np.random.randn(dim, 1)
            )
        elif self.dist_type == 3:  # Uniform on Sphere
            return (
                randu_sphere(dim, 1, self.vmax, self.vmin),
                randu_sphere(dim, 1, self.vmin, -self.vmax),
                randu_sphere(dim, 1, self.vmax, -self.vmax)
            )
        else:
            raise ValueError("Invalid distribution type")

    def _append_metrics(self, dim, a, b, key):
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        self.results[dim][f'cos_{key}'].append(cos_sim(a, b))
        self.results[dim][f'd_{key}'].append(cdist(a.T, b.T)[0][0])
        self.results[dim][f'dnorm_{key}'].append(cdist((a / np.linalg.norm(a)).T, (b / np.linalg.norm(b)).T)[0][0])
        self.results[dim][f'man_{key}'].append(cdist(a.T, b.T, metric='cityblock')[0][0])

    def plot_results(self):
        keys = ['cos', 'dnorm', 'd', 'man']
        types = ['p', 'n', 't']
        for metric in keys:
            plt.figure(figsize=(15, 5))
            for i, t in enumerate(types):
                data = [[float(v) if isinstance(v, np.ndarray) else v for v in self.results[n][f'{metric}_{t}']] for n in self.N]

                ax = plt.subplot(1, 3, i + 1)
                sns.boxplot(data=data)
                ax.set_title(f"{metric.upper()} - Type {t.upper()}")
                ax.set_xticklabels(self.N)
                ax.set_xlabel("Dimensions")
                ax.set_ylabel(metric)
            plt.tight_layout()
            plt.show()

# --- Class for Text Embedding Similarity. Seems everything right to me ---
class TextEmbeddingSimilarity:
    def __init__(self, embedding_1, embedding_2, is_file=False):
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


def estimate_diem_stats(embeddings, num_samples=10000):
    idx = np.random.randint(0, len(embeddings), size=(num_samples, 2))
    distances = np.linalg.norm(embeddings[idx[:, 0]] - embeddings[idx[:, 1]], axis=1)
    return np.median(distances), np.var(distances)

def ztest_manual(data, value=0, std_dev=1):
    mean_sample = np.mean(data)
    n = len(data)
    z = (mean_sample - value) / (std_dev / np.sqrt(n))
    p = 2 * (1 - scipy.stats.norm.cdf(abs(z))) 
    return z, p

# --- Main Execution. @ftessari23 should qualitatively test the results. ---
# --- Similar to 
if __name__ == '__main__':
    N_list = list(range(2, 102, 10))
    dist_type = int(input("Choose a distribution type: (1) Uniform, (2) Gaussian, (3) Uniform on Unit-Sphere): "))
    
    print("Running synthetic vector distance analysis...")
    analyzer = DistanceAnalysis(N_list, vmax=1, vmin=0, dist_type=dist_type)
    analyzer.simulate()
    print("Plotting results...")
    analyzer.plot_results()

    print("\nRunning text embedding similarity analysis...")
    emb1_file = os.path.join('TextEmbeddings', 'embeddings1.csv')
    emb2_file = os.path.join('TextEmbeddings', 'embeddings2.csv')
    
    emb_sim = TextEmbeddingSimilarity(emb1_file, emb2_file, is_file=True)
    
    cosine_sim = emb_sim.compute_cosine_similarity()

    N = emb_sim.sent1.shape[1]
    maxV = 1
    minV = 0

    exp_center, vard, std_one, orth_med, min_DIEM, max_DIEM = DIEM_Stat(N, maxV, minV, fig_flag=0)

    diem_sim = emb_sim.compute_diem_similarity(maxV, minV, exp_center, vard)
    
    emb_sim.plot_comparison(cosine_sim, diem_sim, min_DIEM, max_DIEM)



