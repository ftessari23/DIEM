import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import null_space

def DIEM_Stat(N, maxV, minV, fig_flag):
    d = []
    dort = []

    for _ in range(int(1e5)):
        a = (maxV - minV) * np.random.rand(N, 1) + minV
        b = (maxV - minV) * np.random.rand(N, 1) + minV
        tmp = null_space(a.T)
        ort = tmp[:, 0].reshape(-1, 1)
        d.append(cdist(a.T, b.T, metric='euclidean')[0][0])
        dort.append(cdist(a.T, ort.T, metric='euclidean')[0][0])


    d = np.array(d)
    dort = np.array(dort)
    exp_center = np.median(d)
    vard = np.var(d)
    orth_med = (maxV - minV) * (np.median(dort) - exp_center) / vard
    adjusted_dist = (maxV - minV) * (d - exp_center) / vard
    std_one = np.std(adjusted_dist)
    min_DIEM = -(maxV - minV) * (exp_center / vard)
    max_DIEM = (maxV - minV) * (np.sqrt(N) * (maxV - minV) - exp_center) / vard

    if fig_flag == 1:
        width = 10
        x = np.arange(1, width + 1)
        plt.figure(figsize=(6, 6))
        plt.fill_between(x, -std_one, std_one, color='r', alpha=0.2)
        plt.fill_between(x, -2 * std_one, 2 * std_one, color='r', alpha=0.2)
        plt.fill_between(x, -3 * std_one, 3 * std_one, color='r', alpha=0.2)
        plt.plot(x, np.zeros(width), 'k--', linewidth=1)
        plt.plot(x, np.full(width, orth_med), 'k-.', linewidth=1)
        plt.plot(x, np.full(width, min_DIEM), 'k-.', linewidth=1)
        plt.plot(x, np.full(width, max_DIEM), 'k-.', linewidth=1)
        plt.ylabel('DIEM')
        plt.xticks([])
        plt.box(False)
        plt.show()

    return exp_center, vard, std_one, orth_med, min_DIEM, max_DIEM

def getDIEM(synMat1, synMat2, maxV, minV, exp_center, vard, Plot='off', Text='off', TextSize=10):
    DIEM = (maxV - minV) * (cdist(synMat1.T, synMat2.T, metric='euclidean') - exp_center) / vard

    if np.allclose(DIEM, DIEM.T):
        DIEM = np.triu(DIEM)
        DIEM[DIEM == 0] = np.nan

    ax = None
    if Plot.lower() == 'on':
        ax = plotDIEM(DIEM, Text, TextSize, 1.1 * np.nanmin(DIEM), np.nanmax(DIEM))

    return DIEM, ax

def plotDIEM(DIEM, textToggle, textSize, minD, maxD):
    m, n = DIEM.shape
    plt.figure()
    cax = plt.imshow(DIEM, vmin=minD, vmax=maxD, cmap='autumn')
    plt.colorbar(cax)
    ax = plt.gca()
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))

    if textToggle.lower() == 'on':
        for i in range(m):
            for j in range(n):
                if not np.isnan(DIEM[i, j]):
                    plt.text(j, i, str(int(round(DIEM[i, j]))),
                             ha='center', va='center',
                             fontweight='bold', fontsize=textSize,
                              color='black')

    plt.gcf().patch.set_facecolor('white')
    plt.show()
    return ax
# --- Main function to test the DIEM functions ---
# --- Similar to Example_DIEM.m ---
if __name__ == '__main__':

    # Parameters
    N = 12
    minV = 0
    maxV = 1
    fig_flag = 1

    # Compute DIEM stats
    exp_center, vard, std_one, orth_med, min_DIEM, max_DIEM = DIEM_Stat(N, maxV, minV, fig_flag)

    # Generate synthetic data
    S1 = np.random.rand(N, 5) * (maxV - minV) + minV
    S2 = np.random.rand(N, 5) * (maxV - minV) + minV

    # Compute DIEM matrix
    DIEM, ax = getDIEM(S1, S2, maxV, minV, exp_center, vard, Plot='on', Text='on')
