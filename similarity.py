# similarity.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#유사도 측정하는 함수들
def _to_seq(vec, L, F):
    arr = np.array(vec, dtype=float).reshape(L, F)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _euclid(p, q):
    return float(np.linalg.norm(p - q))

def dtw_distance(A, B):
    n, m = len(A), len(B)
    dp = np.full((n+1, m+1), np.inf, dtype=float); dp[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = _euclid(A[i-1], B[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[n,m] / max(n,m)

def frechet_distance(A, B):
    n, m = len(A), len(B)
    ca = np.full((n, m), -1.0, dtype=float)
    def c(i, j):
        if ca[i,j] > -0.5: return ca[i,j]
        d = _euclid(A[i], B[j])
        if i==0 and j==0: ca[i,j]=d
        elif i==0:        ca[i,j]=max(c(i,j-1), d)
        elif j==0:        ca[i,j]=max(c(i-1,j), d)
        else:             ca[i,j]=max(min(c(i-1,j), c(i-1,j-1), c(i,j-1)), d)
        return ca[i,j]
    return c(n-1, m-1)

def dist_to_sim(dist): 
    return 1.0 / (1.0 + float(dist))

def sim_tier3(vec1, vec2, L=18, F=5, mode='DTW', w_dtw=0.5):
    A = _to_seq(vec1, L, F); B = _to_seq(vec2, L, F)
    if mode == 'DTW':
        return dist_to_sim(dtw_distance(A, B))
    elif mode == 'Frechet':
        return dist_to_sim(frechet_distance(A, B))
    elif mode == 'Hybrid':
        s_dtw = dist_to_sim(dtw_distance(A, B)); s_fr = dist_to_sim(frechet_distance(A, B))
        return float(w_dtw * s_dtw + (1 - w_dtw) * s_fr)
    else:
        return float(cosine_similarity([vec1], [vec2])[0, 0])
