import random as rd
import numpy as np
import scipy.special as sci
import pandas as pd

def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=1, keepdims=True)  # Mean across features
    variance = np.var(x, axis=1, keepdims=True)  # Variance across features
    normalized = (x - mean) / np.sqrt(variance + epsilon)  # Normalization
    return normalized

corpus = np.around(np.random.rand(3,3),3)
corpus.shape
print(f"Corpus shape: {corpus.shape}\n\n")
print(f"Corpus:\n{corpus}")
Q_original = corpus.copy()
Q = corpus.copy()
K = corpus.copy()
V = corpus.copy()

print(f"Q shape: {Q.shape}\n\n")
print(f"Q:\n{Q}")
W_Q = np.around(np.random.rand(3,3),3)
W_K = np.around(np.random.rand(3,3),3)
W_V = np.around(np.random.rand(3,3),3)

print(f"Key weights: {W_K}\n\n")
print(f"Query weights: {W_Q}\n\n")
print(f"Values weights: {W_V}\n\n")
Q = np.dot(Q,W_Q)
K = np.dot(K,W_K)
V = np.dot(V,W_V)

print(f"Key (Post adding weights): \n{K}\n\n")
print(f"Query (Post adding weights): \n{Q}\n\n")
print(f"Values (Post adding weights): \n{V}\n\n")
res = np.dot(Q,K.T)
res = res/np.sqrt(K.shape[1])
res = sci.softmax(res,axis=1)
print(f"Sum = {np.sum(res,axis=1)}")
res = np.dot(res,V)
res = np.around(res,4)

res = layer_norm(res+Q_original)

res_df = pd.DataFrame(res)
res_df.columns=["Amo","el","queso"]
res_df.index=["Amo","el","queso"]
print(f"\n\nEl resultado es: \n{res_df}")