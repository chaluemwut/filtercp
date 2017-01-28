from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

if __name__ == '__main__':
    y = np.array([1,2,3])
    cos = cosine_similarity([[5,4,7], [1,2,3]],[[10,20,30],[3,4,5],[5,6,7]])
    print(cos)
    print(cos.max(axis=1))
    idx = cos.argmax(axis=1)
    print(idx)
    print(y[idx])