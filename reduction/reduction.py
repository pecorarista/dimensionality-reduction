from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA


def apply_ppa(wv: KeyedVectors, top_n: int) -> KeyedVectors:
    V = wv.vectors
    M = V.mean(axis=0)
    V_tilde = V - M
    pca = PCA(n_components=top_n)
    pca.fit(V_tilde)
    W = pca.components_
    V_hat = V_tilde - V_tilde @ W.T @ W

    result = KeyedVectors(wv.vector_size)
    result.add_vectors(list(wv.key_to_index.keys()), V_hat)
    return result


def reduce(
    wv: KeyedVectors,
    new_dimension: int,
    top_n: int
) -> KeyedVectors:

    if new_dimension >= wv.vector_size:
        message = f'''
            `new_dimension` ({new_dimension}) should be less than the original dimension ({wv.vector_size})'
        '''.strip()
        raise ValueError(message)
    wv_ppa = apply_ppa(wv, top_n=top_n)
    pca = PCA(n_components=new_dimension)
    Z = pca.fit_transform(wv_ppa.vectors)

    wv_ppa_pca = KeyedVectors(new_dimension)
    wv_ppa_pca.add_vectors(list(wv.key_to_index.keys()), Z)

    return apply_ppa(wv_ppa_pca, top_n=top_n)


def reduce_by_pca(wv: KeyedVectors, new_dimension: int) -> KeyedVectors:
    result = KeyedVectors(new_dimension)
    pca = PCA(n_components=new_dimension)
    V = pca.fit_transform(wv.vectors)
    result.add_vectors(list(wv.key_to_index.keys()), V)
    return result
