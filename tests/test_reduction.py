import random

import numpy
import pytest
from gensim.models.keyedvectors import KeyedVectors

from reduction.reduction import apply_ppa, reduce, reduce_by_pca

THRESHOLD = 0.3
CHIVE_VECTOR_SIZE = 300


@pytest.fixture(scope='session')
def fix_seeds() -> None:
    random.seed(0)
    numpy.random.seed(0)


@pytest.fixture(scope='session')
def wv() -> KeyedVectors:
    chive = 'chive-1.1-mc5-aunit_gensim/chive-1.1-mc5-aunit.kv'
    return KeyedVectors.load(chive)


@pytest.fixture(scope='session')
def words(wv: KeyedVectors) -> list[str]:
    return random.choices(list(wv.key_to_index.keys()), k=100)


def _jaccard(
    result: KeyedVectors,
    expected: KeyedVectors,
    word: str,
    similarity_top_n: int
) -> float:
    sim1 = set([k for (k, _) in result.most_similar(word, topn=similarity_top_n)])
    sim2 = set([k for (k, _) in expected.most_similar(word, topn=similarity_top_n)])
    return len(sim1.intersection(sim2)) / len(sim1.union(sim2))


@pytest.mark.parametrize('top_n, similarity_top_n', [(3, 10)])
def test_apply_ppa(
    wv: KeyedVectors,
    words: list[str],
    top_n: int,
    similarity_top_n: int
) -> None:
    result = apply_ppa(wv, top_n=top_n)
    assert result.vector_size == CHIVE_VECTOR_SIZE

    naive_score = sum(_jaccard(result, wv, word, similarity_top_n=similarity_top_n) for word in words) / 10
    assert naive_score > THRESHOLD


@pytest.mark.parametrize('new_dimension, top_n, similarity_top_n', [
    (150, 3, 10),
    (150, 7, 10)
])
def test_reduce(
    wv: KeyedVectors,
    words: list[str],
    new_dimension: int,
    top_n: int,
    similarity_top_n: int
) -> None:
    result = reduce(wv, new_dimension=new_dimension, top_n=top_n)
    assert result.vector_size == new_dimension

    naive_score = sum(_jaccard(result, wv, word, similarity_top_n=similarity_top_n) for word in words) / 10
    assert naive_score > THRESHOLD


@pytest.mark.parametrize('new_dimension, similarity_top_n', [
    (150, 10)
])
def test_reduce_by_pca(
    wv: KeyedVectors,
    words: list[str],
    new_dimension: int,
    similarity_top_n: int
) -> None:
    result = reduce_by_pca(wv, new_dimension)
    assert result.vector_size == new_dimension

    naive_score = sum(_jaccard(result, wv, word, similarity_top_n=similarity_top_n) for word in words) / 10
    assert naive_score > THRESHOLD
