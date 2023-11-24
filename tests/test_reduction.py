import pytest
from gensim.models.keyedvectors import KeyedVectors

from reduction.reduction import ppa, reduce


@pytest.fixture(scope='session')
def wv() -> KeyedVectors:
    chive = 'chive-1.1-mc5-aunit_gensim/chive-1.1-mc5-aunit.kv'
    return KeyedVectors.load(chive)


def _jaccard(
    result: KeyedVectors,
    expected: KeyedVectors,
    word: str,
    top_n: int
) -> float:
    sim1 = set([k for (k, _) in result.most_similar(word, topn=top_n)])
    sim2 = set([k for (k, _) in expected.most_similar(word, topn=top_n)])
    return len(sim1.intersection(sim2)) / len(sim1.union(sim2))


def test_ppa(wv: KeyedVectors) -> None:
    result = ppa(wv, top_n=7)
    assert result.vector_size == 300

    for word in ['自動車', '喜ぶ']:
        assert _jaccard(result, wv, word, top_n=10) > 0.6


@pytest.mark.parametrize('new_dimension', [100, 150])
def test_reduce(wv: KeyedVectors, new_dimension: int) -> None:
    result = reduce(wv, new_dimension=new_dimension, top_n=7)
    assert result.vector_size == new_dimension

    for word in ['自動車', '喜ぶ']:
        assert _jaccard(result, wv, word, top_n=10) > 0.6
