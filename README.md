[Effective Dimensionality Reduction for Word Embeddings](https://aclanthology.org/W19-4328) (Raunak et al., RepL4NLP 2019) の実装．

※ ライブラリとして整備する予定はないので，適宜コードを流用して使用してください．

## 環境設定
1. [WorksApplications/chiVe](https://github.com/WorksApplications/chiVe) の `v1.1 mc5 aunit` (gensim) をダウンロードする．
1. 必要なパッケージをインストールする．
    ```bash
    poetry env use YOUR_PYTHON_PATH  # 3.11
    poetry install
    ```
1. 下記のコマンドによってテストを行う．
    ```bash
    poetry run pytest -v
    # OR
    poetry shell
    pytest -v
    ```

## PPA
[All-but-the-Top: Simple and Effective Postprocessing for Word Representations](https://openreview.net/forum?id=HkuGJ3kCb) (Mu et al., ICLR 2018) に基づく．以下の説明は `numpy` に合わせて横ベクトルを中心に記述してある．

### パラメーター
- $`V = \begin{pmatrix} v^{(1)}{}^\mathsf{T} \\ \vdots \\ v^{(n)}{}^\mathsf{T} \end{pmatrix}`$：単語ベクトル $`v^{(1)},\, \ldots,\, v^{(n)} \in \mathbb{R}^p`$ をそれぞれ転置して積み重ねた行列．
- $`K`$：PCA において固有ベクトルを分散を大きくする順に上位何位まで採用するか．

### アルゴリズム
1. $`\displaystyle \overline{v} \leftarrow \frac{1}{n}\sum_{i = 1}^n v^{(i)}`$.  
1. FOR $`i \in \{1,\, \ldots,\, n\}`$  
    DO  
    &nbsp;&nbsp;&nbsp;&nbsp; $`\widetilde{v}^{(i)} \leftarrow v^{(i)} - \overline{v}`$  
    END DO
1. $`
    \widetilde{V} \leftarrow
    \begin{pmatrix}
        \widetilde{v}^{(1)}{}^\mathsf{T} \\
        \vdots \\
        \widetilde{v}^{(n)}{}^\mathsf{T}
    \end{pmatrix}`$.
1. $`W \leftarrow \mathrm{PCA}(\widetilde{V},\, K)`$，ただし $`W =
   \begin{pmatrix}
       w^{(1)}{}^\mathsf{T} \\
       \vdots \\
       w^{(K)}{}^\mathsf{T}
   \end{pmatrix}`$, $`w^{(i)}{}^\mathsf{T} \in \mathbb{R}^{1 \times p}`$ for $`i \in \{1,\, \ldots,\, K\}`$ である．
1. FOR $`i \in \{1, \ldots, n\}`$  
    DO  
    &nbsp;&nbsp;&nbsp;&nbsp; FOR $`K \in \{1,\, \ldots,\, K\}`$  
    &nbsp;&nbsp;&nbsp;&nbsp; DO  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $a_{iK} \leftarrow w^{(K)}{}^\mathsf{T} \widetilde{v}^{(i)}$  
    &nbsp;&nbsp;&nbsp;&nbsp; END DO  
    &nbsp;&nbsp;&nbsp;&nbsp; $\hat{v}^{(i)}{}^\mathsf{T} \leftarrow \widetilde{v}^{(i)}{}^\mathsf{T} - (a_{i1} w^{(1)}{}^\mathsf{T} + \cdots + a_{iK} w^{(K)}{}^\mathsf{T})$  
    END DO
1. $`
    \hat{V} \leftarrow
    \begin{pmatrix}
        \hat{v}^{(1)}{}^\mathsf{T} \\
        \vdots \\
        \hat{v}^{(n)}{}^\mathsf{T}
    \end{pmatrix}
    `$.
1. Return $`\hat{V}`$.

### アルゴリズムに関する補足

ステップ 5 は $` \widetilde{V} = \begin{pmatrix} \widetilde{v}^{(1)}{}^\mathsf{T} \\ \vdots \\ \widetilde{v}^{(n)}{}^\mathsf{T} \end{pmatrix}`$ とすると，行列を用いて
```math
\begin{align}
    \hat{V}
    &= \widetilde{V}
    -
    \begin{pmatrix}
        a_{11} & \cdots & a_{1K} \\
        \vdots & \ddots & \vdots \\
        a_{n1} & \cdots & a_{nK}
    \end{pmatrix}
    W
    \\
    &= \widetilde{V}
    -
    \begin{pmatrix}
        w^{(1)}{}^\mathsf{T} \widetilde{v}^{(1)} & \cdots & w^{(K)}{}^\mathsf{T} \widetilde{v}^{(1)} \\
        \vdots & \ddots & \vdots \\
        w^{(1)}{}^\mathsf{T} \widetilde{v}^{(n)} & \cdots & w^{(K)}{}^\mathsf{T} \widetilde{v}^{(n)}
    \end{pmatrix}
    W \\
    &= \widetilde{V}
    -  \begin{pmatrix}
        v^{(1)}{}^\mathsf{T} \\
        \vdots \\
        v^{(n)}{}^\mathsf{T}
    \end{pmatrix}
    \begin{pmatrix}
        w^{(1)} & \cdots & w^{(K)}
    \end{pmatrix}
    W \\
    &= \widetilde{V} - \widetilde{V} W^\mathsf{T} W
\end{align}
```
のように計算できる．Python では
```python
V_hat = V_tilde - V_tilde @ W.T @ W
```
のように書く．[Mu et al. (2018)](https://openreview.net/forum?id=HkuGJ3kCb) の記述にしたがうと
```python
V_hat = V_tilde - V @ W.T @ W
```
とするのが正しいが，次元削減の実装 [vyraun/Half-Size](https://github.com/vyraun/Half-Size) を見ると前者で計算しているようなので，それに合わせている．

## 次元削減
### パラメーター
- $`V = \begin{pmatrix} v^{(1)}{}^\mathsf{T} \\ \vdots \\ v^{(n)}{}^\mathsf{T} \end{pmatrix}`$：単語ベクトル $`v^{(1)},\, \ldots,\, v^{(n)} \in \mathbb{R}^p`$ をそれぞれ転置して積み重ねた行列．
- $`K`$：PCA において固有ベクトルを分散を大きくする順に上位何位まで採用するか．
- $`d`$：新しい次元．

### アルゴリズム
1. $`X \leftarrow \mathrm{PPA}(V,\, K)`$.  
1. $`Y \leftarrow \mathrm{PCA}(X,\, d)`$.  
ただし $Y$ は $`w^{(1)},\, \ldots,\, w^{(d)} \in \mathbb{R}^{1 \times p}`$ をPCAによって得られた固有ベクトルとしたとき，
次のように表される行列である：$`
   Y =
   \begin{pmatrix}
      w^{(1)}{}^\top x^{(1)} & \cdots  &w^{(d)}{}^\top x^{(1)} \\
      \vdots & \ddots & \vdots \\
      w^{(n)}{}^\top x^{(n)} & \cdots  &w^{(d)}{}^\top x^{(n)}
   \end{pmatrix}.
`$
1. $`V \leftarrow \mathrm{PPA}(Y,\, K)`$.
2. Return $`V`$.
