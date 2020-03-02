ガウス過程と機械学習の第5章、補助変数法でよくわからなくなってしまったので[原著](http://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf)を読んだのでまとめます。スパース近似ガウス過程回帰を統一的視点で眺めるという内容になっています。

[@matsueushiさんが既に綺麗にまとめてくれているの]((https://matsueushi.github.io/posts/ivm/))ですが、数弱の自分は式をパッと見ただけでは理解できなかったので、もうちょっと細かいところもまとめていこうと思います（ほぼ翻訳）。


## ガウス過程回帰
まずガウス過程回帰についておさらいです。ガウス分布の定義は

> 任意次元の入力に対する出力の同時分布が多変量ガウス分布に従うもののこと。  

でした（厳密な言い方では無いと思いますが）。ガウス過程をベイズ的に眺めると、これから紹介する補助変数法の理解がすっきりします。

導入の詳細は[前回の記事](https://hotoke-x.hatenablog.com/entry/2020/02/27/012613)を参照してください。

入力$\mathbf{x_i} = \{x_1, \ldots, x_n\}$と、対応する出力$y_i$$が次のような関係だとします。

$$
y_{i}=f\left(\mathbf{x}_{i}\right)+\varepsilon_{i}, \quad \text { where } \quad \varepsilon_{i} \sim \mathcal{N}\left(0, \sigma_{\text {noise }}^{2}\right)
$$

$\varepsilon_i$はノイズを表し$\sigma^2_{noise}$はノイズの分散です。以降ではこの関係が前提であるとします。

では、ガウス過程回帰をベイズ的に眺めていきます。ガウス過程回帰とは結局のところ、関数$\mathbf{f} = \left[f_1, \ldots, f_n \right], (f_i = f\mathbf(x_i))$について次のような事前分布を導入することと同義です。

$$
p\left(\mathbf{f} | \mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{n}\right)=\mathcal{N}(\mathbf{0}, K)
$$

ここで、行列$K$は$ij$成分が$k(\mathbf{x}_i, \mathbf{x}_j)$となるカーネル行列です。つまり、関数自体が確率変数であると考えるのがガウス過程です。そして、データからその事後分布を推定するのがガウス過程回帰ということになります。  
　では、関数の事後分布$p(\mathbf{f_*} | \mathbf{y})$を導出していきます。ベイズの定理より、

$$
p\left(\mathbf{f}, \mathbf{f}_{*} | \mathbf{y}\right) = \frac{p\left(\mathbf{f}, \mathbf{f}_{*}\right) p(\mathbf{y} | \mathbf{f})}{p(\mathbf{y})} \\
p\left(\mathbf{f}_{*} | \mathbf{y}\right)=\int p\left(\mathbf{f}, \mathbf{f}_{*} | \mathbf{y}\right) \mathrm{d} \mathbf{f}=\frac{1}{p(\mathbf{y})} \int p(\mathbf{y} | \mathbf{f}) p\left(\mathbf{f}, \mathbf{f}_{*}\right) \mathrm{d} \mathbf{f}
$$

ここで、

$$
p\left(\mathbf{f}, \mathbf{f}_{*}\right | \mathbf{y})=\mathcal{N}\left(\mathbf{0},\left[\begin{array}{ll}
K_{\mathrm{f}, \mathrm{f}} & K_{*, \mathrm{f}} \\
K_{\mathrm{f}, *} & K_{*, *}
\end{array}\right]\right), \quad \text { and } \quad p(\mathbf{y} | \mathbf{f})=\mathcal{N}\left(\mathbf{f}, \sigma_{\mathrm{noise}}^{2} I\right)
$$

です。[条件付き多変量ガウス分布の式](https://hotoke-x.hatenablog.com/entry/2020/01/22/145811#%E6%9D%A1%E4%BB%B6%E4%BB%98%E3%81%8D%E5%A4%9A%E5%A4%89%E9%87%8F%E3%82%AC%E3%82%A6%E3%82%B9%E5%88%86%E5%B8%83)から

$$
p\left(\mathbf{f}_{*} | \mathbf{y}\right)=\mathcal{N}\left(K_{*, \mathbf{f}}\left(K_{\mathbf{f}, \mathbf{f}}+\sigma_{\text {noise }}^{2} I\right)^{-1} \mathbf{y}, K_{*, *}-K_{*, \mathbf{f}}\left(K_{\mathbf{f}, \mathbf{f}}+\sigma_{\text {noise }}^{2} I\right)^{-1} K_{\mathbf{f}, *}\right)
$$

となることがわかります。

もちろんこれを直接計算できることに越したことはないのですが、カーネル行列の計算量は$O(N^2)$です。逆行列に至っては$O(N^3)$の計算量とメモリが必要になり、データが多いと計算量が爆発し使い物にならないという問題が発生します。そこで、同時分布$p\left(\mathbf{f}, \mathbf{f}_{*}\right)$を近似しようというのが補助変数法です。ガウス過程回帰だけでなく殆どのスパース近似で用いられる方法とのことです。

## 補助変数法（Inducing Variable Methodl; IVM）
計算量を減らす近似の手段として、補助変数（inducing variable）$\mathbf{u} = \left[ u_1, \ldots, u_m\right]^{\mathrm{T}}$を同時分布$p(\mathbf{f}_*, \mathbf{f})$に導入します。ガウス過程において補助変数を導入するということは、対応する補助入力（inducing inputs）$X_\mathbf{u}$を導入したこと同義です。事後分布では積分消去されるのがミソですが、ただ積分消去するだけでは計算量が増えるだけになってしまいます。そこで、$\mathbf{u}$の導入に際して$\mathbf{f}, \mathbf{f}_*$の条件付き独立を仮定し、以下のような近似を行います。

$$
p\left(\mathbf{f}_{*}, \mathbf{f}\right) \simeq q\left(\mathbf{f}_{*}, \mathbf{f}\right)=\int q\left(\mathbf{f}_{*} | \mathbf{u}\right) q(\mathbf{f} | \mathbf{u}) p(\mathbf{u}) \mathrm{d} \mathbf{u}
$$

この$q(\cdot)$を決めようというのが補助変数法です。


#### The Subset of Data (SoD) Approximation
データの一部だけを使う方法です。データが多すぎるなら削ればよいということです。これは近似と呼ぶのかわかりませんが、計算量を減らす最も簡単な手法ですね。ただ、せっかく取れているデータを使わないのはもったいありません。以降で紹介する4つ（説明するのは3つ）の方法では、データを捨てずに事後分布を近似していきます。

#### The Subset of Regressors (SoR) Approximation
#### The Deterministic Training Conditional (DTC) Approximation
#### The Fully Independent Training Conditional (FITC) Approximation
#### The Partially Independent Training Conditional (PITC) Approximation
書籍「ガウス過程と機械学習」の取り扱い範囲を超えているので今回は取り扱いません。ただ、ここまで読み進めてくれた方ならわかると思いますが、さっきは相関を捨てていましたが、ある程度相関を拾ってあげることで近似精度を上げようという方法です。