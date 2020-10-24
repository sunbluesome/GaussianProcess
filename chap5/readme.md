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
\begin{aligned}
p\left(\mathbf{f}, \mathbf{f}_{*} | \mathbf{y}\right) &= \frac{p\left(\mathbf{f}, \mathbf{f}_{*}\right) p(\mathbf{y} | \mathbf{f})}{p(\mathbf{y})} \\
p\left(\mathbf{f}_{*} | \mathbf{y}\right) &= \int p\left(\mathbf{f}, \mathbf{f}_{*} | \mathbf{y}\right) \mathrm{d} \mathbf{f} \\
&= \frac{1}{p(\mathbf{y})} \int p(\mathbf{y} | \mathbf{f}) p\left(\mathbf{f}, \mathbf{f}_{*}\right) \mathrm{d} \mathbf{f}
\end{aligned}
$$

ここで、

$$
\begin{aligned}
p\left(\mathbf{f}, \mathbf{f}_{*}\right | \mathbf{y}) &= \mathcal{N}\left(\mathbf{0},\left[\begin{array}{ll}
K_{\mathrm{f}, \mathrm{f}} & K_{*, \mathrm{f}} \\
K_{\mathrm{f}, *} & K_{*, *}
\end{array}\right]\right) \\
p(\mathbf{y} | \mathbf{f}) &= \mathcal{N}\left(\mathbf{f}, \sigma_{\mathrm{noise}}^{2} I\right)
\end{aligned}
$$

です。[条件付き多変量ガウス分布の式](https://hotoke-x.hatenablog.com/entry/2020/01/22/145811#%E6%9D%A1%E4%BB%B6%E4%BB%98%E3%81%8D%E5%A4%9A%E5%A4%89%E9%87%8F%E3%82%AC%E3%82%A6%E3%82%B9%E5%88%86%E5%B8%83)から

$$
p\left(\mathbf{f}_{*} | \mathbf{y}\right)=\mathcal{N}\left(K_{*, \mathbf{f}}\left(K_{\mathbf{f}, \mathbf{f}}+\sigma_{\text {noise }}^{2} I\right)^{-1} \mathbf{y}, K_{*, *}-K_{*, \mathbf{f}}\left(K_{\mathbf{f}, \mathbf{f}}+\sigma_{\text {noise }}^{2} I\right)^{-1} K_{\mathbf{f}, *}\right) \label{eq:predict}
$$

となることがわかります。

もちろんこれを直接計算できることに越したことはないのですが、カーネル行列の計算量は$O(N^2)$です。逆行列に至っては$O(N^3)$の計算量とメモリが必要になり、データが多いと計算量が爆発し使い物にならないという問題が発生します。そこで、同時分布$p\left(\mathbf{f}, \mathbf{f}_{*}\right)$を近似しようというのが補助変数法です。ガウス過程回帰だけでなく殆どのスパース近似で用いられる方法とのことです。

## 補助変数法（Inducing Variable Methodl; IVM）
計算量を減らす近似の手段として、補助変数（inducing variable）$\mathbf{u} = \left[ u_1, \ldots, u_m\right]^{\mathrm{T}}$を同時分布$p(\mathbf{f}_*, \mathbf{f})$に導入します。ガウス過程において補助変数を導入するということは、対応する補助入力（inducing inputs）$X_\mathbf{u}$を導入したこと同義です。事後分布では積分消去されるのがミソですが、ただ積分消去するだけでは計算量が増えるだけになってしまいます。そこで、$\mathbf{u}$の導入に際して$\mathbf{f}, \mathbf{f}_*$の条件付き独立を仮定し、以下のような近似を行います。

$$
p\left(\mathbf{f}_{*}, \mathbf{f}\right) \simeq q\left(\mathbf{f}_{*}, \mathbf{f}\right)=\int q\left(\mathbf{f}_{*} | \mathbf{u}\right) q(\mathbf{f} | \mathbf{u}) p(\mathbf{u}) \mathrm{d} \mathbf{u}
$$

この$q(\cdot|\mathbf{u})$を決めようというのが補助変数法です。

また、登場する式のリファレンスとしても便利なので、厳密な条件付き分布についても一度整理しておきましょう。厳密な条件付き分布$p(\cdot|\mathbf{u})$は

$$
\begin{aligned}
p(\mathbf{f} | \mathbf{u}) &=\mathcal{N}\left(K_{f, u} K_{\mathbf{u}, \mathbf{u}}^{-1} \mathbf{u}, \;\; K_{\mathbf{f}, \mathbf{f}}-Q_{\mathbf{f}, \mathbf{f}}\right) \label{eq:ref_train}\\
p\left(\mathbf{f}_{*} | \mathbf{u}\right) &=\mathcal{N}\left(K_{*, \mathbf{u}} K_{\mathbf{u}, \mathbf{u}}^{-1} \mathbf{u}, \;\;K_{*, *}-Q_{*, *}\right) \label{eq:ref_test}
\end{aligned}
$$

です。


#### The Subset of Data (SoD) Approximation
データの一部だけを使う方法です。データが多すぎるなら削ればよいということです。これは近似と呼ぶのかわかりませんが、計算量を減らす最も簡単な手法ですね。ただ、せっかく取れているデータを使わないのはもったいありません。以降で紹介する4つ（説明するのは3つ）の方法では、データを捨てずに事後分布を近似していきます。

#### The Subset of Regressors (SoR) Approximation
SoRは以下の事前重みを持つ線形モデルと解釈することができます。

$$
f_* = K_{*, \mathbf{u}}\mathbf{w_u}, \quad \text{with} \quad p(\mathbf{w_u}) = \mathcal{N}\left(\mathbf{0}, K_{\,\mathbf{u, u}}^{-1}\right)
$$

事前分布$p(\mathbf{w_u})$の分散を$K_{\,\mathbf{u, u}}^{-1}$としているところが重要で、ガウス過程の事前分布が自然と現れます。

$$
\mathbf{u}=K_{\mathbf{u}, \mathbf{u}} \mathbf{w}_{\mathbf{u}} \Rightarrow\left\langle\mathbf{u} \mathbf{u}^{\top}\right\rangle= K_{\mathbf{u}, \mathbf{u}}\left\langle\mathbf{w}_{\mathbf{u}} \mathbf{w}_{\mathbf{u}}^{\top}\right\rangle K_{\mathbf{u}, \mathbf{u}}=K_{\mathbf{u}, \mathbf{u}}
$$

したがって、$\mathbf{w_u}=K_{\mathbf{u,u}}^{-1}\mathbf{u}$より、

$$
\mathbf{f}_{*}=K_{*, \mathbf{u}} K_{\mathbf{u}, \mathbf{u}}^{-1} \mathbf{u}, \quad \text { with } \quad \mathbf{u} \sim \mathcal{N}\left(\mathbf{0}, K_{\mathbf{u}, \mathbf{u}}\right)
$$

となり、\eqref{eq:predict}と見比べると自然な形で式が得られたことがわかります。以上より、$q(\cdot|\mathbf{u})$は

$$
\begin{aligned}
q_{\mathrm{SoR}}(\mathbf{f} | \mathbf{u}) &= \mathcal{N}\left(K_{\mathrm{f}, \mathbf{u}} K_{\mathbf{u}, \mathbf{u}}^{-1} \mathbf{u}, \mathbf{0}\right) \\
q_{\mathrm{SoR}}\left(\mathbf{f}_{*} | \mathbf{u}\right) &= \mathcal{N}\left(K_{*, \mathbf{u}} K_{\mathbf{u}, \mathbf{u}}^{-1} \mathbf{u}, \mathbf{0}\right)
\end{aligned}
$$

で与えられ。

$$
q_{\mathrm{SoR}}\left(\mathbf{f}, \mathbf{f}_{*}\right)=\mathcal{N}\left(\mathbf{0},\left[\begin{array}{ll}
Q_{\mathrm{f}, \mathrm{f}} & Q_{\mathrm{f}, *} \\
Q_{*, f} & Q_{*, *}
\end{array}\right]\right) \\
Q_{\mathbf{a}, \mathbf{b}} \triangleq K_{\mathbf{a}, \mathbf{u}} K_{\mathbf{u}, \mathbf{u}}^{-1} K_{\mathbf{u}, \mathbf{b}} 
$$

となることがわかります。この$Q_{\mathbf{a,b}}$も一応導出しておきましょう。

$$
\begin{aligned}
cov\left(q_{\mathrm{SoR}}(\mathbf{f} | \mathbf{u}), q_{\mathrm{SoR}}(\mathbf{f}_* | \mathbf{u})\right) &= K_{\mathbf{f,u}}K_{\mathbf{u,u}}^{-1}\mathbf{u}\left(K_{\mathbf{f,u}}K_{\mathbf{u,u}}^{-1}\mathbf{u}\right)^{\top} \\
&= K_{\mathbf{f,u}}K_{\mathbf{u,u}}^{-1}\mathbf{u}\mathbf{u}^{\text{T}}K_{\mathbf{u,u}}^{-1}K_{\mathbf{u,*}} \\
&= K_{\mathbf{f,u}}K_{\mathbf{u,u}}^{-1}K_{\mathbf{u,*}}
\end{aligned}
$$

また、近似された事後分布$q_{\mathrm{SoR}}(\mathbf{f_*} | \mathbf{y})$は、$q_{\mathrm{SoR}}(\mathbf{f} | \mathbf{f_*})$の分散共分散行列の対角成分にノイズを足し、条件付き多変量ガウス分布の式を使えば

$$
\begin{aligned}
q_{\mathrm{SoR}}\left(\mathbf{f}_{*} | \mathbf{y}\right) &=\mathcal{N}\left(Q_{*, \mathbf{f}}\left(Q_{\mathbf{f}, \mathbf{f}}+\sigma_{\text {noise }}^{2} I\right)^{-1} \mathbf{y}, Q_{* *}-Q_{*, \mathbf{f}}\left(Q_{\mathbf{f}, \mathbf{f}}+\sigma_{\text {noise }}^{2} I\right)^{-1} Q_{\mathbf{f}, *}\right) \label{eq:sor_iterpret} \\
&=\mathcal{N}\left(\sigma^{-2} K_{*, u} \Sigma K_{\mathbf{u}, \mathbf{f}},\;\; K_{*, u} \Sigma K_{\mathbf{u}, *}\right) \label{eq:sor_mem} \\
\Sigma &= \left(\sigma^{-2} K_{\mathbf{u}, \mathbf{f}} K_{\mathbf{f}, \mathbf{u}}+K_{\mathbf{u}, \mathbf{u}}\right)^{-1}
\end{aligned}
$$

となります。\eqref{eq:sor_mem}の形は実装するときにメモリに優しい形になっています（導出は最後にやります）。理解するだけなら\eqref{eq:sor_iterpret}がわかれば十分です。式をよく見るとわかるのですが、SoRは結局のところカーネル関数を$k_{\mathrm{SoR}}\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=k\left(\mathbf{x}_{i}, \mathbf{u}\right) K_{\mathbf{u}, \mathbf{u}}^{-1} k\left(\mathbf{u}, \mathbf{x}_{j}\right)$と置いた<font color="red">ガウス過程に対応します（ここ重要）</font>。

さて、自然な形で$q(\cdot|\mathbf{u})$が決められたわけですが、この近似によって生じるデメリットもあります。

#### メリット
厳密にガウス過程（これから厳密なガウス過程ではなくなっていくのでメリットとした）

#### デメリット
たまたま得られたデータのばらつきが小さかった時に予測分散が過剰に小さくなるなど、補助変数を介する条件付き分布を導入したことによる自由度の制約を受けます。

以下の手法はこのデメリットを緩和しようとするもので、分散が過剰に小さくならないよう、さらに前提を追加していきます（<font color="red">理論的には破綻するので厳密にはガウス過程ではなくなる</font>）。

#### The Deterministic Training Conditional (DTC) Approximation
DTCは、以下のような近似を行います。

$$
\begin{aligned}
q_{\mathrm{DTC}}(\mathbf{f} | \mathbf{u}) &= \mathcal{N}\left(K_{\mathrm{f}, \mathbf{u}} K_{\mathbf{u}, \mathbf{u}}^{-1} \mathbf{u}, \mathbf{0}\right) \\
q_{\mathrm{DTC}}\left(\mathbf{f}_{*} | \mathbf{u}\right) &= p\left(\mathbf{f}_{*} | \mathbf{u}\right)
\end{aligned}
$$

式を見ればわかる通り、SoRとの本質的な違いは$q_{\mathrm{DTC}}\left(\mathbf{f}_{*} | \mathbf{u}\right)=p\left(\mathbf{f}_{*} | \mathbf{u}\right)$としている部分だけです。すなわち、近似された同時分布$q_{\mathrm{DTC}}\left(\mathbf{f}, \mathbf{f}_{*}\right)$は

$$
q_{\mathrm{DTC}}\left(\mathbf{f}, \mathbf{f}_{*}\right)=\mathcal{N}\left(\mathbf{0},\left[\begin{array}{ll}
Q_{\mathrm{f}, f} & Q_{\mathrm{f}, *} \\
Q_{*, \mathrm{f}} & K_{*, *}
\end{array}\right]\right)
$$

となります。$Q$の中に$K$が混ざっていることからもわかる通り、この時点で厳密にはガウス過程ではなくなっています。あとは、SoRの時と同様に事後分布を計算すれば、

$$
\begin{aligned}
q_{\mathrm{DTC}}\left(\mathbf{f}_{*} | \mathbf{y}\right) &=\mathcal{N}\left(Q_{*, \mathbf{f}}\left(Q_{\mathbf{f}, \mathbf{f}}+\sigma_{\text {noise }}^{2} I\right)^{-1} \mathbf{y}, K_{*, *}-Q_{*, \mathbf{f}}\left(Q_{\mathbf{f}, \mathbf{f}}+\sigma_{\text {noise }}^{2} I\right)^{-1} Q_{\mathbf{f}, *}\right.\\
&=\mathcal{N}\left(\sigma^{-2} K_{*, u} \Sigma K_{\mathbf{u}, \mathbf{f}} \mathbf{y},\;\; K_{*, *}-Q_{*, *}+K_{*, \mathbf{u}} \Sigma K_{*, \mathbf{u}}^{\top}\right)
\end{aligned}
$$

となります。SoRの事後分布と見比べると、分散共分散行列の式中の$Q_{*, *}$が$K_{*, *}$になっていることがわかります。$K_{*, *}-Q_{*, *}$が正定値であることから、SoRよりもDTCの法が予測分の分散が大きくなります。

#### メリット
SoRよりも予測分散が大きくなることが保証される。

#### デメリット
厳密なガウス過程ではなくなる。

#### The Fully Independent Training Conditional (FITC) Approximation
スパースガウス過程とも呼ばれ、以下のように近似を行います。

$$
\begin{aligned}
q_{\mathrm{FTTC}}(\mathbf{f} | \mathbf{u}) &= \prod_{i=1}^{n} p\left(f_{i} | \mathbf{u}\right) = \mathcal{N}\left(K_{\mathrm{f}, \mathbf{u}} K_{\mathbf{u}, \mathbf{u}}^{-1} \mathbf{u}, \operatorname{diag}\left[K_{\mathrm{f}, \mathrm{f}}-Q_{\mathrm{f}, \mathrm{f}}\right]\right)\\
q_{\mathrm{FITC}}\left(f_{*} | \mathbf{u}\right) &= p\left(f_{*} | \mathbf{u}\right)
\end{aligned}
$$

DTCと異なるのは、$q_{\mathrm{FTTC}}(\mathbf{f} | \mathbf{u})$の分散共分散行列$\operatorname{diag}\left[K_{\mathrm{f}, \mathrm{f}}-Q_{\mathrm{f}, \mathrm{f}}\right]$の部分です。FITCがDTCより優れている点は、\eqref{eq:ref_train}と見比べるとよくわかります。FITCは対角成分を捨ててはいるものの、厳密なガウス過程の分散共分散行列の情報を使って近似を行っています（それでも$q_{\mathrm{FTTC}}(\mathbf{f}_* | \mathbf{u})=q_{\mathrm{FTTC}}(\mathbf{f} | \mathbf{u})$としてるので厳密なガウス過程ではない）。

後は今まで通り計算していくと、

$$
\begin{aligned}
q_{\mathrm{FITC}}\left(\mathbf{f}, f_{*}\right) &= \mathcal{N}\left(\mathbf{0},\left[\begin{array}{cc}
Q_{\mathrm{f}, \mathbf{f}}-\operatorname{diag}\left[Q_{\mathrm{f}, \mathrm{f}}-K_{\mathrm{f}, \mathrm{f}}\right] & Q_{\mathrm{f}, *} \\
Q_{*, \mathrm{f}} & K_{*, *}
\end{array}\right]\right) \\
q_{\mathrm{FITC}}\left(f_{*} | \mathbf{y}\right) &=\mathcal{N}\left(Q_{*, \mathbf{f}}\left(Q_{\mathrm{f}, \mathbf{f}}+\Lambda\right)^{-1} \mathbf{y}, K_{*, *}-Q_{*, \mathbf{f}}\left(Q_{\mathrm{f}, \mathbf{f}}+\Lambda\right)^{-1} Q_{\mathrm{f}, *}\right) \\
&=\mathcal{N}\left(K_{*, \mathbf{u}} \Sigma K_{\mathbf{u}, \mathbf{f}} \Lambda^{-1} \mathbf{y}, K_{*, *}-Q_{*, *}+K_{*, \mathbf{u}} \Sigma K_{\mathbf{u}, *}\right)
\end{aligned}
$$

となります。

#### The Partially Independent Training Conditional (PITC) Approximation
書籍「ガウス過程と機械学習」の取り扱い範囲を超えているので今回は取り扱いません。ただ、ここまで読み進めてくれた方ならわかると思いますが、さっきは相関を捨てていましたが、ある程度相関を拾ってあげることで近似精度を上げようという方法です。



## 最後に
メモリに優しい形の式を導出します。

＜近日追記予定＞