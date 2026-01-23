\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{mathtools}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{algorithm}
\usepackage{algpseudocode}

\title{Example-Grounded Prototype Memory Regression on Top of a Frozen Decoder-Only LLM}
\author{}
\date{}

\begin{document}
\maketitle

\section{Problem Setup and Goals}

We are given a supervised regression dataset
\[
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N},
\]
where:
\begin{itemize}[nosep]
    \item $x_i$ is an input text sequence (tokenized for the LLM),
    \item $y_i \in [a,b] \subset \mathbb{R}$ is a continuous scalar label with known bounds $a<b$,
    \item $N \approx 10{,}000$.
\end{itemize}

We assume a \textbf{frozen decoder-only language model} (e.g., LLaMA-like) that maps input tokens to hidden states. Our objective is to build a \textbf{trainable, example-grounded, interpretable prototype layer} that:
\begin{enumerate}[nosep]
    \item avoids caching the full tensor $N \times S \times d_{\text{model}}$ (where $S$ is sequence length),
    \item learns a \textbf{compact memory cache} per training example,
    \item learns to select $K$ \textbf{prototype examples} (no duplicates) from the dataset,
    \item predicts a scalar $\hat{y}$ using an \textbf{inference head} that attends to query and prototype memory,
    \item provides interpretability via attention weights over selected prototypes.
\end{enumerate}

\subsection{Fixed Default Hyperparameters (No Options)}
Throughout, we fix the following default values:
\begin{center}
\begin{tabular}{@{}ll@{}}
\toprule
Parameter & Default \\
\midrule
Max token length & $S = 256$ \\
Backbone hidden size & $d_{\text{model}}$ (given by frozen LLM) \\
Head dimension & $d_h = 256$ \\
Memory tokens per example & $m = 8$ \\
Query compressed tokens & $m_q = 8$ \\
\# prototypes & $K = 128$ \\
Slot candidate pool & $T = 512$ \\
Inference layers & $L = 3$ \\
Attention heads & $H = 8$ \\
FFN expansion & $4 d_h$ \\
Output bounding & $\hat{y}=a + (b-a)\sigma(\cdot)$ \\
Regression loss & Huber, $\delta=0.5$ \\
Stage A epochs & 5 \\
Stage B epochs & 10 \\
Gumbel temperature & $\tau$: linear anneal from 1.0 to 0.1 \\
Label-embedding aux weight (Stage A) & $\lambda_{\text{emb}} = 0.1$ \\
Soft overlap penalty (Stage B) & $\lambda_{\text{ov}} = 1.0$ \\
Repulsion penalty (Stage B) & $\lambda_{\text{rep}} = 0.1$ \\
Cosine margin & $m_{\text{cos}} = 0.2$ \\
\bottomrule
\end{tabular}
\end{center}

\section{Frozen LLM Representation}

Let $\text{LLM}_{\text{frozen}}$ denote the frozen decoder-only transformer. For an input token sequence $x$ of length $S$:
\[
\mathbf{H}(x) = \text{LLM}_{\text{frozen}}(x) \in \mathbb{R}^{S \times d_{\text{model}}}.
\]
We use the \textbf{final layer hidden states} (not logits). In batched form for batch size $B$:
\[
\mathbf{H}_q \in \mathbb{R}^{B \times S \times d_{\text{model}}}.
\]

\paragraph{Important:} The backbone is frozen; gradients do not flow into $\text{LLM}_{\text{frozen}}$.

\section{Attention Primer (for Readers New to LLM/Attention)}

Attention is a mechanism for \textbf{weighted information aggregation}. Given:
\begin{itemize}[nosep]
    \item queries $\mathbf{Q} \in \mathbb{R}^{n_q \times d}$,
    \item keys $\mathbf{K} \in \mathbb{R}^{n_k \times d}$,
    \item values $\mathbf{V} \in \mathbb{R}^{n_k \times d}$,
\end{itemize}
scaled dot-product attention produces:
\[
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
=
\text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}}\right)\mathbf{V}
\in \mathbb{R}^{n_q \times d}.
\]
Intuition:
\begin{enumerate}[nosep]
    \item Compute similarity scores between each query and each key via dot products.
    \item Softmax converts similarities into weights that sum to 1 across keys.
    \item Weighted sum of values returns information relevant to each query.
\end{enumerate}

\subsection{Multi-Head Attention}
Multi-head attention repeats this in $H$ heads with smaller per-head dimension $d/H$ and concatenates head outputs, allowing different heads to focus on different patterns.

\subsection{Cross-Attention vs Self-Attention}
\begin{itemize}[nosep]
    \item \textbf{Self-attention:} $\mathbf{Q},\mathbf{K},\mathbf{V}$ all come from the same sequence.
    \item \textbf{Cross-attention:} $\mathbf{Q}$ comes from one sequence (e.g., a regression token), while $\mathbf{K},\mathbf{V}$ come from another sequence (e.g., memory tokens).
\end{itemize}

\section{Model Overview: Two-Stage Training with Compact Caching}

We define a two-stage pipeline.

\subsection{Stage A: Learn a Compact Memory Writer (Compressor)}
We learn a module that converts the large token-level representation $\mathbf{H}(x)$ into a small set of \textbf{memory tokens} $\mathbf{M}(x)\in \mathbb{R}^{m\times d_h}$.
After Stage A training, we \textbf{cache} $\mathbf{M}(x_i)$ for each training example $i$, avoiding storage of token-level tensors.

\subsection{Stage B: Learn Prototype Selection and Inference Head Using the Cache}
Using the cached memory tokens for all training examples, we:
\begin{enumerate}[nosep]
    \item learn to select $K$ distinct prototype examples via \textbf{masked sequential per-slot selection} (no duplicates in the forward pass),
    \item predict $\hat{y}$ for a query by attending to the selected prototypes using a \textbf{regression token} in an inference head.
\end{enumerate}

\section{Stage A Components (Memory Cache Learning)}

\subsection{Projection to Head Dimension}
We learn linear projections from $d_{\text{model}}$ to $d_h$:
\[
\mathbf{T}(x) = \mathbf{H}(x)\mathbf{W}_m \in \mathbb{R}^{S \times d_h},
\]
where $\mathbf{W}_m \in \mathbb{R}^{d_{\text{model}} \times d_h}$ is trainable.
In batch form:
\[
\mathbf{T} \in \mathbb{R}^{B \times S \times d_h}.
\]

\subsection{Perceiver-Style Memory Compressor (Writer)}
We compress token states into $m$ memory tokens using learnable latent queries.
Let $\mathbf{Q}^{\text{mem}} \in \mathbb{R}^{m \times d_h}$ be trainable latent vectors. Define:
\[
\mathbf{M}(x) = \text{Compressor}(\mathbf{T}(x)) \in \mathbb{R}^{m \times d_h},
\]
implemented as cross-attention from latent queries to token states:
\[
\mathbf{M}(x) = \text{Attn}\!\big(\mathbf{Q}^{\text{mem}},\, \mathbf{T}(x),\, \mathbf{T}(x)\big),
\]
followed by residual connections, layer norms, and a feed-forward network (FFN).

\subsection{Query Compressor (for Efficiency and Symmetry)}
We also compress the query token states into $m_q$ query tokens to reduce computation:
\[
\mathbf{Q}(x) = \text{QueryCompressor}(\mathbf{H}(x)\mathbf{W}_q) \in \mathbb{R}^{m_q \times d_h},
\]
with $\mathbf{W}_q \in \mathbb{R}^{d_{\text{model}} \times d_h}$ trainable and QueryCompressor implemented identically to the memory compressor but with $m_q$ latents.

\subsection{Scalar Label Embedder (Target-Only in Stage A)}
Attention operates in $\mathbb{R}^{d_h}$. We embed a scalar label $y$ into a $d_h$-dimensional vector token:
\[
\mathbf{e}(y) = E(y) \in \mathbb{R}^{d_h}.
\]
We use a small MLP on normalized labels:
\[
\tilde{y} = \frac{y-\mu_y}{\sigma_y + \epsilon},\qquad
\mathbf{e}(y) = \text{MLP}([\tilde{y}]) \in \mathbb{R}^{d_h},
\]
where $(\mu_y,\sigma_y)$ are computed once from the training set.

\paragraph{Key point (anti-cheating).}
In Stage A, $\mathbf{e}(y)$ is \textbf{not appended as an input token} to the memory. It is used only as a \textbf{target} for an auxiliary embedding-prediction loss.

\subsection{Learned Selection Key Readout (Replaces Mean Pooling)}
Mean pooling of memory tokens can erase discriminative information. Instead we learn a key readout to produce a selection key from memory tokens.

Let $\mathbf{q}^{\text{key}}\in\mathbb{R}^{1\times d_h}$ be a trainable query (single latent). Define:
\[
\mathbf{k}(x) = \text{KeyReadout}(\mathbf{M}(x)) \in \mathbb{R}^{d_h},
\qquad
\mathbf{k}(x) = \text{Attn}\big(\mathbf{q}^{\text{key}},\ \mathbf{M}(x),\ \mathbf{M}(x)\big)_{[0,:]}.
\]
We also normalize:
\[
\bar{\mathbf{k}}(x) = \frac{\mathbf{k}(x)}{\|\mathbf{k}(x)\|_2 + \epsilon}.
\]

\subsection{Stage A Inference Head (Training-Only Supervisor)}
To train the compressor to retain label-relevant information, we use a small inference head with a \textbf{single regression token}. This head structure matches Stage B's inference head.

Define:
\begin{itemize}[nosep]
    \item Query tokens: $\mathbf{Q}_q = \mathbf{Q}(x) \in \mathbb{R}^{m_q \times d_h}$
    \item Memory tokens: $\mathbf{M}(x)\in\mathbb{R}^{m\times d_h}$
\end{itemize}
Construct Stage A memory sequence (no label token input):
\[
\mathbf{Mem}^{(A)} = \mathbf{M}(x) \in \mathbb{R}^{m\times d_h}.
\]

\paragraph{Regression token.}
Let $\mathbf{r}_0 \in \mathbb{R}^{d_h}$ be a learned parameter (the regression token initial state). In batch form it is replicated across examples:
\[
\mathbf{R}_0 \in \mathbb{R}^{B \times 1 \times d_h}.
\]

\paragraph{Inference layers.}
For $\ell=1,\dots,L$, update the regression token by two cross-attentions:
\[
\mathbf{R}^{(\ell)} \leftarrow \mathbf{R}^{(\ell-1)} + \text{Attn}\big(\mathbf{R}^{(\ell-1)},\ \mathbf{Q}_q,\ \mathbf{Q}_q\big),
\]
\[
\mathbf{R}^{(\ell)} \leftarrow \mathbf{R}^{(\ell)} + \text{Attn}\big(\mathbf{R}^{(\ell)},\ \mathbf{Mem}^{(A)},\ \mathbf{Mem}^{(A)}\big),
\]
followed by FFN + layer norms (standard transformer block style).

\paragraph{Prediction.}
We map the final regression token to a scalar:
\[
\hat{z} = f_{\text{out}}(\mathbf{R}^{(L)}_{[:,0,:]}) \in \mathbb{R},
\qquad
\hat{y} = a + (b-a)\sigma(\hat{z}) \in [a,b].
\]

\subsection{Auxiliary Label-Embedding Prediction Head (Stage A)}
To explicitly force memory to carry label-relevant information \emph{without providing the label as input},
we predict the label embedding from the learned key readout:
\[
\hat{\mathbf{e}}(x) = g_{\text{emb}}(\mathbf{k}(x)) \in \mathbb{R}^{d_h},
\]
where $g_{\text{emb}}$ is a small MLP.

\section{Stage A Loss and What Is Trained}

\subsection{Huber Regression Loss}
For each example:
\[
\mathcal{L}_{\text{reg}}(\hat{y},y) =
\begin{cases}
\frac{1}{2}(\hat{y}-y)^2 & \text{if } |\hat{y}-y|\le \delta,\\
\delta\left(|\hat{y}-y|-\frac{1}{2}\delta\right) & \text{otherwise},
\end{cases}
\]
with $\delta=0.5$.

\subsection{Auxiliary Embedding Alignment Loss (Stage A)}
We align predicted and target label embeddings:
\[
\mathcal{L}_{\text{emb}}(x,y) = \frac{1}{d_h}\left\|\hat{\mathbf{e}}(x) - \mathbf{e}(y)\right\|_2^2.
\]

\subsection{Stage A Objective}
\[
\mathcal{L}^{(A)} = \frac{1}{B}\sum_{b=1}^{B}\left[
\mathcal{L}_{\text{reg}}(\hat{y}_b,y_b) + \lambda_{\text{emb}}\,\mathcal{L}_{\text{emb}}(x_b,y_b)
\right],
\]
with $\lambda_{\text{emb}}=0.1$.

\subsection{Parameters Updated in Stage A}
Trainable in Stage A:
\begin{itemize}[nosep]
    \item $\mathbf{W}_m, \mathbf{W}_q$ projections,
    \item memory compressor parameters (latent queries + attention projections + FFN),
    \item query compressor parameters,
    \item key readout parameters ($\mathbf{q}^{\text{key}}$ and attention projections),
    \item label embedder $E(\cdot)$ parameters,
    \item auxiliary head $g_{\text{emb}}$ parameters,
    \item inference head parameters including regression token $\mathbf{r}_0$ and output MLP.
\end{itemize}
Frozen in Stage A:
\begin{itemize}[nosep]
    \item all LLM backbone parameters in $\text{LLM}_{\text{frozen}}$.
\end{itemize}

\section{Caching Procedure (After Stage A)}

We now build a cache over the full training set and store only compact representations.

For each training example $(x_i,y_i)$:
\begin{enumerate}[nosep]
    \item Compute token states $\mathbf{H}(x_i)\in\mathbb{R}^{S\times d_{\text{model}}}$ with the frozen LLM.
    \item Project and compress to memory tokens:
    \[
    \mathbf{M}_i = \mathbf{M}(x_i)\in\mathbb{R}^{m\times d_h}.
    \]
    \item Compute a \textbf{learned selection key} via KeyReadout:
    \[
    \mathbf{k}_i = \text{KeyReadout}(\mathbf{M}_i)\in\mathbb{R}^{d_h}.
    \]
    \item Normalize the key:
    \[
    \bar{\mathbf{k}}_i = \frac{\mathbf{k}_i}{\|\mathbf{k}_i\|_2 + \epsilon}.
    \]
\end{enumerate}

We store the following cache:
\[
\text{MemCache} = \{\mathbf{M}_i\}_{i=1}^{N},\quad
\text{KeyCache} = \{\bar{\mathbf{k}}_i\}_{i=1}^{N},\quad
\text{LabelCache} = \{y_i\}_{i=1}^{N}.
\]

\paragraph{Why this removes the memory bottleneck.}
Instead of storing $N\times S\times d_{\text{model}}$, we store $N\times m\times d_h$ where $m \ll S$ and $d_h \ll d_{\text{model}}$ typically. With defaults ($N=10k, m=8, d_h=256$), fp16 storage is on the order of tens of MB.

\section{Stage B: Prototype Selection and Inference Using the Cache}

Stage B trains:
\begin{enumerate}[nosep]
    \item a \textbf{$K$-slot prototype selector} that produces \textbf{unique} prototypes in the forward pass,
    \item an inference head that uses a regression token to attend to the selected prototype memories and the query.
\end{enumerate}
The cache itself is treated as fixed data.

\subsection{$K$-Slot Prototype Selector with Masked Sequential Uniqueness}

We introduce $K$ trainable slot queries:
\[
\mathbf{s}_k \in \mathbb{R}^{d_h},\quad k=1,\dots,K,
\]
and normalize them:
\[
\bar{\mathbf{s}}_k = \frac{\mathbf{s}_k}{\|\mathbf{s}_k\|_2 + \epsilon}.
\]

Each slot scores all cached keys via cosine similarity:
\[
\text{logit}_{k,i} = \bar{\mathbf{s}}_k^\top \bar{\mathbf{k}}_i,\quad i=1,\dots,N.
\]
This yields a logit matrix $\mathbf{L}\in\mathbb{R}^{K\times N}$.

\subsubsection{Candidate Restriction to Top-$T$}
Restrict each slot to its top-$T$ candidates:
\[
\mathcal{I}_k = \text{TopTIndices}(\{\text{logit}_{k,i}\}_{i=1}^N),\quad |\mathcal{I}_k|=T,
\]
and let $\mathbf{v}_k\in\mathbb{R}^{T}$ be the logits for those candidates.

\subsubsection{Masked Sequential Gumbel-Softmax (No Duplicates in Forward Pass)}
We maintain a binary \textbf{availability mask} over dataset indices:
\[
a_i^{(1)} = 1\ \ \forall i,\qquad a_i^{(k)}\in\{0,1\}\ \text{indicates if index $i$ is still available at slot $k$}.
\]

For slot $k$, we mask logits for unavailable candidates:
\[
v'_{k,t} =
\begin{cases}
v_{k,t} & \text{if } a^{(k)}_{\mathcal{I}_k[t]}=1,\\
-\infty & \text{if } a^{(k)}_{\mathcal{I}_k[t]}=0.
\end{cases}
\]

Then apply Gumbel-Softmax:
\[
g_{k,t} = -\log(-\log(u_{k,t}+\epsilon)+\epsilon),\quad u_{k,t}\sim\text{Uniform}(0,1),
\]
\[
\mathbf{p}_k = \text{softmax}\left(\frac{\mathbf{v}'_k + \mathbf{g}_k}{\tau}\right)\in\mathbb{R}^{T}.
\]
Let $t_k^*=\arg\max_t p_{k,t}$ and corresponding dataset index $i_k^*=\mathcal{I}_k[t_k^*]$.
We form a straight-through weight vector
\[
\mathbf{w}_k = \text{onehot}(t_k^*) - \mathbf{p}_k^{\text{stopgrad}} + \mathbf{p}_k.
\]

\paragraph{Mask update (enforces uniqueness).}
We update availability for the next slot by removing the hard-picked index:
\[
a^{(k+1)}_i =
\begin{cases}
0 & \text{if } i = i_k^*,\\
a^{(k)}_i & \text{otherwise}.
\end{cases}
\]
Thus, the \emph{forward} hard selections $\{i_k^*\}_{k=1}^K$ contain \textbf{no duplicates} by construction.

\subsubsection{Sparse Per-Slot Distributions over the Full Dataset}
For soft regularizers, define a sparse distribution over full indices:
\[
q_{k,i} =
\begin{cases}
p_{k,t} & \text{if } i=\mathcal{I}_k[t]\ \text{for some }t,\\
0 & \text{otherwise}.
\end{cases}
\]
This yields $\mathbf{q}_k\in\mathbb{R}^{N}$ with $\sum_i q_{k,i}=1$.

\subsubsection{Differentiable Selected Memory and Label Tokens per Slot}
For each slot $k$, define the selected memory tokens as a weighted combination over its $T$ candidates:
\[
\tilde{\mathbf{M}}_k = \sum_{t=1}^{T} w_{k,t}\ \mathbf{M}_{\mathcal{I}_k[t]}
\in \mathbb{R}^{m\times d_h}.
\]
We also form a differentiable label token (same straight-through weights):
\[
\tilde{\boldsymbol{\ell}}_k = \sum_{t=1}^{T} w_{k,t}\ E(y_{\mathcal{I}_k[t]}) \in \mathbb{R}^{d_h}.
\]

\subsection{Constructing the Prototype Memory for Inference}
Append the selected label token to the selected memory tokens:
\[
\mathbf{P}_k = [\tilde{\mathbf{M}}_k;\ \tilde{\boldsymbol{\ell}}_k] \in \mathbb{R}^{(m+1)\times d_h}.
\]
Concatenate across slots to form the full prototype memory sequence:
\[
\mathbf{Mem}^{(B)} = [\mathbf{P}_1;\mathbf{P}_2;\dots;\mathbf{P}_K] \in \mathbb{R}^{K(m+1)\times d_h}.
\]
In batch form, $\mathbf{Mem}^{(B)}\in\mathbb{R}^{B\times K(m+1)\times d_h}$ is replicated across the batch (since prototypes are global, not query-dependent).

\section{Stage B Query Processing}

Given a query input $x_q$, compute:
\[
\mathbf{H}_q = \text{LLM}_{\text{frozen}}(x_q)\in\mathbb{R}^{S\times d_{\text{model}}},
\]
project to head dimension and compress to $m_q$ query tokens:
\[
\mathbf{Q}_q = \text{QueryCompressor}(\mathbf{H}_q\mathbf{W}_q)\in\mathbb{R}^{m_q\times d_h}.
\]
In batch form: $\mathbf{Q}_q\in\mathbb{R}^{B\times m_q\times d_h}$.

\section{Stage B Inference Head (Regression Token Attends to Query and Prototypes)}

We use the same inference mechanism as in Stage A, except memory now comes from selected prototypes.

Initialize a batch of regression tokens:
\[
\mathbf{R}_0 \in \mathbb{R}^{B\times 1\times d_h}.
\]

For $\ell=1,\dots,L$:
\begin{align}
\mathbf{R}^{(\ell)} &\leftarrow \mathbf{R}^{(\ell-1)} + \text{Attn}\big(\mathbf{R}^{(\ell-1)},\ \mathbf{Q}_q,\ \mathbf{Q}_q\big),\\
\mathbf{R}^{(\ell)} &\leftarrow \mathbf{R}^{(\ell)} + \text{Attn}\big(\mathbf{R}^{(\ell)},\ \mathbf{Mem}^{(B)},\ \mathbf{Mem}^{(B)}\big),
\end{align}
then apply FFN + layer norms.

Finally:
\[
\hat{z} = f_{\text{out}}(\mathbf{R}^{(L)}_{[:,0,:]})\in\mathbb{R},
\qquad
\hat{y} = a + (b-a)\sigma(\hat{z})\in[a,b].
\]

\paragraph{Interpretability via attention.}
The second cross-attention produces attention weights over the $K(m+1)$ memory tokens. By summing the weights belonging to each prototype block $\mathbf{P}_k$, we get a per-prototype importance score. The hard indices $\{i_k^*\}$ provide the example-grounded identities of prototypes.

\section{Stage B Losses and What Is Trained}

\subsection{Regression Loss}
\[
\mathcal{L}_{\text{reg}}^{(B)} = \frac{1}{B}\sum_{b=1}^{B}\mathcal{L}_{\text{reg}}(\hat{y}_b,y_b),
\]
using the same Huber loss as Stage A.

\subsection{Soft Overlap Penalty (Differentiable Duplicate Control)}
Instead of counting hard duplicates, we penalize overlap between the \emph{soft} per-slot selection distributions:
\[
\mathcal{L}_{\text{ov}} = \frac{2}{K(K-1)}\sum_{1\le k<j\le K} \mathbf{q}_k^\top \mathbf{q}_j.
\]
This equals the expected collision probability between independently sampled indices from the slot distributions.

\subsection{Soft Repulsion Penalty (Spread Prototypes in Key Space)}
Define the \emph{expected} normalized key for each slot:
\[
\tilde{\mathbf{k}}_k = \sum_{i=1}^{N} q_{k,i}\ \bar{\mathbf{k}}_i \in \mathbb{R}^{d_h},\qquad
\bar{\tilde{\mathbf{k}}}_k = \frac{\tilde{\mathbf{k}}_k}{\|\tilde{\mathbf{k}}_k\|_2 + \epsilon}.
\]
Form a $K\times K$ cosine similarity matrix:
\[
\tilde{\mathbf{S}}_{k,j} = \bar{\tilde{\mathbf{k}}}_k^\top \bar{\tilde{\mathbf{k}}}_j.
\]
Penalize pairs that are too similar using margin $m_{\text{cos}}$:
\[
\mathcal{L}_{\text{rep}} = \frac{2}{K(K-1)}\sum_{1\le k<j\le K} \max(0,\ \tilde{\mathbf{S}}_{k,j}-m_{\text{cos}}).
\]

\subsection{Stage B Objective}
\[
\mathcal{L}^{(B)} = \mathcal{L}_{\text{reg}}^{(B)} + \lambda_{\text{ov}}\mathcal{L}_{\text{ov}} + \lambda_{\text{rep}}\mathcal{L}_{\text{rep}},
\]
with $\lambda_{\text{ov}}=1.0$ and $\lambda_{\text{rep}}=0.1$.

\subsection{Parameters Updated in Stage B}
Trainable in Stage B:
\begin{itemize}[nosep]
    \item slot queries $\{\mathbf{s}_k\}_{k=1}^K$ (prototype selector),
    \item $\mathbf{W}_q$ and query compressor,
    \item label embedder $E(\cdot)$ (used to construct prototype label tokens),
    \item inference head parameters including regression token and output MLP.
\end{itemize}
Frozen in Stage B:
\begin{itemize}[nosep]
    \item frozen LLM backbone,
    \item cached $\mathbf{M}_i$, $\bar{\mathbf{k}}_i$, and $y_i$.
\end{itemize}

\section{Finalizing Prototypes (Hard Distilled Set)}

After Stage B training, we produce a final set of $K$ hard prototype indices with no duplicates.

Compute logits for all slots against all keys:
\[
\text{logit}_{k,i} = \bar{\mathbf{s}}_k^\top \bar{\mathbf{k}}_i.
\]
Then choose a unique set via greedy masked decoding:
\[
i_k^{\text{final}} = \arg\max_{i \notin \{i_1^{\text{final}},\dots,i_{k-1}^{\text{final}}\}} \text{logit}_{k,i}.
\]
Store $\{i_k^{\text{final}}\}_{k=1}^K$ as the final prototypes.

\section{End-to-End Inference Pipeline (Deployment)}

Given a new query text $x_q$:

\begin{enumerate}[nosep]
    \item \textbf{Query encoding:}
    \[
    \mathbf{H}_q = \text{LLM}_{\text{frozen}}(x_q)\in\mathbb{R}^{S\times d_{\text{model}}}.
    \]
    \item \textbf{Query projection and compression:}
    \[
    \mathbf{Q}_q = \text{QueryCompressor}(\mathbf{H}_q\mathbf{W}_q)\in\mathbb{R}^{m_q\times d_h}.
    \]
    \item \textbf{Load fixed prototype memory from cache:} for each $k=1,\dots,K$ use $i_k^{\text{final}}$ and load cached $\mathbf{M}_{i_k^{\text{final}}}$ and label $y_{i_k^{\text{final}}}$.
    \[
    \mathbf{P}_k = [\mathbf{M}_{i_k^{\text{final}}};\ E(y_{i_k^{\text{final}}})] \in \mathbb{R}^{(m+1)\times d_h}.
    \]
    \[
    \mathbf{Mem}^{\text{fixed}} = [\mathbf{P}_1;\dots;\mathbf{P}_K] \in \mathbb{R}^{K(m+1)\times d_h}.
    \]
    \item \textbf{Run inference head:} initialize regression token $\mathbf{R}_0$ and apply $L$ inference layers of cross-attention to $\mathbf{Q}_q$ and $\mathbf{Mem}^{\text{fixed}}$.
    \item \textbf{Output:}
    \[
    \hat{y} = a + (b-a)\sigma\!\big(f_{\text{out}}(\mathbf{R}^{(L)})\big).
    \]
    \item \textbf{Explanation (prototype attribution):} extract attention weights from the regression token to $\mathbf{Mem}^{\text{fixed}}$, sum weights per prototype block $\mathbf{P}_k$, and report the top contributing prototypes (their original texts and labels).
\end{enumerate}

\section{Algorithms}

\begin{algorithm}[h]
\caption{Stage A Training (Learn Memory Writer + Key Readout)}
\begin{algorithmic}[1]
\For{epoch $=1$ to 5}
\For{minibatch $\{(x_b,y_b)\}_{b=1}^B$}
\State $\mathbf{H}\gets \text{LLM}_{\text{frozen}}(x)$ \Comment{$B\times S\times d_{\text{model}}$}
\State $\mathbf{Q}_q\gets \text{QueryCompressor}(\mathbf{H}\mathbf{W}_q)$ \Comment{$B\times m_q\times d_h$}
\State $\mathbf{M}\gets \text{Compressor}(\mathbf{H}\mathbf{W}_m)$ \Comment{$B\times m\times d_h$}
\State $\hat{y}\gets \text{InferenceHead}(\mathbf{Q}_q,\mathbf{M})$
\State $\mathbf{k}\gets \text{KeyReadout}(\mathbf{M})$ \Comment{$B\times d_h$}
\State $\hat{\mathbf{e}}\gets g_{\text{emb}}(\mathbf{k})$ \Comment{$B\times d_h$}
\State $\mathbf{e}\gets E(y)$ \Comment{$B\times d_h$ (target only)}
\State $\mathcal{L}^{(A)}\gets \text{Huber}(\hat{y},y)+\lambda_{\text{emb}}\frac{1}{d_h}\|\hat{\mathbf{e}}-\mathbf{e}\|_2^2$
\State Update trainable parameters by backprop on $\mathcal{L}^{(A)}$
\EndFor
\EndFor
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[h]
\caption{Cache Construction (After Stage A)}
\begin{algorithmic}[1]
\For{$i=1$ to $N$}
\State $\mathbf{H}_i\gets \text{LLM}_{\text{frozen}}(x_i)$
\State $\mathbf{M}_i\gets \text{Compressor}(\mathbf{H}_i\mathbf{W}_m)$ \Comment{$m\times d_h$}
\State $\mathbf{k}_i \gets \text{KeyReadout}(\mathbf{M}_i)$ \Comment{$d_h$}
\State $\bar{\mathbf{k}}_i \gets \mathbf{k}_i / (\|\mathbf{k}_i\|_2+\epsilon)$
\State Store $\mathbf{M}_i$ (fp16), $\bar{\mathbf{k}}_i$ (fp16), $y_i$
\EndFor
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[h]
\caption{Stage B Training (Unique Prototypes + Soft Regularizers)}
\begin{algorithmic}[1]
\For{epoch $=1$ to 10}
\State Set $\tau \gets 1.0 - \frac{\text{epoch}-1}{9}(1.0-0.1)$
\For{minibatch $\{(x_b,y_b)\}_{b=1}^B$}
\State $\mathbf{H}_q\gets \text{LLM}_{\text{frozen}}(x)$
\State $\mathbf{Q}_q\gets \text{QueryCompressor}(\mathbf{H}_q\mathbf{W}_q)$
\State Initialize availability mask $a_i \gets 1$ for all $i\in\{1,\dots,N\}$
\For{$k=1$ to $K$}
\State Compute logits $\text{logit}_{k,i}=\bar{\mathbf{s}}_k^\top \bar{\mathbf{k}}_i$ for all $i$
\State $\mathcal{I}_k \gets$ top-$T$ indices by $\text{logit}_{k,i}$
\State Mask logits for unavailable indices in $\mathcal{I}_k$ (set to $-\infty$)
\State Sample $p_k$ via Gumbel-Softmax at temperature $\tau$; get hard pick $i_k^*$
\State Update availability: set $a_{i_k^*}\gets 0$
\State Form straight-through weights $w_k$ and build $\tilde{\mathbf{M}}_k$, $\tilde{\boldsymbol{\ell}}_k$
\EndFor
\State Build $\mathbf{Mem}^{(B)}$ by concatenating $[\tilde{\mathbf{M}}_k;\tilde{\boldsymbol{\ell}}_k]$ over $k=1..K$
\State $\hat{y}\gets \text{InferenceHead}(\mathbf{Q}_q,\mathbf{Mem}^{(B)})$
\State $\mathcal{L}_{\text{reg}}\gets \text{Huber}(\hat{y},y)$
\State $\mathcal{L}_{\text{ov}}\gets \frac{2}{K(K-1)}\sum_{k<j}\mathbf{q}_k^\top \mathbf{q}_j$ \Comment{soft overlap}
\State $\mathcal{L}_{\text{rep}}\gets \frac{2}{K(K-1)}\sum_{k<j}\max(0,\tilde{\mathbf{S}}_{k,j}-m_{\text{cos}})$ \Comment{soft repulsion}
\State $\mathcal{L}^{(B)} \gets \mathcal{L}_{\text{reg}} + \lambda_{\text{ov}}\mathcal{L}_{\text{ov}} + \lambda_{\text{rep}}\mathcal{L}_{\text{rep}}$
\State Update Stage B trainable parameters by backprop on $\mathcal{L}^{(B)}$
\EndFor
\EndFor
\end{algorithmic}
\end{algorithm}

\section{Glossary of Variables (Complete)}
\begin{itemize}[nosep]
    \item $N$: number of training examples.
    \item $B$: batch size.
    \item $S$: maximum sequence length after tokenization (default 256).
    \item $d_{\text{model}}$: frozen LLM hidden size.
    \item $d_h$: head dimension for prototype layer (default 256).
    \item $m$: memory tokens per training example in cache (default 8).
    \item $m_q$: compressed query tokens (default 8).
    \item $K$: number of prototypes selected globally (default 128).
    \item $T$: candidate pool size per slot in selection (default 512).
    \item $L$: number of inference head layers (default 3).
    \item $H$: number of attention heads (default 8).
    \item $\mathbf{H}(x)$: token-level hidden states from frozen LLM.
    \item $\mathbf{W}_m,\mathbf{W}_q$: linear projections to $d_h$.
    \item $\mathbf{M}(x)$: compressed memory tokens written for $x$.
    \item $\mathbf{q}^{\text{key}}$: learned key-readout latent query.
    \item $\mathbf{k}(x),\bar{\mathbf{k}}(x)$: learned (normalized) selection key for example $x$.
    \item $\bar{\mathbf{k}}_i$: cached normalized key for example $i$.
    \item $\mathbf{s}_k,\bar{\mathbf{s}}_k$: trainable slot query for prototype slot $k$ (and its normalization).
    \item $\mathbf{Q}_q$: compressed query tokens used by inference head.
    \item $\mathbf{r}_0$: trainable regression token parameter (initial state).
    \item $\mathbf{R}^{(\ell)}$: regression token after inference layer $\ell$.
    \item $E(y)=\mathbf{e}(y)$: scalar label embedder producing a label token in $\mathbb{R}^{d_h}$.
    \item $\hat{\mathbf{e}}(x)$: predicted label embedding from memory/key readout (Stage A auxiliary).
    \item $g_{\text{emb}}$: MLP mapping $\mathbf{k}(x)\mapsto \hat{\mathbf{e}}(x)$.
    \item $\mathbf{Mem}^{(A)}$: Stage A memory (single example memory tokens only).
    \item $\mathbf{Mem}^{(B)}$: Stage B memory (concatenated prototypes).
    \item $\tau$: Gumbel-softmax temperature.
    \item $a_i^{(k)}$: availability mask for uniqueness at slot $k$.
    \item $\mathbf{p}_k$: per-slot soft distribution over top-$T$ candidates.
    \item $\mathbf{q}_k$: sparse per-slot distribution over full dataset indices.
    \item $\tilde{\mathbf{M}}_k$: selected memory tokens for slot $k$ (straight-through).
    \item $\tilde{\boldsymbol{\ell}}_k$: selected label token for slot $k$ (straight-through).
    \item $\tilde{\mathbf{k}}_k$: expected key for slot $k$ (soft).
    \item $\hat{z}$: unbounded scalar output of the head.
    \item $a,b$: known lower and upper bounds of the target range, with $a<b$.
    \item $\hat{y}$: bounded prediction in $[a,b]$.
    \item $\lambda_{\text{emb}}$: weight of Stage A embedding alignment loss.
    \item $\lambda_{\text{ov}},\lambda_{\text{rep}}$: loss weights for Stage B soft overlap and repulsion.
\end{itemize}

\section{Summary (What This System Achieves)}
\begin{itemize}[nosep]
    \item \textbf{Example-grounded prototypes:} Each prototype corresponds to an actual training example index.
    \item \textbf{Compact caching:} Store only $m$ memory tokens per example, not full $S$ token states.
    \item \textbf{Anti-cheating Stage A:} No label token is provided as input; memory must encode label-relevant signal.
    \item \textbf{Learned keys:} Selection keys are produced by a trained key readout (not mean pooling).
    \item \textbf{Unique selection:} A masked sequential selector enforces no duplicates in the forward pass.
    \item \textbf{Stable training:} Soft overlap and soft repulsion losses provide smooth gradients for selection shaping.
    \item \textbf{Expressive inference:} A regression token performs multi-step cross-attention over query and prototypes (not a kNN average).
    \item \textbf{Interpretability:} Attention weights provide prototype-level contribution scores.
\end{itemize}

\end{document}
