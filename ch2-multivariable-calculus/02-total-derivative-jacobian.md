# 02. 전미분과 야코비안

## 🎯 핵심 질문

- $f: \mathbb{R}^n \to \mathbb{R}^m$의 미분이 왜 "선형사상"인가?
- 야코비안 행렬은 어디서 나오는가?
- 편미분 존재 vs 전미분 존재의 차이를 ε-δ로 어떻게 구분하는가?
- PyTorch의 `.jacobian()`이 계산하는 것이 정확히 무엇인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **역전파의 핵심**: 역전파는 각 층에서 야코비안-벡터 곱(VJP)을 계산한다. 야코비안을 이해하지 않으면 역전파의 수학적 근거를 설명할 수 없다.
- **행렬 미분**: $L(W) = \|XW - Y\|^2$에서 $\partial L/\partial W$는 스칼라 함수의 행렬 미분이다. 이것이 야코비안의 특수 경우다.
- **고차원 연쇄법칙**: 딥러닝의 연쇄법칙 $J_{f\circ g} = J_f \cdot J_g$은 야코비안 행렬 곱으로 표현된다.

---

## 📐 수학적 선행 조건

- [Ch1-03. 미분의 정의와 선형근사](../ch1-analysis-foundations/03-derivative-linear-approx.md): 단변수 $f(a+h) = f(a) + f'(a)h + o(h)$
- [Ch2-01. 편미분과 방향도함수](./01-partial-directional-derivative.md)
- 선형대수: 선형사상, 행렬 곱, 행렬식 (→ [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive))

---

## 📖 직관적 이해

단변수에서 $f(a+h) \approx f(a) + f'(a) \cdot h$: 미분 $f'(a)$는 **스칼라**로, $h$에 곱해져 선형 근사를 만든다.

다변수 $f: \mathbb{R}^n \to \mathbb{R}^m$에서는? $f(a+h) \approx f(a) + J \cdot h$: 미분 $J$는 **$m \times n$ 행렬**로, 입력 변화량 $h \in \mathbb{R}^n$을 출력 변화량 $\in \mathbb{R}^m$으로 선형 변환한다.

이 행렬 $J$가 야코비안이고, 그 원소는 $J_{ij} = \frac{\partial f_i}{\partial x_j}$이다.

---

## ✏️ 엄밀한 정의

### 정의 2.4 — 전미분 (Total Derivative)

$f: D \subseteq \mathbb{R}^n \to \mathbb{R}^m$이 점 $a \in D$에서 **전미분가능**하다는 것은, 선형사상 $L: \mathbb{R}^n \to \mathbb{R}^m$이 존재하여:

$$\lim_{\|h\| \to 0} \frac{\|f(a+h) - f(a) - L(h)\|}{\|h\|} = 0$$

이때 $L$을 $f$의 $a$에서의 **전미분(total derivative)**이라 하고, $df(a)$ 또는 $Df(a)$로 쓴다.

### 정의 2.5 — 야코비안 행렬

$f = (f_1, \ldots, f_m): \mathbb{R}^n \to \mathbb{R}^m$이 전미분가능이면, 전미분 $L$의 표준기저에 대한 행렬 표현이 **야코비안(Jacobian)**이다:

$$J_f(a) = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix} \in \mathbb{R}^{m \times n}$$

**특수 경우**:
- $m=1$: 야코비안은 행벡터 $(\nabla f)^\top \in \mathbb{R}^{1 \times n}$
- $n=m$: 정방 행렬, $\det(J)$는 국소 부피 변환율 (변수 변환의 야코비안)

---

## 🔬 정리와 증명

### 정리 2.2 — 전미분의 유일성

**명제**: 전미분이 존재하면 유일하다.

**증명**:  
$L_1$, $L_2$ 모두 전미분이라 하자. 임의의 $v \neq 0$에 대해 $h = tv$ ($t \to 0$)로 놓으면:

$$\frac{\|(L_1 - L_2)(tv)\|}{\|tv\|} = \frac{|t| \cdot \|(L_1 - L_2)(v)\|}{|t| \cdot \|v\|} = \frac{\|(L_1-L_2)(v)\|}{\|v\|}$$

이것이 $\frac{\|f(a+tv)-f(a)-L_1(tv)\|+\|f(a+tv)-f(a)-L_2(tv)\|}{\|tv\|}$에 의해 0으로 수렴하므로 $(L_1 - L_2)(v) = 0$. 모든 $v$에 대해 성립하므로 $L_1 = L_2$. $\square$

---

### 정리 2.3 — 야코비안이 전미분의 행렬 표현임

**명제**: $f$가 전미분가능이면 전미분 $Df(a)$의 행렬 표현은 야코비안 $J_f(a)$이다.

**증명**:  
$e_j$를 $j$번째 표준기저벡터라 하자. $h = te_j$로 놓으면:

$$\frac{\|f(a+te_j) - f(a) - Df(a)(te_j)\|}{|t|} \to 0$$

이는 각 성분에서:
$$\frac{f_i(a+te_j) - f_i(a)}{t} \to [Df(a)]_{ij} \quad (t \to 0)$$

좌변은 정의에 의해 $\frac{\partial f_i}{\partial x_j}(a)$이므로 $[Df(a)]_{ij} = \frac{\partial f_i}{\partial x_j}(a)$. $\square$

---

### 정리 2.4 — $C^1$ 이면 전미분가능

**명제**: $f$의 모든 편미분 $\frac{\partial f_i}{\partial x_j}$이 $a$ 근방에서 존재하고 연속이면 (즉 $f \in C^1$), $f$는 $a$에서 전미분가능하다.

**증명 스케치** (n=2인 경우):  
$f(a + h) - f(a) = [f(a_1 + h_1, a_2 + h_2) - f(a_1, a_2 + h_2)] + [f(a_1, a_2+h_2) - f(a_1, a_2)]$

각 괄호에 MVT 적용:
$= \frac{\partial f}{\partial x_1}(\xi_1, a_2+h_2) h_1 + \frac{\partial f}{\partial x_2}(a_1, \xi_2) h_2$

편미분의 연속성에 의해 $\xi_i \to a_i$이면 편미분값 → $\frac{\partial f}{\partial x_i}(a)$.  
따라서 $f(a+h) - f(a) - J_f(a) h = o(\|h\|)$가 성립. $\square$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp

# ─────────────────────────────────────────────
# 1. 수치 야코비안 계산
# ─────────────────────────────────────────────

def numerical_jacobian(f, x, h=1e-5):
    """중심 차분으로 야코비안 계산 (m×n 행렬 반환)"""
    x = np.array(x, dtype=float)
    n = len(x)
    f0 = np.atleast_1d(f(x))
    m = len(f0)
    J = np.zeros((m, n))
    for j in range(n):
        e_j = np.zeros(n); e_j[j] = 1.0
        f_plus  = np.atleast_1d(f(x + h * e_j))
        f_minus = np.atleast_1d(f(x - h * e_j))
        J[:, j] = (f_plus - f_minus) / (2 * h)
    return J

# 테스트: f: R² → R³
def f_vec(x):
    x1, x2 = x[0], x[1]
    return np.array([
        x1**2 + x2,          # f1
        np.sin(x1) * x2,     # f2
        np.exp(x1 + x2)      # f3
    ])

a = np.array([1.0, 2.0])

# 이론적 야코비안
# J = [[2x1, 1], [cos(x1)*x2, sin(x1)], [exp(x1+x2), exp(x1+x2)]]
J_theory = np.array([
    [2*a[0],           1           ],
    [np.cos(a[0])*a[1], np.sin(a[0])],
    [np.exp(a[0]+a[1]), np.exp(a[0]+a[1])]
])
J_num = numerical_jacobian(f_vec, a)

print("=== f: R² → R³, a = (1, 2) ===")
print("이론 야코비안 (3×2):")
print(J_theory)
print("\n수치 야코비안 (3×2):")
print(J_num)
print(f"\n최대 오차: {np.abs(J_theory - J_num).max():.2e}")

# ─────────────────────────────────────────────
# 2. SymPy로 기호 야코비안
# ─────────────────────────────────────────────

x1, x2 = sp.symbols('x1 x2')
f_sym = sp.Matrix([
    x1**2 + x2,
    sp.sin(x1) * x2,
    sp.exp(x1 + x2)
])

J_sym = f_sym.jacobian([x1, x2])
print("\n[SymPy 야코비안]")
sp.pprint(J_sym)

# a = (1, 2)에서 수치 평가
J_at_a = J_sym.subs([(x1, 1), (x2, 2)])
print("\na=(1,2)에서 야코비안:")
sp.pprint(J_at_a.evalf(4))

# ─────────────────────────────────────────────
# 3. PyTorch jacobian vs 수치 야코비안
# ─────────────────────────────────────────────

try:
    import torch
    
    def f_torch(x):
        return torch.stack([
            x[0]**2 + x[1],
            torch.sin(x[0]) * x[1],
            torch.exp(x[0] + x[1])
        ])
    
    x_t = torch.tensor([1.0, 2.0], requires_grad=True, dtype=torch.float64)
    J_torch = torch.autograd.functional.jacobian(f_torch, x_t)
    
    print("\n[PyTorch jacobian]")
    print(J_torch.detach().numpy())
    print(f"수치 야코비안과 최대 오차: {np.abs(J_torch.detach().numpy() - J_num).max():.2e}")

except ImportError:
    print("PyTorch 미설치")

# ─────────────────────────────────────────────
# 4. 행렬 미분: dL/dW 계산
# ─────────────────────────────────────────────

# L = ||XW - Y||² (선형회귀 손실)
# dL/dW = 2Xᵀ(XW - Y) 

np.random.seed(42)
n_samples, n_in, n_out = 10, 4, 3
X = np.random.randn(n_samples, n_in)   # (10, 4)
W = np.random.randn(n_in, n_out)       # (4, 3)
Y = np.random.randn(n_samples, n_out)  # (10, 3)

residual = X @ W - Y                    # (10, 3)
L = np.sum(residual**2)

# 해석 미분: dL/dW = 2Xᵀ(XW - Y)
dL_dW_theory = 2 * X.T @ residual       # (4, 3)

# 수치 미분 (벡터화)
def loss_fn(w_flat):
    W_mat = w_flat.reshape(n_in, n_out)
    return np.sum((X @ W_mat - Y)**2)

dL_dW_num = numerical_jacobian(
    lambda w: np.array([loss_fn(w)]),
    W.flatten()
).reshape(n_in, n_out)

print(f"\n행렬 미분 검증 (dL/dW):")
print(f"  최대 오차: {np.abs(dL_dW_theory - dL_dW_num).max():.4e}")
```

---

## 🔗 AI/ML 연결

### 역전파는 VJP (Vector-Jacobian Product)

역전파의 핵심 연산: 층 $f$에서 상위 gradient $\delta = \partial L/\partial f(x)$가 내려올 때, 입력 gradient는:

$$\frac{\partial L}{\partial x} = J_f(x)^\top \delta$$

이것이 야코비안 전치-벡터 곱(VJP)이다. $m \times n$ 야코비안 전체를 구성하지 않고 $n$-차원 벡터만 계산하기 때문에 효율적이다.

### 행렬-벡터 연산의 야코비안

| 연산 | 입력 | 출력 | 야코비안 |
|------|------|------|---------|
| $y = Wx$ | $x \in \mathbb{R}^n$ | $y \in \mathbb{R}^m$ | $J = W \in \mathbb{R}^{m\times n}$ |
| $L = \|Wx - y\|^2$ | $W \in \mathbb{R}^{m\times n}$ | 스칼라 | $\nabla_W L = 2(Wx-y)x^\top$ |
| $z = \sigma(x)$ (원소별) | $x \in \mathbb{R}^n$ | $z \in \mathbb{R}^n$ | $J = \text{diag}(\sigma'(x))$ |

---

## ⚖️ 가정과 한계

| 조건 | 보장 |
|------|------|
| 편미분 모두 존재 | 전미분 존재 보장 안 됨 |
| $C^1$ (편미분 연속) | 전미분 존재 보장 |
| 전미분 존재 | 야코비안 = 전미분의 행렬 표현 |
| 정방 야코비안, $\det J \neq 0$ | 역함수 정리(IFT): 국소 역함수 존재 |

---

## 📌 핵심 정리

$$f(a+h) = f(a) + J_f(a)\,h + o(\|h\|), \quad J_f(a) \in \mathbb{R}^{m\times n}$$

$$[J_f(a)]_{ij} = \frac{\partial f_i}{\partial x_j}(a)$$

**VJP**: $\delta^\top J_f(x)$ — 역전파의 핵심 연산

---

## 🤔 생각해볼 문제

**문제 1**: $f(x,y) = (x^2 - y^2,\; 2xy)$의 야코비안을 구하고, $(1, 1)$에서 수치 야코비안과 비교하라.

<details><summary>해설</summary>
$J = \begin{pmatrix} 2x & -2y \\ 2y & 2x \end{pmatrix}$. $(1,1)$에서 $J = \begin{pmatrix}2&-2\\2&2\end{pmatrix}$, $\det J = 8 \neq 0$.
</details>

**문제 2**: 선형 레이어 $y = Wx + b$에서 $\partial y / \partial W$, $\partial y / \partial x$, $\partial y / \partial b$를 구하라.

<details><summary>해설</summary>
$\partial y / \partial x = W$ (야코비안), $\partial L/\partial W = \delta x^\top$ (역전파), $\partial y / \partial b = I$ (항등 행렬).
</details>

**문제 3 (AI 연결)**: PyTorch에서 `torch.autograd.functional.jacobian`을 직접 호출하는 것과 역전파 한 번으로 gradient를 얻는 것의 계산 비용 차이를 설명하라.

<details><summary>해설</summary>
`jacobian`은 $m$번의 역전파 (또는 $n$번의 순전파)가 필요: $O(mn)$ 비용. 스칼라 Loss의 역전파는 1번: $O(n)$ 비용. 스칼라 출력에서 역전파가 훨씬 효율적인 이유다.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 편미분과 방향도함수](./01-partial-directional-derivative.md) | [📚 README](../README.md) | [03. Gradient와 코시-슈바르츠 ▶](./03-gradient-cauchy-schwarz.md) |

</div>
