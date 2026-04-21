# 05. 다변수 연쇄법칙

## 🎯 핵심 질문

- 다변수 연쇄법칙 $J_{f\circ g} = J_f \cdot J_g$는 왜 야코비안 행렬 곱인가?
- 스칼라·벡터·행렬 각각의 연쇄법칙이 어떻게 통일되는가?
- 역전파가 연쇄법칙을 "역순으로" 적용한다는 말의 정확한 의미는?
- 계산 그래프에서 연쇄법칙이 어떻게 구현되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

역전파(Backpropagation)는 **다변수 연쇄법칙의 자동 적용**이다. 딥러닝 프레임워크 전체가 이 한 가지 정리를 효율적으로 구현한 것이다.

- **순전파**: $x \to g(x) \to f(g(x))$ 계산
- **역전파**: $J_{f\circ g} = J_f(g(x)) \cdot J_g(x)$를 역순으로 적용하여 gradient 계산
- 각 층의 야코비안을 명시적으로 저장하지 않고 VJP(Vector-Jacobian Product)만 계산하므로 $O(n)$ 비용

---

## 📐 수학적 선행 조건

- [Ch2-02. 전미분과 야코비안](./02-total-derivative-jacobian.md)
- [Ch1-03. 미분의 정의와 선형근사 — 단변수 연쇄법칙](../ch1-analysis-foundations/03-derivative-linear-approx.md)

---

## 📖 직관적 이해

단변수: $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$ — 스칼라 곱.

다변수: $J_{f \circ g}(x) = J_f(g(x)) \cdot J_g(x)$ — **행렬 곱**.

왜 행렬 곱인가? 선형 근사의 합성이기 때문이다.
- $g(x+h) \approx g(x) + J_g(x) h$
- $f(g(x+h)) \approx f(g(x)) + J_f(g(x)) \cdot [J_g(x) h] = f(g(x)) + [J_f \cdot J_g] h$

합성의 선형 근사는 선형 근사의 합성 = 행렬 곱.

---

## ✏️ 엄밀한 정의와 정리

### 정리 2.11 — 다변수 연쇄법칙

**명제**: $g: \mathbb{R}^n \to \mathbb{R}^k$가 $a$에서 전미분가능, $f: \mathbb{R}^k \to \mathbb{R}^m$이 $g(a)$에서 전미분가능이면, $h = f \circ g: \mathbb{R}^n \to \mathbb{R}^m$은 $a$에서 전미분가능하며:

$$J_h(a) = J_f(g(a)) \cdot J_g(a) \in \mathbb{R}^{m \times n}$$

**증명**:  
$b = g(a)$로 놓자. 전미분가능 가정으로:
$$g(a+s) = b + J_g(a) s + r_g(s), \quad \frac{\|r_g(s)\|}{\|s\|} \to 0$$
$$f(b+t) = f(b) + J_f(b) t + r_f(t), \quad \frac{\|r_f(t)\|}{\|t\|} \to 0$$

$t = J_g(a)s + r_g(s)$로 놓으면:
$$f(g(a+s)) = f(b) + J_f(b)(J_g(a)s + r_g(s)) + r_f(J_g(a)s + r_g(s))$$
$$= f(g(a)) + [J_f(b) \cdot J_g(a)] s + \underbrace{J_f(b) r_g(s) + r_f(J_g(a)s + r_g(s))}_{=: R(s)}$$

$\|R(s)\| / \|s\| \to 0$임을 보이면 된다:
- $\|J_f(b) r_g(s)\| / \|s\| \leq \|J_f(b)\| \cdot \|r_g(s)\| / \|s\| \to 0$
- $\|r_f(J_g s + r_g)\| / \|s\|$: $\|J_g s + r_g\| / \|s\|$는 유계이므로 $r_f / \|t\| \to 0$ 조건에 의해 0으로 수렴

따라서 $J_{f\circ g}(a) = J_f(g(a)) \cdot J_g(a)$. $\square$

---

### 연쇄법칙의 세 가지 형태

**스칼라 → 스칼라** (단변수):
$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

**벡터 → 스칼라** (gradient):
$$\nabla_x [f(g(x))] = J_g(x)^\top \cdot \nabla_y f(g(x)) \in \mathbb{R}^n$$

*역전파의 핵심 공식: upstream gradient $\nabla_y f$에 $J_g^\top$를 곱함*

**벡터 → 벡터** (야코비안):
$$J_{f\circ g}(x) = J_f(g(x)) \cdot J_g(x) \in \mathbb{R}^{m \times n}$$

---

### 역전파는 연쇄법칙의 역순 누적

$L = f_N \circ f_{N-1} \circ \cdots \circ f_1(x)$이면:

$$J_L = J_{f_N} \cdot J_{f_{N-1}} \cdots J_{f_1}$$

$\nabla_x L = J_L^\top \cdot 1 = J_{f_1}^\top \cdot J_{f_2}^\top \cdots J_{f_N}^\top \cdot 1$

역전파는 이 곱을 **오른쪽에서 왼쪽으로** 누적:

$$\delta^{(n)} = J_{f_n}(x^{(n-1)})^\top \delta^{(n+1)}$$

각 단계에서 $m$-차원 벡터 $\delta$에 야코비안 전치를 곱하는 VJP: $O(mn)$ → 전체 $N$층 역전파 비용 $O(N \cdot mn)$.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp

# ─────────────────────────────────────────────
# 1. 야코비안 연쇄법칙 수치 검증
# ─────────────────────────────────────────────

def numerical_jacobian(f, x, h=1e-5):
    x = np.array(x, dtype=float)
    f0 = np.atleast_1d(f(x))
    J = np.zeros((len(f0), len(x)))
    for j in range(len(x)):
        e = np.zeros(len(x)); e[j] = 1
        J[:, j] = (np.atleast_1d(f(x+h*e)) - np.atleast_1d(f(x-h*e))) / (2*h)
    return J

# g: R² → R³,  f: R³ → R²
def g(x): return np.array([x[0]**2, x[0]*x[1], np.sin(x[1])])
def f(y): return np.array([y[0]+y[1], y[1]*y[2]])
def fog(x): return f(g(x))

a = np.array([1.0, 2.0])

J_g = numerical_jacobian(g, a)           # 3×2
J_f = numerical_jacobian(f, g(a))        # 2×3
J_chain = J_f @ J_g                       # 2×2 (연쇄법칙)
J_fog = numerical_jacobian(fog, a)        # 2×2 (직접)

print("=== 연쇄법칙 검증: f∘g ===")
print(f"J_g (3×2):\n{J_g}")
print(f"\nJ_f (2×3):\n{J_f}")
print(f"\nJ_f·J_g (연쇄법칙):\n{J_chain}")
print(f"\nJ_fog (직접계산):\n{J_fog}")
print(f"\n최대 오차: {np.abs(J_chain - J_fog).max():.2e}")

# ─────────────────────────────────────────────
# 2. 스칼라 손실의 역전파 시뮬레이션
# ─────────────────────────────────────────────

def forward_and_backward(x_val, W1, W2):
    """
    2층 선형 네트워크: L = ||W2 · relu(W1 · x) - y||²
    순전파 + 역전파를 수동으로 구현
    """
    # 순전파
    h1 = W1 @ x_val        # (4,)
    a1 = np.maximum(0, h1) # ReLU
    y_pred = W2 @ a1        # (2,)
    y_true = np.ones(2)
    diff = y_pred - y_true
    L = np.sum(diff**2)
    
    # 역전파 (연쇄법칙)
    dL_dy  = 2 * diff                      # (2,)    ∂L/∂y
    dL_da1 = W2.T @ dL_dy                  # (4,)    ∂L/∂a1 = J_W2ᵀ · dL_dy
    dL_dh1 = dL_da1 * (h1 > 0).astype(float)  # ReLU의 subgradient
    dL_dW1 = np.outer(dL_dh1, x_val)      # (4,3)  ∂L/∂W1
    dL_dW2 = np.outer(dL_dy, a1)          # (2,4)  ∂L/∂W2
    
    return L, dL_dW1, dL_dW2

np.random.seed(42)
x_val = np.random.randn(3)
W1 = np.random.randn(4, 3) * 0.1
W2 = np.random.randn(2, 4) * 0.1

L, grad_W1, grad_W2 = forward_and_backward(x_val, W1, W2)
print(f"\n=== 2층 네트워크 역전파 ===")
print(f"손실 L = {L:.6f}")
print(f"‖∂L/∂W1‖ = {np.linalg.norm(grad_W1):.6f}")
print(f"‖∂L/∂W2‖ = {np.linalg.norm(grad_W2):.6f}")

# PyTorch와 비교
try:
    import torch
    x_t  = torch.tensor(x_val,  dtype=torch.float64)
    W1_t = torch.tensor(W1, requires_grad=True, dtype=torch.float64)
    W2_t = torch.tensor(W2, requires_grad=True, dtype=torch.float64)
    
    h1_t = W1_t @ x_t
    a1_t = torch.relu(h1_t)
    y_t  = W2_t @ a1_t
    L_t  = ((y_t - 1)**2).sum()
    L_t.backward()
    
    print(f"\n수동 역전파 vs PyTorch autograd:")
    print(f"  ∂L/∂W1 최대 오차: {np.abs(grad_W1 - W1_t.grad.numpy()).max():.2e}")
    print(f"  ∂L/∂W2 최대 오차: {np.abs(grad_W2 - W2_t.grad.numpy()).max():.2e}")
except ImportError:
    print("PyTorch 미설치")
```

---

## 🔗 AI/ML 연결

### 자동미분 프레임워크의 핵심

모든 딥러닝 프레임워크는 다음을 구현한다:
1. **순전파**: 계산 그래프 구성, 각 노드의 입출력 저장
2. **역전파**: 출력에서 입력 방향으로 VJP 누적

```python
# PyTorch의 역전파 내부 (개념적)
def backward_relu(upstream_grad, saved_input):
    # ReLU의 야코비안 = diag(x > 0)
    # VJP = diag(x > 0)ᵀ · upstream = upstream * (x > 0)
    return upstream_grad * (saved_input > 0)

def backward_linear(upstream_grad, W, x):
    # Linear y = Wx의 야코비안 = W
    # VJP (∂L/∂x) = Wᵀ · upstream
    # ∂L/∂W = upstream · xᵀ (외적)
    return W.T @ upstream_grad, np.outer(upstream_grad, x)
```

### 연쇄법칙과 Gradient Checkpointing

역전파는 순전파 시 모든 중간값을 저장해야 한다 (야코비안-벡터 곱에 필요). 메모리 절약을 위한 Gradient Checkpointing은 일부 중간값만 저장하고, 필요 시 재계산한다.

---

## 📌 핵심 정리

$$J_{f \circ g}(x) = J_f(g(x)) \cdot J_g(x)$$

$$\text{역전파: } \delta^{(n)} = J_{f_n}^\top \cdot \delta^{(n+1)} \quad \text{(VJP)}$$

| 형태 | 공식 | 비용 |
|------|------|------|
| 스칼라-스칼라 | $f' \cdot g'$ | $O(1)$ |
| 벡터-스칼라 | $J_g^\top \nabla_y f$ | $O(kn)$ |
| 벡터-벡터 | $J_f \cdot J_g$ | $O(mn + nk)$ |

---

## 🤔 생각해볼 문제

**문제 1**: $h(x) = \sin(x^2 + 1)$에서 $h'(x)$를 연쇄법칙으로 구하라. $g(x) = x^2+1$, $f(u) = \sin u$.

<details><summary>해설</summary>
$h'(x) = f'(g(x)) \cdot g'(x) = \cos(x^2+1) \cdot 2x$.
</details>

**문제 2**: 3층 네트워크 $L = f_3(f_2(f_1(x)))$에서 $\partial L/\partial x$를 야코비안으로 표현하라. 역전파 시 어떤 순서로 계산하는가?

<details><summary>해설</summary>
$\nabla_x L = J_{f_1}^\top \cdot J_{f_2}^\top \cdot J_{f_3}^\top \cdot 1$. 역전파는 $J_{f_3}^\top \cdot 1 \to J_{f_2}^\top \cdot (\ldots) \to J_{f_1}^\top \cdot (\ldots)$ 순서로 오른쪽부터 계산.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. 야코비안과 헤시안의 기하](./04-jacobian-hessian-geometry.md) | [📚 README](../README.md) | [06. 음함수 정리 ▶](./06-implicit-function-theorem.md) |

</div>
