# 05. 미분가능 함수의 성질과 반례

## 🎯 핵심 질문

- 미분가능 → 연속이지만 역은 왜 거짓인가?
- 모든 점에서 미분불가능한 연속 함수(Weierstrass 함수)가 존재하는가?
- 딥러닝의 ReLU는 $x = 0$에서 미분불가능한데 왜 역전파가 작동하는가?
- Subgradient는 무엇이며, 어떻게 비볼록 최적화를 가능하게 하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

딥러닝에서 사용하는 활성화 함수들은 대부분 **어딘가에서 미분불가능**하다:

- **ReLU**: $x = 0$에서 미분불가능 ($\max(0, x)$의 꺾임)
- **Leaky ReLU, ELU**: 유사한 비매끄러운 점 존재
- **MaxPool**: 최댓값이 여러 입력에서 동시에 달성될 때 비미분

그럼에도 역전파와 경사하강법이 작동하는 이유를 이해하려면 다음 두 개념이 필수:
1. **Subgradient**: 미분불가능 볼록 함수에서의 gradient 대체
2. **Lebesgue 측도 0인 집합**: "거의 어디서나" 미분가능하면 충분

---

## 📐 수학적 선행 조건

- [01. ε-δ 극한의 정의](./01-epsilon-delta.md)
- [02. 연속성과 균등연속성](./02-continuity-uniform.md)
- [03. 미분의 정의와 선형근사](./03-derivative-linear-approx.md)

---

## 📖 직관적 이해

### 미분가능 → 연속의 방향

도함수 $f'(a)$가 존재하면 $f(a+h) = f(a) + f'(a)h + o(h)$. $h \to 0$이면 우변 → $f(a)$. 따라서 자동으로 연속.

### 역이 거짓인 이유

연속이지만 미분불가능한 함수의 예:
- $f(x) = |x|$: $x = 0$에서 왼쪽 기울기 $-1$, 오른쪽 기울기 $+1$ → 극한이 하나로 정해지지 않음
- Weierstrass 함수: **어디서도** 기울기가 하나로 정해지지 않는 연속 함수

### Weierstrass의 충격

1872년 수학계는 "모든 연속 함수는 (어쩌면 유한개의 점을 빼고는) 미분가능하다"고 믿었다. Weierstrass는 이것이 거짓임을 보였다.

$$W(x) = \sum_{n=0}^{\infty} a^n \cos(b^n \pi x), \quad 0 < a < 1,\; b \text: 홀수 정수,\; ab > 1 + \frac{3\pi}{2}$$

이 함수는 연속이지만 **모든 점에서** 미분불가능하다.

> **비유**: 무한히 구겨진 종이. 아무리 확대해도 항상 새로운 주름이 나타나 "접선"을 그을 수 없다.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Subgradient

$f: \mathbb{R}^n \to \mathbb{R}$이 볼록 함수라 하자. 벡터 $g \in \mathbb{R}^n$이 점 $x$에서의 **subgradient**라는 것은:

$$\forall y \in \mathbb{R}^n,\quad f(y) \geq f(x) + g^\top(y - x)$$

집합 $\partial f(x) = \{g : g \text{는 } x\text{에서의 subgradient}\}$를 **subdifferential**이라 한다.

### 정의 5.2 — 거의 어디서나 (Almost Everywhere)

성질 $P$가 집합 $S$에서 **거의 어디서나(a.e.)** 성립한다는 것은, $P$가 성립하지 않는 점들의 집합의 Lebesgue 측도가 0이라는 것이다.

---

## 🔬 정리와 증명

### 정리 5.1 — 미분가능 ⇒ 연속

**명제**: $f$가 $a$에서 미분가능이면 $f$는 $a$에서 연속이다.

**증명**:  
$f'(a)$가 존재하므로:
$$f(a+h) - f(a) = h \cdot \frac{f(a+h) - f(a)}{h} \xrightarrow{h \to 0} 0 \cdot f'(a) = 0$$

따라서 $\lim_{h \to 0} f(a+h) = f(a)$이므로 $f$는 $a$에서 연속이다. $\square$

---

### 반례 5.1 — 연속이지만 미분불가능

**(a) $f(x) = |x|$, $a = 0$**

$f$는 $a = 0$에서 연속: $\lim_{x \to 0} |x| = 0 = f(0)$.

미분불가능:
$$\lim_{h \to 0^+} \frac{|h|}{h} = 1, \quad \lim_{h \to 0^-} \frac{|h|}{h} = -1$$

좌미분 $\neq$ 우미분이므로 미분불가능.

**(b) Weierstrass 함수**

$a = 0.5$, $b = 13$ ($ab = 6.5 > 1 + 3\pi/2 \approx 5.71$)으로 놓으면:

$$W(x) = \sum_{n=0}^{\infty} (0.5)^n \cos(13^n \pi x)$$

이 함수는 $\mathbb{R}$ 위에서 연속이지만 모든 점에서 미분불가능하다.

**증명 스케치** (미분불가능):  
임의의 점 $x_0$에서, $b^n$ 스케일의 진동이 $h = b^{-n}$ 스케일에서의 차분 몫을 발산시킨다. 구체적으로, $h_n = b^{-n}$으로 놓으면:

$$\frac{W(x_0 + h_n) - W(x_0)}{h_n} \quad \text{은 } n \to \infty \text{에서 수렴하지 않는다}$$

두 개의 부분 수열이 서로 다른 값으로 수렴하기 때문이다. $\square$

---

### 정리 5.2 — $|x|$의 Subgradient

**명제**: $f(x) = |x|$의 subdifferential은:
$$\partial f(x) = \begin{cases} \{1\} & x > 0 \\ [-1, 1] & x = 0 \\ \{-1\} & x < 0 \end{cases}$$

**증명** (x = 0인 경우):  
$g \in \partial f(0)$이라는 것은: $\forall y$, $|y| \geq 0 + g \cdot y$.
- $y > 0$이면: $y \geq gy \Rightarrow g \leq 1$
- $y < 0$이면: $-y \geq gy \Rightarrow g \geq -1$

따라서 $g \in [-1, 1]$. 역으로 $g \in [-1, 1]$이면 $|y| \geq gy$ 성립 (삼각부등식). $\square$

---

### 정리 5.3 — Rademacher 정리

**명제**: Lipschitz 연속인 함수 $f: \mathbb{R}^n \to \mathbb{R}$은 거의 어디서나(a.e.) 미분가능하다.

**의미**: 미분불가능점이 존재해도 그 집합의 "크기(측도)"가 0이면, 확률론적·통계적 의미에서 미분이 존재하는 것과 같다.

**딥러닝 응용**: ReLU 네트워크에서 학습 데이터의 preactivation이 정확히 0일 확률은 (연속 분포에서) 0이다. 따라서 거의 어디서나 기울기가 정의되고, SGD가 작동한다.

---

### 정리 5.4 — Subgradient의 최적성 조건

**명제**: 볼록 함수 $f$에서 $x^*$가 전역 최솟값인 것은 $0 \in \partial f(x^*)$인 것과 동치이다.

**증명 (⇒)**: $x^*$가 최솟값이면 $\forall y$, $f(y) \geq f(x^*)$.  
따라서 $g = 0$은 정의 5.1을 만족: $f(y) \geq f(x^*) + 0^\top(y - x^*)$. $\square$

**증명 (⇐)**: $0 \in \partial f(x^*)$이면 $\forall y$, $f(y) \geq f(x^*) + 0 = f(x^*)$. $\square$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ─────────────────────────────────────────────
# 1. Weierstrass 함수 부분합 시각화
# ─────────────────────────────────────────────

def weierstrass(x, a=0.5, b=13, n_terms=30):
    """Weierstrass 함수 부분합"""
    result = np.zeros_like(x, dtype=float)
    for n in range(n_terms):
        result += a**n * np.cos(b**n * np.pi * x)
    return result

x_fine = np.linspace(0, 1, 4000)
W_coarse = weierstrass(x_fine, n_terms=5)
W_medium = weierstrass(x_fine, n_terms=15)
W_fine   = weierstrass(x_fine, n_terms=30)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Weierstrass 함수: 어디서도 미분불가능한 연속 함수', fontsize=13, fontweight='bold')

for ax, W, title in zip(axes,
                         [W_coarse, W_medium, W_fine],
                         ['5항 부분합', '15항 부분합', '30항 부분합']):
    ax.plot(x_fine, W, 'b-', linewidth=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('W(x)')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('05-weierstrass-function.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. 확대해도 항상 주름이 나타남 (자기유사성)
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Weierstrass 함수 확대: 아무리 확대해도 매끄럽지 않다', fontsize=12)

zoom_ranges = [(0, 1), (0.4, 0.6), (0.48, 0.52)]
for ax, (x_lo, x_hi) in zip(axes, zoom_ranges):
    x_zoom = np.linspace(x_lo, x_hi, 5000)
    W_zoom = weierstrass(x_zoom, n_terms=40)
    ax.plot(x_zoom, W_zoom, 'b-', linewidth=0.5)
    ax.set_title(f'구간 [{x_lo}, {x_hi}]', fontsize=10)
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('05-weierstrass-zoom.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. ReLU의 Subgradient와 역전파
# ─────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_subgradient(x, convention='zero'):
    """
    ReLU의 subgradient:
    - x > 0: 1
    - x < 0: 0
    - x = 0: convention에 따라 0 또는 0.5
    """
    grad = np.where(x > 0, 1.0, 0.0)
    if convention == 'half':
        grad = np.where(x == 0, 0.5, grad)
    return grad

x_test = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
print("[ReLU Subgradient 검증]")
print(f"{'x':>8} {'ReLU(x)':>10} {'Subgradient':>15}")
print("-" * 35)
for xi in x_test:
    print(f"{xi:>8.1f} {relu(xi):>10.1f} {relu_subgradient(np.array([xi]))[0]:>15.1f}")

# 그래프
x_plot = np.linspace(-3, 3, 300)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x_plot, relu(x_plot), 'b-', linewidth=2.5, label='ReLU(x)')
axes[0].plot(x_plot, relu_subgradient(x_plot), 'r--', linewidth=2, label='Subgradient')
axes[0].scatter([0], [0.5], color='red', s=100, zorder=5,
                label='x=0에서 임의 선택 (통상 0)')
axes[0].set_title('ReLU와 Subgradient', fontsize=12)
axes[0].set_xlabel('x'); axes[0].set_ylabel('값')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(x_plot, np.abs(x_plot), 'b-', linewidth=2.5, label='|x|')
# subdifferential at x=0: [-1, 1]
axes[1].fill_between([-0.05, 0.05], [-1, -1], [1, 1], alpha=0.3, color='red',
                     label='∂|x|(0) = [-1, 1]')
axes[1].plot(x_plot[x_plot > 0.05], np.ones(sum(x_plot > 0.05)), 'r--', linewidth=2)
axes[1].plot(x_plot[x_plot < -0.05], -np.ones(sum(x_plot < -0.05)), 'r--', linewidth=2,
             label='Subgradient (±1)')
axes[1].set_ylim(-2, 2)
axes[1].set_title('|x|의 Subdifferential', fontsize=12)
axes[1].set_xlabel('x'); axes[1].set_ylabel('Subgradient')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05-subgradient.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 4. 수치 실험: ReLU 미분불가능점에서의 gradient
# ─────────────────────────────────────────────

try:
    import torch
    
    print("\n[PyTorch ReLU gradient at x=0]")
    for x_val in [0.0, 0.0, 0.0]:  # 세 번 반복
        x_t = torch.tensor(x_val, requires_grad=True)
        y = torch.relu(x_t)
        y.backward()
        print(f"x={x_val}, relu(x)={y.item():.1f}, grad={x_t.grad.item():.1f}")
    
    # PyTorch의 기본 관례: x=0에서 gradient = 0
    print("→ PyTorch 관례: x=0에서 ReLU gradient = 0 (subgradient 중 하나)")
    
except ImportError:
    print("PyTorch 미설치 — autograd 실험 생략")
```

**출력**:
```
[ReLU Subgradient 검증]
       x    ReLU(x)     Subgradient
-----------------------------------
    -2.0        0.0             0.0
    -1.0        0.0             0.0
    -0.5        0.0             0.0
     0.0        0.0             0.0
     0.5        0.5             1.0
     1.0        1.0             1.0
     2.0        2.0             1.0

[PyTorch ReLU gradient at x=0]
x=0.0, relu(x)=0.0, grad=0.0
x=0.0, relu(x)=0.0, grad=0.0
x=0.0, relu(x)=0.0, grad=0.0
→ PyTorch 관례: x=0에서 ReLU gradient = 0 (subgradient 중 하나)
```

---

## 🔗 AI/ML 연결

### ReLU가 미분불가능해도 역전파가 작동하는 이유

세 가지 이유가 결합된다:

**1. 측도론적 근거 (Rademacher 정리)**  
실수 분포에서 preactivation이 정확히 0일 확률은 0이다. "거의 어디서나" 미분가능하므로 실질적인 문제가 없다.

**2. Subgradient 하강법**  
미분불가능점에서 subdifferential의 원소 중 하나를 선택해도, 볼록 함수 최적화의 수렴이 보장된다. (비볼록 경우는 실용적으로 작동하지만 이론 보장은 약하다.)

**3. 구현상 관례**  
PyTorch/TensorFlow는 $x = 0$에서 gradient를 0으로 정의한다. 이것이 유일한 올바른 값은 아니지만, 대부분의 실용적 상황에서 학습을 방해하지 않는다.

### Dying ReLU 문제

$x < 0$인 뉴런은 subgradient = 0이 되어 gradient가 통과하지 못한다 ("죽은 뉴런"). 해결책:
- **Leaky ReLU**: $f(x) = \max(0.01x, x)$로 음수 구간에 작은 기울기 유지
- **ELU**: $f(x) = x$ ($x > 0$), $\alpha(e^x - 1)$ ($x \leq 0$)
- **He 초기화**: 뉴런이 처음부터 음수 구간에 빠지지 않도록 초기화

### L1 정규화와 Subgradient Descent

L1 정규화: $\mathcal{L}(W) + \lambda \|W\|_1$에서 $\|W\|_1 = \sum_i |W_i|$.

$W_i \neq 0$이면 gradient = $\text{sign}(W_i)$, $W_i = 0$이면 subgradient $\in [-1, 1]$.  
이것이 Lasso 최적화에서 sparse 해 (많은 가중치가 정확히 0)를 만드는 수학적 이유다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 볼록 함수에서의 Subgradient | 비볼록 함수 (ReLU 네트워크)에서는 수렴 보장이 약하다 |
| $0 \in \partial f(x^*)$ | 비볼록에서는 이 조건이 전역 최솟값을 보장하지 않는다 |
| Rademacher 정리 | 연속 분포 가정. 이산 데이터에서는 직접 적용 불가 |
| PyTorch gradient at 0 | 구현마다 다를 수 있음 (TF와 PyTorch의 관례가 다른 경우 존재) |

---

## 📌 핵심 정리

$$\text{미분가능} \Rightarrow \text{연속} \quad (\text{역은 거짓})$$

| 함수 | 연속성 | 미분가능성 |
|------|--------|-----------|
| $x^2$ | ✅ | ✅ |
| $\|x\|$ | ✅ | ✅ (x≠0), ❌ (x=0) |
| Weierstrass $W(x)$ | ✅ | ❌ (모든 점에서) |

**Subgradient**: 볼록 함수의 미분불가능점에서 gradient를 대체  
$$g \in \partial f(x) \iff \forall y,\; f(y) \geq f(x) + g^\top(y-x)$$

**최적성**: $0 \in \partial f(x^*) \iff x^*$는 볼록 함수의 전역 최솟값

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(x) = \max(x, 0)^2$는 모든 점에서 미분가능한가? $f'(0)$를 정의에 따라 계산하라.

<details>
<summary>해설</summary>

$f(x) = x^2$ ($x \geq 0$), $0$ ($x < 0$).  
$\lim_{h \to 0^+} \frac{h^2}{h} = 0$, $\lim_{h \to 0^-} \frac{0}{h} = 0$. 양쪽 극한이 같으므로 $f'(0) = 0$. 즉 모든 점에서 미분가능 ($C^1$ 함수).

</details>

**문제 2** (심화): $\partial |x|$의 (0에서의) 원소 $g = 0.3$이 실제로 subgradient 정의를 만족하는지 직접 검증하라.

<details>
<summary>해설</summary>

$g = 0.3$: $\forall y$, $|y| \geq |0| + 0.3 \cdot (y - 0) = 0.3y$?  
$y > 0$: $y \geq 0.3y$ ✅ ($0.7y \geq 0$)  
$y < 0$: $-y \geq 0.3y$, 즉 $-1.3y \geq 0$ ✅ ($y < 0$이므로)  
$y = 0$: $0 \geq 0$ ✅  
따라서 $0.3 \in \partial |x|(0)$ 성립.

</details>

**문제 3** (AI 연결): L1 정규화와 L2 정규화에서 최솟값 근방의 subgradient/gradient 차이가 **왜 L1은 희소해를, L2는 작은 값을** 만드는지 수학적으로 설명하라.

<details>
<summary>해설</summary>

L2: $\nabla \|W\|_2^2 = 2W$. 최솟값 조건 $\nabla \mathcal{L} + 2\lambda W = 0$. $W$가 0이 되려면 $\nabla \mathcal{L} = 0$이어야 하므로 정확한 0은 드물다.

L1: $\partial \|W\|_1$은 $W_i = 0$일 때 $[-1, 1]$. 최솟값 조건 $\nabla_i \mathcal{L} + \lambda g_i = 0$에서 $g_i \in [-1, 1]$이 선택 가능. 따라서 $|\nabla_i \mathcal{L}| \leq \lambda$이면 $W_i = 0$을 달성할 수 있어 희소해가 나타난다.

</details>

---

## 🔁 Chapter 1 요약 및 Chapter 2 예고

이 챕터에서 배운 것들:

| 문서 | 핵심 도구 | AI/ML 연결 |
|------|---------|-----------|
| 01. ε-δ 극한 | 정량적 수렴 정의 | 학습률 조건, SGD 수렴 |
| 02. 연속·균등연속 | Lipschitz 조건 | Gradient Clipping, 최솟값 존재 |
| 03. 미분·선형근사 | $f(a+h) = f(a) + f'h + o(h)$ | 역전파, gradcheck |
| 04. MVT·테일러 | $L$-smooth 조건 | GD 학습률 상한, 뉴턴 방법 |
| 05. 미분가능성 반례 | Subgradient | ReLU 역전파, L1 정규화 |

**Chapter 2 예고**: 단변수에서 다변수로 확장한다. $f'(a)$ (스칼라) → $\nabla f(x)$ (벡터) → $Jf(x)$ (야코비안 행렬). "최선의 선형근사"는 행렬로 표현되고, 연쇄법칙은 행렬 곱이 된다. 이것이 역전파의 수학적 본질이다.

---

<div align="center">

| | |
|---|---|
| [◀ 04. 평균값 정리와 테일러 정리](./04-mvt-taylor.md) | [📚 README](../README.md) |

</div>

<div align="center">

🔜 **다음 챕터**: [Chapter 2: 다변수 미적분 — AI가 사는 고차원 공간 →](../ch2-multivariable-calculus/01-partial-directional-derivative.md)

</div>
