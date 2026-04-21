# 04. 평균값 정리와 테일러 정리

## 🎯 핵심 질문

- Rolle → MVT → 테일러 정리는 어떻게 논리적으로 연결되는가?
- 테일러 여분항(Remainder)의 세 가지 형태 (Lagrange, Cauchy, 적분형)는 어떤 경우에 유용한가?
- 경사하강법의 수렴 조건 $\eta < 2/L$이 테일러 전개와 어떻게 연결되는가?
- 2차 테일러 근사가 헤시안을 통해 최적화와 만나는 지점은 어디인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

테일러 정리는 AI/ML 이론 전반에 걸쳐 등장한다:

- **경사하강법 수렴 증명**: $f(x - \eta \nabla f) \leq f(x) - \eta\|\nabla f\|^2 + \frac{\eta^2 L}{2}\|\nabla f\|^2$은 2차 테일러 전개의 직접 응용이다
- **뉴턴 방법**: 2차 테일러 근사를 최소화하면 $x \leftarrow x - H^{-1}\nabla f$가 도출된다
- **손실 함수의 Lipschitz 매끄러움(Smooth) 조건**: $\|∇f(x) - ∇f(y)\| \leq L\|x-y\|$는 $L$-smooth 조건이며, 이는 테일러 2차 여분항의 상한이다
- **Transformer의 학습 안정성**: 사전학습 초기 학습률 워밍업의 이론적 근거는 초기 파라미터에서의 테일러 근사 정확성과 연관된다

---

## 📐 수학적 선행 조건

- [01. ε-δ 극한의 정의](./01-epsilon-delta.md)
- [03. 미분의 정의와 선형근사](./03-derivative-linear-approx.md): $f'(a)$의 정의, $o(h)$ 표기
- 고차 도함수: $f^{(n)}(x)$는 $f$를 $n$번 미분한 것

---

## 📖 직관적 이해

### Rolle → MVT → 테일러의 논리 사슬

```
Rolle 정리           MVT                    테일러 정리
"양 끝이 같으면    "평균 기울기 = 순간     "함수를 다항식으로
 어딘가에 수평     기울기가 되는 점이       근사할 때 오차를
 접선이 있다"       있다"                   정량화한다"
    └──────────────────┘──────────────────────┘
            MVT는 Rolle의 특수화       테일러는 MVT의 반복 적용
```

**핵심 아이디어**: MVT는 "전체 변화량을 한 순간의 변화율로 설명"한다. 이를 반복 적용하면 함수를 임의의 정확도로 다항식으로 근사할 수 있다.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 고차 도함수

$f^{(0)} = f$, $f^{(n)} = (f^{(n-1)})'$으로 재귀적으로 정의. $f$가 $[a, b]$에서 $n$번 미분가능하고 $f^{(n)}$이 연속일 때 $f \in C^n([a, b])$라 쓴다.

### 정의 4.2 — 테일러 다항식

$f \in C^n$이고 점 $a$ 근방에서 다음을 $n$차 테일러 다항식이라 한다:

$$P_n(x; a) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

---

## 🔬 정리와 증명

### 정리 4.1 — Rolle의 정리

**명제**: $f: [a, b] \to \mathbb{R}$이 $[a,b]$에서 연속, $(a,b)$에서 미분가능, $f(a) = f(b)$이면, $c \in (a, b)$가 존재하여 $f'(c) = 0$.

**증명**:  
$f(a) = f(b) = m$이라 하자.

**경우 1**: $f$가 상수 함수 → 모든 점에서 $f'(c) = 0$.

**경우 2**: $f(x) > m$인 점이 존재.  
극값 정리(Extreme Value Theorem)에 의해 $f$는 $[a, b]$에서 최댓값을 가진다. 그 점을 $c \in (a,b)$라 하면 (최댓값은 끝점 $a, b$에서 아니므로):
$$f'(c) = \lim_{h \to 0^+} \frac{f(c+h) - f(c)}{h} \leq 0$$
$$f'(c) = \lim_{h \to 0^-} \frac{f(c+h) - f(c)}{h} \geq 0$$
두 부등식을 동시에 만족하려면 $f'(c) = 0$. $\square$

*(경우 3: $f(x) < m$인 점이 존재 → 최솟값에서 유사하게 증명)*

---

### 정리 4.2 — 평균값 정리 (MVT)

**명제**: $f: [a, b] \to \mathbb{R}$이 $[a,b]$에서 연속, $(a,b)$에서 미분가능이면, $c \in (a,b)$가 존재하여:

$$f'(c) = \frac{f(b) - f(a)}{b - a}$$

**증명 (Rolle 정리 이용)**:  
보조 함수 $g(x) = f(x) - \frac{f(b)-f(a)}{b-a}(x-a)$를 정의하면:
- $g$는 $[a,b]$에서 연속, $(a,b)$에서 미분가능
- $g(a) = f(a)$, $g(b) = f(b) - (f(b)-f(a)) = f(a)$, 즉 $g(a) = g(b)$

Rolle 정리에 의해 $c \in (a,b)$가 존재하여 $g'(c) = 0$.

$g'(x) = f'(x) - \frac{f(b)-f(a)}{b-a}$이므로 $f'(c) = \frac{f(b)-f(a)}{b-a}$. $\square$

---

### 정리 4.3 — 테일러 정리 (Lagrange 여분항)

**명제**: $f \in C^{n+1}([a, b])$이고 $x, a \in [a, b]$이면:

$$f(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k + R_n(x)$$

여기서 **Lagrange 여분항**은 어떤 $c \in (a, x)$ (또는 $(x, a)$)에 대해:

$$R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}$$

**증명 스케치**:  
$R_n(x) = f(x) - P_n(x; a)$로 정의하자.

보조 함수 $g(t) = f(x) - P_n(x; t) - \frac{(x-t)^{n+1}}{(n+1)!} \cdot K$ ($K$는 상수)를 만들어 $g(a) = 0$, $g(x) = 0$이 되도록 $K$를 정한다.

Rolle 정리를 반복 적용하면 $(a, x)$ 안에 $g'(c_1) = 0$인 점이 있고, 이를 정리하면 $K = f^{(n+1)}(c)$가 된다. $\square$

---

### 여분항의 세 가지 형태 비교

| 형태 | 표현 | 언제 유용한가 |
|------|------|-------------|
| **Lagrange** | $\frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}$ | $f^{(n+1)}$의 상한을 알 때 오차 경계 계산 |
| **Cauchy** | $\frac{f^{(n+1)}(c)}{n!}(x-c)^n(x-a)$ | 수렴 반경 분석 |
| **적분형** | $\int_a^x \frac{(x-t)^n}{n!} f^{(n+1)}(t)\,dt$ | 오차의 정밀한 경계, 함수해석에서 주로 사용 |

---

### 정리 4.4 — L-smooth 조건과 테일러 2차 여분항

**정의**: $f$가 **$L$-smooth**라는 것은 기울기 $\nabla f$가 Lipschitz 조건을 만족한다는 것:
$$\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$$

**명제**: $f$가 미분가능하고 $L$-smooth이면:
$$f(y) \leq f(x) + \nabla f(x)^\top (y-x) + \frac{L}{2}\|y-x\|^2$$

**증명**:  
$g(t) = f(x + t(y-x))$으로 놓으면, 기본 정리에 의해:

$$f(y) - f(x) = g(1) - g(0) = \int_0^1 g'(t)\,dt = \int_0^1 \nabla f(x + t(y-x))^\top (y-x)\,dt$$

$$= \nabla f(x)^\top (y-x) + \int_0^1 [\nabla f(x + t(y-x)) - \nabla f(x)]^\top (y-x)\,dt$$

$L$-smooth 조건과 코시-슈바르츠에 의해:

$$\left|\int_0^1 [\nabla f(x+t(y-x))-\nabla f(x)]^\top(y-x)\,dt\right| \leq \int_0^1 Lt\|y-x\|^2\,dt = \frac{L}{2}\|y-x\|^2$$

따라서 $f(y) \leq f(x) + \nabla f(x)^\top(y-x) + \frac{L}{2}\|y-x\|^2$. $\square$

> **핵심 응용**: $y = x - \eta\nabla f(x)$로 놓으면 이 부등식은 경사하강법의 학습률 상한 $\eta < 1/L$을 도출한다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 테일러 다항식 시각화 (차수별 근사)
# ─────────────────────────────────────────────

x = sp.Symbol('x')
f_sym = sp.exp(x)
a = 0  # 전개점

# 차수별 테일러 다항식 생성
degrees = [1, 2, 4, 6, 8]
x_range = np.linspace(-3, 3, 300)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 좌: exp(x)의 테일러 근사
axes[0].plot(x_range, np.exp(x_range), 'k-', linewidth=2.5, label=r'$f(x)=e^x$', zorder=5)

colors = plt.cm.viridis(np.linspace(0, 0.9, len(degrees)))
for i, n in enumerate(degrees):
    P_n = sp.series(f_sym, x, a, n + 1).removeO()
    P_n_func = sp.lambdify(x, P_n, 'numpy')
    y_approx = P_n_func(x_range)
    # 발산하는 부분 clipping
    y_approx = np.clip(y_approx, -20, 30)
    axes[0].plot(x_range, y_approx, '--', color=colors[i], linewidth=1.3,
                 label=f'$P_{n}(x)$', alpha=0.85)

axes[0].set_ylim(-5, 20)
axes[0].set_xlim(-3, 3)
axes[0].set_title(r'$e^x$의 테일러 다항식 (점: $a=0$)', fontsize=12)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 우: 여분항 오차 (log scale)
x_demo = 2.0
true_val = np.exp(x_demo)

ns = list(range(1, 16))
errors = []
for n in ns:
    P_n = sp.series(f_sym, x, a, n + 1).removeO()
    P_n_func = sp.lambdify(x, P_n, 'numpy')
    err = abs(P_n_func(x_demo) - true_val)
    errors.append(err)

axes[1].semilogy(ns, errors, 'ro-', linewidth=2, markersize=6)
axes[1].set_xlabel('테일러 다항식 차수 n')
axes[1].set_ylabel('오차 |P_n(2) - e²|')
axes[1].set_title(r'$x=2$에서의 여분항 오차 감소', fontsize=12)
axes[1].grid(True, alpha=0.3, which='both')
axes[1].annotate(f'e² = {true_val:.6f}', xy=(8, errors[7]),
                 fontsize=10, color='blue')

plt.tight_layout()
plt.savefig('04-taylor-approximation.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. MVT 시각화
# ─────────────────────────────────────────────

def visualize_mvt(f, f_prime, a, b, f_label='f'):
    x_plot = np.linspace(a - 0.2, b + 0.2, 300)
    
    # MVT의 c 수치적 탐색
    avg_slope = (f(b) - f(a)) / (b - a)
    x_search = np.linspace(a + 1e-6, b - 1e-6, 10000)
    c_idx = np.argmin(np.abs(f_prime(x_search) - avg_slope))
    c = x_search[c_idx]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_plot, f(x_plot), 'b-', linewidth=2, label=f'$f(x) = {f_label}$')
    
    # 할선 (secant line)
    secant = f(a) + avg_slope * (x_plot - a)
    ax.plot(x_plot, secant, 'r--', linewidth=1.5, label=f'할선: 기울기={avg_slope:.3f}')
    
    # 접선 (tangent at c)
    tangent = f(c) + avg_slope * (x_plot - c)
    ax.plot(x_plot, tangent, 'g-', linewidth=1.5, label=f'접선 at c={c:.3f}: 기울기={f_prime(c):.3f}')
    
    ax.scatter([a, b], [f(a), f(b)], color='red', s=80, zorder=5)
    ax.scatter([c], [f(c)], color='green', s=120, zorder=5, marker='*',
               label=f'MVT 보장점 c={c:.3f}')
    ax.axvline(x=c, color='green', linestyle=':', alpha=0.5)
    
    ax.set_title(f'평균값 정리 (MVT): [{a}, {b}]', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('04-mvt-visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_mvt(np.sin, np.cos, 0.0, 2.5, f_label=r'\sin(x)')

# ─────────────────────────────────────────────
# 3. L-smooth 조건 검증
# ─────────────────────────────────────────────

def verify_l_smooth(f, grad_f, L, x_range, n_samples=3000, name="f"):
    """
    L-smooth 조건: f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + L/2 ‖y-x‖²
    를 수치적으로 검증
    """
    x_vals = np.random.uniform(*x_range, n_samples)
    y_vals = np.random.uniform(*x_range, n_samples)
    
    lhs = f(y_vals)
    rhs = f(x_vals) + grad_f(x_vals) * (y_vals - x_vals) + (L / 2) * (y_vals - x_vals)**2
    
    violations = (lhs > rhs + 1e-10).mean()
    max_excess = (lhs - rhs).max()
    
    print(f"{name} (L={L}): 위반 비율={violations:.4f}, 최대 초과={max_excess:.4e}")

print("[L-smooth 조건 검증]")
# f(x) = x²/2, 헤시안 = 1이므로 L = 1
verify_l_smooth(lambda x: x**2/2, lambda x: x, L=1.0, x_range=(-3, 3), name="x²/2 (L=1 기대)")
verify_l_smooth(lambda x: x**2/2, lambda x: x, L=0.5, x_range=(-3, 3), name="x²/2 (L=0.5, 너무 작음)")

# f(x) = sin(x): |f''(x)| = |sin(x)| ≤ 1이므로 L = 1
verify_l_smooth(np.sin, np.cos, L=1.0, x_range=(-np.pi, np.pi), name="sin(x) (L=1 기대)")
```

---

## 🔗 AI/ML 연결

### 경사하강법 감소 보장

정리 4.4의 $L$-smooth 부등식에 $y = x - \eta\nabla f(x)$를 대입하면:

$$f(x - \eta\nabla f) \leq f(x) - \eta\|\nabla f\|^2 + \frac{L\eta^2}{2}\|\nabla f\|^2 = f(x) - \eta\left(1 - \frac{L\eta}{2}\right)\|\nabla f\|^2$$

$\eta < 2/L$이면 우변의 괄호 안이 양수 → **한 스텝마다 반드시 손실이 감소**한다. (Chapter 4에서 수렴률 $O(1/k)$까지 증명)

### 뉴턴 방법의 2차 테일러 근사 최소화

점 $x_k$에서 2차 테일러 근사:

$$f(x) \approx f(x_k) + \nabla f(x_k)^\top (x - x_k) + \frac{1}{2}(x-x_k)^\top H_k (x-x_k)$$

이를 $x$에 대해 최소화하면 ($\nabla_x = 0$):

$$x_{k+1} = x_k - H_k^{-1} \nabla f(x_k)$$

이것이 뉴턴 방법의 업데이트 식이다. 2차 수렴의 근거: 3차 여분항 $O(\|x-x_k\|^3)$이 수렴 속도를 결정한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $f \in C^{n+1}$ | 비연속 미분이 존재하면 테일러 정리 성립 안 함 |
| Lagrange 여분항 $c$의 존재 | $c$의 정확한 값은 알 수 없다 (존재성만 보장) |
| 유한 수렴 반경 | 일부 함수 ($\sum x^n$)는 $|x| < 1$에서만 테일러 급수가 수렴 |
| L-smooth 가정 | ReLU는 $C^1$이 아니므로 엄밀히는 $L$-smooth 정의 수정 필요 |

---

## 📌 핵심 정리

$$f(x) = \underbrace{f(a) + f'(a)(x-a) + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n}_{P_n(x;\,a)} + \underbrace{\frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}}_{R_n(x)}$$

| 정리 | 핵심 보장 |
|------|-----------|
| Rolle | $f(a)=f(b)$이면 $f'(c)=0$인 $c \in (a,b)$ 존재 |
| MVT | 평균 변화율 = 어느 순간의 변화율 |
| 테일러 | 함수를 다항식으로 $O(h^{n+1})$ 정확도로 근사 |
| $L$-smooth | $f(y) \leq f(x) + \nabla f \cdot (y-x) + \frac{L}{2}\|y-x\|^2$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(x) = x^3$에 대해 구간 $[-1, 1]$에서 MVT의 $c$를 계산하라.

<details>
<summary>해설</summary>

$f(-1) = -1$, $f(1) = 1$. 평균 기울기 $= \frac{1 - (-1)}{1 - (-1)} = 1$.  
$f'(x) = 3x^2 = 1 \Rightarrow x = \pm 1/\sqrt{3} \approx \pm 0.577$.  
$c = 1/\sqrt{3} \in (-1, 1)$.

</details>

**문제 2** (심화): $e^x$의 테일러 급수가 $\mathbb{R}$ 전체에서 수렴함을 Lagrange 여분항으로 증명하라.

<details>
<summary>해설</summary>

$|R_n(x)| = \left|\frac{e^c}{(n+1)!}x^{n+1}\right| \leq \frac{e^{|x|}|x|^{n+1}}{(n+1)!}$.  
$n \to \infty$이면 $\frac{|x|^{n+1}}{(n+1)!} \to 0$ (factorial이 지수보다 빠르게 증가).  
따라서 $|R_n(x)| \to 0$이므로 $\mathbb{R}$ 전체에서 수렴. $\square$

</details>

**문제 3** (AI 연결): 학습률 $\eta = 0.1$, $L = 1$ ($L$-smooth)인 손실 함수에서, 경사하강법 한 스텝 후 손실 감소량의 하한을 $\|\nabla f(x)\|$로 표현하라.

<details>
<summary>해설</summary>

$f(x - \eta\nabla f) \leq f(x) - \eta(1 - \frac{L\eta}{2})\|\nabla f\|^2 = f(x) - 0.1(1 - 0.05)\|\nabla f\|^2 = f(x) - 0.095\|\nabla f\|^2$.  
즉 한 스텝에 최소 $0.095\|\nabla f(x)\|^2$만큼 감소가 보장된다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. 미분의 정의와 선형근사](./03-derivative-linear-approx.md) | [📚 README](../README.md) | [05. 미분가능 함수의 성질과 반례 ▶](./05-differentiability-properties.md) |

</div>
