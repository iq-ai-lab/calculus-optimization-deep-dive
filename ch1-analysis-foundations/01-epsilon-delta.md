# 01. ε-δ 언어 — 극한의 엄밀한 정의

## 🎯 핵심 질문

- "가까워진다"는 직관적 표현이 왜 수학적으로 부족한가?
- ε-δ 정의는 무엇을 추가로 보장하는가?
- 수열 극한과 함수 극한은 어떻게 연결되는가?
- 발산 함수와 수렴 함수를 ε-δ로 어떻게 구분하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

딥러닝의 거의 모든 이론적 보장은 극한을 토대로 한다.

- **경사하강법 수렴 분석**: "$k \to \infty$일 때 $f(x_k) \to f^*$"라는 주장은 수열 극한의 엄밀한 정의 없이는 성립하지 않는다
- **역전파의 수치 미분 근사**: 유한차분 $\frac{f(x+h) - f(x)}{h}$가 $h \to 0$ 극한에서 도함수와 일치한다는 보장이 필요하다
- **학습률 스케줄링**: $\eta_k \to 0$이 수렴에 필요한 조건이라는 주장은 수열 극한에 대한 이해를 전제한다

"충분히 학습시키면 수렴한다"는 말은 정확히 ε-δ 언어로 번역하면: "임의의 $\varepsilon > 0$에 대해 $N$이 존재해서 $k > N$이면 $f(x_k) - f^* < \varepsilon$이다"가 된다.

---

## 📐 수학적 선행 조건

- 실수의 완비성(completedness): 공집합이 아닌 위로 유계인 실수 집합은 상한(supremum)을 가진다
- 절댓값과 거리: $|a - b|$는 수직선 위의 두 점 사이의 거리

> 선형대수적 선행 지식은 이 챕터에서 불필요합니다.  
> → [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)는 Chapter 2 이후에 필요합니다.

---

## 📖 직관적 이해

### "가까워진다"는 왜 부족한가

$f(x) = \sin(1/x)$를 생각해 보자. $x$를 0에 가까이 할수록 $f(x)$는 $[-1, 1]$ 전체를 진동한다. "가까워진다"는 표현으로는 이 함수가 $x \to 0$에서 극한이 없다는 사실을 엄밀하게 표현할 수 없다.

핵심 문제: **"얼마나 가까워야 가까운가?"**

ε-δ 언어는 이 질문에 정량적으로 답한다.

> **비유**: 궁수가 과녁의 중심에서 ε 이내로 맞히려면, 활을 쏘는 각도를 δ 이내로 조정하면 충분하다는 보장이 ε-δ 정의다. 이 보장이 **모든** ε > 0에 대해 성립할 때 "수렴한다"고 한다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 함수의 극한 (ε-δ 정의)

$f: D \to \mathbb{R}$이 $D \subseteq \mathbb{R}$에서 정의되고, $a$가 $D$의 집적점(accumulation point)이라 하자. 다음이 성립할 때 $\lim_{x \to a} f(x) = L$이라 한다:

$$\forall \varepsilon > 0,\; \exists \delta > 0 \text{ s.t. } 0 < |x - a| < \delta \Rightarrow |f(x) - L| < \varepsilon$$

**각 기호의 의미**:
- $\varepsilon$ (엡실론): 출력값의 허용 오차. "얼마나 정확한 답을 원하는가"
- $\delta$ (델타): 입력값의 허용 범위. "입력을 얼마나 좁혀야 하는가"
- $0 < |x - a| < \delta$: $x$는 $a$와 다르되 $\delta$보다 가까움 ($x = a$ 자체는 제외)
- $|f(x) - L| < \varepsilon$: 출력이 $L$에서 $\varepsilon$ 이내

### 정의 1.2 — 수열의 극한

수열 $(a_n)_{n=1}^\infty$에 대해 다음이 성립할 때 $\lim_{n \to \infty} a_n = L$이라 한다:

$$\forall \varepsilon > 0,\; \exists N \in \mathbb{N} \text{ s.t. } n > N \Rightarrow |a_n - L| < \varepsilon$$

### 정의 1.3 — 극한의 부재

$\lim_{x \to a} f(x)$가 **존재하지 않는다**는 것은:

$$\forall L \in \mathbb{R},\; \exists \varepsilon > 0 \text{ s.t. } \forall \delta > 0,\; \exists x \text{ with } 0 < |x-a| < \delta \text{ and } |f(x) - L| \geq \varepsilon$$

---

## 🔬 정리와 증명

### 정리 1.1 — 극한의 유일성

**명제**: $\lim_{x \to a} f(x) = L$이고 $\lim_{x \to a} f(x) = M$이면 $L = M$이다.

**증명**:  
$L \neq M$이라 가정하고 $\varepsilon = |L - M|/2 > 0$으로 놓자.

$\lim_{x \to a} f(x) = L$이므로, 이 $\varepsilon$에 대해 $\delta_1 > 0$이 존재하여:
$$0 < |x - a| < \delta_1 \Rightarrow |f(x) - L| < \varepsilon$$

$\lim_{x \to a} f(x) = M$이므로, 이 $\varepsilon$에 대해 $\delta_2 > 0$이 존재하여:
$$0 < |x - a| < \delta_2 \Rightarrow |f(x) - M| < \varepsilon$$

$\delta = \min(\delta_1, \delta_2)$로 놓으면, $0 < |x - a| < \delta$인 $x$에 대해:

$$|L - M| \leq |L - f(x)| + |f(x) - M| < \varepsilon + \varepsilon = |L - M|$$

이는 $|L - M| < |L - M|$이라는 모순이다. 따라서 $L = M$. $\square$

---

### 정리 1.2 — 수열 판정법 (Heine의 정리)

**명제**: $\lim_{x \to a} f(x) = L$인 것은, $a$로 수렴하는 임의의 수열 $(x_n)$ ($x_n \neq a$)에 대해 $\lim_{n \to \infty} f(x_n) = L$인 것과 동치이다.

**증명 (⇒ 방향)**:  
$\lim_{x \to a} f(x) = L$이라 하자. 임의의 $\varepsilon > 0$에 대해 $\delta > 0$이 존재하여 $0 < |x - a| < \delta$이면 $|f(x) - L| < \varepsilon$.

$x_n \to a$ ($x_n \neq a$)인 수열이 있으면, 이 $\delta$에 대해 $N$이 존재하여 $n > N$이면 $|x_n - a| < \delta$.  
따라서 $n > N$이면 $|f(x_n) - L| < \varepsilon$. 즉 $f(x_n) \to L$. $\square$

**증명 (⇐ 방향, 대우)**:  
$\lim_{x \to a} f(x) \neq L$이라 가정하면, $\exists \varepsilon_0 > 0$ s.t. $\forall \delta > 0$, $0 < |x - a| < \delta$이고 $|f(x) - L| \geq \varepsilon_0$인 $x$가 존재한다.

각 $n$에 대해 $\delta = 1/n$으로 놓으면, $0 < |x_n - a| < 1/n$이고 $|f(x_n) - L| \geq \varepsilon_0$인 $x_n$이 존재한다.  
이 수열은 $x_n \to a$이지만 $f(x_n) \not\to L$. 대우 성립. $\square$

> **응용**: $\lim_{x \to 0} \sin(1/x)$가 존재하지 않음을 증명하려면, $x_n = 1/(2\pi n) \to 0$이면 $\sin(1/x_n) = 0$이고, $y_n = 1/(\pi/2 + 2\pi n) \to 0$이면 $\sin(1/y_n) = 1$이므로, 두 수열이 다른 극한값을 준다.

---

### 증명 예제 5개

**예제 1**: $\lim_{x \to 3} (2x + 1) = 7$

**증명**: $|f(x) - 7| = |2x + 1 - 7| = |2x - 6| = 2|x - 3|$.  
임의의 $\varepsilon > 0$에 대해 $\delta = \varepsilon/2$로 놓으면:
$$0 < |x - 3| < \delta \Rightarrow |2x + 1 - 7| = 2|x - 3| < 2\delta = \varepsilon \quad \square$$

**예제 2**: $\lim_{x \to 2} x^2 = 4$

**증명**: $|x^2 - 4| = |x-2||x+2|$.  
$|x - 2| < 1$이면 $|x| < 3$이므로 $|x + 2| < 5$.  
$\delta = \min(1, \varepsilon/5)$으로 놓으면:
$$|x - 2| < \delta \Rightarrow |x^2 - 4| = |x-2||x+2| < \frac{\varepsilon}{5} \cdot 5 = \varepsilon \quad \square$$

**예제 3**: $\lim_{x \to 0} \frac{\sin x}{x} = 1$

**증명 스케치**: 기하학적 부등식 $\cos x \leq \frac{\sin x}{x} \leq 1$ ($0 < x < \pi/2$)을 이용한다.  
$x \to 0$일 때 $\cos x \to 1$이므로 샌드위치 정리(Squeeze Theorem)에 의해 극한은 1이다.  
*(샌드위치 정리 증명은 아래 생략 — ε-δ 직접 적용)*

**예제 4**: $\lim_{n \to \infty} \frac{1}{n} = 0$ (수열 극한)

**증명**: 임의의 $\varepsilon > 0$에 대해 아르키메데스 성질에 의해 $N > 1/\varepsilon$인 자연수 $N$이 존재한다.  
$n > N$이면 $\left|\frac{1}{n} - 0\right| = \frac{1}{n} < \frac{1}{N} < \varepsilon$. $\square$

**예제 5**: $\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = 2$

**증명**: $x \neq 1$이면 $\frac{x^2 - 1}{x - 1} = x + 1$.  
$\left|\frac{x^2-1}{x-1} - 2\right| = |x + 1 - 2| = |x - 1|$.  
$\delta = \varepsilon$으로 놓으면 $0 < |x - 1| < \delta \Rightarrow |f(x) - 2| < \varepsilon$. $\square$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ─────────────────────────────────────────────
# 1. ε-δ 조건을 수치적으로 확인하는 실험
# ─────────────────────────────────────────────

def check_epsilon_delta(f, a, L, epsilon, delta, n_samples=10000):
    """
    0 < |x - a| < delta 범위에서 샘플링하여
    |f(x) - L| < epsilon 조건이 몇 %나 만족되는지 확인
    """
    # a를 제외한 (a-delta, a+delta) 범위에서 균등 샘플링
    x_samples = np.random.uniform(a - delta, a + delta, n_samples)
    x_samples = x_samples[np.abs(x_samples - a) > 1e-10]  # x ≠ a 제거
    
    f_vals = f(x_samples)
    satisfied = np.abs(f_vals - L) < epsilon
    ratio = satisfied.mean()
    
    print(f"ε = {epsilon:.4f}, δ = {delta:.4f}")
    print(f"조건 만족 비율: {ratio:.4f} ({ratio*100:.1f}%)")
    return ratio

# 예제 1: f(x) = 2x + 1, a = 3, L = 7
print("=" * 50)
print("예제 1: lim_{x→3} (2x + 1) = 7")
f1 = lambda x: 2 * x + 1
for eps in [0.1, 0.01, 0.001]:
    delta = eps / 2  # 증명에서 δ = ε/2
    check_epsilon_delta(f1, a=3, L=7, epsilon=eps, delta=delta)
print()

# 예제 2: f(x) = sin(x)/x, a = 0, L = 1
print("=" * 50)
print("예제 2: lim_{x→0} sin(x)/x = 1")
f2 = lambda x: np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / x)
for eps in [0.1, 0.01, 0.001]:
    delta = np.sqrt(6 * eps)  # 근사적 δ
    check_epsilon_delta(f2, a=0, L=1, epsilon=eps, delta=delta)

# ─────────────────────────────────────────────
# 2. sin(1/x)의 극한 불존재 시각화
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 좌: sin(1/x) — 극한 없음
x_pos = np.linspace(0.001, 0.5, 5000)
axes[0].plot(x_pos, np.sin(1 / x_pos), 'b-', linewidth=0.5, alpha=0.8)
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].set_xlim(0, 0.5)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_title(r'$f(x) = \sin(1/x)$: 극한 없음', fontsize=13)
axes[0].set_xlabel(r'$x$')
axes[0].set_ylabel(r'$f(x)$')
axes[0].grid(True, alpha=0.3)
axes[0].annotate('x → 0 에서 진동\n극한 불존재', xy=(0.05, 0.8),
                 fontsize=10, color='red',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 우: sin(x)/x — 극한 존재
x_all = np.linspace(-2 * np.pi, 2 * np.pi, 2000)
y_sinc = np.where(np.abs(x_all) < 1e-10, 1.0, np.sin(x_all) / x_all)
axes[1].plot(x_all, y_sinc, 'g-', linewidth=1.5)
axes[1].axhline(y=1, color='r', linestyle='--', linewidth=1, label='극한값 L=1')
axes[1].scatter([0], [1], color='red', s=80, zorder=5, label=r'$\lim_{x\to 0} = 1$')
axes[1].set_title(r'$f(x) = \sin(x)/x$: 극한 존재', fontsize=13)
axes[1].set_xlabel(r'$x$')
axes[1].set_ylabel(r'$f(x)$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01-epsilon-delta-comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. SymPy로 극한을 기호 계산
# ─────────────────────────────────────────────

x = sp.Symbol('x')
n = sp.Symbol('n', positive=True, integer=True)

limits = {
    r"\lim_{x \to 3}(2x+1)":   sp.limit(2*x + 1, x, 3),
    r"\lim_{x \to 0}\sin(x)/x": sp.limit(sp.sin(x)/x, x, 0),
    r"\lim_{x \to 1}(x^2-1)/(x-1)": sp.limit((x**2 - 1)/(x - 1), x, 1),
    r"\lim_{x \to \infty}(1+1/x)^x":  sp.limit((1 + 1/x)**x, x, sp.oo),
    r"\lim_{n \to \infty} 1/n":        sp.limit(1/n, n, sp.oo),
}

print("\n[SymPy 극한 계산 결과]")
for expr, val in limits.items():
    print(f"  {expr} = {val}")
```

**출력 예시**:
```
[SymPy 극한 계산 결과]
  \lim_{x \to 3}(2x+1) = 7
  \lim_{x \to 0}\sin(x)/x = 1
  \lim_{x \to 1}(x^2-1)/(x-1) = 2
  \lim_{x \to \infty}(1+1/x)^x = E
  \lim_{n \to \infty} 1/n = 0
```

---

## 🔗 AI/ML 연결

### 경사하강법 수렴의 수열 극한 표현

SGD의 수렴 정리는 정확히 수열 극한으로 서술된다:

$$\lim_{k \to \infty} \mathbb{E}[f(x_k)] = f^*$$

이는 "임의의 $\varepsilon > 0$에 대해 $N$이 존재하여 $k > N$이면 $\mathbb{E}[f(x_k)] - f^* < \varepsilon$"을 의미한다.

### 역전파에서의 수치 미분과 극한

PyTorch의 `gradcheck` 함수는 다음 수치 미분이 autograd와 일치하는지 확인한다:

$$\frac{f(x + h\cdot e_i) - f(x - h\cdot e_i)}{2h} \xrightarrow{h \to 0} \frac{\partial f}{\partial x_i}$$

이 수렴이 ε-δ 언어로 엄밀히 정당화되지 않으면, 작은 $h$ 값에서 부동소수점 오차가 개입한다.

### 학습률 스케줄링의 조건

SGD 수렴을 위한 Robbins-Monro 조건:

$$\sum_{k=1}^\infty \eta_k = \infty, \quad \sum_{k=1}^\infty \eta_k^2 < \infty$$

첫 번째 조건은 수열의 급수 발산, 두 번째는 수열 급수 수렴이다. 둘 다 ε-δ 정의를 바탕으로 한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $a$가 집적점이어야 한다 | 고립점에서는 극한 정의가 공허하게 성립하거나 무의미하다 |
| 단일 극한값 $L$ | 좌극한 ≠ 우극한이면 (양방향) 극한이 존재하지 않는다 |
| 실수 범위 | 복소수 함수의 극한은 모든 방향에서의 접근을 고려해야 한다 |

**수치적 주의**: $h \to 0$ 극한에서 $h = 10^{-15}$ 정도면 부동소수점 cancellation 오차가 실제 미분 값을 압도한다. 수치 미분의 적절한 $h$ 선택이 필요한 이유다.

---

## 📌 핵심 정리

$$\lim_{x \to a} f(x) = L \iff \forall\varepsilon > 0,\; \exists\delta > 0 \text{ s.t. } 0 < |x-a| < \delta \Rightarrow |f(x)-L| < \varepsilon$$

| 개념 | 핵심 |
|------|------|
| ε | "목표 정확도" — 출력의 허용 오차 |
| δ | "입력 제어량" — ε을 만족시키는 입력 범위 |
| 극한의 유일성 | 두 개의 극한값을 가질 수 없다 |
| 수열 판정법 | 수렴하는 임의의 수열이 같은 값으로 함수값이 수렴 |
| 극한 부재 증명 | 다른 극한값을 주는 두 수열을 찾으면 된다 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\lim_{x \to 4} \sqrt{x} = 2$를 ε-δ로 증명하라.

<details>
<summary>힌트 및 해설</summary>

$|\sqrt{x} - 2| = \frac{|x - 4|}{|\sqrt{x} + 2|}$.  
$|x - 4| < 1$이면 $x > 3$이므로 $\sqrt{x} + 2 > \sqrt{3} + 2 > 3$.  
따라서 $|\sqrt{x} - 2| < \frac{|x-4|}{3}$.  
$\delta = \min(1, 3\varepsilon)$으로 놓으면 증명 완성.

</details>

**문제 2** (심화): $f(x) = \sin(1/x)$의 $x \to 0$ 극한이 존재하지 않음을 Heine 정리를 이용해 증명하라.

<details>
<summary>힌트 및 해설</summary>

$x_n = \frac{1}{2\pi n}$이면 $f(x_n) = 0$이고, $y_n = \frac{1}{\pi/2 + 2\pi n}$이면 $f(y_n) = 1$.  
두 수열 모두 0으로 수렴하지만 함수값의 극한이 다르다.  
Heine 정리의 역(대우)에 의해 극한이 존재하지 않는다.

</details>

**문제 3** (AI 연결): PyTorch의 `gradcheck`이 `h = 1e-6`을 기본값으로 사용하는 이유를 부동소수점 정밀도와 극한의 관점에서 설명하라.

<details>
<summary>힌트 및 해설</summary>

중심 차분법의 오차는 $O(h^2)$이다. $h$가 너무 크면 미분 근사 오차, 너무 작으면 부동소수점 cancellation 오차가 지배한다. 두 오차의 균형점이 float64 기준 $h \approx 10^{-6} \sim 10^{-7}$이다.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. 연속성과 균등연속성 ▶](./02-continuity-uniform.md) |

</div>
