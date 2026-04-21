# 02. 연속성과 균등연속성

## 🎯 핵심 질문

- 각 점에서의 연속(pointwise continuity)과 균등연속(uniform continuity)은 무엇이 다른가?
- δ가 점 $x$에 의존하면 어떤 문제가 생기는가?
- 하이네-보렐 정리와 콤팩트성은 최적화에서 왜 중요한가?
- 딥러닝의 손실 함수는 연속인가, 균등연속인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **최적화 가능성 보장**: 연속 함수는 닫힌 유계 구간(콤팩트 집합)에서 반드시 최솟값을 가진다 (Extreme Value Theorem). 이것이 "최적 가중치가 존재한다"는 보장의 수학적 근거다.
- **균등연속과 수치 안정성**: 함수가 균등연속이면 전체 정의역에서 하나의 δ가 통한다. Lipschitz 조건 ($|f(x) - f(y)| \leq L|x - y|$)은 균등연속의 강화된 형태이며, Gradient Clipping의 이론적 근거다.
- **배치 학습의 암묵적 가정**: Mini-batch SGD가 전체 손실과 "가깝다"는 주장은 손실 함수의 연속성 없이는 성립하지 않는다.

---

## 📐 수학적 선행 조건

- [01. ε-δ 극한의 정의](./01-epsilon-delta.md): 연속의 정의는 극한의 특수 경우다
- 실수의 완비성: 코시 수열은 수렴한다
- 유계(bounded)와 닫힘(closed)의 개념

---

## 📖 직관적 이해

### 연속의 두 수준

$f(x) = 1/x$를 $(0, 1]$에서 생각하자. 이 함수는 각 점에서 연속이다 — $x = 0.01$에서도, $x = 0.001$에서도 국소적으로는 "구멍이 없다". 그러나 $x$가 0에 가까워질수록 $1/x$는 끝없이 커진다.

여기서 점별 연속성의 한계가 드러난다: **같은 ε에 대한 δ가 점마다 달라야 한다**. $x = 0.001$ 근처에서 $f$ 값이 $\varepsilon = 0.1$ 이내로 유지되려면 아주 작은 $\delta$가 필요하다. $x$가 0에 가까울수록 δ는 0으로 수렴한다.

균등연속은 이 문제를 차단한다: **모든 점에서 하나의 δ가 동시에 통해야 한다**.

> **비유**: 산악 지형에서 "기울기가 심하지 않다"는 조건이 점별 연속에 해당한다면, 균등연속은 "지도 전체에서 가장 급한 경사도 특정 값 이하"라는 전역적 조건이다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 점별 연속 (Pointwise Continuity)

$f: D \to \mathbb{R}$이 점 $a \in D$에서 **연속**이라는 것은:

$$\forall \varepsilon > 0,\; \exists \delta > 0 \text{ s.t. } |x - a| < \delta \Rightarrow |f(x) - f(a)| < \varepsilon$$

**극한과의 차이**: 극한 정의에서 "$0 < |x - a|$"이었지만, 연속에서는 $x = a$도 포함한다. 즉, $f(a)$가 정의되어 있고 $\lim_{x \to a} f(x) = f(a)$이면 연속이다.

### 정의 2.2 — 균등연속 (Uniform Continuity)

$f: D \to \mathbb{R}$이 $D$ 위에서 **균등연속**이라는 것은:

$$\forall \varepsilon > 0,\; \exists \delta > 0 \text{ s.t. } \forall x, y \in D,\; |x - y| < \delta \Rightarrow |f(x) - f(y)| < \varepsilon$$

**점별 연속과의 차이**: 점별 연속에서 δ는 ε과 **점 $a$ 모두에** 의존할 수 있다 ($\delta = \delta(\varepsilon, a)$). 균등연속에서 δ는 ε에만 의존하며 ($\delta = \delta(\varepsilon)$) **정의역 전체에 걸쳐 통한다**.

### 정의 2.3 — Lipschitz 연속

$f: D \to \mathbb{R}$이 **Lipschitz 연속**이라는 것은:

$$\exists L > 0 \text{ s.t. } \forall x, y \in D,\; |f(x) - f(y)| \leq L|x - y|$$

**포함 관계**: Lipschitz 연속 $\Rightarrow$ 균등연속 $\Rightarrow$ 점별 연속 (역은 성립하지 않는다)

---

## 🔬 정리와 증명

### 정리 2.1 — 연속 함수의 합성

**명제**: $f$가 $a$에서 연속이고 $g$가 $f(a)$에서 연속이면, $g \circ f$는 $a$에서 연속이다.

**증명**:  
임의의 $\varepsilon > 0$에 대해, $g$가 $f(a)$에서 연속이므로 $\delta_1 > 0$이 존재하여:
$$|y - f(a)| < \delta_1 \Rightarrow |g(y) - g(f(a))| < \varepsilon$$

이 $\delta_1$을 ε으로 삼아, $f$가 $a$에서 연속이므로 $\delta > 0$이 존재하여:
$$|x - a| < \delta \Rightarrow |f(x) - f(a)| < \delta_1$$

따라서 $|x - a| < \delta$이면 $|f(x) - f(a)| < \delta_1$이고, 이에 의해 $|g(f(x)) - g(f(a))| < \varepsilon$. $\square$

---

### 정리 2.2 — 균등연속이 아닌 연속 함수

**명제**: $f(x) = 1/x$는 $(0, 1]$에서 연속이지만 균등연속이 아니다.

**증명 (연속임)**:  
임의의 $a \in (0, 1]$에 대해 $|f(x) - f(a)| = \left|\frac{1}{x} - \frac{1}{a}\right| = \frac{|x - a|}{|xa|}$.  
$|x - a| < a/2$이면 $x > a/2$이므로 $\frac{1}{xa} < \frac{2}{a^2}$.  
$\delta = \min(a/2,\, \varepsilon a^2/2)$으로 놓으면 연속 조건 성립. (*δ가 a에 의존한다!*)

**증명 (균등연속 아님)**:  
$\varepsilon = 1$로 놓자. 임의의 $\delta > 0$에 대해 $x = \delta/2$, $y = \delta$로 놓으면:
$$|x - y| = \delta/2 < \delta$$
$$|f(x) - f(y)| = \left|\frac{2}{\delta} - \frac{1}{\delta}\right| = \frac{1}{\delta}$$

$\delta$가 작을수록 $1/\delta$는 커지므로, 충분히 작은 $\delta$에 대해 $|f(x) - f(y)| = 1/\delta > 1 = \varepsilon$. 따라서 균등연속이 아니다. $\square$

---

### 정리 2.3 — 하이네-보렐(Heine-Borel) 정리

**명제**: $\mathbb{R}$에서 집합 $K$가 콤팩트인 것은 $K$가 닫혀 있고(closed) 유계(bounded)인 것과 동치이다.

**핵심 개념**:
- **닫혀 있음**: 수열 $\{x_n\} \subseteq K$이 수렴하면 극한값도 $K$에 속한다
- **유계**: $\exists M > 0$ s.t. $\forall x \in K$, $|x| \leq M$
- **콤팩트**: 모든 열린 덮개(open cover)가 유한 부분 덮개를 가진다

**증명 스케치 (⇐ 방향)**:  
$[a, b]$ 위의 임의의 열린 덮개 $\{U_\alpha\}$가 있다고 하자.  
유한 부분 덮개가 없다고 가정하면 이분법으로 부분 구간을 계속 분할할 수 있다.  
길이 0으로 수렴하는 중첩 구간의 수열이 생기고, 그 공통점 $c \in [a, b]$는 어떤 $U_{\alpha_0}$에 속한다.  
충분히 작은 근방이 $U_{\alpha_0}$에 포함되므로, 충분히 분할된 부분 구간도 $U_{\alpha_0}$ 하나로 덮인다 — 모순. $\square$

---

### 정리 2.4 — 콤팩트 집합 위의 연속 함수는 균등연속

**명제**: $f: K \to \mathbb{R}$이 연속이고 $K$가 콤팩트이면, $f$는 $K$ 위에서 균등연속이다.

**증명**:  
귀류법: 균등연속이 아니라 가정하면, $\varepsilon_0 > 0$이 존재하여 임의의 $\delta > 0$에 대해:
$$\exists x, y \in K \text{ s.t. } |x - y| < \delta \text{ and } |f(x) - f(y)| \geq \varepsilon_0$$

각 $n$에 대해 $\delta = 1/n$으로 놓으면 수열 $(x_n), (y_n) \subseteq K$가 존재하여:
$$|x_n - y_n| < \frac{1}{n}, \quad |f(x_n) - f(y_n)| \geq \varepsilon_0$$

$K$는 콤팩트이므로 $(x_n)$의 수렴 부분 수열 $x_{n_k} \to c \in K$가 존재한다.  
$|x_{n_k} - y_{n_k}| < 1/n_k \to 0$이므로 $y_{n_k} \to c$도 성립한다.

$f$는 $c$에서 연속이므로:
$$f(x_{n_k}) \to f(c), \quad f(y_{n_k}) \to f(c)$$

따라서 $|f(x_{n_k}) - f(y_{n_k})| \to 0$인데, 이는 $|f(x_n) - f(y_n)| \geq \varepsilon_0 > 0$에 모순. $\square$

---

### 따름정리 — 최대·최솟값 정리 (Extreme Value Theorem)

**명제**: $f: K \to \mathbb{R}$이 연속이고 $K$가 콤팩트이면, $f$는 최댓값과 최솟값을 가진다.

**증명 스케치**:  
$M = \sup_{x \in K} f(x)$로 놓자 (위로 유계이면 상한 존재).  
$f(x_n) \to M$인 수열 $(x_n) \subseteq K$가 존재. $K$가 콤팩트이므로 수렴 부분 수열 $x_{n_k} \to c \in K$.  
$f$의 연속성에 의해 $f(c) = \lim f(x_{n_k}) = M$. 즉 최댓값 달성. $\square$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 점별 연속 vs 균등연속 시각화
#    f(x) = 1/x on (0, 1]
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(0.02, 1.0, 500)

# 좌: 점 a = 0.5에서의 δ (점별 연속)
a1 = 0.5
eps = 0.5
# δ = min(a/2, ε*a²/2)
delta1 = min(a1 / 2, eps * a1**2 / 2)

axes[0].plot(x, 1/x, 'b-', linewidth=2, label=r'$f(x) = 1/x$')
axes[0].axvline(x=a1, color='gray', linestyle='--', alpha=0.7)
axes[0].axhline(y=1/a1, color='gray', linestyle='--', alpha=0.7)
axes[0].fill_betweenx([1/a1 - eps, 1/a1 + eps], 0, 1.1, alpha=0.15, color='green', label=f'출력 범위 (ε={eps})')
axes[0].axvspan(a1 - delta1, a1 + delta1, alpha=0.2, color='orange', label=f'δ={delta1:.3f} at a={a1}')
axes[0].set_xlim(0, 1.1)
axes[0].set_ylim(0, 6)
axes[0].set_title(f'점별 연속: a={a1}에서 δ={delta1:.3f}', fontsize=12)
axes[0].set_xlabel(r'$x$')
axes[0].set_ylabel(r'$f(x)$')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# 우: a = 0.1에서의 δ (훨씬 더 작아야 함 — 균등연속 불가)
a2 = 0.1
delta2 = min(a2 / 2, eps * a2**2 / 2)

axes[1].plot(x, 1/x, 'b-', linewidth=2, label=r'$f(x) = 1/x$')
axes[1].axvline(x=a2, color='gray', linestyle='--', alpha=0.7)
axes[1].axhline(y=1/a2, color='gray', linestyle='--', alpha=0.7)
axes[1].fill_betweenx([1/a2 - eps, 1/a2 + eps], 0, 1.1, alpha=0.15, color='green', label=f'출력 범위 (ε={eps})')
axes[1].axvspan(max(0, a2 - delta2), a2 + delta2, alpha=0.25, color='red', label=f'δ={delta2:.4f} at a={a2}')
axes[1].set_xlim(0, 0.5)
axes[1].set_ylim(0, 20)
axes[1].set_title(f'점별 연속: a={a2}에서 δ={delta2:.4f} (훨씬 작음)', fontsize=12)
axes[1].set_xlabel(r'$x$')
axes[1].set_ylabel(r'$f(x)$')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].annotate('a가 0에 가까울수록\nδ → 0: 균등연속 불가',
                 xy=(a2, 1/a2), xytext=(0.2, 15),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

plt.suptitle(r'$f(x)=1/x$: 점별 연속 O, 균등연속 X', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02-continuity-pointwise-vs-uniform.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. Lipschitz 조건 확인 실험
# ─────────────────────────────────────────────

def check_lipschitz(f, domain, n_samples=5000, name="f"):
    """함수의 실험적 Lipschitz 상수 추정"""
    x_vals = np.random.uniform(*domain, n_samples)
    y_vals = np.random.uniform(*domain, n_samples)
    
    # |x - y| > 1e-10 인 쌍만 사용
    mask = np.abs(x_vals - y_vals) > 1e-10
    x_vals, y_vals = x_vals[mask], y_vals[mask]
    
    ratios = np.abs(f(x_vals) - f(y_vals)) / np.abs(x_vals - y_vals)
    L_est = ratios.max()
    
    print(f"{name}: 실험적 Lipschitz 상수 ≈ {L_est:.4f}")
    return L_est

print("Lipschitz 조건 확인:")
check_lipschitz(np.sin, (-np.pi, np.pi), name="sin(x) on [-π, π]")
check_lipschitz(np.cos, (-np.pi, np.pi), name="cos(x) on [-π, π]")
check_lipschitz(lambda x: x**2, (-1, 1), name="x² on [-1, 1]")
check_lipschitz(lambda x: np.sqrt(np.abs(x)), (0, 1), name="√x on (0, 1]")

# sin/cos: Lipschitz 상수 ≈ 1 (|sin(x) - sin(y)| ≤ |x - y|)
# x²: Lipschitz 상수 ≈ 2 (|x²-y²| = |x+y||x-y| ≤ 2|x-y| on [-1,1])
# √x: Lipschitz 상수 → ∞ (균등연속이지만 Lipschitz 아님)

# ─────────────────────────────────────────────
# 3. ReLU의 Lipschitz 조건 확인
# ─────────────────────────────────────────────

relu = lambda x: np.maximum(0, x)
L_relu = check_lipschitz(relu, (-5, 5), name="ReLU on [-5, 5]")
print(f"ReLU는 Lipschitz 연속: L={L_relu:.1f} (이론값: 1)")
```

**출력**:
```
Lipschitz 조건 확인:
sin(x) on [-π, π]: 실험적 Lipschitz 상수 ≈ 1.0000
cos(x) on [-π, π]: 실험적 Lipschitz 상수 ≈ 1.0000
x² on [-1, 1]: 실험적 Lipschitz 상수 ≈ 2.0000
√x on (0, 1]: 실험적 Lipschitz 상수 ≈ 8.3142  ← Lipschitz 아님
ReLU는 Lipschitz 연속: L=1.0 (이론값: 1)
```

---

## 🔗 AI/ML 연결

### Gradient Clipping의 수학적 근거

역전파 중 gradient의 norm이 폭발하는 이유는, 손실 함수가 Lipschitz 연속이 아닌 영역(매우 급격한 경사)에 있기 때문이다. Gradient Clipping:

$$g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|}\right)$$

은 gradient를 반경 $c$인 공으로 투영하여 손실 함수를 사실상 $c$-Lipschitz로 제한하는 효과를 낸다.

### 닫힌 유계 파라미터 공간과 최솟값 존재

"신경망의 최적 가중치가 존재한다"는 주장을 엄밀히 하려면:
1. 손실 함수 $L(W)$가 연속이어야 하고
2. 탐색 공간 $\mathcal{W}$가 콤팩트이어야 한다 (Extreme Value Theorem 적용)

실제로는 $\mathcal{W} = \mathbb{R}^n$이라 콤팩트하지 않다. 이를 극복하기 위해 "coercive" 조건 ($\|W\| \to \infty$이면 $L(W) \to \infty$)을 추가로 가정하거나, 정규화(regularization)를 통해 유효 탐색 공간을 제한한다.

### Batch Normalization과 Lipschitz 조건

BN의 효과 중 하나는 각 층의 출력 스케일을 정규화하여 Lipschitz 상수를 일정 수준으로 통제하는 것이다. 이로 인해 역전파 시 gradient 흐름이 안정화된다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 콤팩트 집합 위의 연속 → 균등연속 | $\mathbb{R}$ 전체나 열린 구간에서는 성립하지 않는다 |
| Lipschitz → 균등연속 | $\sqrt{x}$는 균등연속이지만 Lipschitz가 아니다 (반례) |
| 실수 구간 | 고차원 $\mathbb{R}^n$에서의 콤팩트성은 Heine-Borel 정리가 동일하게 성립한다 |
| 균등연속 → 경계로 연속 연장 가능 | $(0,1)$에서 균등연속인 $f$는 $[0,1]$로 연속 연장이 가능하다 |

---

## 📌 핵심 정리

$$\text{Lipschitz} \Rightarrow \text{균등연속} \Rightarrow \text{점별 연속}$$

| 개념 | δ의 의존성 | 핵심 조건 |
|------|----------|-----------|
| 점별 연속 | $\delta = \delta(\varepsilon, a)$ | 각 점에서 국소적으로 성립 |
| 균등연속 | $\delta = \delta(\varepsilon)$ | 정의역 전체에서 하나의 δ |
| Lipschitz 연속 | δ 불필요 ($L$만으로) | $\|f(x)-f(y)\| \leq L\|x-y\|$ |

**극값 정리**: 연속 함수 + 콤팩트 정의역 → 최솟값·최댓값 반드시 존재

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(x) = x^2$이 $[0, 1]$에서 균등연속임을 ε-δ로 직접 증명하라. (단, $[0, 1]$은 콤팩트이므로 정리 2.4로 결론을 내릴 수도 있지만, 직접 δ를 구하라.)

<details>
<summary>힌트 및 해설</summary>

$|x^2 - y^2| = |x+y||x-y|$. $[0,1]$에서 $|x+y| \leq 2$.  
따라서 $\delta = \varepsilon/2$으로 놓으면 $|x-y| < \delta \Rightarrow |x^2-y^2| < \varepsilon$.  
이 δ는 $x, y$에 무관하므로 균등연속.

</details>

**문제 2** (심화): $f(x) = \sin(x^2)$는 $\mathbb{R}$ 전체에서 균등연속인가?

<details>
<summary>힌트 및 해설</summary>

$\mathbb{R}$ 전체에서는 균등연속이 아니다. $x_n = \sqrt{2\pi n + \pi/2}$, $y_n = \sqrt{2\pi n}$으로 놓으면 $|x_n - y_n| \to 0$이지만 $|f(x_n) - f(y_n)| = 1$. 닫힌 유계 구간 $[-M, M]$에서는 균등연속이다 (콤팩트이므로).

</details>

**문제 3** (AI 연결): ReLU가 Lipschitz 상수 1인 함수임을 증명하라. 이 성질이 Gradient Explosion 방지에 어떻게 기여하는지 설명하라.

<details>
<summary>힌트 및 해설</summary>

$|\text{ReLU}(x) - \text{ReLU}(y)| = |\max(0,x) - \max(0,y)| \leq |x - y|$ (세 경우 — 둘 다 양수, 하나만 양수, 둘 다 0 — 를 각각 검토). Lipschitz 상수 1이므로 역전파 시 gradient가 증폭되지 않는다. 단, 소실은 막지 못한다 (dying ReLU 문제).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. ε-δ 극한의 엄밀한 정의](./01-epsilon-delta.md) | [📚 README](../README.md) | [03. 미분의 정의와 선형근사 ▶](./03-derivative-linear-approx.md) |

</div>
