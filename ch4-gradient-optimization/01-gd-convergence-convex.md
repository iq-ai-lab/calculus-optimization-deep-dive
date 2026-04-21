# 01. 경사하강법 수렴 — 볼록·매끄러운 경우

## 🎯 핵심 질문

- $L$-smooth 가정 하에서 GD의 수렴률이 정확히 $O(1/k)$인 이유는?
- 강볼록(strongly convex) 경우의 선형 수렴 $O((1-\mu/L)^k)$는 어떻게 증명하는가?
- 각 가정(볼록성, 매끄러움)이 깨지면 수렴이 어떻게 실패하는가?
- 실제 딥러닝 손실 함수는 이 조건들을 만족하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

GD 수렴 이론은 학습률 설정과 수렴 속도 예측의 이론적 기반이다:
- "$k$번 학습 후 얼마나 좋아지는가"를 정량화: $f(x_k) - f^* \leq O(1/k)$
- 학습률 상한 $\eta \leq 1/L$의 이론적 근거
- 강볼록 손실 (Logistic Regression 등)에서 선형 수렴 보장

---

## 📐 수학적 선행 조건

- [Ch3-01. 다변수 테일러 정리](../ch3-taylor-quadratic/01-multivariate-taylor.md) — $L$-smooth 조건
- [Ch3-03. 볼록성 판정](../ch3-taylor-quadratic/03-saddle-points-convexity.md) — 볼록 함수의 정의

---

## ✏️ 정의와 핵심 도구

### 정의 4.1 — $L$-smooth와 $\mu$-강볼록

$f \in C^1$이 **$L$-smooth**: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$

$f \in C^1$이 **$\mu$-강볼록**: $f(y) \geq f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2$

헤시안과의 관계: $f \in C^2$이면
$$\mu I \preceq H_f(x) \preceq L I \quad \forall x$$

---

## 🔬 정리와 증명

### 보조정리 4.1 — Descent Lemma

**명제**: $f$가 $L$-smooth이면:
$$f(x - \eta\nabla f(x)) \leq f(x) - \eta\left(1 - \frac{\eta L}{2}\right)\|\nabla f(x)\|^2$$

**증명**: Ch1-04 정리 4.4의 결과. $y = x - \eta\nabla f(x)$로 놓으면:
$$f(y) \leq f(x) + \nabla f(x)^\top(y-x) + \frac{L}{2}\|y-x\|^2$$
$= f(x) - \eta\|\nabla f\|^2 + \frac{\eta^2 L}{2}\|\nabla f\|^2 = f(x) - \eta(1 - \frac{\eta L}{2})\|\nabla f\|^2 \quad \square$

$\eta \leq 1/L$이면 항상 $f(x_{k+1}) \leq f(x_k)$ (단조 감소).

---

### 정리 4.1 — 볼록 $L$-smooth에서 $O(1/k)$ 수렴

**가정**: $f$ 볼록, $L$-smooth, 최솟값 $x^*$ 존재. $\eta = 1/L$.

**명제**: $f(x_k) - f^* \leq \frac{L\|x_0 - x^*\|^2}{2k}$

**증명**:  
볼록성으로부터: $f(x) - f^* \leq \nabla f(x)^\top (x - x^*)$ (1차 특성)

$x_{k+1} = x_k - \eta\nabla f(x_k)$에서:

$$\|x_{k+1} - x^*\|^2 = \|x_k - \eta\nabla f(x_k) - x^*\|^2$$
$$= \|x_k - x^*\|^2 - 2\eta\nabla f(x_k)^\top(x_k - x^*) + \eta^2\|\nabla f(x_k)\|^2$$

볼록성: $\nabla f(x_k)^\top(x_k - x^*) \geq f(x_k) - f^*$. 따라서:

$$\|x_{k+1}-x^*\|^2 \leq \|x_k-x^*\|^2 - 2\eta(f(x_k)-f^*) + \eta^2\|\nabla f\|^2$$

Descent Lemma ($\eta = 1/L$): $\eta^2\|\nabla f\|^2 \leq 2\eta(f(x_k)-f(x_{k+1}))$을 대입:

$$f(x_k) - f^* \leq \frac{\|x_k - x^*\|^2 - \|x_{k+1}-x^*\|^2}{2\eta}$$

$k=0$부터 $k-1$까지 합산 (telescoping):

$$\sum_{t=0}^{k-1}(f(x_t) - f^*) \leq \frac{\|x_0-x^*\|^2}{2\eta}$$

GD는 단조 감소이므로 $f(x_k) \leq f(x_t)$ for $t \leq k$:

$$k \cdot (f(x_k) - f^*) \leq \sum_{t=0}^{k-1}(f(x_t)-f^*) \leq \frac{L\|x_0-x^*\|^2}{2}$$

따라서 $f(x_k) - f^* \leq \frac{L\|x_0-x^*\|^2}{2k} = O(1/k)$. $\square$

---

### 정리 4.2 — $\mu$-강볼록 $L$-smooth에서 선형 수렴

**가정**: $f$가 $\mu$-강볼록, $L$-smooth. $\eta = 2/(\mu+L)$.

**명제**: $\|x_k - x^*\|^2 \leq \left(\frac{\kappa-1}{\kappa+1}\right)^{2k}\|x_0-x^*\|^2$ ($\kappa = L/\mu$)

**증명 스케치**:  
$P_L(x) = x - \frac{1}{L}\nabla f(x)$ (GD 스텝 사상)으로 놓으면:

강볼록과 $L$-smooth 조건으로부터 $P_L$이 수축 사상임을 보일 수 있다:
$$\|P_L(x) - x^*\| \leq \frac{\kappa-1}{\kappa+1}\|x-x^*\|$$

반복 적용하면 $\|x_k - x^*\| \leq \left(\frac{\kappa-1}{\kappa+1}\right)^k\|x_0-x^*\|$. $\square$

---

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 볼록 L-smooth: O(1/k) 수렴 검증
# ─────────────────────────────────────────────

# f(x) = ||Ax - b||² / 2 (볼록, L-smooth)
np.random.seed(42)
n = 20
A = np.random.randn(n, n)
b = np.random.randn(n)
H = A.T @ A  # 헤시안 = AᵀA (PSD)
c = -A.T @ b

# f(x) = x'Hx/2 + c'x + const
L = np.linalg.eigvalsh(H).max()
x_star = np.linalg.solve(H, -c)
f_star = 0.5*x_star@H@x_star + c@x_star

def f(x): return 0.5*x@H@x + c@x
def grad(x): return H@x + c

# GD with η = 1/L
x = np.zeros(n)
gaps = []
for k in range(1, 500):
    x -= (1/L) * grad(x)
    gaps.append(f(x) - f_star)

# 이론적 상한: L*||x0-x*||² / (2k)
R0 = np.linalg.norm(x_star)**2  # ||x0-x*||² (x0=0)
k_vals = np.arange(1, 500)
upper = L * R0 / (2 * k_vals)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].loglog(k_vals, gaps, 'b-', linewidth=2, label='GD 실제 수렴')
axes[0].loglog(k_vals, upper, 'r--', linewidth=1.5, label=r'이론 상한 $O(1/k)$')
axes[0].set_xlabel('반복 횟수 k'); axes[0].set_ylabel('f(x_k) - f*')
axes[0].set_title(r'볼록 L-smooth GD: $O(1/k)$ 수렴', fontsize=11)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, which='both')

# ─────────────────────────────────────────────
# 2. 강볼록: 선형 수렴 검증
# ─────────────────────────────────────────────

# L-smooth 조건이 깨지는 반례 (학습률이 너무 클 때)
mu_val = 1.0; L_val = 20.0
kappa = L_val / mu_val
H_sc = np.diag(np.linspace(mu_val, L_val, n))

def f_sc(x): return 0.5*x@H_sc@x
def grad_sc(x): return H_sc@x

rho = ((kappa-1)/(kappa+1))**2
x_opt = 2/( mu_val + L_val )  # 최적 학습률

x = np.ones(n)
gaps_sc = []
for _ in range(200):
    x -= x_opt * grad_sc(x)
    gaps_sc.append(f_sc(x))

k_sc = np.arange(1, 201)
upper_sc = f_sc(np.ones(n)) * rho**k_sc

axes[1].semilogy(k_sc, gaps_sc, 'b-', linewidth=2, label='GD 실제')
axes[1].semilogy(k_sc, upper_sc, 'r--', linewidth=1.5,
                 label=f'이론: ρ={(rho):.3f}^k')
axes[1].set_xlabel('반복 횟수 k'); axes[1].set_ylabel('f(x_k)')
axes[1].set_title(f'강볼록 선형 수렴 (κ={kappa:.0f}, ρ={rho:.3f})', fontsize=11)
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('ch4-01-gd-convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. L-smooth 위반 시 발산
# ─────────────────────────────────────────────

x0 = np.ones(n)
for eta_frac, label, color in [(0.5,'η=1/L (안정)','blue'),
                                 (0.9,'η=1.8/L (진동)','orange'),
                                 (1.1,'η=2.2/L (발산)','red')]:
    x = x0.copy()
    vals = [f(x)]
    for _ in range(80):
        x -= (eta_frac * 2 / L) * grad(x)
        vals.append(min(f(x), 1e6))
    axes[0].figure.axes[-1] if False else None

fig2, ax2 = plt.subplots(figsize=(8, 5))
for eta_frac, label, color in [(0.5,'η=0.5/L (안정)','blue'),
                                 (0.9,'η=0.9/L (수렴)','green'),
                                 (1.1,'η=1.1/L (발산!)','red')]:
    x = np.ones(n)
    vals = []
    stable = True
    for _ in range(80):
        x -= (eta_frac / L) * grad(x)
        fv = f(x)
        vals.append(min(fv, 1e6) if stable else 1e6)
        if fv > 1e5: stable = False
    ax2.semilogy(range(80), np.clip(vals, 1e-10, 1e6), color=color, linewidth=2, label=label)

ax2.set_xlabel('반복 횟수'); ax2.set_ylabel('f(x_k)')
ax2.set_title('학습률 선택: 수렴 vs 발산 (η < 2/L 조건)', fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('ch4-01-learning-rate-convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔗 AI/ML 연결

### 비볼록 딥러닝에서 이 정리의 의미

딥러닝 손실은 볼록이 아니다. 하지만:
- **국소 볼록 근사**: 충분히 좁은 근방에서 2차 테일러 근사는 볼록 (H PD이면)
- **PL 조건(Polyak-Łojasiewicz)**: 볼록성보다 약한 $\|\nabla f\|^2 \geq 2\mu(f-f^*)$ 가정으로도 선형 수렴 증명 가능
- **실용적 적용**: Adam + Cosine LR Schedule의 경험적 조율은 수렴 이론의 실용적 구현

---

## 📌 핵심 정리

| 조건 | 수렴률 | 정확한 의미 |
|------|--------|-----------|
| 볼록 + $L$-smooth | $O(1/k)$ | $k$번 후 $\varepsilon$ 근사에 $O(1/\varepsilon)$ 반복 |
| $\mu$-강볼록 + $L$-smooth | $O(\rho^k)$, $\rho < 1$ | 기하급수적 수렴, $O(\log(1/\varepsilon))$ 반복 |

**수렴 조건**: $\eta < 2/L$ (필수), 최적 $\eta = 1/L$ (볼록) 또는 $2/(\mu+L)$ (강볼록)

---

## 🤔 생각해볼 문제

**문제 1**: $f(x) = \frac{1}{2}x^2$에서 $\eta = 0.5$, $\eta = 1.5$, $\eta = 2.5$로 GD를 1차원에서 직접 시뮬레이션하고, $\eta < 2/L = 2$의 경계를 확인하라.

**문제 2**: PL 조건 $\|\nabla f(x)\|^2 \geq 2\mu(f(x)-f^*)$가 성립하면 $\eta = 1/L$의 GD가 선형 수렴함을 증명하라.

<details><summary>해설</summary>
Descent Lemma: $f(x_{k+1}) \leq f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2$. PL 조건 대입: $f(x_{k+1}) \leq f(x_k) - \frac{\mu}{L}(f(x_k)-f^*)$. 따라서 $f(x_{k+1})-f^* \leq (1-\mu/L)(f(x_k)-f^*)$. 선형 수렴.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch3-04. 조건수와 최적화 속도](../ch3-taylor-quadratic/04-condition-number-optimization.md) | [📚 README](../README.md) | [02. 학습률의 역할 ▶](./02-learning-rate-analysis.md) |

</div>
