# 02. 헤시안의 고유값과 국소 기하

## 🎯 핵심 질문

- 헤시안의 고유값 부호가 극솟값/극댓값/안장점을 어떻게 결정하는가?
- 2차 충분조건을 Spectral Theorem으로 어떻게 증명하는가?
- 딥러닝 Loss Landscape의 "거의 대부분이 안장점"이라는 주장의 수학적 근거는?
- Flat Minima vs Sharp Minima의 차이는 고유값으로 어떻게 표현되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **SGD의 안장점 회피**: 확률적 잡음이 안장점 탈출을 돕는다는 이론 (Ge 2015, Jin 2017)은 헤시안 부정부호에 기반
- **일반화와 Flat Minima**: $\lambda_{\max}(H)$가 작은 해가 더 좋은 일반화를 보인다는 실험적·이론적 증거 (Keskar 2016)
- **학습률과 수렴 속도**: 헤시안 고유값 최댓값 $L$이 학습률 상한을 결정, 최솟값 $\mu$가 수렴 속도 결정

---

## 📐 수학적 선행 조건

- [Ch3-01. 다변수 테일러 정리](./01-multivariate-taylor.md)
- [Ch2-04. 야코비안과 헤시안의 기하](../ch2-multivariable-calculus/04-jacobian-hessian-geometry.md)
- 선형대수: 고유값, 대칭행렬 대각화, Spectral Theorem

---

## 🔬 정리와 증명

### 정리 3.3 — 2차 충분조건 (Second-Order Sufficient Conditions)

**명제**: $f \in C^2$이고 $\nabla f(a) = 0$ (정류점)이라 하자.

1. $H_f(a)$가 **양의 정부호(PD)**: $a$는 엄격한 국소 최솟값
2. $H_f(a)$가 **음의 정부호(ND)**: $a$는 엄격한 국소 최댓값
3. $H_f(a)$가 **부정부호(Indefinite)**: $a$는 안장점

**증명 (1번)**:  
Spectral Theorem에 의해 $H = Q\Lambda Q^\top$, 모든 $\lambda_i > 0$.  
최솟값 $\mu = \min_i \lambda_i > 0$.

임의의 단위벡터 $v = Q^\top h / \|h\|$에 대해:
$$h^\top H h = h^\top Q\Lambda Q^\top h = \|h\|^2 v^\top \Lambda v = \|h\|^2 \sum_i \lambda_i v_i^2 \geq \mu \|h\|^2$$

2차 테일러 전개 ($\nabla f(a) = 0$이므로 1차 항 없음):
$$f(a+h) = f(a) + \frac{1}{2}h^\top H h + o(\|h\|^2) \geq f(a) + \frac{\mu}{2}\|h\|^2 + o(\|h\|^2)$$

충분히 작은 $\|h\|$에서 $o(\|h\|^2)/\|h\|^2 \to 0$이므로 $f(a+h) > f(a)$.  
따라서 $a$는 엄격한 국소 최솟값. $\square$

**증명 (3번 — 안장점)**:  
$\lambda_+ > 0 > \lambda_-$인 고유값이 존재. 대응 고유벡터를 $v_+$, $v_-$라 하면:

방향 $h = tv_+$: $f(a + tv_+) = f(a) + \frac{t^2}{2}\lambda_+ + o(t^2) > f(a)$ (충분히 작은 $t$)  
방향 $h = tv_-$: $f(a + tv_-) = f(a) + \frac{t^2}{2}\lambda_- + o(t^2) < f(a)$ (충분히 작은 $t$)

따라서 $a$는 극소도 극대도 아닌 안장점. $\square$

---

### 정리 3.4 — 필요조건 vs 충분조건

**명제 (필요조건)**: $a$가 국소 최솟값이면 $\nabla f(a) = 0$이고 $H_f(a)$는 PSD.

**주의**: 충분조건 ≠ 필요조건. 반례:
- $f(x) = x^4$: $f''(0) = 0$ (PSD 아님, 경계)이지만 $x=0$은 전역 최솟값
- $f(x,y) = (y-x^2)(y-3x^2)$: $(0,0)$에서 $\nabla f = 0$, $H$ 부정부호이지만 임의의 직선 경로에서는 최솟값

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# 1. 세 가지 정류점 타입 3D 시각화
# ─────────────────────────────────────────────

cases = [
    ("극솟값 (PD): λ=[2,4]",   lambda x, y: x**2 + 2*y**2, (0, 0)),
    ("안장점 (Indef): λ=[2,-2]", lambda x, y: x**2 - y**2,   (0, 0)),
    ("원숭이 안장점",             lambda x, y: x**3 - 3*x*y**2, (0, 0)),
]

fig = plt.figure(figsize=(15, 5))
x_r = np.linspace(-1.5, 1.5, 60)
y_r = np.linspace(-1.5, 1.5, 60)
X, Y = np.meshgrid(x_r, y_r)

for i, (title, func, crit_pt) in enumerate(cases, 1):
    Z = func(X, Y)
    ax = fig.add_subplot(1, 3, i, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.85, linewidth=0)
    ax.scatter(*crit_pt, func(*crit_pt), color='red', s=100, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f')

plt.suptitle('헤시안 고유값과 정류점 유형', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ch3-02-hessian-critical-points.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. 고차원 Loss Landscape에서의 안장점 분포
# (Dauphin 2014 재현: 랜덤 행렬 이론)
# ─────────────────────────────────────────────

def count_saddle_fraction(n_dim, n_trials=200):
    """n차원 가우시안 랜덤 행렬의 고유값 분포"""
    neg_frac = []
    for _ in range(n_trials):
        # 랜덤 대칭 행렬 (GOE)
        A = np.random.randn(n_dim, n_dim)
        H = (A + A.T) / (2 * np.sqrt(n_dim))
        eigvals = np.linalg.eigvalsh(H)
        # 안장점: 양과 음의 고유값 모두 존재
        has_pos = (eigvals > 0).any()
        has_neg = (eigvals < 0).any()
        neg_frac.append(1.0 if (has_pos and has_neg) else 0.0)
    return np.mean(neg_frac)

dims = [2, 5, 10, 20, 50, 100]
saddle_fracs = [count_saddle_fraction(d) for d in dims]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(dims, saddle_fracs, 'ro-', linewidth=2, markersize=8)
axes[0].set_xlabel('차원 n')
axes[0].set_ylabel('안장점 비율')
axes[0].set_title('차원 증가 → 안장점이 지배적', fontsize=11)
axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1.1)

# 특정 차원의 고유값 분포
n = 100
A = np.random.randn(n, n)
H = (A + A.T) / (2 * np.sqrt(n))
eigvals = np.linalg.eigvalsh(H)

axes[1].hist(eigvals, bins=30, color='steelblue', alpha=0.7, density=True)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='λ=0')
x_sc = np.linspace(-2.5, 2.5, 200)
# Wigner semicircle law
sc = np.sqrt(np.maximum(0, 4 - x_sc**2)) / (2 * np.pi)
axes[1].plot(x_sc, sc, 'r-', linewidth=2, label='Wigner 반원 법칙')
axes[1].set_xlabel('고유값 λ')
axes[1].set_ylabel('밀도')
axes[1].set_title(f'n={n}차원 랜덤 헤시안 고유값 분포', fontsize=11)
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch3-02-saddle-point-distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. Flat vs Sharp Minima 비교
# ─────────────────────────────────────────────

x_r = np.linspace(-3, 3, 500)
flat_min   = 0.1 * x_r**2 + 0.05 * x_r**4    # 작은 λ_max
sharp_min  = 2.0 * x_r**2 + 0.5  * x_r**4    # 큰 λ_max

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x_r, flat_min,  'b-', linewidth=2.5, label='Flat Minimum (λ_max 작음, 일반화 좋음)')
ax.plot(x_r, sharp_min, 'r-', linewidth=2.5, label='Sharp Minimum (λ_max 큼, 과적합 위험)')

# 분포 변화 시각화 (학습 ↔ 테스트)
shift = 0.3
ax.plot(x_r, 0.1*(x_r-shift)**2 + 0.05*(x_r-shift)**4, 'b--', alpha=0.5,
        label='Flat: 분포 이동 후 손실 소폭 증가')
ax.plot(x_r, 2.0*(x_r-shift)**2 + 0.5*(x_r-shift)**4, 'r--', alpha=0.5,
        label='Sharp: 분포 이동 후 손실 대폭 증가')

ax.set_ylim(0, 10); ax.set_xlim(-2, 2)
ax.set_title('Flat vs Sharp Minima: 분포 이동에 대한 강건성', fontsize=12)
ax.set_xlabel('파라미터 공간'); ax.set_ylabel('손실')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ch3-02-flat-vs-sharp.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔗 AI/ML 연결

### 헤시안 고유값 스펙트럼과 학습률

$L$-smooth ($\lambda_{\max}(H) \leq L$): 학습률 상한 $\eta < 1/L$  
$\mu$-strongly convex ($\lambda_{\min}(H) \geq \mu$): 선형 수렴률 $1 - \mu/L$

**조건수 $\kappa = L/\mu$**: 클수록 수렴 느림 (다음 문서 Ch3-04에서 자세히 분석)

### Batch Size와 Flat Minima

큰 배치는 sharp minima, 작은 배치는 flat minima로 수렴하는 경향이 있다는 관측 (Keskar 2016). 이유: 작은 배치의 gradient 잡음이 날카로운 극솟값에서 탈출을 돕는다.

---

## 📌 핵심 정리

| 헤시안 부호 | 정류점 유형 | 학습 의미 |
|-----------|-----------|---------|
| 모든 $\lambda_i > 0$ | 국소 최솟값 | 수렴 성공 |
| 모든 $\lambda_i < 0$ | 국소 최댓값 | GD로 접근 불가 |
| 양/음 혼재 | 안장점 | 고차원에서 지배적 |
| $\lambda_i \geq 0$ | 불명확 (퇴화) | 추가 분석 필요 |

고차원에서 안장점이 지배적인 이유: 모든 고유값이 동일 부호일 확률은 지수적으로 감소.

---

## 🤔 생각해볼 문제

**문제 1**: $f(x,y) = x^4 - 2x^2 + y^2$의 모든 정류점을 찾고, 각각의 유형을 헤시안으로 판정하라.

<details><summary>해설</summary>
$\nabla f = (4x^3-4x, 2y) = 0$. 정류점: $(0,0)$, $(\pm 1, 0)$.  
$H = \begin{pmatrix}12x^2-4&0\\0&2\end{pmatrix}$.  
$(0,0)$: $H=\begin{pmatrix}-4&0\\0&2\end{pmatrix}$ 부정부호 → 안장점.  
$(\pm 1, 0)$: $H=\begin{pmatrix}8&0\\0&2\end{pmatrix}$ PD → 국소 최솟값.
</details>

**문제 2 (AI 연결)**: 딥러닝 학습에서 SGD가 Adam보다 더 flat한 minima를 찾는 경향이 있다는 주장을 헤시안 관점에서 설명하라.

<details><summary>해설</summary>
SGD의 gradient 잡음이 sharp minima ($\lambda_{\max}$가 큰 영역)에서 더 큰 탈출력을 만든다. Adam의 적응적 학습률이 오히려 sharp minima로 수렴하도록 유도할 수 있다. Gradient 잡음의 covariance가 헤시안과 연관된다는 이론적 연결이 있다.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 다변수 테일러 정리](./01-multivariate-taylor.md) | [📚 README](../README.md) | [03. 안장점과 볼록성 판정 ▶](./03-saddle-points-convexity.md) |

</div>
