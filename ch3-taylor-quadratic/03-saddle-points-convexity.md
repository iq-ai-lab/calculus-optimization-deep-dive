# 03. 안장점과 볼록성 판정

## 🎯 핵심 질문

- 2차 판별식 $D = f_{xx}f_{yy} - f_{xy}^2$은 어떻게 유도되는가?
- 볼록 함수의 정의와 헤시안 PSD는 어떻게 동치인가?
- 딥러닝 Loss에 안장점이 "넘쳐난다"는 Dauphin 2014의 핵심 주장은?
- 비볼록 문제에서 경사하강법이 안장점에 수렴하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **비볼록 최적화의 현실**: 딥러닝 손실 함수는 비볼록이다. "국소 최솟값에 갇히는 것" 보다 "안장점에 느리게 수렴하는 것"이 실제 문제다.
- **볼록성과 수렴 보장**: 볼록 함수에서만 "정류점 = 전역 최솟값"이 보장된다. Logistic Regression은 볼록, ReLU 네트워크는 비볼록.
- **SGD vs GD**: SGD의 잡음은 안장점 탈출을 도와 비볼록 최적화에서 오히려 유리할 수 있다.

---

## 📐 수학적 선행 조건

- [Ch3-01. 다변수 테일러 정리](./01-multivariate-taylor.md)
- [Ch3-02. 헤시안의 고유값과 국소 기하](./02-hessian-eigenvalues-geometry.md)

---

## ✏️ 엄밀한 정의와 정리

### 정의 3.1 — 볼록 함수

$f: C \to \mathbb{R}$ ($C$: 볼록 집합)가 **볼록(convex)**이라는 것은:

$$\forall x, y \in C,\; \forall \lambda \in [0,1]:\quad f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

**등호**: $f$가 **강볼록(strictly convex)**이면 $\lambda \in (0,1)$에서 등호 없음.

### 정리 3.5 — 볼록성 ↔ 헤시안 PSD

**명제**: $f \in C^2$이면:

$$f \text{ 볼록} \iff H_f(x) \text{ PSD} \; \forall x$$

**증명 (⇒)**:  
볼록이면 임의의 방향 $v$에 대해 $g(t) = f(x+tv)$는 볼록 (단변수). 볼록 $\Rightarrow$ $g''(t) \geq 0$.  
$g''(0) = v^\top H_f(x) v \geq 0$이므로 $H_f(x)$ PSD. $\square$

**증명 (⇐)**:  
$H_f$ PSD이면 2차 테일러 전개에서:
$$f(y) = f(x) + \nabla f(x)^\top(y-x) + \frac{1}{2}(y-x)^\top H_f(\xi)(y-x) \geq f(x) + \nabla f(x)^\top(y-x)$$

이것이 볼록성의 1차 조건 (First-Order Characterization)이므로 볼록. $\square$

---

### 정리 3.6 — 2×2 헤시안 판별식

**명제**: $f: \mathbb{R}^2 \to \mathbb{R}$, $\nabla f(a,b) = 0$, $D = f_{xx}f_{yy} - f_{xy}^2$이면:

| 조건 | 결론 |
|------|------|
| $D > 0$, $f_{xx} > 0$ | 국소 최솟값 |
| $D > 0$, $f_{xx} < 0$ | 국소 최댓값 |
| $D < 0$ | 안장점 |
| $D = 0$ | 불명확 |

**유도**: 헤시안 $H = \begin{pmatrix}a&b\\b&c\end{pmatrix}$의 고유값:

$$\lambda_{1,2} = \frac{(a+c) \pm \sqrt{(a-c)^2 + 4b^2}}{2}$$

$\det H = ac - b^2 = \lambda_1 \lambda_2 = D$, $\text{tr} H = a+c = \lambda_1 + \lambda_2$.

- $D > 0$: 두 고유값이 같은 부호 → $\text{tr}H$의 부호로 PD/ND 판정
- $D < 0$: 두 고유값이 다른 부호 → 부정부호 → 안장점

---

### 정리 3.7 — 볼록 함수의 정류점 = 전역 최솟값

**명제**: $f$가 볼록이고 $\nabla f(x^*) = 0$이면 $x^*$는 전역 최솟값.

**증명**:  
볼록 함수의 1차 특성: $f(y) \geq f(x^*) + \nabla f(x^*)^\top(y-x^*) = f(x^*) + 0 = f(x^*)$. 모든 $y$에 대해 성립. $\square$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 판별식으로 정류점 자동 분류
# ─────────────────────────────────────────────

x, y = sp.symbols('x y')

def classify_critical_points(f_sym):
    """모든 정류점을 찾아 헤시안으로 분류"""
    fx = sp.diff(f_sym, x)
    fy = sp.diff(f_sym, y)
    
    # 정류점 (∇f = 0)
    critical = sp.solve([fx, fy], [x, y], dict=True)
    
    fxx = sp.diff(f_sym, x, 2)
    fxy = sp.diff(f_sym, x, y)
    fyy = sp.diff(f_sym, y, 2)
    D_sym = fxx * fyy - fxy**2
    
    print(f"f(x,y) = {f_sym}")
    for pt in critical:
        D_val  = D_sym.subs(pt)
        fxx_val = fxx.subs(pt)
        if D_val > 0 and fxx_val > 0:
            ptype = "국소 최솟값"
        elif D_val > 0 and fxx_val < 0:
            ptype = "국소 최댓값"
        elif D_val < 0:
            ptype = "안장점"
        else:
            ptype = "불명확"
        print(f"  정류점 {pt}: D={float(D_val):.3f}, f_xx={float(fxx_val):.3f} → {ptype}")
    print()

classify_critical_points(x**3 - 3*x + y**3 - 3*y)
classify_critical_points(x**2 - y**2 + x*y)
classify_critical_points(x**4 + y**4 - 4*x*y)

# ─────────────────────────────────────────────
# 2. 경사하강법의 안장점 탈출 (잡음 효과)
# ─────────────────────────────────────────────

def saddle_function(p): return p[0]**2 - p[1]**2
def saddle_grad(p):     return np.array([2*p[0], -2*p[1]])

def gd_trajectory(grad_fn, start, lr=0.05, n_steps=200, noise=0.0, seed=42):
    np.random.seed(seed)
    path = [np.array(start, dtype=float)]
    x = np.array(start, dtype=float)
    for _ in range(n_steps):
        g = grad_fn(x) + noise * np.random.randn(*x.shape)
        x = x - lr * g
        path.append(x.copy())
    return np.array(path)

# 안장점 (0,0) 근방에서 시작
start = np.array([0.01, 0.01])  # 약간의 perturbation

path_gd   = gd_trajectory(saddle_grad, start, noise=0.0)
path_sgd  = gd_trajectory(saddle_grad, start, noise=0.3, seed=0)

x_r = np.linspace(-1.5, 1.5, 100)
y_r = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x_r, y_r)
Z = X**2 - Y**2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, path, title, color in [
    (axes[0], path_gd,  'GD (no noise): 안장점 근처 정체', 'blue'),
    (axes[1], path_sgd, 'SGD (noise=0.3): 안장점 탈출', 'red')
]:
    cp = ax.contourf(X, Y, Z, levels=20, cmap='RdBu', alpha=0.6)
    ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
    ax.plot(path[:,0], path[:,1], '-', color=color, linewidth=1.5, alpha=0.8)
    ax.scatter(*path[0],  s=100, color='green', zorder=5, label='시작')
    ax.scatter(*path[-1], s=100, color='red',   zorder=5, marker='*', label='종점')
    ax.scatter(0, 0, s=150, color='black', zorder=6, marker='x', linewidths=3, label='안장점')
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.suptitle(r'안장점 $f=x^2-y^2$: GD 정체 vs SGD 탈출', fontsize=12)
plt.tight_layout()
plt.savefig('ch3-03-saddle-escape.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. 볼록 vs 비볼록 함수 비교
# ─────────────────────────────────────────────

x_r = np.linspace(-3, 3, 300)

funcs = {
    r"볼록: $x^2$ (H=2>0)":     (lambda x: x**2, 'blue'),
    r"비볼록: $x^4-2x^2$ (안장점 포함)": (lambda x: x**4 - 2*x**2, 'red'),
    r"볼록: $e^x$ (H=e^x>0)":   (lambda x: np.exp(x), 'green'),
}

fig, ax = plt.subplots(figsize=(9, 5))
for label, (fn, color) in funcs.items():
    ax.plot(x_r, np.clip(fn(x_r), -5, 15), color=color, linewidth=2, label=label)

ax.set_ylim(-5, 12); ax.axhline(0, color='k', linewidth=0.5)
ax.set_title('볼록 vs 비볼록 함수', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ch3-03-convex-vs-nonconvex.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔗 AI/ML 연결

### Dauphin 2014: 딥러닝의 안장점 문제

논문의 핵심 주장: 딥러닝의 국소 최솟값은 손실이 다양하게 다르지 않다. 문제는 **안장점 근방에서의 느린 수렴**이다.

수학적 근거: 랜덤 행렬 이론 (Wigner 반원 법칙) — $n$차원 GOE 랜덤 행렬의 고유값이 모두 양수일 확률은 $e^{-\Theta(n^2)}$으로 감소. 즉 고차원에서 국소 최솟값은 지수적으로 드물다.

### 볼록 문제: Logistic Regression

이진 분류에서 Cross-Entropy Loss + Sigmoid:
$$L(w) = -\sum_i [y_i \log \sigma(w^\top x_i) + (1-y_i)\log(1-\sigma(w^\top x_i))]$$

헤시안 $H = X^\top \text{diag}(\sigma_i(1-\sigma_i)) X$ — PSD이므로 볼록. GD는 전역 최솟값으로 수렴.

---

## 📌 핵심 정리

| | 볼록 | 비볼록 |
|---|------|-------|
| 정류점 | 반드시 전역 최솟값 | 극소/극대/안장점 가능 |
| 헤시안 | PSD | 부정부호 가능 |
| GD 수렴 | 전역 최솟값 | 안장점에 느리게 수렴 가능 |
| SGD 역할 | 불필요한 잡음 | 안장점 탈출에 유리 |

$$f \text{ 볼록} \iff H_f \text{ PSD} \iff f(y) \geq f(x) + \nabla f^\top(y-x)$$

---

## 🤔 생각해볼 문제

**문제 1**: $f(x,y) = \sin x + \sin y$ ($-\pi < x,y < \pi$)의 모든 정류점을 찾아 분류하라.

<details><summary>해설</summary>
$\nabla f = (\cos x, \cos y) = 0 \Rightarrow x,y \in \{\pm\pi/2\}$. 4개 정류점. $H = \text{diag}(-\sin x, -\sin y)$. $(\pi/2,\pi/2)$: $H=\text{diag}(-1,-1)$ ND → 극대. $(-\pi/2,-\pi/2)$: PD → 극소. $(\pm\pi/2, \mp\pi/2)$: 부정부호 → 안장점.
</details>

**문제 2 (AI 연결)**: Saddle-Free Newton 방법은 헤시안의 음의 고유값을 어떻게 활용하는가?

<details><summary>해설</summary>
뉴턴 방법 $x \leftarrow x - H^{-1}\nabla f$는 안장점에서 음의 고유값 방향으로 이동해야 하는데, $H^{-1}$의 음의 고유값이 올바른 방향을 제공한다. Saddle-Free Newton (Dauphin 2014)은 $|H|^{-1}\nabla f$를 사용하여 고유값의 절댓값을 취함으로써 안장점에서 항상 하강하도록 수정한다.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 헤시안의 고유값과 국소 기하](./02-hessian-eigenvalues-geometry.md) | [📚 README](../README.md) | [04. 조건수와 최적화 속도 ▶](./04-condition-number-optimization.md) |

</div>
