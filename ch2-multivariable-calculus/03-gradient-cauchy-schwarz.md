# 03. Gradient와 코시-슈바르츠

## 🎯 핵심 질문

- $D_v f = \nabla f \cdot v$는 어떻게 유도되는가?
- 코시-슈바르츠 부등식은 Gradient 방향이 최대 증가임을 어떻게 증명하는가?
- 경사하강법은 이 성질을 어떻게 사용하는가?
- Gradient가 0인 점에서 방향도함수는 어떻게 되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **경사하강법의 정당성**: "$-\nabla f$ 방향으로 움직이는 것이 최적"이라는 경사하강법의 핵심 주장은 코시-슈바르츠 등호 조건에서 나온다.
- **Gradient 크기 = 민감도**: $\|\nabla f(x)\|$는 점 $x$에서 함수가 얼마나 빠르게 변할 수 있는지의 최댓값이다. Gradient norm이 작으면 어느 방향으로 움직여도 함수 변화가 작다.
- **Gradient Flow**: "Gradient가 0인 방향으로 서서히 이동"하는 연속 시간 GD $\dot{x} = -\nabla f(x)$는 Gradient가 최대 감소 방향이라는 사실의 극한 표현이다.

---

## 📐 수학적 선행 조건

- [Ch2-01. 편미분과 방향도함수](./01-partial-directional-derivative.md)
- [Ch2-02. 전미분과 야코비안](./02-total-derivative-jacobian.md)
- 내적과 코시-슈바르츠 부등식 (선형대수)

---

## 📖 직관적 이해

> **비유**: 산에 서서 어느 방향으로 올라갈지 결정한다. 방향도함수는 각 방향의 "즉각적인 경사"다. Gradient는 모든 방향 중 경사가 가장 가파른 방향을 가리키는 나침반이다. 경사하강법은 이 나침반의 반대방향으로 걷는 것이다.

---

## ✏️ 엄밀한 정의

### 정의 2.6 — 코시-슈바르츠 부등식

$u, v \in \mathbb{R}^n$에 대해:
$$|u \cdot v| \leq \|u\| \cdot \|v\|$$

등호는 $u \parallel v$ (평행, 즉 $u = \lambda v$ for some $\lambda \in \mathbb{R}$)일 때 성립.

---

## 🔬 정리와 증명

### 정리 2.5 — Gradient는 최대 증가 방향

**명제**: $f: \mathbb{R}^n \to \mathbb{R}$이 전미분가능하고 $\nabla f(a) \neq 0$이면:

1. 임의의 단위벡터 $v$에 대해 $D_v f(a) = \nabla f(a) \cdot v \leq \|\nabla f(a)\|$
2. 등호는 $v = \frac{\nabla f(a)}{\|\nabla f(a)\|}$일 때 성립
3. 최소 방향도함수 $= -\|\nabla f(a)\|$는 $v = -\frac{\nabla f(a)}{\|\nabla f(a)\|}$에서 달성

**증명**:  
전미분가능이면 $D_v f(a) = \nabla f(a) \cdot v$ (Ch2-01, 정리 2.1).

코시-슈바르츠에 의해:
$$D_v f(a) = \nabla f(a) \cdot v \leq \|\nabla f(a)\| \cdot \|v\| = \|\nabla f(a)\|$$

등호 조건: $v \parallel \nabla f(a)$이고 같은 방향, 즉 $v = \frac{\nabla f(a)}{\|\nabla f(a)\|}$.

최솟값: $D_v f = \nabla f \cdot v \geq -\|\nabla f\|$ (코시-슈바르츠의 반대 방향), 등호는 $v = -\frac{\nabla f(a)}{\|\nabla f(a)\|}$. $\square$

---

### 정리 2.6 — 코시-슈바르츠 부등식 증명

**증명**:  
임의의 $t \in \mathbb{R}$에 대해 $\|u + tv\|^2 \geq 0$:

$$\|u\|^2 + 2t(u \cdot v) + t^2\|v\|^2 \geq 0$$

이것은 $t$에 대한 이차식이고, 항상 $\geq 0$이므로 판별식 $\leq 0$:

$$4(u \cdot v)^2 - 4\|u\|^2\|v\|^2 \leq 0 \implies |u \cdot v| \leq \|u\|\|v\| \quad \square$$

등호: 판별식 $= 0 \iff \|u + t^*v\|^2 = 0 \iff u = -t^* v$, 즉 $u \parallel v$.

---

### 정리 2.7 — Gradient와 등위면의 수직성

**명제**: $c = f(a)$에서의 등위면 $S = \{x : f(x) = c\}$ 위의 모든 접선벡터 $w$에 대해 $\nabla f(a) \cdot w = 0$.

**증명**:  
$S$ 위의 매끄러운 곡선 $\gamma(t)$로 $\gamma(0) = a$이면, $f(\gamma(t)) = c$ (상수).  
양변 미분: $\nabla f(\gamma(t)) \cdot \gamma'(t) = 0$.  
$t=0$에서: $\nabla f(a) \cdot \gamma'(0) = 0$.  
$w = \gamma'(0)$은 임의의 접선벡터이므로 $\nabla f(a) \perp S$. $\square$

> **기하학적 의미**: Gradient는 등위면에 수직으로 튀어나온다. 경사하강법은 등위면을 "가장 빨리 가로지르는" 방향으로 이동한다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ─────────────────────────────────────────────
# 1. 방향도함수와 코시-슈바르츠 시각화
# ─────────────────────────────────────────────

def f(x): return x[0]**2 + 3*x[1]**2   # 타원형 등위선
def grad_f(x): return np.array([2*x[0], 6*x[1]])

a = np.array([1.0, 1.0])
grad = grad_f(a)
grad_norm = np.linalg.norm(grad)

# 모든 방향의 방향도함수 계산
thetas = np.linspace(0, 2*np.pi, 360)
D_v_vals = [grad @ np.array([np.cos(t), np.sin(t)]) for t in thetas]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 좌: 방향도함수 vs 방향
axes[0].plot(np.degrees(thetas), D_v_vals, 'b-', linewidth=2, label=r'$D_v f(a) = \nabla f \cdot v$')
axes[0].axhline(y=grad_norm, color='r', linestyle='--', linewidth=1.5,
                label=f'최댓값 = ‖∇f‖ = {grad_norm:.3f}')
axes[0].axhline(y=-grad_norm, color='g', linestyle='--', linewidth=1.5,
                label=f'최솟값 = -‖∇f‖ = {-grad_norm:.3f}')
max_theta = np.degrees(np.arctan2(grad[1], grad[0]))
axes[0].axvline(x=max_theta, color='r', linestyle=':', alpha=0.7)
axes[0].set_xlabel('방향 θ (도)')
axes[0].set_ylabel(r'$D_v f(a)$')
axes[0].set_title(r"$f=x^2+3y^2$, $a=(1,1)$: 방향도함수의 방향 의존성", fontsize=11)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# 우: 등위선과 Gradient 시각화
x_rng = np.linspace(-2, 2, 200)
X, Y  = np.meshgrid(x_rng, x_rng)
Z     = X**2 + 3*Y**2

ax = axes[1]
cp = ax.contour(X, Y, Z, levels=15, colors='gray', alpha=0.5)
ax.clabel(cp, inline=True, fontsize=7)
# Gradient 화살표
ax.quiver(a[0], a[1], grad[0], grad[1], color='red', scale=20,
          width=0.005, label=r'$\nabla f$ (최대 증가)')
ax.quiver(a[0], a[1], -grad[0], -grad[1], color='blue', scale=20,
          width=0.005, label=r'$-\nabla f$ (GD 방향)')
ax.scatter(*a, color='black', s=80, zorder=5)
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_title('등위선과 Gradient (수직)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch2-03-gradient-cauchy-schwarz.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. 코시-슈바르츠 수치 검증
# ─────────────────────────────────────────────

np.random.seed(0)
n_tests = 1000
violations = 0
for _ in range(n_tests):
    u = np.random.randn(10)
    v = np.random.randn(10)
    lhs = abs(np.dot(u, v))
    rhs = np.linalg.norm(u) * np.linalg.norm(v)
    if lhs > rhs + 1e-10:
        violations += 1
print(f"코시-슈바르츠 위반 횟수: {violations}/{n_tests} (0이어야 함)")

# ─────────────────────────────────────────────
# 3. Gradient Flow 시뮬레이션
# ─────────────────────────────────────────────

def gradient_descent_path(f_grad, start, lr=0.1, n_steps=50):
    path = [np.array(start)]
    x = np.array(start, dtype=float)
    for _ in range(n_steps):
        x = x - lr * f_grad(x)
        path.append(x.copy())
    return np.array(path)

# Rosenbrock 함수: f(x,y) = (1-x)² + 100(y-x²)²
def rosenbrock_grad(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

path = gradient_descent_path(rosenbrock_grad, [-1.0, 0.5], lr=0.001, n_steps=2000)

x_r = np.linspace(-2, 2, 200)
y_r = np.linspace(-0.5, 3.0, 200)
Xr, Yr = np.meshgrid(x_r, y_r)
Zr = (1 - Xr)**2 + 100*(Yr - Xr**2)**2

fig, ax = plt.subplots(figsize=(8, 6))
cp = ax.contourf(Xr, Yr, np.log(Zr + 1), levels=30, cmap='viridis', alpha=0.7)
ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=1, alpha=0.8, label='GD 경로')
ax.scatter([1], [1], color='gold', s=150, zorder=5, marker='*', label='최적점 (1,1)')
ax.scatter(*path[0], color='white', s=80, zorder=5, label='시작점')
ax.set_title('Rosenbrock 함수에서의 경사하강법 경로', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
plt.colorbar(cp, label='log(f+1)')
plt.tight_layout()
plt.savefig('ch2-03-gradient-descent-path.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔗 AI/ML 연결

### 경사하강법의 단계별 정당성

**단계 1**: 전미분가능이면 $D_v f = \nabla f \cdot v$ (Ch2-01, 정리 2.1)  
**단계 2**: 코시-슈바르츠에 의해 $D_v f \geq -\|\nabla f\|$ (정리 2.5)  
**단계 3**: 등호는 $v = -\nabla f / \|\nabla f\|$에서 달성  
**결론**: $-\nabla f$ 방향이 가장 빠르게 감소하는 방향이므로, $x \leftarrow x - \eta \nabla f$는 국소적으로 최적

**주의**: 이것은 **무한소** 스텝에 대한 주장이다. 유한 $\eta$에서는 L-smooth 조건이 추가로 필요하다 (Ch4-02).

### Gradient Norm과 수렴 진단

학습 중 $\|\nabla f(x_k)\|$를 모니터링하는 이유:
- $\|\nabla f\| \approx 0$: 정류점(stationary point) 근방 — 극솟값, 극댓값, 안장점 중 하나
- $\|\nabla f\|$이 감소하지 않음: 발산 또는 학습률 문제

---

## 📌 핵심 정리

$$D_v f(a) = \nabla f(a) \cdot v \leq \|\nabla f(a)\| \quad (\|v\|=1)$$

| 방향 $v$ | $D_v f(a)$ |
|----------|-----------|
| $+\nabla f / \|\nabla f\|$ | $+\|\nabla f\|$ (최대) |
| $-\nabla f / \|\nabla f\|$ | $-\|\nabla f\|$ (최소, GD 방향) |
| $\perp \nabla f$ | $0$ (등위면 접선 방향) |

---

## 🤔 생각해볼 문제

**문제 1**: $f(x,y) = e^{x^2+y^2}$에서 점 $(1,0)$의 최대 증가 방향과 그 방향도함수값을 구하라.

<details><summary>해설</summary>
$\nabla f = 2xe^{x^2+y^2}(1,0)|_{(1,0)} = (2e, 0)$. 최대 증가 방향 $v=(1,0)$, $D_v f = 2e \approx 5.44$.
</details>

**문제 2**: $f(x,y,z) = x^2 + y^2 + z^2$의 등위면은 구(sphere)다. $(1,1,1)$에서 Gradient가 등위구에 수직임을 확인하라.

<details><summary>해설</summary>
$\nabla f(1,1,1) = (2,2,2)$. 구의 법벡터는 $(1,1,1)$ 방향으로, $(2,2,2) = 2(1,1,1)$이므로 실제로 수직.
</details>

**문제 3 (AI 연결)**: Adam 옵티마이저는 $v = -\nabla f / \|\nabla f\|$ (정규화된 gradient 방향)와 비슷한 방향을 사용한다고 볼 수 있는가? 어떤 점에서 다른가?

<details><summary>해설</summary>
Adam의 업데이트 $m_t / (\sqrt{v_t} + \epsilon)$는 각 좌표를 독립적으로 스케일링한다. 이는 등방성(isotropic) 정규화가 아니라 좌표별 적응 스케일링이다. $\|\nabla f\|$로 나누는 것과는 다르며, 오히려 대각 Hessian의 역수에 가깝다.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 전미분과 야코비안](./02-total-derivative-jacobian.md) | [📚 README](../README.md) | [04. 야코비안과 헤시안의 기하 ▶](./04-jacobian-hessian-geometry.md) |

</div>
