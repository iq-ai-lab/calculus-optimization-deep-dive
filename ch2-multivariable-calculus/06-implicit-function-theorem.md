# 06. 음함수 정리

## 🎯 핵심 질문

- $g(x, y) = 0$ 조건에서 야코비안이 가역이면 왜 $y = \phi(x)$로 표현 가능한가?
- 제약 최적화에서 제약 곡면의 접평면이 왜 $\ker(Dg)$인가?
- Deep Equilibrium Models에서 음함수 정리가 어떻게 gradient를 제공하는가?
- 역함수 정리(Inverse Function Theorem)와 어떻게 연결되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Deep Equilibrium Model (DEQ)**: 무한 깊이 네트워크를 고정점 방정식 $z^* = f_\theta(z^*)$으로 정의. $\nabla_\theta z^*$를 음함수 정리로 계산하면 무한 깊이를 실제로 펼치지 않아도 된다.
- **Optimization as a Layer (OptNet, CVXPY Layers)**: 내부 최적화 문제의 해 $y^*(x)$에 대한 gradient를 KKT 조건 미분(= 음함수 정리)으로 계산.
- **제약 최적화의 이론적 기반**: 라그랑주 승수법의 기하학적 정당성 (Ch6-01)이 음함수 정리에서 나온다.

---

## 📐 수학적 선행 조건

- [Ch2-02. 전미분과 야코비안](./02-total-derivative-jacobian.md): 야코비안의 가역성
- [Ch2-05. 다변수 연쇄법칙](./05-chain-rule-general.md)
- 선형대수: 역행렬, 행렬식

---

## 📖 직관적 이해

$g(x, y) = x^2 + y^2 - 1 = 0$: 단위원. 점 $(0, 1)$ 근방에서 $y = \sqrt{1-x^2}$로 명시적으로 쓸 수 있다. 하지만 점 $(\pm 1, 0)$ 근방에서는 $y$가 $x$의 함수로 쓰이지 않는다 ($y$가 두 값을 가짐).

음함수 정리는 이런 "국소적으로 함수로 쓸 수 있는" 조건을 야코비안의 가역성으로 정확히 표현한다.

---

## ✏️ 엄밀한 정의와 정리

### 정리 2.12 — 음함수 정리 (Implicit Function Theorem)

**명제**: $F: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^m$이 $C^1$이고 $(a, b) \in \mathbb{R}^n \times \mathbb{R}^m$에서 $F(a, b) = 0$이라 하자. $y$ 부분의 야코비안:

$$\frac{\partial F}{\partial y}(a, b) \in \mathbb{R}^{m \times m}$$

이 가역(invertible)이면, $(a, b)$ 근방에서 유일한 $C^1$ 함수 $\phi: U \to \mathbb{R}^m$ ($U$: $a$의 근방)가 존재하여:

1. $\phi(a) = b$
2. $F(x, \phi(x)) = 0$ for all $x \in U$

그리고 $\phi$의 야코비안:

$$\frac{\partial \phi}{\partial x}(a) = -\left[\frac{\partial F}{\partial y}(a, b)\right]^{-1} \frac{\partial F}{\partial x}(a, b)$$

**증명 스케치**:  
$G(x, y) = y - \phi(x)$ 형태로 보조 함수를 만들어 축소 사상 정리(Banach Fixed Point Theorem)를 적용. $F$의 $y$-야코비안 가역성이 축소 조건을 보장함. $\square$

---

### 따름정리 — 야코비안 공식 도출

$F(x, y) = 0$을 $x$에 대해 미분하면 (연쇄법칙):

$$\frac{\partial F}{\partial x} + \frac{\partial F}{\partial y} \frac{\partial y}{\partial x} = 0$$

$\frac{\partial F}{\partial y}$ 가역이면:

$$\frac{\partial y}{\partial x} = -\left[\frac{\partial F}{\partial y}\right]^{-1} \frac{\partial F}{\partial x}$$

---

### 정리 2.13 — 제약 곡면의 접평면

**명제**: $g: \mathbb{R}^n \to \mathbb{R}^k$ ($k < n$)가 $C^1$이고 $S = \{x : g(x) = 0\}$이라 하자. $a \in S$에서 $Dg(a)$의 행 벡터가 선형독립이면, $S$의 $a$에서의 접평면(접공간)은:

$$T_a S = \ker(Dg(a)) = \{v \in \mathbb{R}^n : Dg(a) v = 0\}$$

**증명**:  
$S$ 위의 곡선 $\gamma(t)$, $\gamma(0) = a$. $g(\gamma(t)) = 0$을 미분하면 $Dg(a) \cdot \gamma'(0) = 0$.  
따라서 모든 접선 $\gamma'(0) \in \ker(Dg(a))$.  
차원 계산으로 $\dim(T_a S) = n - k = \dim(\ker Dg)$이므로 등호. $\square$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 음함수 정리: 단위원 예제
# ─────────────────────────────────────────────

x, y = sp.symbols('x y')
F = x**2 + y**2 - 1   # 단위원 방정식

# 음함수 정리: dy/dx = -(∂F/∂y)⁻¹ · (∂F/∂x)
dF_dx = sp.diff(F, x)
dF_dy = sp.diff(F, y)
dy_dx = -dF_dx / dF_dy   # = -x/y

print("[음함수 정리: 단위원 x² + y² = 1]")
print(f"dF/dx = {dF_dx}")
print(f"dF/dy = {dF_dy}")
print(f"dy/dx = -(∂F/∂y)⁻¹ · ∂F/∂x = {dy_dx}")

# 점 (0, 1)에서 검증: 명시적 함수 y = √(1-x²)
point = {x: 0, y: 1}
dy_dx_val = dy_dx.subs(point)
print(f"\n(0,1)에서 dy/dx = {dy_dx_val}")
# 명시적: d/dx √(1-x²)|_{x=0} = -x/√(1-x²)|_{x=0} = 0 (일치)

# ─────────────────────────────────────────────
# 2. DEQ (Deep Equilibrium) gradient 계산
# ─────────────────────────────────────────────

def deq_implicit_gradient(f_theta, z_star, theta, dL_dz):
    """
    DEQ: z* = f_θ(z*) → F(z, θ) = z - f_θ(z) = 0
    음함수 정리: ∂z*/∂θ = -(∂F/∂z)⁻¹ · (∂F/∂θ)
                          = (I - J_f)⁻¹ · J_f_θ
    
    dL/dθ = (dL/dz*) · (∂z*/∂θ)
    """
    n = len(z_star)
    
    # 수치 야코비안 ∂f/∂z at z*
    J_f_z = np.zeros((n, n))
    h = 1e-5
    for j in range(n):
        e_j = np.zeros(n); e_j[j] = 1
        J_f_z[:, j] = (f_theta(z_star + h*e_j, theta) 
                       - f_theta(z_star - h*e_j, theta)) / (2*h)
    
    # (I - J_f_z)을 사용: ∂F/∂z = I - J_f_z
    IJ = np.eye(n) - J_f_z
    
    # 음함수 정리: dL/dz* · (I - J_f)⁻¹
    # 전치된 선형계: (I - J_f)ᵀ v = dL/dz*
    v = np.linalg.solve(IJ.T, dL_dz)
    return v

# 예제: 간단한 고정점 반복
def f_simple(z, theta):
    """f(z) = tanh(theta * z)"""
    return np.tanh(theta * z)

# 고정점 찾기
def find_fixed_point(f_theta, theta, z_init=0.5, n_iter=100):
    z = z_init
    for _ in range(n_iter):
        z = f_theta(np.array([z]), np.array([theta]))[0]
    return z

theta_val = np.array([0.5])
z_star_val = np.array([find_fixed_point(f_simple, theta_val[0])])
dL_dz_val = np.array([1.0])

grad = deq_implicit_gradient(f_simple, z_star_val, theta_val, dL_dz_val)
print(f"\n[DEQ Implicit Gradient]")
print(f"고정점 z* = {z_star_val}")
print(f"음함수 정리 gradient dL/dθ ≈ {grad}")

# ─────────────────────────────────────────────
# 3. 제약 곡면 접평면 시각화
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 단위원 위의 여러 점에서 접선(ker Dg)과 법선(∇g) 시각화
theta_pts = np.linspace(0, 2*np.pi, 8, endpoint=False)
circle_x  = np.cos(np.linspace(0, 2*np.pi, 300))
circle_y  = np.sin(np.linspace(0, 2*np.pi, 300))

axes[0].plot(circle_x, circle_y, 'b-', linewidth=2, label='단위원 g(x,y)=0')
for t in theta_pts:
    px, py = np.cos(t), np.sin(t)
    # 법선 = ∇g = (2x, 2y) 방향
    axes[0].annotate('', xy=(px + 0.3*px, py + 0.3*py), xytext=(px, py),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    # 접선 = ker(Dg) 방향: (-y, x)
    tx, ty = -py * 0.3, px * 0.3
    axes[0].plot([px-tx, px+tx], [py-ty, py+ty], 'g-', linewidth=1.5, alpha=0.7)

axes[0].set_aspect('equal'); axes[0].set_xlim(-1.8, 1.8); axes[0].set_ylim(-1.8, 1.8)
axes[0].legend(fontsize=9)
from matplotlib.lines import Line2D
handles = [Line2D([0],[0],color='red',lw=2,label='∇g (법선)'),
           Line2D([0],[0],color='green',lw=2,label='ker(Dg) (접선)')]
axes[0].legend(handles=handles, fontsize=9)
axes[0].set_title('제약 곡면 접평면 = ker(Dg)', fontsize=11)
axes[0].grid(True, alpha=0.3)

# 음함수 정리: dy/dx = -Fx/Fy 시각화 (타원)
a_e, b_e = 2.0, 1.0
t_vals = np.linspace(0, 2*np.pi, 300)
ex = a_e * np.cos(t_vals)
ey = b_e * np.sin(t_vals)
axes[1].plot(ex, ey, 'b-', linewidth=2, label=f'타원 x²/{a_e}² + y²/{b_e}² = 1')

pt = np.pi / 4
px2, py2 = a_e*np.cos(pt), b_e*np.sin(pt)
slope = -(px2/a_e**2) / (py2/b_e**2)  # dy/dx = -(∂F/∂x)/(∂F/∂y)
dx_span = np.linspace(-0.5, 0.5, 50)
axes[1].plot(px2 + dx_span, py2 + slope*dx_span, 'r-', linewidth=2,
             label=f'접선 기울기 = {slope:.3f}')
axes[1].scatter([px2],[py2], s=100, color='red', zorder=5)
axes[1].set_aspect('equal'); axes[1].legend(fontsize=9)
axes[1].set_title('음함수 정리: 타원 위 접선 기울기', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch2-06-implicit-function.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔗 AI/ML 연결

### DEQ의 역전파

$$z^* = f_\theta(z^*) \implies \frac{dz^*}{d\theta} = \left(I - \frac{\partial f}{\partial z}\bigg|_{z^*}\right)^{-1} \frac{\partial f}{\partial \theta}\bigg|_{z^*}$$

행렬 역수를 명시적으로 구하지 않고 Conjugate Gradient나 Anderson acceleration으로 선형계를 푼다.

### CVXPY Layers (OptNet)

$$y^*(x) = \arg\min_y \frac{1}{2}\|y\|^2 \;\text{s.t.}\; Ax = b$$

KKT 조건 미분 = 음함수 정리 적용:
$$\frac{\partial y^*}{\partial x} = -A^\dagger \frac{\partial (Ax-b)}{\partial x} = -A^\dagger A$$

---

## 📌 핵심 정리

$$F(x, y) = 0,\; \frac{\partial F}{\partial y} \text{ 가역} \implies \frac{\partial y}{\partial x} = -\left[\frac{\partial F}{\partial y}\right]^{-1} \frac{\partial F}{\partial x}$$

$$T_a S = \ker(Dg(a)) \quad (S = \{g = 0\}\text{의 접평면})$$

---

## 🤔 생각해볼 문제

**문제 1**: $F(x, y) = y^3 + xy - 1 = 0$에서 $(0, 1)$ 근방 음함수 $dy/dx$를 구하라.

<details><summary>해설</summary>
$\partial F/\partial x = y$, $\partial F/\partial y = 3y^2 + x$. 점 $(0,1)$: $dy/dx = -1/(3+0) = -1/3$.
</details>

**문제 2**: 구면 $x^2+y^2+z^2=1$ 위의 점 $(1/\sqrt{3}, 1/\sqrt{3}, 1/\sqrt{3})$에서 접평면을 구하라.

<details><summary>해설</summary>
$\nabla g = (2x, 2y, 2z)|_p = (2/\sqrt{3})(1,1,1)$. 접평면: $x+y+z = \sqrt{3}$.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. 다변수 연쇄법칙](./05-chain-rule-general.md) | [📚 README](../README.md) | [Ch3-01. 다변수 테일러 정리 ▶](../ch3-taylor-quadratic/01-multivariate-taylor.md) |

</div>
