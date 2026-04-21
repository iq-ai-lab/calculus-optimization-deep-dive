# 01. 편미분과 방향도함수

## 🎯 핵심 질문

- 편미분은 방향도함수의 어떤 특수 경우인가?
- 모든 방향 미분이 존재해도 전미분이 존재하지 않는 함수가 있는가?
- 방향도함수가 $\nabla f \cdot v$로 쓰이는 조건은 무엇인가?
- 딥러닝에서 "gradient 방향으로 내려간다"는 것이 정확히 무슨 의미인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **경사하강법의 출발점**: $x_{k+1} = x_k - \eta \nabla f(x_k)$는 "gradient 방향의 반대"로 움직인다. 이 방향이 왜 가장 빠르게 내려가는지는 방향도함수의 개념 없이 설명할 수 없다.
- **편미분 ≠ 전미분**: PyTorch autograd가 계산하는 것은 편미분 모음이 아니라 전미분(야코비안)이다. 편미분이 모두 존재해도 전미분이 없을 수 있다는 사실은, 수치 gradient만으로 "미분가능성"을 확인할 수 없음을 시사한다.
- **Directional Derivative와 Finite Difference**: 수치 최적화에서 탐색 방향 $d$를 따라 $\frac{f(x + \alpha d) - f(x)}{\alpha}$를 계산하는 것이 바로 방향도함수의 수치 근사다.

---

## 📐 수학적 선행 조건

- [Ch1-01. ε-δ 극한의 정의](../ch1-analysis-foundations/01-epsilon-delta.md)
- [Ch1-03. 미분의 정의와 선형근사](../ch1-analysis-foundations/03-derivative-linear-approx.md): 단변수 $f'(a)$의 정의
- 벡터와 내적의 기본 개념 (고등학교 수준)

---

## 📖 직관적 이해

### 편미분: 한 방향만 보는 미분

$f(x, y)$를 산악 지형이라 하면, 편미분 $\frac{\partial f}{\partial x}$는 **동쪽 방향으로만** 걸어갈 때의 경사, $\frac{\partial f}{\partial y}$는 **북쪽 방향으로만** 걸어갈 때의 경사다.

### 방향도함수: 임의 방향의 경사

단위벡터 $v$를 따라 걷는다면, 그 방향의 경사가 방향도함수 $D_v f$다. 동쪽과 북쪽만이 아니라 임의의 방향을 모두 다룬다.

### 핵심 문제

편미분 두 개를 알면 모든 방향도함수를 알 수 있는가?  
답: **조건이 필요하다.** 전미분이 존재하면 $D_v f = \nabla f \cdot v$가 성립하지만, 전미분이 없으면 편미분이 존재해도 이 공식이 깨진다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 편미분

$f: \mathbb{R}^n \to \mathbb{R}$의 점 $a = (a_1, \ldots, a_n)$에서 $x_i$ 방향의 **편미분**:

$$\frac{\partial f}{\partial x_i}(a) = \lim_{h \to 0} \frac{f(a + h e_i) - f(a)}{h}$$

여기서 $e_i$는 $i$번째 표준기저벡터. **핵심**: $x_i$를 제외한 모든 변수를 상수로 고정하고 단변수 미분을 취하는 것이다.

### 정의 2.2 — 방향도함수

단위벡터 $v \in \mathbb{R}^n$ ($\|v\| = 1$) 방향의 **방향도함수**:

$$D_v f(a) = \lim_{t \to 0} \frac{f(a + tv) - f(a)}{t}$$

**관계**: 편미분은 $v = e_i$인 방향도함수의 특수 경우:
$$\frac{\partial f}{\partial x_i}(a) = D_{e_i} f(a)$$

### 정의 2.3 — Gradient

$f$의 모든 편미분을 모아 만든 벡터:

$$\nabla f(a) = \left(\frac{\partial f}{\partial x_1}(a),\; \frac{\partial f}{\partial x_2}(a),\; \ldots,\; \frac{\partial f}{\partial x_n}(a)\right)$$

> ⚠️ **주의**: Gradient의 정의는 편미분의 모음이다. 그러나 $D_v f = \nabla f \cdot v$가 성립하려면 **전미분 존재**라는 추가 조건이 필요하다.

---

## 🔬 정리와 증명

### 정리 2.1 — 전미분이 존재하면 $D_v f = \nabla f \cdot v$

**명제**: $f: \mathbb{R}^n \to \mathbb{R}$이 $a$에서 전미분가능(fully differentiable)이면:

$$D_v f(a) = \nabla f(a) \cdot v \quad \forall v \text{ (단위벡터)}$$

**증명**:  
전미분가능이란 선형사상 $L: \mathbb{R}^n \to \mathbb{R}$이 존재하여:
$$f(a + h) = f(a) + L(h) + o(\|h\|) \quad (h \to 0)$$

$h = tv$로 놓으면 ($t \to 0$):
$$\frac{f(a + tv) - f(a)}{t} = \frac{L(tv) + o(t)}{t} = L(v) + \frac{o(t)}{t} \xrightarrow{t \to 0} L(v)$$

$v = e_i$로 놓으면 $L(e_i) = \frac{\partial f}{\partial x_i}(a)$이고, 선형사상의 행렬 표현에 의해:
$$L(v) = \sum_i L(e_i) v_i = \sum_i \frac{\partial f}{\partial x_i}(a) v_i = \nabla f(a) \cdot v \quad \square$$

---

### 반례 2.1 — 편미분 존재, 방향도함수 존재, 전미분 없음

**함수**: 

$$f(x, y) = \begin{cases} \frac{x^2 y}{x^2 + y^2} & (x, y) \neq (0, 0) \\ 0 & (x, y) = (0, 0) \end{cases}$$

**편미분 (0,0)에서 존재**:
$$\frac{\partial f}{\partial x}(0,0) = \lim_{h \to 0}\frac{f(h,0)-f(0,0)}{h} = \lim_{h\to 0}\frac{0}{h} = 0$$
$$\frac{\partial f}{\partial y}(0,0) = \lim_{h \to 0}\frac{f(0,h)-f(0,0)}{h} = \lim_{h\to 0}\frac{0}{h} = 0$$

따라서 $\nabla f(0,0) = (0, 0)$.

**방향도함수**: $v = (\cos\theta, \sin\theta)$로 놓으면:
$$D_v f(0,0) = \lim_{t \to 0} \frac{f(t\cos\theta, t\sin\theta)}{t} = \lim_{t \to 0} \frac{t^2\cos^2\theta \cdot t\sin\theta}{(t^2\cos^2\theta + t^2\sin^2\theta) \cdot t} = \cos^2\theta\sin\theta$$

**모순**: $D_v f = \nabla f \cdot v = 0 \cdot \cos\theta + 0 \cdot \sin\theta = 0$이어야 하지만, 실제로는 $\cos^2\theta\sin\theta \neq 0$ ($\theta \neq 0, \pi/2, \pi$). 따라서 **전미분이 존재하지 않는다**.

---

### 반례 2.2 — 더 강한 반례: 방향도함수도 선형이 아닌 함수

$$f(x, y) = \begin{cases} \frac{xy^2}{x^2 + y^4} & (x, y) \neq (0, 0) \\ 0 & (x, y) = (0, 0) \end{cases}$$

직선 경로 $y = tx$: $f(x, tx) = \frac{x(tx)^2}{x^2 + (tx)^4} = \frac{t^2 x^3}{x^2(1 + t^4 x^2)} \xrightarrow{x \to 0} 0$.

포물선 경로 $x = t^2$, $y = t$: $f(t^2, t) = \frac{t^2 \cdot t^2}{t^4 + t^4} = \frac{1}{2}$.

즉, 원점으로 가는 서로 다른 경로에서 극한값이 다르다 → **(0,0)에서 연속도 아니다!** 편미분이 존재하더라도 연속성조차 보장되지 않는다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# 1. 방향도함수 수치 계산
# ─────────────────────────────────────────────

def directional_derivative_numerical(f, point, direction, h=1e-5):
    """수치 방향도함수: (f(a + h*v) - f(a - h*v)) / (2h)"""
    point = np.array(point, dtype=float)
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)  # 단위벡터화
    return (f(point + h * direction) - f(point - h * direction)) / (2 * h)

def gradient_numerical(f, point, h=1e-5):
    """수치 gradient: 각 방향 편미분"""
    point = np.array(point, dtype=float)
    n = len(point)
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n); e_i[i] = 1.0
        grad[i] = (f(point + h * e_i) - f(point - h * e_i)) / (2 * h)
    return grad

# 테스트 함수: f(x,y) = x² + 2xy + y³
def f_test(p): return p[0]**2 + 2*p[0]*p[1] + p[1]**3
a = np.array([1.0, 2.0])

# 이론적 gradient: (2x + 2y, 2x + 3y²)
grad_theory = np.array([2*a[0] + 2*a[1], 2*a[0] + 3*a[1]**2])
grad_num    = gradient_numerical(f_test, a)

print("=== f(x,y) = x² + 2xy + y³, a = (1, 2) ===")
print(f"이론 gradient:  {grad_theory}")
print(f"수치 gradient:  {grad_num}")
print(f"오차:           {np.abs(grad_theory - grad_num)}")

# 임의 방향 v = (1/√2, 1/√2)
v = np.array([1, 1]) / np.sqrt(2)
D_v_formula  = grad_theory @ v          # ∇f · v
D_v_numerical = directional_derivative_numerical(f_test, a, v)

print(f"\n방향 v = {v}")
print(f"∇f · v (공식):  {D_v_formula:.8f}")
print(f"수치 방향도함수: {D_v_numerical:.8f}")
print(f"일치 여부: {np.isclose(D_v_formula, D_v_numerical)}")

# ─────────────────────────────────────────────
# 2. 반례: 편미분 있어도 전미분 없는 함수 시각화
# ─────────────────────────────────────────────

def f_counterexample(p):
    """f(x,y) = x²y / (x² + y²), f(0,0) = 0"""
    x, y = p[0], p[1]
    denom = x**2 + y**2
    if denom < 1e-15:
        return 0.0
    return x**2 * y / denom

# 방향에 따른 방향도함수 값 (이론: cos²θ · sinθ)
thetas = np.linspace(0, 2 * np.pi, 360)
D_v_numerical_vals = []
D_v_formula_vals   = []

for theta in thetas:
    v = np.array([np.cos(theta), np.sin(theta)])
    D_num = directional_derivative_numerical(f_counterexample,
                                              [0.0, 0.0], v, h=1e-6)
    D_v_numerical_vals.append(D_num)
    D_v_formula_vals.append(0.0)  # ∇f(0,0) = (0,0) → ∇f·v = 0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 좌: 방향도함수가 방향에 따라 달라짐 (전미분 없음)
axes[0].plot(np.degrees(thetas), D_v_numerical_vals, 'b-', linewidth=1.5,
             label=r'$D_v f(0,0)$ (수치)')
axes[0].plot(np.degrees(thetas), D_v_formula_vals, 'r--', linewidth=1.5,
             label=r'$\nabla f \cdot v = 0$ (공식, 틀림!)')
axes[0].fill_between(np.degrees(thetas), D_v_numerical_vals, D_v_formula_vals,
                     alpha=0.2, color='red', label='오차 (전미분 없음)')
axes[0].set_xlabel('방향 θ (도)')
axes[0].set_ylabel(r'$D_v f(0,0)$')
axes[0].set_title(r'$f=x^2y/(x^2+y^2)$: 편미분 O, 전미분 X', fontsize=11)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# 우: 3D 곡면 시각화
X = np.linspace(-1, 1, 60)
Y = np.linspace(-1, 1, 60)
Xg, Yg = np.meshgrid(X, Y)
Z = np.where(Xg**2 + Yg**2 < 1e-15, 0.0, Xg**2 * Yg / (Xg**2 + Yg**2))

ax3d = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax3d.plot_surface(Xg, Yg, Z, cmap='coolwarm', alpha=0.8, linewidth=0)
ax3d.set_title(r'$f(x,y) = \frac{x^2 y}{x^2+y^2}$', fontsize=11)
ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('f(x,y)')

plt.tight_layout()
plt.savefig('ch2-01-partial-directional.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. SymPy로 기호 편미분 검증
# ─────────────────────name ──────────────────
import sympy as sp

x, y = sp.symbols('x y')
funcs = {
    "x²y/(x²+y²)": x**2*y / (x**2 + y**2),
    "sin(xy)":      sp.sin(x*y),
    "exp(x²+y²)":   sp.exp(x**2 + y**2),
}

print("\n[SymPy 편미분]")
for name, f_sym in funcs.items():
    df_dx = sp.diff(f_sym, x)
    df_dy = sp.diff(f_sym, y)
    print(f"f = {name}")
    print(f"  ∂f/∂x = {sp.simplify(df_dx)}")
    print(f"  ∂f/∂y = {sp.simplify(df_dy)}\n")
```

**출력**:
```
=== f(x,y) = x² + 2xy + y³, a = (1, 2) ===
이론 gradient:  [ 6. 14.]
수치 gradient:  [ 6. 14.]
오차:           [1.33e-10  0.00e+00]

방향 v = [0.707 0.707]
∇f · v (공식):  14.14213562
수치 방향도함수: 14.14213562
일치 여부: True
```

---

## 🔗 AI/ML 연결

### 경사하강법의 방향 선택 근거

방향도함수 $D_v f = \nabla f \cdot v$를 최소화하는 방향 $v$는 $-\nabla f / \|\nabla f\|$다 (다음 문서 [Ch2-03]에서 코시-슈바르츠로 엄밀히 증명).

### Numerical Gradient Check

PyTorch의 `gradcheck`은 다음을 비교한다:
```python
# 방향 e_i의 방향도함수 수치 계산
numerical = (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)
# vs autograd의 편미분
analytical = grad_via_autograd[i]
```

편미분이 모두 맞아도 전미분이 없다면 `gradcheck`이 특정 방향에서 실패할 수 있다. (실제로는 매끄러운 함수를 사용하므로 이런 일은 드물다.)

### Second-Order Directional Derivative

뉴턴 방법에서 방향 $v$의 **2차 방향도함수**:
$$D_v^2 f(x) = v^\top H(x) v$$

이것이 양수이면 $v$ 방향이 "위로 볼록"하여 극솟값 방향, 음수이면 안장점 또는 극댓값 방향이다.

---

## ⚖️ 가정과 한계

| 조건 | 보장 |
|------|------|
| 편미분 존재 | 방향도함수 존재를 보장하지 않음 |
| 모든 방향도함수 존재 | 전미분 존재를 보장하지 않음 |
| 전미분 존재 | $D_v f = \nabla f \cdot v$ 성립, 연속 보장 |
| 편미분이 연속 ($C^1$) | 전미분 존재 보장 (충분조건) |

---

## 📌 핵심 정리

$$\text{편미분 존재} \not\Rightarrow \text{전미분 존재} \not\Rightarrow \text{연속}$$
$$\text{전미분 존재} \Rightarrow D_v f = \nabla f \cdot v \quad (\text{모든 방향 } v)$$
$$C^1 \Rightarrow \text{전미분 존재} \Rightarrow \text{연속} \Rightarrow \text{편미분 존재}$$

---

## 🤔 생각해볼 문제

**문제 1**: $f(x,y) = \sqrt{|xy|}$는 $(0,0)$에서 편미분이 존재하는가? 전미분은?

<details>
<summary>해설</summary>

$\partial f/\partial x(0,0) = \lim_{h\to 0} \sqrt{|h \cdot 0|}/h = 0$. 편미분 존재.  
방향 $v = (1/\sqrt{2}, 1/\sqrt{2})$: $D_v f = \lim_{t\to 0} \sqrt{t^2/2}/t = 1/\sqrt{2} \neq \nabla f \cdot v = 0$. 전미분 없음.

</details>

**문제 2**: $f(x,y) = x^2 + y^2$에서 점 $(1,1)$에서 방향 $v = (3/5, 4/5)$의 방향도함수를 구하라.

<details>
<summary>해설</summary>

$\nabla f = (2x, 2y) = (2, 2)$. $D_v f = (2, 2) \cdot (3/5, 4/5) = 6/5 + 8/5 = 14/5 = 2.8$.

</details>

**문제 3 (AI 연결)**: 딥러닝에서 gradient를 계산할 때 "편미분의 모음"이 아니라 "전미분"이 필요한 이유를 설명하라. 어떤 상황에서 양자가 다를 수 있는가?

<details>
<summary>해설</summary>

합성 함수의 역전파에서 연쇄법칙 $J_{f\circ g} = J_f \cdot J_g$은 전미분(야코비안)의 합성이다. 만약 중간 함수가 전미분가능하지 않으면 이 공식이 성립하지 않아 역전파가 틀린 결과를 낼 수 있다. Custom autograd 함수 구현 시 `forward`와 `backward`를 올바르게 정의해야 하는 이유다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch1-05. 미분가능 함수의 성질과 반례](../ch1-analysis-foundations/05-differentiability-properties.md) | [📚 README](../README.md) | [02. 전미분과 야코비안 ▶](./02-total-derivative-jacobian.md) |

</div>
