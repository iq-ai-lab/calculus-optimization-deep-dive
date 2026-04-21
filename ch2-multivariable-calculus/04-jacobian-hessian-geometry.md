# 04. 야코비안과 헤시안의 기하

## 🎯 핵심 질문

- 야코비안의 행렬식이 국소 부피 변환율인 이유는?
- 헤시안이 "2차 곡률"이라는 말의 기하학적 의미는?
- 헤시안의 고유값과 고유벡터는 무엇을 말해주는가?
- 딥러닝에서 헤시안을 어떻게 근사하여 활용하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **뉴턴 방법**: $x \leftarrow x - H^{-1}\nabla f$는 헤시안을 이용해 2차 수렴을 달성. L-BFGS는 헤시안을 근사
- **Loss Landscape 해석**: 학습된 모델의 헤시안 고유값 분포는 일반화 능력과 연관됨 (Flat Minima 이론)
- **Natural Gradient**: Fisher Information Matrix는 파라미터 공간의 헤시안에 해당하며, 효율적인 학습을 가능케 함

---

## 📐 수학적 선행 조건

- [Ch2-02. 전미분과 야코비안](./02-total-derivative-jacobian.md)
- [Ch1-04. 테일러 정리](../ch1-analysis-foundations/04-mvt-taylor.md)
- 선형대수: 고유값, 대칭행렬의 스펙트럼 분해 (→ [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive))

---

## ✏️ 엄밀한 정의

### 정의 2.7 — 헤시안 행렬

$f: \mathbb{R}^n \to \mathbb{R}$이 $C^2$ 함수이면, **헤시안(Hessian)** $H_f(a) \in \mathbb{R}^{n \times n}$:

$$[H_f(a)]_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}(a)$$

슈바르츠 정리(Schwarz's Theorem)에 의해 $f \in C^2$이면 $H$는 대칭행렬: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$.

### 정의 2.8 — 이차형식

$H$가 대칭행렬일 때, 함수 $Q(v) = v^\top H v$를 **이차형식(quadratic form)**이라 한다.

- $H$가 **양의 정부호(PD)**: $\forall v \neq 0$, $v^\top H v > 0$
- $H$가 **반양정부호(PSD)**: $\forall v$, $v^\top H v \geq 0$
- $H$가 **부정부호(Indefinite)**: $v^\top H v > 0$인 $v$와 $< 0$인 $v$ 둘 다 존재

---

## 🔬 정리와 증명

### 정리 2.8 — 헤시안은 2차 테일러 항의 계수

**명제**: $f \in C^2$이면:

$$f(a + h) = f(a) + \nabla f(a)^\top h + \frac{1}{2} h^\top H_f(a) h + o(\|h\|^2)$$

**증명**:  
$g(t) = f(a + th)$로 놓으면 $g \in C^2$이고:

$$g(1) = g(0) + g'(0) + \frac{1}{2}g''(0) + o(1) \quad (t=0 \text{ 근방 테일러})$$

$g'(t) = \nabla f(a+th)^\top h$이므로 $g'(0) = \nabla f(a)^\top h$.

$g''(t) = h^\top H_f(a+th) h$이므로 $g''(0) = h^\top H_f(a) h$.

대입하면 위 식 성립. $\square$

---

### 정리 2.9 — Spectral Theorem (대칭행렬의 고유값 분해)

**명제**: 대칭행렬 $H \in \mathbb{R}^{n\times n}$은 **직교 대각화** 가능하다:

$$H = Q \Lambda Q^\top, \quad Q^\top Q = I, \quad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$$

- 고유값 $\lambda_i \in \mathbb{R}$ (모두 실수)
- 고유벡터 $q_i$ (서로 직교)

**헤시안의 기하학적 해석**:  
고유벡터 방향 $q_i$에서 함수의 "곡률(curvature)"은 대응 고유값 $\lambda_i$:

$$q_i^\top H q_i = \lambda_i$$

- $\lambda_i > 0$: $q_i$ 방향으로 위로 볼록 (2차 증가)
- $\lambda_i < 0$: $q_i$ 방향으로 아래로 볼록 (2차 감소)
- $\lambda_i = 0$: 해당 방향의 곡률이 0 (평탄)

---

### 정리 2.10 — 야코비안의 행렬식 = 부피 변환율

**명제**: $f: \mathbb{R}^n \to \mathbb{R}^n$이 전미분가능이면, 점 $a$ 근방에서 $f$는 부피를 $|\det J_f(a)|$배 변환한다.

**직관**: $n$-차원 초직육면체 $[a, a+h_1 e_1] \times \cdots \times [a, a+h_n e_n]$의 부피는 $\prod h_i$이고, $f$를 통해 변환된 이미지의 부피는 근사적으로 $|\det J_f(a)| \prod h_i$이다.

**응용**: 다변수 적분의 변수 변환:
$$\int_{f(D)} g(y)\, dy = \int_D g(f(x)) |\det J_f(x)|\, dx$$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# 1. SymPy로 헤시안 기호 계산
# ─────────────────────────────────────────────

x, y = sp.symbols('x y')

test_funcs = {
    "x² + y²":           x**2 + y**2,
    "x² - y²":           x**2 - y**2,
    "x²y + y³":          x**2*y + y**3,
    "exp(x²+y²)":        sp.exp(x**2 + y**2),
}

print("[SymPy 헤시안 계산]")
for name, f_sym in test_funcs.items():
    H = sp.hessian(f_sym, [x, y])
    eigenvals = list(H.eigenvals().keys())
    print(f"\nf = {name}")
    print(f"  헤시안 = {H}")
    print(f"  고유값 = {[sp.simplify(ev) for ev in eigenvals]}")

# ─────────────────────────────────────────────
# 2. 3D 곡면과 헤시안 고유값 시각화
# ─────────────────────────────────────────────

def f_bowl(p):    return p[0]**2 + 4*p[1]**2                 # PD: λ=[2,8]
def f_saddle(p):  return p[0]**2 - 2*p[1]**2                 # Indef: λ=[2,-4]
def f_valley(p):  return p[0]**2 + 0.1*p[1]**2               # PD, 조건수 큼

funcs_3d = [
    (f_bowl,   "볼 (PD): λ₁=2, λ₂=8",   np.array([[2,0],[0,8]])),
    (f_saddle, "안장점 (Indef): λ₁=2, λ₂=-4", np.array([[2,0],[0,-4]])),
    (f_valley, "계곡 (PD, κ=40): λ₁=2, λ₂=0.2", np.array([[2,0],[0,0.2]])),
]

fig = plt.figure(figsize=(15, 5))

x_r = np.linspace(-2, 2, 50)
y_r = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x_r, y_r)

for i, (func, title, H_theory) in enumerate(funcs_3d, 1):
    Z = np.array([[func(np.array([xi, yi])) for xi in x_r] for yi in y_r])
    Z = np.clip(Z, -10, 10)
    
    ax = fig.add_subplot(1, 3, i, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, linewidth=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
    ax.set_zlim(-10, 10)

plt.suptitle('헤시안 고유값과 함수 기하학', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ch2-04-hessian-geometry-3d.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. 수치 헤시안 계산
# ─────────────────────────────────────────────

def numerical_hessian(f, x, h=1e-4):
    """2차 유한차분으로 헤시안 계산"""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i = np.zeros(n); e_i[i] = 1
            e_j = np.zeros(n); e_j[j] = 1
            H[i,j] = (f(x+h*e_i+h*e_j) - f(x+h*e_i-h*e_j)
                     - f(x-h*e_i+h*e_j) + f(x-h*e_i-h*e_j)) / (4*h**2)
    return H

def f_test(x): return x[0]**2 + 3*x[0]*x[1] + 2*x[1]**2

H_theory = np.array([[2, 3], [3, 4]])
H_num    = numerical_hessian(f_test, np.array([0.0, 0.0]))

print("\n[수치 헤시안 검증]")
print(f"이론 헤시안:\n{H_theory}")
print(f"수치 헤시안:\n{H_num}")
print(f"최대 오차: {np.abs(H_theory - H_num).max():.2e}")

eigenvals, eigenvecs = np.linalg.eigh(H_theory)
print(f"\n고유값: {eigenvals}")
print(f"고유벡터:\n{eigenvecs}")
kappa = eigenvals.max() / eigenvals.min()
print(f"조건수 κ = {kappa:.2f}")
```

---

## 🔗 AI/ML 연결

### Hessian-Vector Product (HVP)

헤시안 전체 $O(n^2)$ 저장 없이 특정 방향의 2차 정보를 얻는 방법:
$$Hv = \nabla(\nabla f \cdot v)$$

PyTorch에서 두 번의 역전파로 계산: `torch.autograd.grad(grad_f @ v, x)`

### 딥러닝에서 헤시안 활용

| 방법 | 헤시안 활용 | 비용 |
|------|-----------|------|
| 뉴턴 방법 | $H^{-1}\nabla f$ | $O(n^3)$ — 대규모 불가 |
| L-BFGS | $H^{-1}$ 근사 (벡터 쌍) | $O(mk)$ — 중규모 |
| Adam | 대각 Hessian 근사 | $O(n)$ — 대규모 가능 |
| Natural Gradient | Fisher = $\mathbb{E}[J^\top J]$ 역수 | $O(n^2)$ — 근사 필요 |

### Loss Landscape의 Flat Minima

손실의 헤시안 고유값 $\lambda_{\max}$가 작은 "flat minimum"은 일반화가 좋다는 주장 (Hochreiter & Schmidhuber 1997, Keskar 2016):
- Flat: 파라미터 섭동에 손실이 덜 민감 → 테스트 도메인 shift에 robust
- Sharp: 훈련 데이터에 과적합, 작은 분포 변화에 큰 성능 저하

---

## 📌 핵심 정리

$$f(a+h) = f(a) + \nabla f^\top h + \frac{1}{2}h^\top H h + o(\|h\|^2)$$

| 헤시안 부호 | 의미 | 최적화에서 |
|-----------|------|-----------|
| PD ($\lambda_i > 0$) | 국소 최솟값 방향 | 볼록, 안정적 수렴 |
| PSD ($\lambda_i \geq 0$) | 최솟값 가능 (퇴화) | 수렴 보장 약화 |
| Indefinite | 안장점 | GD 탈출 필요 |
| ND ($\lambda_i < 0$) | 국소 최댓값 | GD로 접근 불가 |

---

## 🤔 생각해볼 문제

**문제 1**: $f(x,y) = x^2 + 4xy + 4y^2$의 헤시안을 구하라. 이 함수의 최솟값이 유일한가?

<details><summary>해설</summary>
$H = \begin{pmatrix}2&4\\4&8\end{pmatrix}$. $\det H = 16-16=0$, 즉 PSD이지만 PD 아님. $f=(x+2y)^2 \geq 0$. 최솟값 0은 직선 $x=-2y$ 전체에서 달성 (유일하지 않음).
</details>

**문제 2**: 헤시안 $H = \begin{pmatrix}3&1\\1&3\end{pmatrix}$의 고유값과 고유벡터를 구하고, 이차형식의 등위선(타원) 주축 방향을 설명하라.

<details><summary>해설</summary>
$\lambda_1=4$, $\lambda_2=2$. 고유벡터 $(1,1)/\sqrt{2}$, $(1,-1)/\sqrt{2}$. 등위선 타원의 단축은 $\lambda=4$ 방향 (더 급한 곡률), 장축은 $\lambda=2$ 방향.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Gradient와 코시-슈바르츠](./03-gradient-cauchy-schwarz.md) | [📚 README](../README.md) | [05. 다변수 연쇄법칙 ▶](./05-chain-rule-general.md) |

</div>
