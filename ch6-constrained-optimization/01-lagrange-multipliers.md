# 01. 라그랑주 승수법

## 🎯 핵심 질문
등식 제약 조건이 있을 때 최적화 문제를 어떻게 풀 수 있을까? 제약 조건 없는 최적화와 어떻게 다를까?

## 🔍 왜 이 개념이 AI에서 중요한가

라그랑주 승수법은 **제약 조건이 있는 최적화**의 기초이다. AI/ML에서 나타나는 많은 문제들이 제약을 포함한다:

- **주성분 분석 (PCA)**: 단위 노름 제약 하에서 분산 최대화
- **가중치 정규화**: L2 norm 제약 하의 손실 함수 최소화
- **정책 최적화**: 거리 제약 하의 보상 최대화
- **신경망 구조 탐색 (NAS)**: 리소스 제약 하의 정확도 최적화

라그랑주 승수법을 이해하면 이들 문제의 수학적 구조를 파악하고 효율적인 알고리즘을 설계할 수 있다.

## 📐 수학적 선행 조건

다음 개념에 대한 이해가 필요하다:

- **벡터 미분**: 그래디언트 $\nabla f$, 야코비안, 헤시안
- **선형 대수**: 벡터 공간, 선형 독립성, 고유값/고유벡터
- **다변수 함수**: 연쇄 법칙 (chain rule), 내재 미분
- **제약 다양체**: $g(x) = 0$를 만족하는 점들의 집합의 기하학적 성질

## ✏️ 정의와 핵심 도구

### 등식 제약 최적화 문제
$$\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad g(x) = 0$$

여기서:
- $f: \mathbb{R}^n \to \mathbb{R}$ (목적함수, 스칼라 값)
- $g: \mathbb{R}^n \to \mathbb{R}^m$ (제약함수, $m$개의 등식 제약)
- $m \leq n$ (제약이 많으면 문제가 과결정됨)

### 라그랑지안
$$\mathcal{L}(x, \lambda) = f(x) + \lambda^\top g(x)$$

여기서 $\lambda \in \mathbb{R}^m$을 **라그랑주 승수(Lagrange multiplier)** 라고 부른다.

### 기하적 직관
최적점 $x^*$에서 목적함수의 그래디언트 $\nabla f(x^*)$는 제약함수의 그래디언트들의 선형결합 방향이어야 한다:
$$\nabla f(x^*) = \sum_{i=1}^{m} \lambda_i \nabla g_i(x^*)$$

**왜?** 제약 다양체 위에서 $x^*$가 국소 최솟값이려면, $x^*$에서 제약을 만족하면서 목적함수를 감소시킬 수 없어야 한다. 이는 $\nabla f(x^*)$가 제약 다양체의 접공간(tangent space)에 수직이어야 함을 의미한다.

## 🔬 정의와 증명

### 정리 1: 라그랑주 필요조건 (KKT 1차 조건, 등식 케이스)

**가정:**
- $x^*$가 $\min f(x)$ s.t. $g(x) = 0$의 국소 최솟값
- LICQ (Linear Independence Constraint Qualification) 만족: $\nabla g_1(x^*), \ldots, \nabla g_m(x^*)$가 선형독립

**결론:** 존재하는 $\lambda^* \in \mathbb{R}^m$이 있어서:
$$\nabla f(x^*) + \sum_{i=1}^{m} \lambda_i^* \nabla g_i(x^*) = 0$$
또는 $\nabla_x \mathcal{L}(x^*, \lambda^*) = 0$

**증명:**

$x^*$가 제약 다양체 위의 국소 최솟값이므로, 제약을 만족하는 임의의 방향 $d$ (접선 방향)에 대해:
$$\nabla f(x^*)^\top d = 0$$

제약 조건 $g(x) = 0$을 전미분하면:
$$\nabla g(x^*)^\top d = 0$$

즉, 접선 방향은 $\{\nabla g_i(x^*)\}$의 직교 보공간에 속한다. LICQ 하에서 $\nabla g_1(x^*), \ldots, \nabla g_m(x^*)$가 선형독립이므로, 제약 다양체의 접공간 차원은 $n - m$이다.

$\nabla f(x^*)$가 접공간의 모든 벡터 $d$에 수직이므로:
$$\nabla f(x^*) \in \text{span}\{\nabla g_1(x^*), \ldots, \nabla g_m(x^*)\}$$

따라서 계수 $\lambda_1^*, \ldots, \lambda_m^*$이 존재해서:
$$\nabla f(x^*) = -\sum_{i=1}^{m} \lambda_i^* \nabla g_i(x^*)$$

정리하면:
$$\nabla_x \mathcal{L}(x^*, \lambda^*) = \nabla f(x^*) + \sum_{i=1}^{m} \lambda_i^* \nabla g_i(x^*) = 0$$

### 정리 2: 2차 충분조건

**가정:**
- $\nabla_x \mathcal{L}(x^*, \lambda^*) = 0$
- Bordered Hessian이 제약 다양체의 접공간에서 양정부호:
$$d^\top \nabla_{xx}^2 \mathcal{L}(x^*, \lambda^*) d > 0 \quad \forall d \neq 0, \; \nabla g(x^*)^\top d = 0$$

**결론:** $x^*$는 제약 최적화 문제의 국소 최솟값이다.

### 주요 경우: PCA와 최대 분산 방향

**문제:** 단위 노름 제약 하에서 최대 분산 찾기
$$\max_{x} x^\top \Sigma x \quad \text{s.t.} \quad x^\top x = 1$$

여기서 $\Sigma = \frac{1}{n} X^\top X$ (표본 공분산)

**라그랑지안:**
$$\mathcal{L}(x, \lambda) = x^\top \Sigma x - \lambda(x^\top x - 1)$$

**1차 조건:**
$$\nabla_x \mathcal{L} = 2\Sigma x - 2\lambda x = 0$$
$$\Sigma x = \lambda x$$

즉, $x$는 $\Sigma$의 고유벡터이고, 최대 분산은 최대 고유값 $\lambda_{\max}$이다!

**라일리 몫(Rayleigh quotient):**
$$\max_x \frac{x^\top \Sigma x}{x^\top x} = \lambda_{\max}(\Sigma)$$

제약 없이도 동일한 결과를 얻는다는 것을 보이자. 아래 함수를 정의하면:
$$R(x) = \frac{x^\top \Sigma x}{x^\top x}$$

미분하면:
$$\nabla R(x) = \frac{2\Sigma x(x^\top x) - 2x(x^\top \Sigma x)}{(x^\top x)^2}$$

$\nabla R(x) = 0$이 되려면:
$$\Sigma x (x^\top x) = x(x^\top \Sigma x)$$

양변에 $x^\top$를 곱하고, $\nabla R(x) = 0$인 점에서 $x^\top \Sigma x = \lambda x^\top x$이므로:
$$\Sigma x = \lambda x$$

따라서 라그랑주 승수법이 정확히 라일리 몫의 최적점을 찾는다.

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# 설정
np.random.seed(42)
n_samples = 100

# 제약이 있는 최적화 문제 정의
def objective(x):
    """목적함수: f(x) = (x1 - 1)^2 + (x2 - 2)^2"""
    return (x[0] - 1)**2 + (x[1] - 2)**2

def constraint(x):
    """제약조건: g(x) = x1^2 + x2^2 - 1 = 0 (단위원)"""
    return x[0]**2 + x[1]**2 - 1.0

# scipy를 이용한 제약 최적화
from scipy.optimize import NonlinearConstraint
constraint_obj = {'type': 'eq', 'fun': constraint}
x0 = np.array([0.1, 0.1])
result = minimize(objective, x0, method='SLSQP', constraints=constraint_obj)
x_opt = result.x
print(f"제약 최적해: x = {x_opt}")
print(f"최적값: f(x*) = {objective(x_opt):.6f}")

# 라그랑주 조건 검증
def lagrangian_gradient(x, lam):
    """라그랑지안 그래디언트"""
    grad_f = np.array([2*(x[0] - 1), 2*(x[1] - 2)])
    grad_g = np.array([2*x[0], 2*x[1]])
    return grad_f + lam * grad_g

# 최적점에서 라그랑주 승수 계산
# ∇f(x*) = λ ∇g(x*)에서:
grad_f_opt = np.array([2*(x_opt[0] - 1), 2*(x_opt[1] - 2)])
grad_g_opt = np.array([2*x_opt[0], 2*x_opt[1]])

# λ 계산: ∇f · ∇g / (∇g · ∇g)
lambda_opt = np.dot(grad_f_opt, grad_g_opt) / np.dot(grad_g_opt, grad_g_opt)
print(f"\n라그랑주 승수: λ* = {lambda_opt:.6f}")

# 라그랑지안 1차 조건 확인
lagrangian_grad = lagrangian_gradient(x_opt, lambda_opt)
print(f"∇L(x*, λ*) = {lagrangian_grad}")
print(f"|∇L| = {np.linalg.norm(lagrangian_grad):.2e}")

# === 시각화 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 왼쪽: 등위선과 제약곡선
ax = axes[0]
x1_range = np.linspace(-2, 2, 200)
x2_range = np.linspace(-2.5, 2.5, 200)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = (X1 - 1)**2 + (X2 - 2)**2

# 목적함수 등위선
contours = ax.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.7)
ax.clabel(contours, inline=True, fontsize=8)

# 제약곡선 (단위원)
theta = np.linspace(0, 2*np.pi, 1000)
x1_circle = np.cos(theta)
x2_circle = np.sin(theta)
ax.plot(x1_circle, x2_circle, 'r-', linewidth=2.5, label='제약: $x_1^2 + x_2^2 = 1$')

# 최적점
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=20, label=f'최적점 $x^*$')

# 최적점에서의 그래디언트 시각화
scale = 0.3
ax.arrow(x_opt[0], x_opt[1], scale*grad_f_opt[0], scale*grad_f_opt[1], 
         head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
ax.text(x_opt[0] + 0.3, x_opt[1] + 0.3, r'$\nabla f(x^*)$', fontsize=11, color='blue')

ax.arrow(x_opt[0], x_opt[1], scale*grad_g_opt[0], scale*grad_g_opt[1], 
         head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.7)
ax.text(x_opt[0] - 0.4, x_opt[1] - 0.25, r'$\nabla g(x^*)$', fontsize=11, color='green')

ax.set_xlim(-2, 2)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('등식 제약 최적화: 등위선 vs 제약곡선', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_aspect('equal')

# 오른쪽: PCA 예제
ax = axes[1]

# 표본 생성 (2D)
mean = np.array([0, 0])
cov = np.array([[2, 0.8], [0.8, 1]])
X = np.random.multivariate_normal(mean, cov, 300)

# 산점도
ax.scatter(X[:, 0], X[:, 1], alpha=0.4, s=20, color='gray')

# 공분산 행렬
Sigma = np.cov(X.T)

# 고유값 분해
eigenvalues, eigenvectors = np.linalg.eig(Sigma)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\n=== PCA: 라그랑주 승수법 적용 ===")
print(f"공분산 행렬의 고유값: {eigenvalues}")
print(f"최대 고유값: {eigenvalues[0]:.4f}")
print(f"최대 고유벡터: {eigenvectors[:, 0]}")

# 최대 분산 방향 시각화
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]
scale = 2.0

ax.arrow(0, 0, scale*v1[0]*np.sqrt(eigenvalues[0]), 
         scale*v1[1]*np.sqrt(eigenvalues[0]), 
         head_width=0.15, head_length=0.15, 
         fc='red', ec='red', linewidth=2, label='PC1 (최대 분산)')
ax.arrow(0, 0, scale*v2[0]*np.sqrt(eigenvalues[1]), 
         scale*v2[1]*np.sqrt(eigenvalues[1]), 
         head_width=0.15, head_length=0.15, 
         fc='blue', ec='blue', linewidth=2, label='PC2')

# 단위원 (제약)
circle = Circle((0, 0), 1, fill=False, edgecolor='black', 
                linewidth=2, linestyle='--', label='제약: $\|v\|=1$')
ax.add_patch(circle)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel('$v_1$', fontsize=12)
ax.set_ylabel('$v_2$', fontsize=12)
ax.set_title('PCA: 라그랑주 승수법으로 고유벡터 찾기', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/tmp/lagrange_multipliers.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장됨: /tmp/lagrange_multipliers.png")
plt.show()

# === 수치 검증 ===
print("\n=== 라그랑주 조건 검증 ===")
print(f"제약 조건 만족도: g(x*) = {constraint(x_opt):.2e}")
print(f"기울기 평행성: ∇f ∥ ∇g")
print(f"  ∇f(x*) / |∇f| = {grad_f_opt / np.linalg.norm(grad_f_opt)}")
print(f"  ∇g(x*) / |∇g| = {grad_g_opt / np.linalg.norm(grad_g_opt)}")
```

## 🔗 AI/ML 연결

### 1. 주성분 분석 (PCA)
주어진 데이터의 최대 분산 방향을 찾는 것이 PCA의 핵심이다. 라그랑주 승수법은 이를 수학적으로 정확히 푼다.

### 2. 가중치 정규화 제약
신경망에서 $\|w\|_2 \leq r$ 제약 하에서 손실함수 최소화는 라그랑주 승수법과 동치이다:
$$\min L(w) \text{ s.t. } \|w\|_2 \leq r \equiv \min L(w) + \lambda \|w\|_2^2$$

### 3. Variational Inference
잠재변수 분포 $q(z)$를 최적화할 때 정규화 제약 $\int q(z)dz = 1$이 자동으로 처리된다:
$$\mathcal{L} = \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z)] + \lambda(\int q(z)dz - 1)$$

## 📌 핵심 정리

| 개념 | 설명 |
|------|------|
| **라그랑지안** | $\mathcal{L}(x,\lambda) = f(x) + \lambda^\top g(x)$ |
| **1차 필요조건** | $\nabla_x \mathcal{L} = 0$, $g(x) = 0$ |
| **기하적 의미** | 최적점에서 $\nabla f \parallel \nabla g$ |
| **2차 충분조건** | Bordered Hessian이 제약 다양체에서 양정부호 |
| **PCA 응용** | 고유벡터는 라그랑주 승수 조건의 해 |

## 🤔 생각해볼 문제

1. **여러 제약의 경우**: $m$개의 등식 제약 $g_i(x) = 0$ ($i=1,\ldots,m$)이 있을 때, 라그랑지안과 1차 조건을 쓰시오.

2. **LICQ 조건**: LICQ가 실패하는 경우를 생각해보자. 예: $g(x) = x_1 = 0$, $h(x) = x_1^2 = 0$. 두 번째 제약의 그래디언트는?

3. **2차 조건의 의미**: "Bordered Hessian이 제약 다양체에서 양정부호"는 무엇을 의미하는가? 신경해야 할 고유값은 어떤 범위인가?

4. **제약 없는 등가 문제**: 등식 제약을 페널티 항으로 바꾸면: $\min f(x) + \rho \|g(x)\|^2$. 이것이 원래 문제와 어떤 관계가 있는가?

5. **대규모 시스템**: 고차원에서 라그랑주 승수법의 계산 복잡도는? 헤시안 계산이 불가능하면 어떻게 할까?

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch5-05. Autograd 구현](../ch5-backprop-autograd/05-autograd-numpy-implementation.md) | [📚 README](../README.md) | [02. 부등식 제약과 KKT ▶](./02-inequality-kkt.md) |

</div>
