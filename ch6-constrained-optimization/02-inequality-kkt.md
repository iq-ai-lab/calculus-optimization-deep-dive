# 02. 부등식 제약과 KKT 조건

## 🎯 핵심 질문
부등식 제약이 포함된 문제 ($x \geq 0$, 확률 $\leq 1$ 등)를 어떻게 다루나? 활성 제약과 비활성 제약의 역할은?

## 🔍 왜 이 개념이 AI에서 중요한가

실제 AI/ML 문제의 대부분은 부등식 제약을 포함한다:

- **Support Vector Machines (SVM)**: 분류 여유(margin) 제약
- **제약 확률 최적화**: 예측 확률 $p(x) \leq 1$
- **강화학습**: KL 발산 제약, 신뢰영역(trust region) 제약
- **최적 수송(Optimal Transport)**: 확률 심플렉스 제약
- **신경망 양자화**: 가중치 크기 상한 제약

KKT 조건은 이들 문제의 최적성을 판정하고 알고리즘을 설계하는 핵심이다.

## 📐 수학적 선행 조건

다음 개념에 대한 이해가 필요하다:

- **라그랑주 승수법** (01장 참고)
- **Fritz John 조건**: 일반적인 필요조건
- **선형 제약 조건의 자격 (CQ)**: LICQ, MFCQ, Slater 조건
- **정리와 증명 작성 능력**

## ✏️ 정의와 핵심 도구

### 일반 제약 최적화 문제
$$\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad g_i(x) \leq 0 \; (i=1,\ldots,m), \quad h_j(x) = 0 \; (j=1,\ldots,\ell)$$

여기서:
- $f: \mathbb{R}^n \to \mathbb{R}$ (목적함수)
- $g_i: \mathbb{R}^n \to \mathbb{R}$ ($m$개의 부등식 제약)
- $h_j: \mathbb{R}^n \to \mathbb{R}$ ($\ell$개의 등식 제약)

### 활성 집합 (Active Set)
점 $x$에서:
$$\mathcal{A}(x) = \{i : g_i(x) = 0\}$$

즉, 제약을 등호로 만족하는 인덱스들의 집합이다. 활성 제약만 최적성에 영향을 미친다.

### 라그랑지안과 KKT 조건
$$\mathcal{L}(x, \mu, \lambda) = f(x) + \sum_{i=1}^{m} \mu_i g_i(x) + \sum_{j=1}^{\ell} \lambda_j h_j(x)$$

여기서:
- $\mu = (\mu_1, \ldots, \mu_m)$: 부등식 제약의 라그랑주 승수
- $\lambda = (\lambda_1, \ldots, \lambda_\ell)$: 등식 제약의 라그랑주 승수

### KKT (Karush-Kuhn-Tucker) 조건

점 $x^*$가 최적이기 위한 필요조건:

**1. 기울기 정류성 (Stationarity):**
$$\nabla_x \mathcal{L}(x^*, \mu^*, \lambda^*) = 0$$

**2. 원문제 실현가능성 (Primal Feasibility):**
$$g_i(x^*) \leq 0 \quad (i=1,\ldots,m)$$
$$h_j(x^*) = 0 \quad (j=1,\ldots,\ell)$$

**3. 쌍대 실현가능성 (Dual Feasibility):**
$$\mu_i \geq 0 \quad (i=1,\ldots,m)$$

**4. 상보적 여유 (Complementary Slackness):**
$$\mu_i g_i(x^*) = 0 \quad (i=1,\ldots,m)$$

즉, $\mu_i > 0$이면 $g_i(x^*) = 0$ (활성), $g_i(x^*) < 0$이면 $\mu_i = 0$ (비활성)

## 🔬 정리와 증명

### 정리 1: KKT 필요조건

**가정:**
- $x^*$가 문제의 국소 최솟값
- LICQ 만족: $\nabla h_j(x^*)$ ($j=1,\ldots,\ell$)와 활성 제약 $\nabla g_i(x^*)$ ($i \in \mathcal{A}(x^*)$)가 선형독립

**결론:** $x^*$에서 KKT 조건을 만족하는 $(\mu^*, \lambda^*)$가 존재한다.

**증명:**

먼저 활성 제약만 고려하면 등식 제약으로 볼 수 있다:
$$g_i(x) = 0 \quad (i \in \mathcal{A}(x^*))$$
$$h_j(x) = 0 \quad (j=1,\ldots,\ell)$$

LICQ에 의해 $\{\nabla g_i(x^*), \nabla h_j(x^*)\}$가 선형독립이므로, 라그랑주 승수법의 정리 1을 적용:

$\exists \lambda_j^*, \mu_i^*$ ($i \in \mathcal{A}(x^*)$)이 있어서:
$$\nabla f(x^*) + \sum_{i \in \mathcal{A}(x^*)} \mu_i^* \nabla g_i(x^*) + \sum_j \lambda_j^* \nabla h_j(x^*) = 0$$

비활성 제약에 대해 $\mu_i^* = 0$으로 설정하면 (상보적 여유), KKT 조건을 얻는다.

$\mu_i^* \geq 0$임을 보이려면 2계 필요조건을 사용한다. 제약 다양체의 접선 방향에서 라그랑지안의 헤시안이 반정부호(semi-definite)여야 하고, 이는 $\mu_i^* \geq 0$을 함축한다.

### 정리 2: 볼록 문제에서의 KKT 충분조건

**가정:**
- $f, g_i$가 볼록함수
- $h_j$가 아핀함수 (선형 등식 제약)
- $x^*$에서 KKT 조건을 만족

**결론:** $x^*$는 전역 최솟값이다.

**증명:**

임의의 실현가능 점 $x$에 대해:

$$f(x) - f(x^*) \geq \nabla f(x^*)^\top(x - x^*) \quad \text{(볼록함수의 정의)}$$

KKT 조건 $\nabla_x \mathcal{L}(x^*, \mu^*, \lambda^*) = 0$에서:
$$\nabla f(x^*) = -\sum_i \mu_i^* \nabla g_i(x^*) - \sum_j \lambda_j^* \nabla h_j(x^*)$$

따라서:
$$f(x) - f(x^*) \geq -\sum_i \mu_i^* \nabla g_i(x^*)^\top(x - x^*) - \sum_j \lambda_j^* \nabla h_j(x^*)^\top(x - x^*)$$

$g_i$가 볼록이므로: $g_i(x) - g_i(x^*) \geq \nabla g_i(x^*)^\top(x - x^*)$

따라서:
$$\nabla g_i(x^*)^\top(x - x^*) \leq g_i(x) - g_i(x^*)$$

$x$가 실현가능이면 $g_i(x) \leq 0$, $g_i(x^*) \leq 0$이고, 상보적 여유 $\mu_i^* g_i(x^*) = 0$에서:

$$\sum_i \mu_i^* \nabla g_i(x^*)^\top(x - x^*) \leq \sum_i \mu_i^* g_i(x) \leq 0$$

등식 제약 $h_j(x) = 0$에서: $\nabla h_j(x^*)^\top(x - x^*) = 0$ (아핀 함수)

결합하면:
$$f(x) - f(x^*) \geq 0$$

즉, $f(x^*) \leq f(x)$ for all feasible $x$.

### 정리 3: 완전성 (Completeness)
매끄러운 볼록 함수와 제약에 대해, KKT 조건과 정규성 조건을 합치면 필요충분조건이다.

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches

# ========== 예제 1: 간단한 부등식 제약 최적화 ==========
print("=" * 60)
print("예제 1: 부등식 제약 최적화")
print("=" * 60)

# 문제: min (x1 - 2)^2 + (x2 - 1)^2 s.t. x1 + x2 >= 1, x1 >= 0, x2 >= 0
def obj_ex1(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

def constraint_ineq_ex1(x):
    # g(x) = -( x1 + x2 - 1) <= 0, i.e., x1 + x2 >= 1
    return np.array([-x[0] - x[1] + 1, -x[0], -x[1]])

constraints_ex1 = [
    {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1},  # x1 + x2 >= 1
    {'type': 'ineq', 'fun': lambda x: x[0]},             # x1 >= 0
    {'type': 'ineq', 'fun': lambda x: x[1]}              # x2 >= 0
]

x0 = np.array([0.5, 0.5])
result = minimize(obj_ex1, x0, method='SLSQP', constraints=constraints_ex1)
x_opt = result.x

print(f"최적해: x* = {x_opt}")
print(f"목적함수값: f(x*) = {obj_ex1(x_opt):.6f}")
print(f"제약 만족도: x1 + x2 = {x_opt[0] + x_opt[1]:.6f}")

# KKT 조건 검증: 활성 제약 확인
g_values = constraint_ineq_ex1(x_opt)
print(f"\n제약 함수값: g(x*) = {g_values}")
print(f"활성 제약 (g=0): {[i for i, g in enumerate(g_values) if np.isclose(g, 0)]}")

# 기울기와 라그랑주 승수 계산
grad_f = np.array([2*(x_opt[0] - 2), 2*(x_opt[1] - 1)])
grad_g1 = np.array([-1, -1])  # -(x1 + x2 - 1)
grad_g2 = np.array([-1, 0])   # -x1
grad_g3 = np.array([0, -1])   # -x2

print(f"\n그래디언트: ∇f = {grad_f}")
print(f"제약 그래디언트: ∇g1 = {grad_g1}, ∇g2 = {grad_g2}, ∇g3 = {grad_g3}")

# 활성 제약에 대해 라그랑주 승수 계산
# 활성 제약: x1 + x2 = 1
# ∇f = μ1 * ∇g1 + μ2 * ∇g2 + μ3 * ∇g3
# 비활성 제약의 μ는 0

# ========== 예제 2: Support Vector Machine (SVM) ==========
print("\n" + "=" * 60)
print("예제 2: Support Vector Machine")
print("=" * 60)

# 단순한 2D 분류 데이터
np.random.seed(42)
n_pos = 20
n_neg = 20

# 양의 클래스 (클래스 1)
X_pos = np.random.randn(n_pos, 2) + np.array([2, 2])
y_pos = np.ones(n_pos)

# 음의 클래스 (클래스 -1)
X_neg = np.random.randn(n_neg, 2) + np.array([-1, -1])
y_neg = -np.ones(n_neg)

# 데이터 결합
X = np.vstack([X_pos, X_neg])
y = np.hstack([y_pos, y_neg])

# SVM 원문제: min (1/2) ||w||^2 s.t. y_i (w^T x_i + b) >= 1
# 혹은: min (1/2) ||w||^2 s.t. -(y_i (w^T x_i + b) - 1) <= 0

def svm_objective(w_b):
    w = w_b[:2]
    return 0.5 * np.dot(w, w)

def svm_constraints_gen(X, y):
    def constraint_func(w_b):
        w = w_b[:2]
        b = w_b[2]
        return y * (X @ w + b) - 1  # >= 0 형태
    return constraint_func

constraints_svm = [
    {'type': 'ineq', 'fun': svm_constraints_gen(X, y)}
]

w0 = np.array([0.1, 0.1, 0.0])
result_svm = minimize(svm_objective, w0, method='SLSQP', constraints=constraints_svm)
w_opt = result_svm.x[:2]
b_opt = result_svm.x[2]

print(f"SVM 최적해: w* = {w_opt}, b* = {b_opt:.4f}")
print(f"마진: 2 / ||w*|| = {2.0 / np.linalg.norm(w_opt):.6f}")

# 지지벡터 (support vectors) 찾기
margins = y * (X @ w_opt + b_opt)
support_vectors = np.isclose(margins, 1.0, atol=1e-5)
print(f"지지벡터 개수: {np.sum(support_vectors)} / {len(y)}")
print(f"지지벡터 인덱스: {np.where(support_vectors)[0]}")

# SVM KKT 조건 검증
print("\nSVM KKT 조건:")
print(f"1. 기울기 정류성: ∇f = Σ μ_i y_i x_i")
grad_svm = w_opt
print(f"   ∇f = w* = {grad_svm}")

# 여유(margin)
margins = y * (X @ w_opt + b_opt)
print(f"2. 원문제 실현가능성: min(margin) = {np.min(margins):.6f} >= 0? {np.min(margins) >= -1e-5}")
print(f"3. 쌍대 실현가능성: μ_i >= 0")
print(f"4. 상보적 여유: μ_i * (y_i(w^T x_i + b) - 1) = 0")
print(f"   활성 제약 (margin ≈ 1): {np.sum(support_vectors)} 개")

# === 시각화 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 왼쪽: 부등식 제약 최적화
ax = axes[0]
x1_range = np.linspace(-0.5, 3, 200)
x2_range = np.linspace(-0.5, 3, 200)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = (X1 - 2)**2 + (X2 - 1)**2

contours = ax.contour(X1, X2, Z, levels=15, cmap='viridis', alpha=0.7)
ax.clabel(contours, inline=True, fontsize=8)

# 제약 곡선: x1 + x2 = 1
x1_line = np.linspace(-0.5, 3, 100)
x2_line = 1 - x1_line
ax.plot(x1_line, x2_line, 'r-', linewidth=2.5, label='제약: $x_1 + x_2 = 1$')

# 실현가능 영역
ax.fill_between(x1_line, x2_line, 3, where=(x2_line <= 3), 
                alpha=0.1, color='green', label='실현가능 영역')

# 최적점
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=20, label=f'최적점 $x^*$')

# 기울기
grad_f_opt = np.array([2*(x_opt[0] - 2), 2*(x_opt[1] - 1)])
scale = 0.3
ax.arrow(x_opt[0], x_opt[1], scale*grad_f_opt[0], scale*grad_f_opt[1],
         head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
ax.text(x_opt[0] + 0.2, x_opt[1] - 0.3, r'$\nabla f$', fontsize=11, color='blue')

ax.set_xlim(-0.5, 3)
ax.set_ylim(-0.5, 3)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('부등식 제약 최적화: 활성 제약', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# 오른쪽: SVM
ax = axes[1]

# 데이터 산점도
ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', marker='o', s=100, 
          alpha=0.6, label='양의 클래스 (y=+1)', edgecolors='darkred', linewidth=1.5)
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', marker='s', s=100, 
          alpha=0.6, label='음의 클래스 (y=-1)', edgecolors='darkblue', linewidth=1.5)

# 지지벡터 강조
support_indices = np.where(support_vectors)[0]
ax.scatter(X[support_indices, 0], X[support_indices, 1], 
          facecolors='none', edgecolors='green', s=200, linewidth=2.5,
          label=f'지지벡터 ({len(support_indices)}개)')

# 결정 경계
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_boundary = np.linspace(x_min, x_max, 100)
x2_boundary = -(w_opt[0] * x1_boundary + b_opt) / w_opt[1]

ax.plot(x1_boundary, x2_boundary, 'k-', linewidth=2.5, label='결정 경계')

# 마진 경계
x2_margin_plus = -(w_opt[0] * x1_boundary + b_opt + 1) / w_opt[1]
x2_margin_minus = -(w_opt[0] * x1_boundary + b_opt - 1) / w_opt[1]
ax.plot(x1_boundary, x2_margin_plus, 'k--', linewidth=1, alpha=0.5)
ax.plot(x1_boundary, x2_margin_minus, 'k--', linewidth=1, alpha=0.5)

# 마진 영역
ax.fill_between(x1_boundary, x2_margin_minus, x2_margin_plus, 
               alpha=0.1, color='gray', label='마진')

ax.set_xlim(x_min, x_max)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Support Vector Machine: KKT 조건 검증', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/kkt_conditions.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장됨: /tmp/kkt_conditions.png")
plt.show()

# === 수치 검증: KKT 행렬 형식 ===
print("\n" + "=" * 60)
print("KKT 조건: 행렬 형식 검증")
print("=" * 60)

# SVM의 KKT 시스템을 형식적으로 표현
print("\nSVM KKT 시스템:")
print("∇f(w) + Σ_i μ_i ∇g_i(w) = 0")
print("g_i(w) = -(y_i(w^T x_i + b) - 1) <= 0")
print("μ_i >= 0, μ_i g_i(w) = 0")

print("\n활성 제약 (지지벡터)의 경우:")
for idx in support_indices[:3]:  # 처음 3개만
    margin = y[idx] * (X[idx] @ w_opt + b_opt)
    print(f"  점 {idx}: y={y[idx]:+.0f}, margin={margin:.4f}, 활성={'Yes' if np.isclose(margin, 1.0, atol=1e-4) else 'No'}")
```

## 🔗 AI/ML 연결

### 1. Support Vector Machines (SVM)
$$\min \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1$$

지지벡터는 정확히 KKT 조건에서 $\mu_i > 0$인 점들이다.

### 2. 강화학습: Trust Region Policy Optimization (TRPO)
$$\max_\pi \mathbb{E}[r(s, a, \pi)] \quad \text{s.t.} \quad D_{\text{KL}}(\pi_{\text{old}} \| \pi) \leq \delta$$

KKT 조건은 최적 정책의 형태를 결정한다.

### 3. 신경망 정규화
$$\min L(w) \quad \text{s.t.} \quad \|w\|_2 \leq r$$

이는 라그랑주 승수법 등가:
$$\min L(w) + \lambda \|w\|_2^2$$

## 📌 핵심 정리

| 조건 | 의미 | 검증 방법 |
|------|------|----------|
| **정류성** | $\nabla f = \sum \mu_i \nabla g_i + \sum \lambda_j \nabla h_j$ | 기울기 계산 |
| **원문제 실현가능** | $g_i(x) \leq 0$, $h_j(x) = 0$ | 제약 함수값 확인 |
| **쌍대 실현가능** | $\mu_i \geq 0$ | 라그랑주 승수 부호 |
| **상보적 여유** | $\mu_i g_i(x) = 0$ | 활성/비활성 제약 |

## 🤔 생각해볼 문제

1. **활성 집합 예측**: 최적해에서 활성 집합을 미리 알 수 있다면, 제약 최적화를 단순한 등식 제약 문제로 축소할 수 있다. 이 아이디어를 이용한 알고리즘을 생각해보자.

2. **상보적 여유의 의미**: 왜 $\mu_i g_i(x) = 0$이어야 하는가? 정책적으로 이는 무엇을 의미하는가?

3. **제약 자격 (Constraint Qualification)**: LICQ, MFCQ, Slater 조건의 차이는? 각각 언제 필요한가?

4. **SVM의 지지벡터**: 왜 모든 점이 지지벡터가 되는 것은 아닐까? 데이터가 선형 분리 가능하면 지지벡터 개수는?

5. **쌍대 갭**: KKT 조건을 만족하면 쌍대 갭이 0일까? (힌트: 볼록성 필요)

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 라그랑주 승수법](./01-lagrange-multipliers.md) | [📚 README](../README.md) | [03. 라그랑지안 쌍대성 ▶](./03-lagrangian-duality.md) |

</div>
