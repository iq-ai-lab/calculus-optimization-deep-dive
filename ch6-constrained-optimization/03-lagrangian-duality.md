# 03. 라그랑지안 쌍대성

## 🎯 핵심 질문
어려운 원문제(primal problem)를 더 쉬운 쌍대문제(dual problem)로 바꿀 수 있을까? 두 문제의 최적값이 같을까?

## 🔍 왜 이 개념이 AI에서 중요한가

라그랑지안 쌍대성은 **계산 효율성**과 **이론적 통찰**을 제공한다:

- **SVM**: 원문제는 이차계획법(QP)이지만, 쌍대문제는 더 효율적인 SMO 알고리즘 가능
- **Optimal Transport**: Monge 문제 → 선형계획법 쌍대 문제
- **강화학습**: Policy gradient ← 쌍대성을 통한 정책 업데이트 유도
- **분산 최적화**: ADMM, 근위 알고리즘 ← 증강 라그랑지안
- **신경망 압축**: 자원 제약 최적화 ← 쌍대성 통해 효율적 풀이

## 📐 수학적 선행 조건

다음 개념에 대한 이해가 필요하다:

- **라그랑주 승수법과 KKT 조건** (01-02장 참고)
- **오목함수 (Concave Function)**: 선형결합의 성질
- **컨벡스 분석**: 볼록 집합, 볼록 함수, 아핀 함수
- **선형계획법 기초**: 심플렉스 알고리즘, 이원성

## ✏️ 정의와 핵심 도구

### 원문제와 쌍대 함수

원문제:
$$p^* = \min_x f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \; h_j(x) = 0$$

라그랑지안:
$$\mathcal{L}(x, \mu, \lambda) = f(x) + \sum_i \mu_i g_i(x) + \sum_j \lambda_j h_j(x)$$

쌍대 함수:
$$q(\mu, \lambda) = \inf_x \mathcal{L}(x, \mu, \lambda)$$

여기서:
- $\mu \in \mathbb{R}^m_+$ (쌍대 실현가능성: $\mu_i \geq 0$)
- $\lambda \in \mathbb{R}^\ell$ (등식 제약의 승수는 부호 제약 없음)

### 쌍대 문제
$$d^* = \max_{\mu,\lambda} q(\mu, \lambda) \quad \text{s.t.} \quad \mu \geq 0$$

## 🔬 정리와 증명

### 정리 1: 약한 쌍대성 (Weak Duality)

**명제:**
$$d^* \leq p^*$$

**증명:**

임의의 쌍대 실현가능 $(\mu, \lambda)$ ($\mu \geq 0$)에 대해:

$$q(\mu, \lambda) = \inf_x \mathcal{L}(x, \mu, \lambda)$$

따라서 임의의 $x$에 대해:
$$q(\mu, \lambda) \leq \mathcal{L}(x, \mu, \lambda) = f(x) + \sum_i \mu_i g_i(x) + \sum_j \lambda_j h_j(x)$$

이제 $x$가 원문제의 실현가능 해라고 하자: $g_i(x) \leq 0$, $h_j(x) = 0$.

그러면:
$$\sum_i \mu_i g_i(x) \leq 0 \quad \text{(왜냐하면 $\mu_i \geq 0$, $g_i(x) \leq 0$)}$$
$$\sum_j \lambda_j h_j(x) = 0$$

따라서:
$$q(\mu, \lambda) \leq f(x) + \sum_i \mu_i g_i(x) + \sum_j \lambda_j h_j(x) \leq f(x)$$

양변에 모든 원문제 실현가능 $x$에 대해 우변을 최소화하면:
$$q(\mu, \lambda) \leq \min_x f(x) = p^*$$

이는 모든 쌍대 실현가능 $(\mu, \lambda)$에 대해 성립하므로:
$$d^* = \max_{\mu,\lambda} q(\mu, \lambda) \leq p^*$$

### 정리 2: 쌍대 함수의 오목성

**명제:** $q(\mu, \lambda)$는 $(\mu, \lambda)$에 대한 오목함수(concave function)이다.

**증명:**

$\mu, \mu' \in \mathbb{R}^m_+$, $\lambda, \lambda' \in \mathbb{R}^\ell$, $\theta \in [0,1]$에 대해:

$$q(\theta \mu + (1-\theta)\mu', \theta \lambda + (1-\theta)\lambda')$$
$$= \inf_x \{ f(x) + \sum_i (\theta \mu_i + (1-\theta)\mu_i') g_i(x) + \sum_j (\theta \lambda_j + (1-\theta)\lambda_j') h_j(x) \}$$

라그랑지안이 $(\mu, \lambda)$에서 선형이므로:
$$= \inf_x \{ \theta[\cdots] + (1-\theta)[\cdots] \}$$

$\inf$가 선형 함수들의 가중합에 대해 오목성을 보존하므로:
$$\geq \theta \inf_x [\cdots] + (1-\theta) \inf_x [\cdots]$$
$$= \theta q(\mu, \lambda) + (1-\theta) q(\mu', \lambda')$$

따라서 $q$는 오목함수이고, 쌍대 문제는 **항상 볼록 최적화**이다!

### 정리 3: 강한 쌍대성과 Slater 조건

**정의 (Slater 조건):** 
존재하는 $\tilde{x}$가 있어서:
$$g_i(\tilde{x}) < 0 \quad (i=1,\ldots,m)$$
$$h_j(\tilde{x}) = 0 \quad (j=1,\ldots,\ell)$$

즉, 부등식 제약을 **엄격히**(strictly) 만족하는 내점(interior point)이 존재한다.

**정리 (강한 쌍대성, Strong Duality):**

$f, g_i$가 볼록함수이고 $h_j$가 아핀함수일 때, Slater 조건이 성립하면:
$$d^* = p^*$$

그리고 쌍대 최적값 $(\mu^*, \lambda^*)$가 존재하며, KKT 조건이 최적성 필요충분조건이 된다.

**증명 개요:**

Slater 조건은 제약 자격(constraint qualification)이다. 이 조건 하에서:

1. 원문제에 대해 라그랑주 승수 $(\mu^*, \lambda^*)$ 존재
2. KKT 조건이 필요충분조건
3. 쌍대 함수의 최댓값이 원문제의 최솟값과 같음

엄밀한 증명은 **Farkas 보조정리**를 사용하는데, 이는 선형계획법의 핵심 정리이다.

**Farkas 보조정리:** 선형 시스템 $Ax = b$ ($x \geq 0$)이 해를 가질 필요충분조건은 $y^\top A \geq 0$을 만족하는 모든 $y$에 대해 $y^\top b \geq 0$이다.

### 정리 4: 쌍대 갭 (Duality Gap)

**정의:**
$$\text{Duality Gap} = p^* - d^*$$

약한 쌍대성에 의해 gap $\geq 0$이다.

**특성:**
- Strong duality 성립 (Slater 조건) ⇒ gap = 0
- KKT 조건 만족 ⇒ gap = 0 (볼록 문제)

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
from mpl_toolkits.mplot3d import Axes3D

# ========== 예제 1: 이차계획법 (Quadratic Programming) ==========
print("=" * 70)
print("예제 1: 이차계획법과 쌍대 문제")
print("=" * 70)

# 원문제: min (1/2) x^T Q x + c^T x s.t. A x <= b
# Q = [[2, 0], [0, 2]], c = [0, 0], A = [[1, 1]], b = [1]
# 즉, min (1/2)(x1^2 + x2^2) s.t. x1 + x2 <= 1

Q = np.array([[2.0, 0.0], [0.0, 2.0]])
c = np.array([0.0, 0.0])
A = np.array([[1.0, 1.0]])
b = np.array([1.0])

# 원문제 풀이
def objective_primal(x):
    return 0.5 * x @ Q @ x + c @ x

constraints = LinearConstraint(A, -np.inf, b)
bounds = Bounds([-np.inf, -np.inf], [np.inf, np.inf])
x0 = np.array([0.0, 0.0])

result_primal = minimize(objective_primal, x0, method='SLSQP', 
                         constraints={'type': 'ineq', 'fun': lambda x: -A @ x + b})
x_primal = result_primal.x
p_star = objective_primal(x_primal)

print(f"원문제 최적해: x* = {x_primal}")
print(f"원문제 최적값: p* = {p_star:.6f}")

# 쌍대 함수 정의
# q(μ) = inf_x [(1/2) x^T Q x + c^T x + μ (A x - b)]
# = inf_x [(1/2) x^T Q x + (c^T + μ A^T) x - μ b]
# 미분: Q x + c + μ A^T = 0 => x(μ) = -Q^{-1}(c + μ A^T)
# q(μ) = (1/2) x(μ)^T Q x(μ) + c^T x(μ) + μ (A x(μ) - b)

def dual_function(mu):
    """쌍대 함수 q(μ)"""
    if mu < 0:
        return -np.inf
    x_mu = -np.linalg.solve(Q, c + mu * A.T)
    q_val = 0.5 * x_mu @ Q @ x_mu + c @ x_mu + mu * (A @ x_mu - b)
    return q_val, x_mu

# 쌍대 문제: max q(μ) s.t. μ >= 0
def neg_dual_function(mu):
    return -dual_function(mu[0])[0]

mu0 = np.array([0.1])
result_dual = minimize(neg_dual_function, mu0, method='SLSQP',
                      constraints={'type': 'ineq', 'fun': lambda mu: mu})
mu_dual = result_dual.x[0]
q_star, x_dual = dual_function(mu_dual)

print(f"\n쌍대 최적 승수: μ* = {mu_dual:.6f}")
print(f"쌍대 최적값: d* = {q_star:.6f}")
print(f"쌍대 갭: p* - d* = {p_star - q_star:.2e}")
print(f"강한 쌍대성 만족? {np.isclose(p_star, q_star)}")

# 쌍대 함수 시각화
mu_range = np.linspace(0, 2, 100)
q_values = []
for mu in mu_range:
    q_val, _ = dual_function(mu)
    q_values.append(q_val)

# === 시각화 ===
fig = plt.figure(figsize=(16, 5))

# 왼쪽: 원문제 (등위선)
ax1 = fig.add_subplot(131)
x1_range = np.linspace(-0.5, 1.5, 150)
x2_range = np.linspace(-0.5, 1.5, 150)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = 0.5 * (X1**2 + X2**2)

contours = ax1.contour(X1, X2, Z, levels=15, cmap='viridis', alpha=0.7)
ax1.clabel(contours, inline=True, fontsize=8)

# 제약
x1_line = np.linspace(-0.5, 1.5, 100)
x2_line = 1 - x1_line
ax1.plot(x1_line, x2_line, 'r-', linewidth=2.5, label='제약: $x_1 + x_2 = 1$')
ax1.fill_between(x1_line, -0.5, x2_line, alpha=0.1, color='green')

# 최적점
ax1.plot(x_primal[0], x_primal[1], 'r*', markersize=20, label='원문제 최적점')

ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlabel('$x_1$', fontsize=12)
ax1.set_ylabel('$x_2$', fontsize=12)
ax1.set_title('원문제: $\\min \\frac{1}{2}(x_1^2 + x_2^2)$', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_aspect('equal')

# 중간: 쌍대 함수
ax2 = fig.add_subplot(132)
ax2.plot(mu_range, q_values, 'b-', linewidth=2.5, label='쌍대 함수 $q(\\mu)$')
ax2.axhline(p_star, color='r', linestyle='--', linewidth=2, label=f'원문제 최적값 $p^*$ = {p_star:.3f}')
ax2.plot(mu_dual, q_star, 'go', markersize=12, label=f'쌍대 최적점: $\\mu^*$ = {mu_dual:.3f}')

ax2.set_xlabel('$\\mu$', fontsize=12)
ax2.set_ylabel('$q(\\mu)$', fontsize=12)
ax2.set_title('쌍대 함수와 강한 쌍대성', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# 오른쪽: 약한 쌍대성 시각화
ax3 = fig.add_subplot(133)
mu_test = np.linspace(0, 2, 50)
gaps = []

for mu in mu_test:
    q_val, _ = dual_function(mu)
    gap = p_star - q_val
    gaps.append(gap)

ax3.fill_between(mu_test, 0, gaps, alpha=0.3, color='orange', label='쌍대 갭')
ax3.plot(mu_test, gaps, 'o-', color='orange', linewidth=2, markersize=5)
ax3.plot(mu_dual, 0, 'go', markersize=12, label='$\\mu^*$ (강한 쌍대성)')
ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)

ax3.set_xlabel('$\\mu$', fontsize=12)
ax3.set_ylabel('쌍대 갭: $p^* - q(\\mu)$', fontsize=12)
ax3.set_title('약한 쌍대성: 항상 gap $\\geq$ 0', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

plt.tight_layout()
plt.savefig('/tmp/duality.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장됨: /tmp/duality.png")
plt.show()

# ========== 예제 2: SVM 원/쌍대 문제 ==========
print("\n" + "=" * 70)
print("예제 2: SVM 원문제와 쌍대 문제")
print("=" * 70)

np.random.seed(42)
n_pos = 15
n_neg = 15

X_pos = np.random.randn(n_pos, 2) + np.array([2, 2])
X_neg = np.random.randn(n_neg, 2) + np.array([-1, -1])

X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(n_pos), -np.ones(n_neg)])
n_samples = len(y)

# 원문제: min (1/2)||w||^2 s.t. y_i(w^T x_i + b) >= 1
def svm_primal_objective(w_b):
    w = w_b[:2]
    return 0.5 * np.dot(w, w)

constraints_svm = [
    {'type': 'ineq', 'fun': lambda w_b: y * (X @ w_b[:2] + w_b[2]) - 1}
]

w_b_init = np.array([0.1, 0.1, 0.0])
result_svm_primal = minimize(svm_primal_objective, w_b_init, method='SLSQP',
                            constraints=constraints_svm)
w_primal = result_svm_primal.x[:2]
b_primal = result_svm_primal.x[2]
p_svm = svm_primal_objective(result_svm_primal.x)

print(f"SVM 원문제 최적해: w* = {w_primal}, b* = {b_primal:.4f}")
print(f"원문제 최적값: {p_svm:.6f}")

# 쌍대 문제: max Σ_i α_i - (1/2) Σ_i,j α_i α_j y_i y_j x_i^T x_j
#            s.t. 0 <= α_i, Σ_i α_i y_i = 0
def svm_dual_objective(alpha):
    # 목적함수: Σ α_i - (1/2) Σ_{i,j} α_i α_j y_i y_j x_i^T x_j
    first_term = np.sum(alpha)
    K = X @ X.T  # 커널 행렬
    second_term = 0.5 * alpha @ np.diag(y) @ K @ np.diag(y) @ alpha
    return first_term - second_term

def neg_svm_dual(alpha):
    return -svm_dual_objective(alpha)

alpha_init = np.ones(n_samples) * 0.01
constraints_svm_dual = [
    {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)},  # Σ α_i y_i = 0
    {'type': 'ineq', 'fun': lambda alpha: alpha},           # α_i >= 0
]

result_svm_dual = minimize(neg_svm_dual, alpha_init, method='SLSQP',
                          constraints=constraints_svm_dual, options={'maxiter': 1000})
alpha_dual = result_svm_dual.x
d_svm = svm_dual_objective(alpha_dual)

print(f"\nSVM 쌍대 문제 최적 승수: α* shape = {alpha_dual.shape}")
print(f"쌍대 최적값: {d_svm:.6f}")
print(f"쌍대 갭: {p_svm - d_svm:.2e}")

# 지지벡터
support_vectors = alpha_dual > 1e-4
n_support = np.sum(support_vectors)
print(f"지수 벡터 개수: {n_support} / {n_samples}")
print(f"상보적 여유 확인: 활성 제약의 α_i > 0")

# === KKT 조건 검증 ===
print("\n=== KKT 조건 검증 ===")
print(f"제약 조건 1: Σ_i α_i y_i = {np.dot(alpha_dual, y):.2e} (= 0?)")
print(f"제약 조건 2: min(α_i) = {np.min(alpha_dual):.2e} >= 0?")
print(f"기울기 조건: w* = Σ_i α_i y_i x_i")
w_from_dual = np.sum(alpha_dual[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
print(f"  원문제에서: w* = {w_primal}")
print(f"  쌍대에서: w* = {w_from_dual}")
print(f"  차이: {np.linalg.norm(w_primal - w_from_dual):.2e}")

# === 그래프 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# SVM 결과
ax = axes[0]
colors = ['red' if label > 0 else 'blue' for label in y]
sizes = [200 if sv else 50 for sv in support_vectors]
alphas = [0.9 if sv else 0.3 for sv in support_vectors]

for i in range(n_samples):
    ax.scatter(X[i, 0], X[i, 1], c=colors[i], s=sizes[i], alpha=alphas[i],
              edgecolors='black' if support_vectors[i] else 'none',
              linewidth=2 if support_vectors[i] else 0)

# 결정 경계
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_boundary = np.linspace(x_min, x_max, 100)
x2_boundary = -(w_primal[0] * x1_boundary + b_primal) / w_primal[1]

ax.plot(x1_boundary, x2_boundary, 'k-', linewidth=2, label='결정 경계')
x2_margin_plus = -(w_primal[0] * x1_boundary + b_primal + 1) / w_primal[1]
x2_margin_minus = -(w_primal[0] * x1_boundary + b_primal - 1) / w_primal[1]
ax.plot(x1_boundary, x2_margin_plus, 'k--', linewidth=1, alpha=0.5)
ax.plot(x1_boundary, x2_margin_minus, 'k--', linewidth=1, alpha=0.5)

ax.set_xlim(x_min, x_max)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('SVM: 원문제 최적해', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# 라그랑주 승수 분포
ax = axes[1]
ax.bar(np.arange(n_samples), alpha_dual, color=['green' if sv else 'gray' for sv in support_vectors])
ax.set_xlabel('샘플 인덱스', fontsize=12)
ax.set_ylabel('$\\alpha_i$ (쌍대 변수)', fontsize=12)
ax.set_title(f'SVM 쌍대 변수: {n_support}개 지지벡터', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/svm_duality.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장됨: /tmp/svm_duality.png")
plt.show()
```

## 🔗 AI/ML 연결

### 1. SVM과 쌍대 문제
원문제는 고차원에서 이차계획법이 복잡하지만, 쌍대 문제는 **커널 트릭**을 사용할 수 있다:
$$d^* = \max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

### 2. Optimal Transport (최적 수송)
Monge 문제 (연속 함수 찾기)의 쌍대는 Kantorovich 선형계획법이다.

### 3. 분산 최적화와 ADMM
증강 라그랑지안과 쌍대성을 이용해 중앙 서버 없이 여러 에이전트가 협력하는 알고리즘 설계.

## 📌 핵심 정리

| 개념 | 정의 |
|------|------|
| **쌍대 함수** | $q(\mu,\lambda) = \inf_x \mathcal{L}(x,\mu,\lambda)$ |
| **약한 쌍대성** | $d^* \leq p^*$ (항상 성립) |
| **강한 쌍대성** | $d^* = p^*$ (Slater 조건 하에) |
| **쌍대 갭** | $p^* - d^*$ (강한 쌍대성 하 = 0) |
| **쌍대 함수 성질** | 오목함수 → 쌍대 문제는 항상 볼록 |

## 🤔 생각해볼 문제

1. **쌍대 함수의 오목성**: 왜 $q(\mu, \lambda)$가 오목함수일까? 이것의 의미는?

2. **Slater 조건**: 왜 내점(interior point) 존재 조건이 필요할까? 반례를 생각해보자.

3. **쌍대 갭과 근사**: 쌍대 갭이 0이 아니면 원문제 해의 근사율은? 얼마나 떨어져 있을까?

4. **SVM의 쌍대성**: 쌍대 문제가 원문제보다 쉬운 이유는? (힌트: 변수 개수와 제약 형태)

5. **계산 복잡도**: 고차원($n$ 크다) vs 많은 샘플($m$ 크다)일 때, 원/쌍대 중 어느 것을 선택할까?

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 부등식 제약과 KKT](./02-inequality-kkt.md) | [📚 README](../README.md) | [04. 음함수 정리 응용 ▶](./04-implicit-function-constrained.md) |

</div>
