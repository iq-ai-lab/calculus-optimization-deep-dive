# 05. AI에서의 제약 최적화 응용

## 🎯 핵심 질문
AI/ML에서 제약 최적화 이론을 실제로 어떻게 사용할까? 어떤 알고리즘이 어떤 문제를 푸나?

## 🔍 왜 이 개념이 AI에서 중요한가

이전 4개 장에서 배운 라그랑주 승수법, KKT 조건, 쌍대성, 민감도 분석을 **실제 AI 알고리즘에 적용**한다:

- **최적화 알고리즘**: Proximal Gradient, Frank-Wolfe, Projected GD
- **머신러닝**: SVM, Elastic Net, 제약 클러스터링
- **강화학습**: RLHF (Reinforcement Learning from Human Feedback), Trust Region Policy Optimization
- **신경망 구조 탐색**: DARTS (Differentiable Architecture Search)
- **분산 최적화**: ADMM (Alternating Direction Method of Multipliers)

이 모든 것의 수학적 기초는 제약 최적화이다.

## 📐 수학적 선행 조건

다음 개념에 대한 이해가 필요하다:

- **Convex Analysis**: 근위 함수(proximal function), 부분 미분(subdifferential)
- **KKT 조건과 쌍대성** (01-03장 참고)
- **Gradient Descent**: 기본 구조와 수렴 분석
- **확률 모델**: 로짓 회귀, 정책 분포

## ✏️ 정의와 핵심 도구

### 1. Proximal Gradient Method
복합 최적화:
$$\min_x \{ f(x) + h(x) \}$$

여기서:
- $f$: 매끄러운 볼록함수 (미분 가능)
- $h$: 비매끄러운 볼록함수 (L1 정규화 등)

**근위 연산자 (Proximal Operator):**
$$\text{prox}_{\eta h}(v) = \text{argmin}_x \left\{ h(x) + \frac{1}{2\eta} \|x - v\|_2^2 \right\}$$

해석적으로 계산 가능한 경우들:
- **L1 정규화**: $h(x) = \lambda \|x\|_1$ → Soft-thresholding
- **L2 정규화**: $h(x) = \lambda \|x\|_2$ → 스케일링
- **제약 영역**: $h(x) = \mathbb{I}_C(x)$ (Indicator function) → Projection

**알고리즘:**
$$x_{k+1} = \text{prox}_{\eta_k h}\left( x_k - \eta_k \nabla f(x_k) \right)$$

### 2. Frank-Wolfe (Conditional Gradient)
제약:
$$\min_{x \in C} f(x)$$

여기서 $C$는 컨벡스 집합 (일반적으로 제약 최적화가 어려운 형태).

**원리**: 제약된 영역 내 선형함수 최소화 → 선형 계획법 사용 가능!

**알고리즘:**
$$s_k = \text{argmin}_{s \in C} \nabla f(x_k)^\top s$$
$$x_{k+1} = x_k + \gamma_k(s_k - x_k), \quad \gamma_k \in [0, 1]$$

**특성**:
- 희소 해 (atomic set에서 선택)
- 제약된 영역의 projection 불필요
- 저주파 근사 (Low-rank approximation) 가능

### 3. Projected Gradient Descent
$$x_{k+1} = \Pi_C(x_k - \eta \nabla f(x_k))$$

여기서 $\Pi_C(x) = \text{argmin}_{z \in C} \|z - x\|$

**일반적인 Projection:**
- **L2 Ball**: $\|x\|_2 \leq r$ → $\Pi(x) = \min(1, \frac{r}{\|x\|}) x$
- **Simplex**: $\sum x_i = 1, x_i \geq 0$ → Softmax 연결
- **Non-negative**: $x \geq 0$ → ReLU

## 🔬 정리와 증명

### 정리 1: Soft-Thresholding (L1 정규화의 근위 연산자)

**명제:**
L1 정규화 $h(x) = \lambda \|x\|_1$에 대해:
$$\text{prox}_{\eta h}(v) = \begin{cases}
v_i - \eta\lambda & \text{if } v_i > \eta\lambda \\
0 & \text{if } |v_i| \leq \eta\lambda \\
v_i + \eta\lambda & \text{if } v_i < -\eta\lambda
\end{cases}$$

또는 간단히: $S_\tau(v) = \text{sign}(v) \max(|v| - \tau, 0)$, 여기서 $\tau = \eta\lambda$.

**증명:**

근위 연산자의 정의에서:
$$\text{prox}_{\eta h}(v) = \text{argmin}_x \left\{ \lambda \|x\|_1 + \frac{1}{2\eta} \|x - v\|_2^2 \right\}$$

1차 필요조건 (부분미분 포함):
$$0 \in \lambda \partial \|x\|_1 + \frac{1}{\eta}(x - v)$$

$\partial \|x\|_1 = \{ g : g_i \in \text{sign}(x_i) \cup [-1,1] \text{ if } x_i = 0 \}$이므로, 성분별로:

**경우 1: $x_i > 0$**
$$0 \in \lambda + \frac{1}{\eta}(x_i - v_i)$$
$$x_i = v_i - \eta\lambda$$

이는 $v_i > \eta\lambda$일 때만 $x_i > 0$을 만족한다.

**경우 2: $x_i < 0$**
$$0 \in -\lambda + \frac{1}{\eta}(x_i - v_i)$$
$$x_i = v_i + \eta\lambda$$

이는 $v_i < -\eta\lambda$일 때만 $x_i < 0$을 만족한다.

**경우 3: $x_i = 0$**
$$0 \in \lambda g_i + \frac{1}{\eta}(0 - v_i), \quad g_i \in [-1, 1]$$
$$v_i \in [-\eta\lambda, \eta\lambda]$$

즉, $|v_i| \leq \eta\lambda$이면 $x_i = 0$이다.

### 정리 2: Proximal Gradient Method의 수렴성

**가정:**
- $f$: $L$-smooth (Lipschitz 그래디언트)
- $h$: 볼록함수
- $\eta < 1/L$

**결론:**
$$\mathbb{E}[f(x_k) + h(x_k)] - (f^* + h^*) = O(1/k)$$

(비강볼록의 경우)

### 정리 3: Frank-Wolfe의 희소성

**특성:** 알고리즘이 생성하는 해는 **원자(atoms)의 가중합** 형태이다:
$$x_k = \sum_{i=1}^k \gamma_i s_i, \quad s_i \in \text{extreme points of } C$$

특히 nuclear norm ball 제약에서는 low-rank 해를 생성한다.

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ========== 예제 1: Soft-Thresholding (L1 정규화) ==========
print("=" * 70)
print("예제 1: Proximal Gradient Method with Soft-Thresholding")
print("=" * 70)

# 문제: min (1/2)||Ax - b||^2 + λ||x||_1
# 이는 Elastic Net의 단순화

np.random.seed(42)
m, n = 50, 20
A = np.random.randn(m, n)
x_true = np.zeros(n)
x_true[:5] = np.random.randn(5)  # 5개 특성만 활성
b = A @ x_true + 0.1 * np.random.randn(m)

lam = 0.1
eta = 1.0 / (np.linalg.norm(A, 2)**2 + 0.01)  # 스텝 사이즈

# Soft-thresholding 함수
def soft_threshold(v, tau):
    """Soft-thresholding 연산자"""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0)

# Proximal Gradient Method
def proximal_gradient(A, b, lam, eta, n_iters=100):
    """근위 그래디언트 알고리즘"""
    x = np.zeros(A.shape[1])
    losses = []
    
    for k in range(n_iters):
        # Gradient step for smooth part
        grad = A.T @ (A @ x - b)
        x_temp = x - eta * grad
        
        # Proximal step for L1
        x = soft_threshold(x_temp, eta * lam)
        
        # 손실 계산
        loss = 0.5 * np.sum((A @ x - b)**2) + lam * np.sum(np.abs(x))
        losses.append(loss)
    
    return x, losses

x_pg, losses_pg = proximal_gradient(A, b, lam, eta, n_iters=200)

print(f"Proximal Gradient 해:")
print(f"  활성 특성 개수 (||x||_0): {np.sum(np.abs(x_pg) > 1e-5)}")
print(f"  ||x||_1: {np.sum(np.abs(x_pg)):.6f}")
print(f"  최종 손실: {losses_pg[-1]:.6f}")
print(f"\n참 해:")
print(f"  활성 특성 개수: {np.sum(np.abs(x_true) > 1e-5)}")
print(f"  ||x||_1: {np.sum(np.abs(x_true)):.6f}")

# === 시각화 ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 왼쪽 위: Soft-thresholding 함수
ax = axes[0, 0]
v = np.linspace(-2, 2, 100)
tau = 0.5
s = soft_threshold(v, tau)

ax.plot(v, v, 'k--', alpha=0.3, linewidth=1, label='No thresholding: $v$')
ax.plot(v, s, 'b-', linewidth=2.5, label=f'Soft-thresholding: $S_{{{tau}}}(v)$')
ax.axvline(-tau, color='r', linestyle=':', alpha=0.5)
ax.axvline(tau, color='r', linestyle=':', alpha=0.5)
ax.fill_betweenx([-2, 2], -tau, tau, alpha=0.1, color='red')

ax.set_xlabel('입력 $v$', fontsize=12)
ax.set_ylabel('출력 $S_\\tau(v)$', fontsize=12)
ax.set_title('Soft-Thresholding 함수: L1 근위 연산자', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# 왼쪽 아래: 수렴 곡선
ax = axes[1, 0]
ax.semilogy(losses_pg, 'b-', linewidth=2.5, label='근위 그래디언트 법')
ax.set_xlabel('반복 횟수 $k$', fontsize=12)
ax.set_ylabel('손실함수', fontsize=12)
ax.set_title('Proximal Gradient Method 수렴', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11)

# 오른쪽 위: 해의 희소성
ax = axes[0, 1]
indices = np.arange(n)
ax.bar(indices - 0.2, x_true, width=0.4, label='참 해', alpha=0.7)
ax.bar(indices + 0.2, x_pg, width=0.4, label='복구된 해', alpha=0.7)
ax.set_xlabel('특성 인덱스', fontsize=12)
ax.set_ylabel('가중치', fontsize=12)
ax.set_title('Sparse Recovery: 처음 10개 특성', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(-1, 10)

# 오른쪽 아래: L1 정규화 효과
ax = axes[1, 1]
x_norms_pg = np.cumsum(np.abs(x_pg[np.argsort(-np.abs(x_pg))]))
x_norms_true = np.cumsum(np.abs(x_true[np.argsort(-np.abs(x_true))]))

ax.plot(range(1, n+1), x_norms_pg, 'b-', linewidth=2, marker='o', 
       markersize=5, label='복구된 해')
ax.plot(range(1, n+1), x_norms_true, 'r--', linewidth=2, marker='s', 
       markersize=5, label='참 해')

ax.set_xlabel('상위 $k$개 특성 누적', fontsize=12)
ax.set_ylabel('누적 L1 norm', fontsize=12)
ax.set_title('희소성: 상위 특성 집중도', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/tmp/proximal_gradient.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장됨: /tmp/proximal_gradient.png")
plt.show()

# ========== 예제 2: Frank-Wolfe (희소 해) ==========
print("\n" + "=" * 70)
print("예제 2: Frank-Wolfe Method - L2 Ball 제약")
print("=" * 70)

# 문제: min ||x - b||^2 s.t. ||x||_2 <= r
b_target = np.array([1.0, 0.5, -0.3, 0.2])
r = 1.0  # L2 ball 반지름
n_dim = len(b_target)

def frank_wolfe_l2ball(b, r, n_iters=50):
    """Frank-Wolfe 알고리즘 (L2 ball 제약)"""
    x = np.zeros(n_dim)
    iterates = [x.copy()]
    losses = []
    
    for k in range(n_iters):
        # 1. Linear minimization oracle
        # min_s<∈ C> <∇f(x), s> => s = -r * ∇f(x) / ||∇f(x)||
        grad = 2 * (x - b)
        s = -r * grad / (np.linalg.norm(grad) + 1e-10)
        
        # 2. Step size
        gamma = 2.0 / (k + 2)
        
        # 3. Update
        x = x + gamma * (s - x)
        
        loss = np.sum((x - b)**2)
        losses.append(loss)
        iterates.append(x.copy())
    
    return np.array(iterates), losses

iterates_fw, losses_fw = frank_wolfe_l2ball(b_target, r, n_iters=100)
x_fw = iterates_fw[-1]

print(f"Frank-Wolfe 해:")
print(f"  ||x||_2: {np.linalg.norm(x_fw):.6f} (제약: <= {r})")
print(f"  손실: {losses_fw[-1]:.6f}")
print(f"  비영 성분 개수: {np.sum(np.abs(x_fw) > 1e-5)}")

# 최적해 (정사영)
x_proj = r * b_target / np.linalg.norm(b_target)
loss_proj = np.sum((x_proj - b_target)**2)
print(f"\n정사영 해 (참값):")
print(f"  ||x||_2: {np.linalg.norm(x_proj):.6f}")
print(f"  손실: {loss_proj:.6f}")

# === 시각화 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: Frank-Wolfe 궤적 (2D 시각화)
ax = axes[0]

# L2 ball 경계
theta = np.linspace(0, 2*np.pi, 200)
circle_x = r * np.cos(theta)
circle_y = r * np.sin(theta)
ax.plot(circle_x, circle_y, 'k-', linewidth=2, label='제약: $\\|x\\|_2 \\leq r$')

# 목표점
ax.scatter([b_target[0]], [b_target[1]], c='red', s=200, marker='*', 
          zorder=5, label='목표 $b$', edgecolors='darkred', linewidth=2)

# Frank-Wolfe 궤적
colors = plt.cm.Blues(np.linspace(0.3, 1, len(iterates_fw)))
for i in range(0, len(iterates_fw), 5):
    ax.scatter(iterates_fw[i, 0], iterates_fw[i, 1], c=[colors[i]], s=50, zorder=4)
    if i > 0:
        ax.arrow(iterates_fw[i-5, 0], iterates_fw[i-5, 1],
                iterates_fw[i, 0] - iterates_fw[i-5, 0],
                iterates_fw[i, 1] - iterates_fw[i-5, 1],
                head_width=0.05, head_length=0.03, fc='blue', ec='blue', alpha=0.5)

ax.scatter(iterates_fw[0, 0], iterates_fw[0, 1], c='green', s=100, marker='o', 
          zorder=5, label='시작점', edgecolors='darkgreen', linewidth=2)
ax.scatter(x_fw[0], x_fw[1], c='purple', s=100, marker='s', 
          zorder=5, label='Frank-Wolfe 해', edgecolors='darkviolet', linewidth=2)
ax.scatter(x_proj[0], x_proj[1], c='orange', s=100, marker='^', 
          zorder=5, label='최적 정사영', edgecolors='darkorange', linewidth=2)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Frank-Wolfe: 선형 최적화의 반복', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_aspect('equal')

# 오른쪽: 수렴 곡선
ax = axes[1]
ax.semilogy(losses_fw, 'b-', linewidth=2.5, label='Frank-Wolfe')
ax.axhline(loss_proj, color='r', linestyle='--', linewidth=2, label='최적값')

ax.set_xlabel('반복 횟수 $k$', fontsize=12)
ax.set_ylabel('손실함수', fontsize=12)
ax.set_title('Frank-Wolfe 수렴성: $O(1/k)$', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/tmp/frank_wolfe.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장됨: /tmp/frank_wolfe.png")
plt.show()

# ========== 예제 3: RLHF (강화학습 인간 피드백) ==========
print("\n" + "=" * 70)
print("예제 3: RLHF - KL 제약 최적화")
print("=" * 70)

# 간단한 모델: 2개 행동, 2개 상태
n_actions = 3
n_states = 5

# 참조 정책 (언어모델)
pi_ref = np.array([
    [0.3, 0.5, 0.2],
    [0.4, 0.3, 0.3],
    [0.2, 0.3, 0.5],
    [0.5, 0.2, 0.3],
    [0.3, 0.3, 0.4],
])

# 보상 함수
r = np.array([
    [1.0, -0.5, 0.0],
    [-0.5, 1.0, 0.5],
    [0.0, 0.5, 1.0],
    [1.5, 0.0, -0.5],
    [0.2, 1.2, 0.0],
])

beta = 0.1  # KL 제약 강도

def kl_divergence(pi, pi_ref):
    """정책 사이의 KL 발산"""
    return np.sum(pi * (np.log(pi + 1e-10) - np.log(pi_ref + 1e-10)))

def rlhf_objective(pi_flat, pi_ref, r, beta):
    """RLHF 목적함수 (음수로 최소화)"""
    pi = pi_flat.reshape(pi_ref.shape)
    # 정규화
    pi = np.clip(pi, 1e-10, 1)
    pi = pi / np.sum(pi, axis=1, keepdims=True)
    
    reward_term = np.sum(pi * r)  # E[r]
    kl_term = np.sum(pi * (np.log(pi + 1e-10) - np.log(pi_ref + 1e-10)))
    
    return -(reward_term - beta * kl_term)

def rlhf_gradient(pi_flat, pi_ref, r, beta):
    """그래디언트"""
    eps = 1e-7
    grad = np.zeros_like(pi_flat)
    for i in range(len(pi_flat)):
        pi_plus = pi_flat.copy()
        pi_plus[i] += eps
        pi_minus = pi_flat.copy()
        pi_minus[i] -= eps
        grad[i] = (rlhf_objective(pi_plus, pi_ref, r, beta) - 
                   rlhf_objective(pi_minus, pi_ref, r, beta)) / (2 * eps)
    return grad

# 최적화
pi_init = np.ones((n_states, n_actions)) / n_actions
pi_init_flat = pi_init.flatten()

# scipy로 풀기
def constraint_simplex(pi_flat):
    """제약: 각 행의 합이 1"""
    pi = pi_flat.reshape(-1, n_actions)
    return np.ones(n_states) - np.sum(pi, axis=1)

def constraint_positive(pi_flat):
    """제약: pi >= 0"""
    return pi_flat

result_rlhf = minimize(
    lambda x: rlhf_objective(x, pi_ref, r, beta),
    pi_init_flat,
    method='SLSQP',
    constraints=[
        {'type': 'ineq', 'fun': constraint_positive},
        {'type': 'eq', 'fun': constraint_simplex}
    ]
)

pi_optimal = result_rlhf.x.reshape(n_states, n_actions)
# 정규화
pi_optimal = np.clip(pi_optimal, 1e-10, 1)
pi_optimal = pi_optimal / np.sum(pi_optimal, axis=1, keepdims=True)

print(f"RLHF 최적 정책:")
print(pi_optimal)

# KKT 조건: 최적 정책은 exp(r/β) π_ref에 비례
pi_kkt = pi_ref * np.exp(r / (beta + 0.01))  # numerical stability
pi_kkt = pi_kkt / np.sum(pi_kkt, axis=1, keepdims=True)

print(f"\nKKT 조건으로부터:")
print(pi_kkt)

print(f"\nKL 발산:")
print(f"  초기: {kl_divergence(pi_init, pi_ref):.6f}")
print(f"  최적: {kl_divergence(pi_optimal, pi_ref):.6f}")

print(f"\n예상 보상:")
print(f"  초기: {np.sum(pi_init * r):.6f}")
print(f"  최적: {np.sum(pi_optimal * r):.6f}")

# === 시각화 ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 왼쪽 위: 초기 정책
ax = axes[0, 0]
im1 = ax.imshow(pi_ref, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
ax.set_title('참조 정책 $\\pi_{ref}$', fontsize=13, fontweight='bold')
ax.set_xlabel('행동', fontsize=11)
ax.set_ylabel('상태', fontsize=11)
for i in range(n_states):
    for j in range(n_actions):
        ax.text(j, i, f'{pi_ref[i, j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im1, ax=ax)

# 오른쪽 위: 최적 정책
ax = axes[0, 1]
im2 = ax.imshow(pi_optimal, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
ax.set_title('RLHF 최적 정책', fontsize=13, fontweight='bold')
ax.set_xlabel('행동', fontsize=11)
ax.set_ylabel('상태', fontsize=11)
for i in range(n_states):
    for j in range(n_actions):
        ax.text(j, i, f'{pi_optimal[i, j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im2, ax=ax)

# 왼쪽 아래: 보상 함수
ax = axes[1, 0]
im3 = ax.imshow(r, cmap='RdYlGn', aspect='auto', vmin=np.min(r), vmax=np.max(r))
ax.set_title('보상 함수 $r(s, a)$', fontsize=13, fontweight='bold')
ax.set_xlabel('행동', fontsize=11)
ax.set_ylabel('상태', fontsize=11)
for i in range(n_states):
    for j in range(n_actions):
        ax.text(j, i, f'{r[i, j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im3, ax=ax)

# 오른쪽 아래: 정책 변화
ax = axes[1, 1]
change = pi_optimal - pi_ref
im4 = ax.imshow(change, cmap='RdBu_r', aspect='auto', vmin=-0.4, vmax=0.4)
ax.set_title('정책 변화: $\\pi^* - \\pi_{ref}$', fontsize=13, fontweight='bold')
ax.set_xlabel('행동', fontsize=11)
ax.set_ylabel('상태', fontsize=11)
for i in range(n_states):
    for j in range(n_actions):
        ax.text(j, i, f'{change[i, j]:+.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im4, ax=ax)

plt.tight_layout()
plt.savefig('/tmp/rlhf_constrained.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장됨: /tmp/rlhf_constrained.png")
plt.show()

print("\n" + "=" * 70)
print("요약: AI에서의 제약 최적화 응용")
print("=" * 70)
print("""
1. Proximal Gradient + Soft-Thresholding:
   - 희소 학습 (Sparse Learning)
   - Elastic Net, Lasso
   - 신호 처리, 이미지 복원
   
2. Frank-Wolfe:
   - SVM, Structural SVM
   - Low-rank 근사
   - 추천 시스템 (제약된 인수분해)
   
3. RLHF (KL 제약):
   - 대규모 언어모델 미세조정
   - 가치 정렬 (Value Alignment)
   - 정책 최적화
   
모두 제약 최적화의 원리에 기반!
""")
```

## 🔗 AI/ML 연결

### 1. Elastic Net과 Sparse Learning
$$\min \|Ax - b\|_2^2 + \lambda_1 \|x\|_1 + \lambda_2 \|x\|_2^2$$

Soft-thresholding으로 선택적으로 특성을 제거.

### 2. SVM과 Frank-Wolfe
쌍대 문제에서 선형 최소화:
$$\min_{s \in \{\pm 1\}^m} \alpha^\top s$$

제약된 영역의 구조를 활용한 효율적 풀이.

### 3. RLHF와 KL 제약
최적 정책은 다음 형태:
$$\pi^*(a|s) \propto \pi_{\text{ref}}(a|s) \exp\left(\frac{r(s,a)}{\beta}\right)$$

이는 KKT 조건에서 직접 유도된다!

## 📌 핵심 정리

| 알고리즘 | 목적 | 제약 | 응용 |
|---------|------|------|------|
| **Proximal GD** | $f + h$ 최소화 | 비매끄러운 정규화 | Sparse learning |
| **Frank-Wolfe** | $f$ 최소화 | 컨벡스 집합 | SVM, Low-rank |
| **Projected GD** | $f$ 최소화 | 간단한 제약 | 확률 심플렉스 |
| **ADMM** | 분산 최적화 | 블록 분리 가능 | 대규모 문제 |

## 🤔 생각해볼 문제

1. **Soft-thresholding의 해석**: 왜 L1 정규화가 희소성을 유도할까? (기하학적 직관)

2. **Frank-Wolfe의 희소성**: 왜 알고리즘이 극값(extreme points)만 선택할까?

3. **RLHF의 최적 정책**: $\pi^* \propto \pi_{ref} \exp(r/\beta)$를 어떻게 유도하는가? (변분법)

4. **계산 복잡도**: Projected vs Proximal vs Frank-Wolfe 중 언제 어느 것을 선택할까?

5. **분산 최적화**: ADMM (교대 방향 승수법)이 왜 효율적일까? (쌍대성)

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. 음함수 정리 응용](./04-implicit-function-constrained.md) | [📚 README](../README.md) | [Ch7-01. Softmax 야코비안 ▶](../ch7-advanced-dl-calculus/01-softmax-jacobian.md) |

</div>
