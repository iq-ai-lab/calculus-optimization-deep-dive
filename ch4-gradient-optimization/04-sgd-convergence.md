# 04. SGD 수렴 분석

## 🎯 핵심 질문
- 미니배치 그래디언트가 잡음을 포함할 때 왜 수렴하는가?
- GD의 $O(1/k)$ vs SGD의 $O(1/\sqrt{k})$: 이 차이는 피할 수 없는가?
- 분산 감소(SVRG)가 수렴을 $O(\log k / k)$로 만드는 방법은?
- 학습률 스케줄이 왜 $\eta_k = c/\sqrt{k}$여야 하는가?

## 🔍 왜 이 개념이 AI에서 중요한가

모든 딥러닝은 **확률적 그래디언트**에 기반한다:
- 전체 배치는 계산 비용이 엄청남 ($O(n)$ per step)
- 미니배치만 사용하면 $O(b)$ (배치 크기 $b \ll n$)
- 하지만 **잡음 도입**: 수렴이 느려지거나 불안정
- **이론과 실전의 갭**: 실제로는 미니배치가 더 효율적 (더 빨리 찾음)

## 📐 수학적 선행 조건

- **비편향 그래디언트**: $\mathbb{E}[\tilde{g}_k] = \nabla f(x_k)$
- **유계 분산**: $\mathbb{E}[\|\tilde{g}_k - \nabla f(x_k)\|^2] \leq \sigma^2$
- **Lyapunov 함수**: $V_k = f(x_k) - f^*$
- **Random variables의 수렴**: Almost sure vs in expectation

## ✏️ 정의와 핵심 도구

### 정의 1: 확률적 경사하강법 (SGD)

$$x_{k+1} = x_k - \eta_k \tilde{\nabla}f(x_k)$$

여기서:
- $\tilde{\nabla}f(x_k) = \nabla f(x_i(x_k)) + \xi_k$ (잡음)
- $\xi_k = \tilde{\nabla}f(x_k) - \nabla f(x_k)$ (오차항)

### 정의 2: 통계적 가정

**비편향성**: $\mathbb{E}[\tilde{\nabla}f(x_k) \mid x_k] = \nabla f(x_k)$

**유계 분산** (Assumption A1):
$$\mathbb{E}[\|\tilde{\nabla}f(x_k) - \nabla f(x_k)\|^2 \mid x_k] \leq \sigma^2$$

**그래디언트 노름 유계성** (Assumption A2):
$$\|\nabla f(x)\| \leq G \text{ for all } x$$

## 🔬 정리와 증명

### 정리 1: 볼록 경우 SGD의 $O(1/\sqrt{k})$ 수렴

**설정:**
- 볼록 함수 $f$
- L-smooth: $f(y) \leq f(x) + \nabla f(x)^\top(y-x) + \frac{L}{2}\|y-x\|^2$
- 비편향 그래디언트, 분산 $\sigma^2$
- 학습률: $\eta_k = c/\sqrt{k}$ with $c = 1/(2L)$

**정리:**
$$\mathbb{E}[f(\bar{x}_k) - f(x^*)] \leq \frac{R_0^2 + \sigma^2 \sum_{t=1}^k \eta_t^2}{2\sum_{t=1}^k \eta_t}$$

$\eta_k = c/\sqrt{k}$일 때:
$$\sum_{t=1}^k \eta_t \approx 2c\sqrt{k}, \quad \sum_{t=1}^k \eta_t^2 \approx c^2 \log k$$

따라서:
$$\mathbb{E}[f(\bar{x}_k) - f(x^*)] = O\left(\frac{c^2 \log k}{c\sqrt{k}}\right) = O\left(\frac{\log k}{\sqrt{k}}\right) = O(1/\sqrt{k})$$

실제로는 첫 번째 항이 dominant하므로:
$$\mathbb{E}[f(\bar{x}_k) - f(x^*)] \leq O(1/\sqrt{k})$$

**증명:**

Descent Lemma를 기댓값으로 확장:
$$f(x_{k+1}) = f(x_k) - \eta_k \nabla f(x_k)^\top \tilde{\nabla}f(x_k) + O(\eta_k^2)$$

기댓값 취하면:
$$\mathbb{E}[f(x_{k+1})] \leq \mathbb{E}[f(x_k)] - \eta_k \mathbb{E}[\|\nabla f(x_k)\|^2] + \frac{\eta_k^2 L}{2}\mathbb{E}[\|\tilde{\nabla}f(x_k)\|^2]$$

비편향성에서 $\mathbb{E}[\|\tilde{\nabla}f\|^2] = \|\nabla f\|^2 + \sigma^2$이므로:
$$\mathbb{E}[f(x_{k+1})] \leq \mathbb{E}[f(x_k)] - \eta_k \mathbb{E}[\|\nabla f(x_k)\|^2] + \frac{\eta_k^2 L}{2}(\mathbb{E}[\|\nabla f(x_k)\|^2] + \sigma^2)$$

$$= \mathbb{E}[f(x_k)] - \eta_k(1 - \frac{\eta_k L}{2})\mathbb{E}[\|\nabla f(x_k)\|^2] + \frac{\eta_k^2 L\sigma^2}{2}$$

볼록성에서:
$$\mathbb{E}[\|\nabla f(x_k)\|^2] \geq \frac{2}{\text{diam}^2}(\mathbb{E}[f(x_k)] - f(x^*))$$

(복잡한 증명 생략, 합을 취해서):

$$\mathbb{E}[f(\bar{x}_k)] - f(x^*) \leq \frac{R_0^2 + \sigma^2\sum \eta_t^2}{2\sum \eta_t}$$

$\square$

### 정리 2: 강볼록 경우 SGD의 $O(\log k / k)$ 수렴

**설정:**
- $\mu$-강볼록 함수
- 학습률: $\eta_k = \frac{2}{\mu(k+1)}$

**정리:**
$$\mathbb{E}[\|x_k - x^*\|^2] \leq C\left(\frac{\log k}{k}\right)$$

**증명:**

강볼록성에서:
$$\mathbb{E}[\|x_{k+1} - x^*\|^2] = \mathbb{E}[\|x_k - \eta_k \tilde{\nabla}f(x_k) - x^*\|^2]$$

$$= \mathbb{E}[\|x_k - x^*\|^2] - 2\eta_k \mathbb{E}[\nabla f(x_k)^\top(x_k - x^*)] + \eta_k^2 \mathbb{E}[\|\tilde{\nabla}f\|^2]$$

강볼록성:
$$\nabla f(x_k)^\top(x_k - x^*) \geq \frac{\mu}{2}\|x_k - x^*\|^2$$

따라서:
$$\mathbb{E}[\|x_{k+1} - x^*\|^2] \leq (1 - \mu\eta_k)\mathbb{E}[\|x_k - x^*\|^2] + \eta_k^2(\sigma^2 + \text{bias})$$

$\eta_k = \frac{2}{\mu(k+1)}$를 대입:
$$(1 - \mu\eta_k) = 1 - \frac{2}{k+1} = \frac{k-1}{k+1}$$

재귀적으로:
$$\mathbb{E}[\|x_k - x^*\|^2] \leq \prod_{t=0}^{k-1}(1-\mu\eta_t) \|x_0-x^*\|^2 + \sum_{t=0}^{k-1} \prod_{s=t+1}^{k-1}(1-\mu\eta_s) \eta_t^2 \sigma^2$$

첫 번째 항: 텔레스코핑
$$\prod_{t=0}^{k-1}\frac{t}{t+3} \approx \frac{1}{k}$$

두 번째 항: 분석적으로 $O(\log k / k^2)$

합치면: $O(\log k / k)$

$\square$

### 정리 3: SVRG (Stochastic Variance Reduced Gradient)

**문제점**: 표준 SGD는 $O(1/\sqrt{k})$이지만 SVRG는 $O(\log k / k)$를 달성

**알고리즘:**

outer loop (m 스텝):
- $\tilde{x} = x_{\text{current}}$
- $\tilde{\nabla} = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\tilde{x})$ (full gradient)

inner loop (m 스텝):
$$x_{k+1} = x_k - \eta\left(\nabla f_{i_k}(x_k) - \nabla f_{i_k}(\tilde{x}) + \tilde{\nabla}\right)$$

**정리:**
SVRG 수렴: $\mathbb{E}[f(x_k) - f^*] = O(\log k / k)$ (같은 조건수)

**증명 아이디어:**

분산 감소:
$$\mathbb{E}[\|\nabla f_{i_k}(x_k) - \nabla f_{i_k}(\tilde{x}) + \tilde{\nabla} - \nabla f(x_k)\|^2]$$

$$= \mathbb{E}[\|\nabla f_{i_k}(x_k) - \nabla f(x_k)\|^2] + \|\nabla f_{i_k}(\tilde{x}) - \tilde{\nabla}\|^2 + \ldots$$

두 번째 항은 $x_k \to \tilde{x}$일수록 감소 (inner loop 진행)

$\square$

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 테스트 함수: 로지스틱 회귀 손실
np.random.seed(42)

def generate_data(n_samples=1000, n_features=20, n_noise=10):
    """분류 문제 생성"""
    X_signal = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features + n_noise)
    X_noise = np.random.randn(n_samples, n_noise)
    X = np.hstack([X_signal, X_noise])
    y = np.sign(X @ w_true)  # 이진 분류
    return X, y, w_true

X, y, w_true = generate_data(n_samples=500, n_features=10, n_noise=5)

def logistic_loss(X, y, w):
    """전체 데이터에 대한 로지스틱 손실"""
    z = y * (X @ w)
    return np.mean(np.log(1 + np.exp(-z)))

def logistic_grad_full(X, y, w):
    """전체 그래디언트"""
    z = y * (X @ w)
    g = -y * np.exp(-z) / (1 + np.exp(-z))
    return X.T @ g / len(y)

def logistic_grad_sample(X, y, w, idx):
    """샘플별 그래디언트"""
    z = y[idx] * (X[idx] @ w)
    g = -y[idx] * np.exp(-z) / (1 + np.exp(-z))
    return X[idx] * g

def gradient_descent(X, y, w0, eta, max_iter=200):
    """배치 경사하강법 (결정적)"""
    w = w0.copy()
    trajectory = [w.copy()]
    losses = [logistic_loss(X, y, w)]
    
    for k in range(max_iter):
        grad = logistic_grad_full(X, y, w)
        w = w - eta * grad
        trajectory.append(w.copy())
        losses.append(logistic_loss(X, y, w))
    
    return np.array(trajectory), np.array(losses)

def sgd(X, y, w0, eta_k_fn, max_iter=5000):
    """확률적 경사하강법"""
    w = w0.copy()
    n = len(y)
    trajectory = [w.copy()]
    losses = [logistic_loss(X, y, w)]
    grads_norms = [np.linalg.norm(logistic_grad_full(X, y, w))]
    
    for k in range(max_iter):
        # 샘플 하나 선택
        idx = np.random.randint(n)
        grad_sample = logistic_grad_sample(X, y, w, idx)
        
        # 학습률 (스케줄)
        eta_k = eta_k_fn(k)
        
        # 업데이트
        w = w - eta_k * grad_sample
        trajectory.append(w.copy())
        losses.append(logistic_loss(X, y, w))
        
        if k % 10 == 0:
            grads_norms.append(np.linalg.norm(logistic_grad_full(X, y, w)))
    
    return np.array(trajectory), np.array(losses), np.array(grads_norms)

def minibatch_sgd(X, y, w0, batch_size, eta_k_fn, max_iter=1000):
    """미니배치 SGD"""
    w = w0.copy()
    n = len(y)
    trajectory = [w.copy()]
    losses = [logistic_loss(X, y, w)]
    
    for k in range(max_iter):
        # 미니배치 샘플링
        indices = np.random.choice(n, batch_size, replace=False)
        grad_batch = logistic_grad_full(X[indices], y[indices], w)
        
        eta_k = eta_k_fn(k)
        w = w - eta_k * grad_batch
        trajectory.append(w.copy())
        losses.append(logistic_loss(X, y, w))
    
    return np.array(trajectory), np.array(losses)

# 초기값
w0 = np.random.randn(X.shape[1]) * 0.1

print("="*70)
print("SGD CONVERGENCE ANALYSIS")
print("="*70)

# 1. GD (배치)
print("\n1. Batch Gradient Descent (GD)")
traj_gd, loss_gd = gradient_descent(X, y, w0, eta=0.01, max_iter=200)
print(f"   Final loss: {loss_gd[-1]:.6e}")

# 2. SGD with 1/sqrt(k) schedule
print("\n2. SGD with η_k = 0.1/√k")
eta_sqrt = lambda k: 0.1 / np.sqrt(k + 1)
traj_sgd_sqrt, loss_sgd_sqrt, grad_sgd_sqrt = sgd(X, y, w0, eta_sqrt, max_iter=5000)
print(f"   Final loss: {loss_sgd_sqrt[-1]:.6e}")

# 3. SGD with 1/k schedule
print("\n3. SGD with η_k = 0.01/k")
eta_k = lambda k: 0.01 / (k + 1)
traj_sgd_k, loss_sgd_k, grad_sgd_k = sgd(X, y, w0, eta_k, max_iter=5000)
print(f"   Final loss: {loss_sgd_k[-1]:.6e}")

# 4. Minibatch SGD (batch_size=32)
print("\n4. Minibatch SGD (batch_size=32)")
loss_mb32 = minibatch_sgd(X, y, w0, batch_size=32, 
                          eta_k_fn=lambda k: 0.05/np.sqrt(k+1), max_iter=1000)
print(f"   Final loss: {loss_mb32[-1]:.6e}")

# 5. Minibatch SGD (batch_size=64)
print("\n5. Minibatch SGD (batch_size=64)")
loss_mb64 = minibatch_sgd(X, y, w0, batch_size=64, 
                          eta_k_fn=lambda k: 0.05/np.sqrt(k+1), max_iter=1000)
print(f"   Final loss: {loss_mb64[-1]:.6e}")

print()

# 시각화
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. 손실 곡선 비교
ax1 = fig.add_subplot(gs[0, :2])

# GD
ax1.semilogy(loss_gd, 'b-', linewidth=2.5, label='GD (batch)', marker='o', markersize=5, markevery=10)

# SGD variants
k_sgd = np.arange(len(loss_sgd_sqrt))
ax1.semilogy(k_sgd[::10], loss_sgd_sqrt[::10], 'r^-', linewidth=2.5, label='SGD (η=0.1/√k)', markersize=5)
ax1.semilogy(k_sgd[::10], loss_sgd_k[::10], 'g^-', linewidth=2.5, label='SGD (η=0.01/k)', markersize=5)

# 이론적 수렴률
k_theory = np.arange(1, len(loss_sgd_sqrt))
theory_sqrt = loss_sgd_sqrt[0] / np.sqrt(k_theory)
theory_k = loss_sgd_sqrt[0] / k_theory

ax1.semilogy(k_theory, theory_sqrt, 'r:', linewidth=2, alpha=0.6, label='Theory: O(1/√k)')
ax1.semilogy(k_theory, theory_k, 'g:', linewidth=2, alpha=0.6, label='Theory: O(1/k)')

ax1.set_xlabel('Iteration k', fontsize=11)
ax1.set_ylabel('Loss f(x_k)', fontsize=11)
ax1.set_title('SGD Convergence Rates', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim([0, 2000])

# 2. 수렴 속도 확인 (log-log)
ax2 = fig.add_subplot(gs[0, 2])
k_range = np.arange(1, min(500, len(loss_sgd_sqrt)))
ax2.loglog(k_range, loss_sgd_sqrt[:500], 'r-', linewidth=2.5, label='SGD (√k schedule)', marker='^', markersize=5)

# 참조선
k_ref = np.arange(10, 500)
ax2.loglog(k_ref, loss_sgd_sqrt[0] / np.sqrt(k_ref), 'r--', linewidth=1.5, alpha=0.6, label='1/√k')
ax2.loglog(k_ref, loss_sgd_sqrt[0] / k_ref, 'g--', linewidth=1.5, alpha=0.6, label='1/k')

ax2.set_xlabel('Iteration k (log)', fontsize=11)
ax2.set_ylabel('Loss (log)', fontsize=11)
ax2.set_title('Rate Verification (log-log)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

# 3. 학습률 스케줄
ax3 = fig.add_subplot(gs[1, 0])
k_sched = np.arange(0, 2000)
eta_sqrt_vals = 0.1 / np.sqrt(k_sched + 1)
eta_k_vals = 0.01 / (k_sched + 1)
eta_const = 0.01 * np.ones_like(k_sched)

ax3.loglog(k_sched[10:], eta_sqrt_vals[10:], 'r-', linewidth=2.5, label='η=0.1/√k')
ax3.loglog(k_sched[10:], eta_k_vals[10:], 'g-', linewidth=2.5, label='η=0.01/k')
ax3.loglog(k_sched, eta_const, 'b--', linewidth=2, alpha=0.6, label='Constant η')
ax3.set_xlabel('Iteration k (log)', fontsize=11)
ax3.set_ylabel('Learning Rate (log)', fontsize=11)
ax3.set_title('Learning Rate Schedules', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, which='both')

# 4. 그래디언트 노름 (SGD)
ax4 = fig.add_subplot(gs[1, 1])
k_grad = np.arange(0, len(grad_sgd_sqrt))
ax4.semilogy(k_grad[::10], grad_sgd_sqrt[::10], 'r-', linewidth=2.5, label='SGD (√k)', marker='o', markersize=5)

# GD 참조
grad_gd = np.array([np.linalg.norm(logistic_grad_full(X, y, traj_gd[k])) for k in range(len(traj_gd))])
k_gd_grad = np.arange(len(grad_gd))
ax4.semilogy(k_gd_grad, grad_gd, 'b-', linewidth=2.5, label='GD (batch)', marker='s', markersize=5, markevery=5)

ax4.set_xlabel('Iteration k', fontsize=11)
ax4.set_ylabel('||∇f(x)||', fontsize=11)
ax4.set_title('Gradient Norm Evolution', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')

# 5. 미니배치 효과
ax5 = fig.add_subplot(gs[1, 2])
k_mb = np.arange(len(loss_mb32))
ax5.semilogy(k_mb, loss_mb32, 'b-', linewidth=2.5, label='Batch=32', marker='o', markersize=5)
ax5.semilogy(k_mb, loss_mb64, 'r-', linewidth=2.5, label='Batch=64', marker='^', markersize=5)
ax5.semilogy(k_mb, loss_gd[:len(loss_mb32)], 'g--', linewidth=2.5, label='Full batch (GD)', alpha=0.7)

ax5.set_xlabel('Epoch (iteration)', fontsize=11)
ax5.set_ylabel('Loss', fontsize=11)
ax5.set_title('Minibatch SGD Comparison', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, which='both')

# 6. 손실 vs 반복 (early iterations, log scale)
ax6 = fig.add_subplot(gs[2, 0])
k_early = np.arange(1, min(200, len(loss_sgd_sqrt)))
ax6.semilogy(k_early, loss_sgd_sqrt[1:200], 'r-', linewidth=2, label='SGD', marker='^', markersize=4, markevery=10)
ax6.semilogy(k_early, loss_gd[1:200], 'b-', linewidth=2, label='GD', marker='o', markersize=4, markevery=10)

ax6.set_xlabel('Iteration k', fontsize=11)
ax6.set_ylabel('Loss', fontsize=11)
ax6.set_title('Early Iterations (Zoom)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, which='both')
ax6.set_xlim([0, 200])

# 7. 표준화된 손실 (GD와 SGD)
ax7 = fig.add_subplot(gs[2, 1])
# GD를 100 스텝까지 확장 (비교 가능하도록)
loss_gd_norm = (loss_gd - loss_gd[-1]) / (loss_gd[0] - loss_gd[-1])
loss_sgd_norm = (loss_sgd_sqrt - loss_sgd_sqrt[-50]) / (loss_sgd_sqrt[0] - loss_sgd_sqrt[-50])

ax7.semilogy(loss_gd_norm[1:200], 'b-', linewidth=2.5, label='GD', marker='o', markersize=5, markevery=10)
ax7.semilogy(loss_sgd_norm[1:2000], 'r-', linewidth=2.5, label='SGD', marker='^', markersize=5, markevery=100)

ax7.set_xlabel('Iteration k', fontsize=11)
ax7.set_ylabel('Normalized Loss', fontsize=11)
ax7.set_title('Normalized Loss Comparison', fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3, which='both')

# 8. 배치 크기 vs 분산 감소
ax8 = fig.add_subplot(gs[2, 2])

# 여러 배치 크기로 실험
batch_sizes = [1, 4, 16, 32, 64]
final_losses_mb = []

for bs in batch_sizes:
    if bs == 1:
        loss_mb = loss_sgd_sqrt
    else:
        loss_mb = minibatch_sgd(X, y, w0, batch_size=bs, 
                               eta_k_fn=lambda k: 0.05/np.sqrt(k+1), 
                               max_iter=1000)
    final_losses_mb.append(loss_mb[-1])

ax8.semilogy(batch_sizes, final_losses_mb, 'bo-', linewidth=2.5, markersize=10, label='Final Loss')
ax8.set_xlabel('Batch Size', fontsize=11)
ax8.set_ylabel('Final Loss', fontsize=11)
ax8.set_title('Effect of Batch Size', fontsize=12, fontweight='bold')
ax8.set_xscale('log')
ax8.grid(True, alpha=0.3, which='both')

plt.savefig('/sessions/exciting-gifted-mccarthy/mnt/outputs/ch4-gradient-optimization/04_sgd_convergence.png',
            dpi=150, bbox_inches='tight')
print("✓ Visualization saved: 04_sgd_convergence.png")
plt.close()

# 분산 분석
print("\n" + "="*70)
print("VARIANCE ANALYSIS")
print("="*70)

fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

# 그래디언트 분산 추정
def estimate_gradient_variance(X, y, w, n_samples=100):
    """샘플 그래디언트의 분산 추정"""
    n = len(y)
    grads = []
    for _ in range(n_samples):
        idx = np.random.randint(n)
        g = logistic_grad_sample(X, y, w, idx)
        grads.append(np.linalg.norm(g)**2)
    return np.var(grads)

# 초기, 중간, 후기 포인트에서 분산 추정
w_init = w0
w_mid = traj_sgd_sqrt[2500]
w_late = traj_sgd_sqrt[-1]

var_init = estimate_gradient_variance(X, y, w_init)
var_mid = estimate_gradient_variance(X, y, w_mid)
var_late = estimate_gradient_variance(X, y, w_late)

axes[0].bar(['Initial', 'Mid', 'Late'], [var_init, var_mid, var_late], color=['blue', 'green', 'red'])
axes[0].set_ylabel('Gradient Variance', fontsize=11)
axes[0].set_title('Variance During Training', fontsize=12, fontweight='bold')
axes[0].set_yscale('log')

# 배치 크기별 분산 감소
batch_sizes_var = np.array([1, 2, 4, 8, 16, 32])
variances = []
for bs in batch_sizes_var:
    vars_bs = []
    for _ in range(50):
        indices = np.random.choice(len(y), bs, replace=False)
        grad_batch = logistic_grad_full(X[indices], y[indices], w_mid)
        grad_full = logistic_grad_full(X, y, w_mid)
        vars_bs.append(np.linalg.norm(grad_batch - grad_full)**2)
    variances.append(np.mean(vars_bs))

variances = np.array(variances)
axes[1].loglog(batch_sizes_var, variances, 'bo-', linewidth=2.5, markersize=10)
axes[1].loglog(batch_sizes_var, variances[0]/batch_sizes_var, 'r--', linewidth=2, alpha=0.6, label='1/B (theory)')
axes[1].set_xlabel('Batch Size B', fontsize=11)
axes[1].set_ylabel('Variance ||∇_B - ∇||²', fontsize=11)
axes[1].set_title('Variance Reduction with Batch Size', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, which='both')

# 이동 평균 손실과 그래디언트 노름
axes[2].semilogy(k_grad[::10], grad_sgd_sqrt[::10], 'b-', linewidth=2.5, label='Gradient norm', marker='o', markersize=5)
ax2_twin = axes[2].twinx()
loss_smooth = np.convolve(loss_sgd_sqrt[::10], np.ones(10)/10, mode='same')
ax2_twin.semilogy(np.arange(len(loss_smooth))*10, loss_smooth, 'r-', linewidth=2.5, label='Loss (smoothed)', alpha=0.7, marker='^', markersize=5)
axes[2].set_xlabel('Iteration k', fontsize=11)
axes[2].set_ylabel('Gradient Norm', fontsize=11, color='b')
ax2_twin.set_ylabel('Loss', fontsize=11, color='r')
axes[2].set_title('Gradient & Loss Trajectories', fontsize=12, fontweight='bold')
axes[2].tick_params(axis='y', labelcolor='b')
ax2_twin.tick_params(axis='y', labelcolor='r')

plt.tight_layout()
plt.savefig('/sessions/exciting-gifted-mccarthy/mnt/outputs/ch4-gradient-optimization/04_sgd_variance_analysis.png',
            dpi=150, bbox_inches='tight')
print("✓ Visualization saved: 04_sgd_variance_analysis.png")
plt.close()
```

## 🔗 AI/ML 연결

### PyTorch SGD 예제
```python
import torch
import torch.optim as optim

# 기본 SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 학습률 스케줄
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # 에포크마다 학습률 감소
```

## 📌 핵심 정리

1. **수렴 속도**:
   - GD (배치): $O(1/k)$
   - SGD (확률적): $O(1/\sqrt{k})$ (정보-이론적 한계)
   - SVRG (분산 감소): $O(\log k / k)$

2. **학습률 선택**:
   - $\eta_k = c/\sqrt{k}$: $O(1/\sqrt{k})$ 수렴
   - $\eta_k = c/k$: $O(\log k / k)$ (강볼록)

3. **배치 크기와 분산**:
   - 분산 $\propto 1/B$ (배치 크기 $B$)
   - 더 큰 배치 = 더 안정적이지만 계산 비용 증가

4. **실전 가이드**:
   - GD: 최종 모델 평가용 (검증 손실 최소화)
   - SGD: 큰 규모 학습
   - Minibatch: 배치 크기 32-256이 경험적으로 최적

## 🤔 생각해볼 문제

1. **문제 1**: SGD의 수렴률이 GD보다 느린 ($O(1/\sqrt{k})$ vs $O(1/k)$) 이유는?
   - 정보-이론적 관점에서 설명하시오.

2. **문제 2**: SVRG가 $O(\log k/k)$를 달성하려면 매 epoch마다 전체 그래디언트를 계산해야 하는데, 이것이 실전에서 잘 안 쓰이는 이유는?

3. **문제 3**: 미니배치 크기가 크면 클수록 좋은가?
   - 장점과 단점을 나열하시오.

4. **문제 4**: Learning rate schedule $\eta_k = c/\sqrt{k}$는 어떻게 설계되었나?
   - 왜 이 형태인가?

5. **문제 5** (구현): 주어진 손실 함수에서 분산을 추정하고, 최적 배치 크기를 자동으로 추천하는 알고리즘을 작성하시오.

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. 모멘텀과 네스테로프](./03-momentum-nesterov.md) | [📚 README](../README.md) | [05. 적응형 학습률 ▶](./05-adam-rmsprop-adagrad.md) |

</div>
