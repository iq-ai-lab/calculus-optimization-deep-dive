# 05. 적응형 학습률: Adagrad, RMSProp, Adam

## 🎯 핵심 질문
- 왜 모든 파라미터에 같은 학습률을 사용하면 안 되는가?
- Adagrad의 학습률 감소 문제를 RMSProp과 Adam이 어떻게 해결하는가?
- Adam의 bias correction이 정확히 무엇을 보정하는가?
- Adam이 GD보다 일반화 성능이 낮을 때가 왜 있는가?

## 🔍 왜 이 개념이 AI에서 중요한가

**적응형 옵티마이저가 지배하는 현대 AI**:
- NLP: Adam이 표준 (BERT, GPT, T5)
- Vision: SGD with Momentum이 여전히 경쟁력 있음
- 그래디언트가 매우 희소한 문제: Adagrad 계열 필수
- 하이퍼파라미터 튜닝의 난제: Adam은 학습률에 덜 민감

## 📐 수학적 선행 조건

- **그래디언트 통계**: 스케일이 다른 파라미터들의 처리
- **Exponential moving average**: 지수 가중 평균의 성질
- **행렬식과 양정치성**: Hessian approximation
- **편향 (Bias)**: 초기 스텝에서의 지수 평균 편향

## ✏️ 정의와 핵심 도구

### 정의 1: Adagrad

$$G_t = \sum_{\tau=1}^t g_\tau g_\tau^\top$$

**업데이트:**
$$x_{t+1} = x_t - \frac{\eta}{\sqrt{\text{diag}(G_t) + \epsilon}} \odot g_t$$

여기서 $\odot$는 원소별 곱셈 (element-wise multiplication)

**효과:**
- 그래디언트가 자주 나타나는 차원: 학습률 감소
- 그래디언트가 드물게 나타나는 차원: 높은 학습률 유지

### 정의 2: RMSProp

지수 이동 평균으로 Adagrad의 단조 감소 문제 해결:

$$v_t = \rho v_{t-1} + (1-\rho)g_t^2$$

**업데이트:**
$$x_{t+1} = x_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot g_t$$

### 정의 3: Adam (Adaptive Moment Estimation)

첫 번째 모멘트 (평균)와 두 번째 모멘트 (분산)를 동시에 추적:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

**편향 보정 (Bias Correction):**
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**업데이트:**
$$x_{t+1} = x_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

기본값: $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$

## 🔬 정리와 증명

### 정리 1: Adagrad의 $O(1/\sqrt{T})$ 수렴 (강볼록)

**설정:**
- $\mu$-강볼록 함수
- G-bounded gradient: $\|g_t\| \leq G$
- Adagrad 적응형 학습률

**정리:**
$$\mathbb{E}[\|x_T - x^*\|^2] = O\left(\frac{\sqrt{\log T}}{T}\right)$$

또는 손실 기준:
$$f(x_T) - f(x^*) = O\left(\frac{\sqrt{\log T}}{\sqrt{T}}\right)$$

**증명:**

Adagrad는 각 파라미터 $i$에 대해:
$$x^i_{t+1} = x^i_t - \frac{\eta}{\sqrt{G_t^{ii} + \epsilon}} g_t^i$$

$G_t^{ii} = \sum_{\tau=1}^t (g_\tau^i)^2$이므로:

일반적 SGD 분석에서 지터(jitter)항이 학습률로 조정됨:

$$\mathbb{E}[\|x_{t+1} - x^*\|^2] \leq (1-\mu\eta/\sqrt{t})\mathbb{E}[\|x_t-x^*\|^2] + \frac{\eta^2 \sigma^2}{\sqrt{t}}$$

$\eta$를 최적화하면:
$$\sum_{t=1}^T \frac{\eta}{\sqrt{G_t^{ii}}} = O(\sqrt{T})$$

그래서 final convergence:
$$\mathbb{E}[\|x_T - x^*\|^2] \leq O\left(\frac{\sqrt{\log T}}{T}\right)$$

$\square$

### 정리 2: Adam의 수렴 (비볼록)

**설정:**
- 비볼록 함수 (신경망)
- 그래디언트 bounded: $\|g_t\| \leq G$
- 분산 bounded: $\mathbb{E}[\|g_t - \nabla f\|^2] \leq \sigma^2$

**정리:**
Adam을 다음과 같이 구성하면:
$$\alpha_t = \alpha / \sqrt{t}$$ (감소하는 스텝 크기)

다음을 만족:
$$\min_{1 \leq t \leq T} \mathbb{E}[\|\nabla f(x_t)\|^2] \leq O\left(\frac{1}{\sqrt{T}}\right)$$

즉, $O(1/\sqrt{T})$ 수렴 (SGD와 동일)

**증명 스케치:**

Adam의 핵심은 $\hat{m}_t / \sqrt{\hat{v}_t}$가 **신호-대-잡음비 (SNR) 정규화**를 수행한다는 것:

$$\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} = \frac{\text{EMA of } g_t}{\sqrt{\text{EMA of } g_t^2}}$$

이는 개별 성분의 scale을 자동으로 정규화:

$$\alpha_t \frac{\hat{m}_t^i}{\sqrt{\hat{v}_t^i} + \epsilon} \approx \alpha_t \cdot \text{sign}(g_t^i)$$

큰 분산을 가진 성분은 단위 크기 업데이트로 조정됨.

표준 강하강법 분석을 적용하면:

$$\mathbb{E}[f(x_t)] - f(x^*) \leq \text{const} \cdot \frac{1}{\sqrt{T}}$$

$\square$

### 정리 3: Bias Correction의 필요성

**문제:**
초기에 $m_0 = v_0 = 0$으로 시작하면:
$$m_1 = (1-\beta_1) g_1 \approx 0.1 \cdot g_1 \quad (\beta_1 = 0.9)$$
$$v_1 = (1-\beta_2) g_1^2 \approx 0.001 \cdot g_1^2 \quad (\beta_2 = 0.999)$$

따라서 매우 작은 업데이트만 발생.

**Bias Correction 후:**
$$\hat{m}_1 = \frac{0.1 \cdot g_1}{1-0.9} = g_1$$
$$\hat{v}_1 = \frac{0.001 \cdot g_1^2}{1-0.999} = g_1^2$$

올바른 스케일로 복원.

**정리 (Bias Correction):**
편향 보정 없이는 처음 수백 스텝에서 수렴이 느림. 보정과 함께:
$$\mathbb{E}[\|\hat{m}_t\|^2] = O(\mathbb{E}[\|g_t\|^2])$$
$$\mathbb{E}[\hat{v}_t] = O(\mathbb{E}[\|g_t^2\|])$$

초기부터 올바른 스케일 유지.

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Rosenbrock 함수 (non-convex, feature scaling 필요)
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    dfdx0 = 400*x[0]*(x[0]**2 - x[1]) + 2*(x[0]-1)
    dfdx1 = 200*(x[1] - x[0]**2)
    return np.array([dfdx0, dfdx1])

# Optimizer 구현들

def gradient_descent(grad_fn, x0, eta, max_iter=5000):
    """표준 경사하강법"""
    x = x0.copy()
    trajectory = [x.copy()]
    losses = [rosenbrock(x)]
    
    for k in range(max_iter):
        g = grad_fn(x)
        x = x - eta * g
        trajectory.append(x.copy())
        losses.append(rosenbrock(x))
        
        if np.linalg.norm(g) < 1e-8:
            break
    
    return np.array(trajectory), np.array(losses)

def adagrad(grad_fn, x0, alpha, max_iter=5000, epsilon=1e-8):
    """Adagrad"""
    x = x0.copy()
    G = np.zeros_like(x)  # 누적 제곱 그래디언트
    trajectory = [x.copy()]
    losses = [rosenbrock(x)]
    
    for k in range(max_iter):
        g = grad_fn(x)
        G += g * g  # 누적
        
        # 적응형 업데이트
        x = x - alpha / np.sqrt(G + epsilon) * g
        trajectory.append(x.copy())
        losses.append(rosenbrock(x))
        
        if np.linalg.norm(g) < 1e-8:
            break
    
    return np.array(trajectory), np.array(losses)

def rmsprop(grad_fn, x0, alpha, beta=0.9, max_iter=5000, epsilon=1e-8):
    """RMSProp"""
    x = x0.copy()
    v = np.zeros_like(x)  # 이동 평균 제곱
    trajectory = [x.copy()]
    losses = [rosenbrock(x)]
    
    for k in range(max_iter):
        g = grad_fn(x)
        v = beta * v + (1 - beta) * (g ** 2)
        
        x = x - alpha / np.sqrt(v + epsilon) * g
        trajectory.append(x.copy())
        losses.append(rosenbrock(x))
        
        if np.linalg.norm(g) < 1e-8:
            break
    
    return np.array(trajectory), np.array(losses)

def adam(grad_fn, x0, alpha, beta1=0.9, beta2=0.999, max_iter=5000, epsilon=1e-8):
    """Adam"""
    x = x0.copy()
    m = np.zeros_like(x)  # 첫 번째 모멘트 (평균)
    v = np.zeros_like(x)  # 두 번째 모멘트 (분산)
    trajectory = [x.copy()]
    losses = [rosenbrock(x)]
    
    for k in range(max_iter):
        g = grad_fn(x)
        
        # 모멘트 업데이트
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        # 편향 보정
        m_hat = m / (1 - beta1**(k+1))
        v_hat = v / (1 - beta2**(k+1))
        
        # 업데이트
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        trajectory.append(x.copy())
        losses.append(rosenbrock(x))
        
        if np.linalg.norm(g) < 1e-8:
            break
    
    return np.array(trajectory), np.array(losses)

def adamw(grad_fn, x0, alpha, beta1=0.9, beta2=0.999, lambda_reg=0.01, max_iter=5000, epsilon=1e-8):
    """AdamW (Weight Decay, 정규화 분리)"""
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    losses = [rosenbrock(x)]
    
    for k in range(max_iter):
        g = grad_fn(x)
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        m_hat = m / (1 - beta1**(k+1))
        v_hat = v / (1 - beta2**(k+1))
        
        # Adam 스텝
        adam_step = alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Weight decay (그래디언트와 독립적으로 적용)
        x = (1 - lambda_reg * alpha) * x - adam_step
        trajectory.append(x.copy())
        losses.append(rosenbrock(x))
        
        if np.linalg.norm(g) < 1e-8:
            break
    
    return np.array(trajectory), np.array(losses)

# 초기점
x0 = np.array([-1.5, 2.5])

print("="*70)
print("ADAPTIVE OPTIMIZER COMPARISON")
print("="*70)

# 각 옵티마이저 실행
results = {}

# GD (for comparison)
print("\nGradient Descent (η=0.001)")
traj_gd, loss_gd = gradient_descent(rosenbrock_grad, x0, eta=0.001, max_iter=10000)
results['GD'] = (traj_gd, loss_gd)
print(f"  Iterations: {len(loss_gd)-1}, Final loss: {loss_gd[-1]:.6e}")

# Adagrad
print("\nAdagrad (α=0.1)")
traj_ada, loss_ada = adagrad(rosenbrock_grad, x0, alpha=0.1, max_iter=5000)
results['Adagrad'] = (traj_ada, loss_ada)
print(f"  Iterations: {len(loss_ada)-1}, Final loss: {loss_ada[-1]:.6e}")

# RMSProp
print("\nRMSProp (α=0.01, β=0.9)")
traj_rms, loss_rms = rmsprop(rosenbrock_grad, x0, alpha=0.01, beta=0.9, max_iter=5000)
results['RMSProp'] = (traj_rms, loss_rms)
print(f"  Iterations: {len(loss_rms)-1}, Final loss: {loss_rms[-1]:.6e}")

# Adam
print("\nAdam (α=0.01, β₁=0.9, β₂=0.999)")
traj_adam, loss_adam = adam(rosenbrock_grad, x0, alpha=0.01, max_iter=5000)
results['Adam'] = (traj_adam, loss_adam)
print(f"  Iterations: {len(loss_adam)-1}, Final loss: {loss_adam[-1]:.6e}")

# AdamW
print("\nAdamW (α=0.01, β₁=0.9, β₂=0.999, λ=0.01)")
traj_adamw, loss_adamw = adamw(rosenbrock_grad, x0, alpha=0.01, lambda_reg=0.01, max_iter=5000)
results['AdamW'] = (traj_adamw, loss_adamw)
print(f"  Iterations: {len(loss_adamw)-1}, Final loss: {loss_adamw[-1]:.6e}")

print()

# 시각화
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. 손실 곡선 비교
ax1 = fig.add_subplot(gs[0, :2])
for name, (_, loss) in results.items():
    ax1.semilogy(loss[:min(500, len(loss))], label=name, linewidth=2.5, marker='o', markersize=4, markevery=50)

ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Loss (log)', fontsize=11)
ax1.set_title('Convergence Comparison: All Optimizers', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# 2. 수렴 속도 (처음 100스텝)
ax2 = fig.add_subplot(gs[0, 2])
for name, (_, loss) in results.items():
    ax2.semilogy(loss[:100], label=name, linewidth=2, marker='o', markersize=5, markevery=10)

ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Early Convergence (zoom)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

# 3. 2D 트래젝토리
ax3 = fig.add_subplot(gs[1, 0])

# Contour plot
x_range = np.linspace(-2, 3, 100)
y_range = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = 100 * (Y - X**2)**2 + (1 - X)**2

contours = ax3.contour(X, Y, Z, levels=20, alpha=0.3, colors='gray')
ax3.clabel(contours, inline=True, fontsize=7)

colors = {'GD': 'blue', 'Adagrad': 'green', 'RMSProp': 'red', 'Adam': 'purple', 'AdamW': 'orange'}
for name, (traj, _) in results.items():
    ax3.plot(traj[:min(50, len(traj)), 0], traj[:min(50, len(traj)), 1], 
             'o-', color=colors[name], label=name, linewidth=2, markersize=4, alpha=0.8)

ax3.plot(x0[0], x0[1], 'g*', markersize=15, label='Start')
ax3.plot(1, 1, 'r+', markersize=15, markeredgewidth=2, label='Optimum')
ax3.set_xlabel('x₁', fontsize=11)
ax3.set_ylabel('x₂', fontsize=11)
ax3.set_title('2D Trajectories (Rosenbrock)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-2, 2])
ax3.set_ylim([-0.5, 3.5])

# 4. Adam의 편향 보정 효과
ax4 = fig.add_subplot(gs[1, 1])

# Adam without bias correction
m = np.zeros_like(x0)
v = np.zeros_like(x0)
loss_no_bc = []
loss_with_bc = []

x = x0.copy()
for k in range(300):
    g = rosenbrock_grad(x)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * (g ** 2)
    
    # Without bias correction
    x_no_bc = x - 0.01 * m / (np.sqrt(v) + 1e-8)
    loss_no_bc.append(rosenbrock(x_no_bc))
    
    # With bias correction
    m_hat = m / (1 - 0.9**(k+1))
    v_hat = v / (1 - 0.999**(k+1))
    x = x - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
    loss_with_bc.append(rosenbrock(x))

ax4.semilogy(loss_no_bc[:200], 'r-', linewidth=2.5, label='Without bias correction', marker='^', markersize=4, markevery=20)
ax4.semilogy(loss_with_bc[:200], 'b-', linewidth=2.5, label='With bias correction', marker='o', markersize=4, markevery=20)
ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('Loss', fontsize=11)
ax4.set_title('Adam: Bias Correction Effect', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')

# 5. 적응형 학습률 (Adam)
ax5 = fig.add_subplot(gs[1, 2])

beta1, beta2 = 0.9, 0.999
m = np.zeros(2)
v = np.zeros(2)
effective_lr = []

x = x0.copy()
for k in range(500):
    g = rosenbrock_grad(x)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)
    
    m_hat = m / (1 - beta1**(k+1))
    v_hat = v / (1 - beta2**(k+1))
    
    # 각 파라미터의 effective learning rate
    lr_effective = 0.01 / (np.sqrt(v_hat) + 1e-8)
    effective_lr.append(np.mean(lr_effective))
    
    x = x - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)

ax5.semilogy(effective_lr, 'b-', linewidth=2.5, marker='o', markersize=5, markevery=50)
ax5.axhline(y=0.01, color='r', linestyle='--', linewidth=2, alpha=0.6, label='Base learning rate')
ax5.set_xlabel('Iteration', fontsize=11)
ax5.set_ylabel('Effective Learning Rate', fontsize=11)
ax5.set_title('Adam: Adaptive Learning Rate', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, which='both')

# 6. 파라미터별 그래디언트 크기
ax6 = fig.add_subplot(gs[2, 0])

x = x0.copy()
grad_norms = [[], []]  # 각 파라미터의 그래디언트

for k in range(1000):
    g = rosenbrock_grad(x)
    grad_norms[0].append(abs(g[0]))
    grad_norms[1].append(abs(g[1]))
    x = x - 0.001 * g

ax6.semilogy(grad_norms[0][:300], 'b-', linewidth=2, label='∂f/∂x₁', marker='o', markersize=3, markevery=30)
ax6.semilogy(grad_norms[1][:300], 'r-', linewidth=2, label='∂f/∂x₂', marker='^', markersize=3, markevery=30)
ax6.set_xlabel('Iteration', fontsize=11)
ax6.set_ylabel('Gradient Magnitude', fontsize=11)
ax6.set_title('Per-parameter Gradient Evolution', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, which='both')

# 7. Adagrad의 학습률 감소
ax7 = fig.add_subplot(gs[2, 1])

x = x0.copy()
G = np.zeros(2)
learning_rates_adagrad = [[], []]

for k in range(500):
    g = rosenbrock_grad(x)
    G += g * g
    
    # 각 파라미터의 적응형 학습률
    lr_adapt = 0.1 / np.sqrt(G + 1e-8)
    learning_rates_adagrad[0].append(lr_adapt[0])
    learning_rates_adagrad[1].append(lr_adapt[1])
    
    x = x - lr_adapt * g

ax7.semilogy(learning_rates_adagrad[0][:300], 'b-', linewidth=2, label='x₁ learning rate', marker='o', markersize=3, markevery=30)
ax7.semilogy(learning_rates_adagrad[1][:300], 'r-', linewidth=2, label='x₂ learning rate', marker='^', markersize=3, markevery=30)
ax7.set_xlabel('Iteration', fontsize=11)
ax7.set_ylabel('Adaptive Learning Rate', fontsize=11)
ax7.set_title('Adagrad: Monotonic LR Decay', fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3, which='both')

# 8. RMSProp vs Adagrad 학습률
ax8 = fig.add_subplot(gs[2, 2])

# RMSProp
x = x0.copy()
v = np.zeros(2)
learning_rates_rms = [[], []]

for k in range(500):
    g = rosenbrock_grad(x)
    v = 0.9 * v + 0.1 * (g ** 2)
    lr_adapt = 0.01 / np.sqrt(v + 1e-8)
    learning_rates_rms[0].append(lr_adapt[0])
    learning_rates_rms[1].append(lr_adapt[1])
    x = x - lr_adapt * g

ax8.semilogy(learning_rates_rms[0][:300], 'g-', linewidth=2.5, label='RMSProp x₁', marker='o', markersize=4, markevery=30)
ax8.semilogy(learning_rates_rms[1][:300], 'b-', linewidth=2.5, label='RMSProp x₂', marker='^', markersize=4, markevery=30)
ax8.semilogy(learning_rates_adagrad[0][:300], 'r--', linewidth=1.5, alpha=0.6, label='Adagrad x₁', markevery=30)
ax8.semilogy(learning_rates_adagrad[1][:300], 'orange', linewidth=1.5, alpha=0.6, linestyle='--', label='Adagrad x₂', markevery=30)
ax8.set_xlabel('Iteration', fontsize=11)
ax8.set_ylabel('Adaptive Learning Rate', fontsize=11)
ax8.set_title('RMSProp vs Adagrad: Learning Rate', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3, which='both')

plt.savefig('/sessions/exciting-gifted-mccarthy/mnt/outputs/ch4-gradient-optimization/05_adam_rmsprop_adagrad.png',
            dpi=150, bbox_inches='tight')
print("✓ Visualization saved: 05_adam_rmsprop_adagrad.png")
plt.close()
```

## 🔗 AI/ML 연결

### PyTorch 구현
```python
import torch
import torch.optim as optim

model = torch.nn.Linear(10, 1)

# Adagrad
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# RMSProp
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (권장: L2 정규화 분리)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## 📌 핵심 정리

1. **Adagrad**:
   - 희소 그래디언트에 강함 (자연어 처리)
   - 학습률이 단조 감소 → 나중에 거의 진전 없음

2. **RMSProp**:
   - Adagrad의 단조 감소 문제 해결
   - 지수 이동 평균으로 "recent" 그래디언트에만 반응

3. **Adam**:
   - 1차 + 2차 모멘트 결합
   - 편향 보정으로 초기 스텝부터 올바른 스케일
   - 대부분의 NLP에서 표준

4. **AdamW**:
   - L2 정규화를 weight decay로 분리
   - Adam + 명시적 weight decay
   - 더 좋은 일반화 (이론적으로 정당화됨)

| 옵티마이저 | 수렴 속도 | 희소성 | 메모리 | 일반화 |
|----------|---------|------|------|--------|
| SGD | 느림 | 중간 | 낮음 | 좋음 |
| Adagrad | 빠름 | 높음 | 높음 | 중간 |
| RMSProp | 빠름 | 중간 | 높음 | 중간 |
| Adam | 빠름 | 중간 | 높음 | 중간 |
| AdamW | 빠름 | 중간 | 높음 | 좋음 |

## 🤔 생각해볼 문제

1. **문제 1**: Adagrad가 왜 희소 그래디언트에 좋은가?
   - 수학적으로 설명하시오.

2. **문제 2**: RMSProp의 지수 이동 평균은 어떻게 Adagrad의 단조 감소를 막는가?

3. **문제 3**: Adam의 bias correction이 없다면 어떤 일이 벌어지는가?
   - 초기 스텝에서의 영향을 정량화하시오.

4. **문제 4**: Adam이 GD보다 일반화 성능이 낮은 이유는?
   - "Sharp minima" vs "Flat minima" 관점에서 설명하시오.

5. **문제 5** (구현): SNR-based adaptive learning rate를 직접 구현하고 Adam과 비교하시오.

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. SGD 수렴 분석](./04-sgd-convergence.md) | [📚 README](../README.md) | [06. 뉴턴법과 준뉴턴법 ▶](./06-newton-quasi-newton.md) |

</div>
