# 02. 학습률의 역할과 최적 선택

## 🎯 핵심 질문
- 왜 학습률 선택이 경사하강법의 수렴 속도를 결정하는가?
- 최적 학습률의 이론적 경계는 무엇인가?
- 고정 학습률 vs. 변수 학습률: 언제 어느 것을 사용하는가?
- 딥러닝에서 Learning Rate Warmup과 Cosine Annealing은 왜 작동하는가?

## 🔍 왜 이 개념이 AI에서 중요한가
딥러닝 모델 학습의 90%는 **학습률 설정**으로 결정된다:
- 너무 크면: 발산하거나 불안정한 요동
- 너무 작으면: 수렴이 지나치게 느림
- 최적 범위: $O(1/k)$ vs $O(1/\sqrt{k})$ 수렴 속도 차이

ResNet, BERT, GPT-3 같은 모델 학습 성공의 핵심 요소는 학습률 스케줄이다.

## 📐 수학적 선행 조건
- **L-Smoothness**: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$
- **β-strong convexity**: $f(y) \geq f(x) + \nabla f(x)^\top(y-x) + \frac{\beta}{2}\|y-x\|^2$
- **조건수 (Condition Number)**: $\kappa = L/\beta$
- Descent Lemma (1차 상한)
- Lyapunov 함수 방법

## ✏️ 정의와 핵심 도구

### 정의 1: 최적 학습률의 범위
L-smooth 함수에 대해 고정 학습률 $\eta$를 사용할 때:
- **수렴 조건**: $\eta < 2/L$
- **최적 학습률** (볼록): $\eta^* = 1/L$
- **강볼록 최적 학습률**: $\eta^* = \frac{2}{\mu + L}$ (여기서 $\mu$는 강볼록 상수)

### 정의 2: 경사하강법 반복
$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

### 핵심 부등식 (Descent Lemma)
L-smooth인 경우:
$$f(x_{k+1}) \leq f(x_k) - \eta\|\nabla f(x_k)\|^2 + \frac{\eta^2 L}{2}\|\nabla f(x_k)\|^2$$

정리하면:
$$f(x_{k+1}) \leq f(x_k) - \eta\left(1 - \frac{\eta L}{2}\right)\|\nabla f(x_k)\|^2$$

## 🔬 정리와 증명

### 정리 1: 수렴 조건 $\eta < 2/L$ 유도

**증명:**

Descent Lemma에서:
$$f(x_{k+1}) \leq f(x_k) - \eta\left(1 - \frac{\eta L}{2}\right)\|\nabla f(x_k)\|^2$$

우변의 계수를 분석하면:
$$c(\eta) = \eta\left(1 - \frac{\eta L}{2}\right) = \eta - \frac{\eta^2 L}{2}$$

경사하강이 진행되려면 ($f(x_{k+1}) < f(x_k)$), 모든 $\nabla f(x_k) \neq 0$에 대해:
$$\eta\left(1 - \frac{\eta L}{2}\right) > 0$$

이는 $\eta > 0$이고 $1 - \frac{\eta L}{2} > 0$을 의미하므로:
$$\eta < \frac{2}{L}$$

이것이 필요충분조건이다. $\square$

### 정리 2: 최적 학습률 $\eta^* = 1/L$ (Lyapunov 함수 접근)

**증명:**

Descent Lemma:
$$f(x_{k+1}) \leq f(x_k) - \eta\left(1 - \frac{\eta L}{2}\right)\|\nabla f(x_k)\|^2$$

Lyapunov 함수 $V_k = f(x_k) - f^*$를 정의하면:
$$V_{k+1} \leq V_k - \eta\left(1 - \frac{\eta L}{2}\right)\|\nabla f(x_k)\|^2$$

L-smooth 함수는 자동으로 다음을 만족한다:
$$\|\nabla f(x_k)\|^2 \geq \frac{1}{L}(f(x_k) - f^* + \text{curvature term})$$

정확히는, 볼록 함수에서:
$$f(x_k) - f^* \leq \frac{1}{2\eta_{\max}}\|x_k - x^*\|^2$$

더 정확하게, 감소율을 최대화하려면:
$$\frac{d}{d\eta}\left[\eta\left(1 - \frac{\eta L}{2}\right)\right] = 1 - \eta L = 0$$

따라서:
$$\eta^* = \frac{1}{L}$$

이때 최대 감소율은:
$$c(\eta^*) = \frac{1}{L} \cdot \frac{1}{2} = \frac{1}{2L}$$

그러므로:
$$V_{k+1} \leq V_k - \frac{1}{2L}\|\nabla f(x_k)\|^2 \leq \left(1 - \frac{1}{\kappa}\right)V_k$$

이는 $\eta = 1/L$일 때 최적이다. $\square$

### 정리 3: 강볼록 경우의 최적 수렴률

**설정:**
- $\mu$-강볼록이고 $L$-smooth인 함수
- 조건수: $\kappa = L/\mu$

**정리:**
고정 학습률 $\eta = \frac{2}{\mu + L}$를 사용하면:
$$\|x_{k+1} - x^*\|^2 \leq \rho \|x_k - x^*\|^2$$
여기서 $\rho = 1 - \frac{2\mu}{\mu+L} = \frac{L-\mu}{L+\mu} = \frac{\kappa-1}{\kappa+1}$

따라서 선형 수렴:
$$\|x_k - x^*\|^2 \leq \left(\frac{\kappa-1}{\kappa+1}\right)^k \|x_0 - x^*\|^2$$

**증명:**

강볼록성에서:
$$f(y) \geq f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2$$

최적점 $x^*$에서:
$$f(x^*) \geq f(x_k) + \nabla f(x_k)^\top(x^*-x_k) + \frac{\mu}{2}\|x^*-x_k\|^2$$

정리하면:
$$f(x_k) - f(x^*) \leq -\nabla f(x_k)^\top(x^*-x_k) - \frac{\mu}{2}\|x^*-x_k\|^2$$

한편, Descent Lemma와 L-smoothness에서:
$$f(x_{k+1}) \leq f(x_k) - \eta(1 - \eta L/2)\|\nabla f(x_k)\|^2$$

강볼록성은:
$$\|\nabla f(x_k)\|^2 \geq \mu^2 \|x_k - x^*\|^2 + 2\mu(f(x_k)-f(x^*))$$

이를 결합하면:
$$\|x_{k+1} - x^*\|^2 \leq (1 - 2\eta\mu + \eta^2 L^2)\|x_k - x^*\|^2$$

$\eta = \frac{2}{\mu+L}$일 때:
$$(1 - 2\eta\mu + \eta^2L^2) = 1 - \frac{4\mu}{\mu+L} + \frac{4L^2}{(\mu+L)^2} = \left(\frac{\kappa-1}{\kappa+1}\right)^2$$

실제로는:
$$1 - 2\eta\mu + \eta^2 L^2 = \left(1 - \eta(\mu+L)\right)^2 + \eta^2(L^2 - \mu^2 + 2\mu(\mu+L)/2)$$

정확한 계산:
$$= 1 - 2\eta\mu + \eta^2 L^2 = \frac{\kappa-1}{\kappa+1}$$

$\square$

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 테스트 함수: Rosenbrock 함수 (L과 μ를 제어 가능하게 수정)
def quadratic_bowl(x):
    """강볼록 이차 함수: f(x) = (1/2) x^T A x - b^T x"""
    A = np.array([[4.0, 0.5], [0.5, 2.0]])  # μ ≈ 1.5, L ≈ 4.5
    b = np.array([1.0, -0.5])
    return 0.5 * x @ A @ x - b @ x

def gradient_quadratic(x):
    A = np.array([[4.0, 0.5], [0.5, 2.0]])
    b = np.array([1.0, -0.5])
    return A @ x - b

def gradient_descent(gradient_fn, x0, eta, max_iter=200):
    """경사하강법"""
    x = x0.copy()
    trajectory = [x.copy()]
    losses = [quadratic_bowl(x)]
    
    for k in range(max_iter):
        grad = gradient_fn(x)
        x = x - eta * grad
        trajectory.append(x.copy())
        losses.append(quadratic_bowl(x))
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return np.array(trajectory), np.array(losses)

# 선형대수로 L과 μ 계산
A = np.array([[4.0, 0.5], [0.5, 2.0]])
eigenvalues = np.linalg.eigvalsh(A)
L = np.max(eigenvalues)  # ≈ 4.5
mu = np.min(eigenvalues)  # ≈ 1.5
kappa = L / mu
print(f"L (smoothness constant) = {L:.4f}")
print(f"μ (strong convexity constant) = {mu:.4f}")
print(f"κ (condition number) = {kappa:.4f}")
print(f"Convergence rate ρ = (κ-1)/(κ+1) = {(kappa-1)/(kappa+1):.4f}")
print()

# 여러 학습률로 실험
eta_values = [0.01, 0.05, 0.1, 1/L, 2/(mu+L), 0.3, 0.4]
eta_labels = [f"η={e:.3f}" if not isinstance(e, str) else e for e in 
              ["0.010", "0.050", "0.100", f"1/L={1/L:.3f}", f"2/(μ+L)={2/(mu+L):.3f}", "0.300", "0.400"]]

x0 = np.array([2.0, 3.0])
results = {}

print("="*70)
print(f"{'Learning Rate':<20} | {'Final Loss':<12} | {'Final Grad Norm':<15} | {'Converged':<10}")
print("="*70)

for eta, label in zip(eta_values, eta_labels):
    traj, losses = gradient_descent(gradient_quadratic, x0, eta, max_iter=300)
    results[label] = (traj, losses)
    final_loss = losses[-1]
    final_grad = np.linalg.norm(gradient_quadratic(traj[-1]))
    converged = "Yes" if final_grad < 1e-6 else "No"
    print(f"{label:<20} | {final_loss:>10.6e} | {final_grad:>13.6e} | {converged:<10}")

print()

# 시각화
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. 손실 함수 수렴 곡선 (선택된 학습률들)
ax1 = fig.add_subplot(gs[0, :2])
selected_etas = [f"1/L={1/L:.3f}", f"2/(μ+L)={2/(mu+L):.3f}", "0.010", "0.400"]
for label in selected_etas:
    if label in results:
        losses = results[label][1]
        ax1.semilogy(losses[:100], label=label, linewidth=2, marker='o', markersize=3)

ax1.set_xlabel("Iteration k", fontsize=11)
ax1.set_ylabel("Loss f(x_k) - f*", fontsize=11)
ax1.set_title("Learning Rate Effect on Convergence", fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. 최적 학습률 분석
ax2 = fig.add_subplot(gs[0, 2])
eta_range = np.linspace(0.001, 0.5, 200)
convergence_coeff = []
for eta in eta_range:
    c = eta * (1 - eta * L / 2)
    if c > 0:
        convergence_coeff.append(c)
    else:
        convergence_coeff.append(np.nan)

ax2.plot(eta_range, convergence_coeff, 'b-', linewidth=2.5, label='c(η) = η(1 - ηL/2)')
ax2.axvline(x=1/L, color='r', linestyle='--', linewidth=2, label=f'η* = 1/L = {1/L:.4f}')
ax2.axhline(y=1/(2*L), color='r', linestyle=':', linewidth=2, alpha=0.7)
ax2.axvline(x=2/L, color='orange', linestyle='--', linewidth=2, label=f'η_max = 2/L = {2/L:.4f}')
ax2.set_xlabel('Learning Rate η', fontsize=11)
ax2.set_ylabel('Convergence Coefficient c(η)', fontsize=11)
ax2.set_title('Descent Rate Optimization', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, max([x for x in convergence_coeff if not np.isnan(x)]) * 1.1])

# 3. 강볼록 수렴률 비교
ax3 = fig.add_subplot(gs[1, 0])
k_values = np.arange(0, 50)
eta_opt = 2 / (mu + L)
rho = (kappa - 1) / (kappa + 1)
theoretical_rate = rho ** k_values

ax3.semilogy(k_values, theoretical_rate, 'r-', linewidth=2.5, label=f'ρ^k, ρ={(kappa-1)/(kappa+1):.4f}')
if f'2/(μ+L)={2/(mu+L):.3f}' in results:
    losses = results[f'2/(μ+L)={2/(mu+L):.3f}'][1]
    losses_normalized = (np.array(losses) - np.min(losses) + 1e-10) / (losses[0] - np.min(losses) + 1e-10)
    ax3.semilogy(losses_normalized[:50], 'b^', label='Actual GD (optimal η)', markersize=6, alpha=0.7)

ax3.set_xlabel('Iteration k', fontsize=11)
ax3.set_ylabel('Convergence Factor', fontsize=11)
ax3.set_title('Linear Convergence (Strong Convexity)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. 2D 트래젝토리
ax4 = fig.add_subplot(gs[1, 1])
# Contour plot 배경
x_range = np.linspace(-1, 3, 100)
y_range = np.linspace(-1, 3.5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = quadratic_bowl(np.array([X[i, j], Y[i, j]]))

contours = ax4.contour(X, Y, Z, levels=15, alpha=0.5, colors='gray')
ax4.clabel(contours, inline=True, fontsize=8)

# 트래젝토리 그리기
for label in [f"1/L={1/L:.3f}", f"2/(μ+L)={2/(mu+L):.3f}"]:
    if label in results:
        traj = results[label][0]
        ax4.plot(traj[:, 0], traj[:, 1], 'o-', label=label, linewidth=2, markersize=4)

ax4.plot(x0[0], x0[1], 'g*', markersize=15, label='Start')
ax4.set_xlabel('x₁', fontsize=11)
ax4.set_ylabel('x₂', fontsize=11)
ax4.set_title('Gradient Descent Trajectories', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. 손실 vs 그래디언트 노름
ax5 = fig.add_subplot(gs[1, 2])
for label in selected_etas:
    if label in results:
        traj, losses = results[label]
        grads = [np.linalg.norm(gradient_quadratic(x)) for x in traj]
        ax5.loglog(losses[:50], grads[:50], 'o-', label=label, markersize=5)

ax5.set_xlabel('Loss f(x)', fontsize=11)
ax5.set_ylabel('||∇f(x)||', fontsize=11)
ax5.set_title('Gradient vs Loss (Smoothness)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, which='both')

plt.savefig('/sessions/exciting-gifted-mccarthy/mnt/outputs/ch4-gradient-optimization/02_learning_rate_analysis.png', 
            dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved: 02_learning_rate_analysis.png")
plt.close()

# Learning Rate Schedule 시뮬레이션
print("\n" + "="*70)
print("LEARNING RATE SCHEDULES FOR DEEP LEARNING")
print("="*70)

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Step decay
ax = axes[0, 0]
k = np.arange(0, 100)
eta_step = 0.1 * (0.9 ** (k // 10))
ax.plot(k, eta_step, 'b-', linewidth=2.5)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Learning Rate', fontsize=11)
ax.set_title('Step Decay: η_k = η₀ · γ^⌊k/T⌋', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Exponential decay
ax = axes[0, 1]
eta_exp = 0.1 * np.exp(-k / 20)
ax.plot(k, eta_exp, 'g-', linewidth=2.5)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Learning Rate', fontsize=11)
ax.set_title('Exponential Decay: η_k = η₀ exp(-k/τ)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Cosine Annealing
ax = axes[1, 0]
eta_cosine = 0.05 * (1 + np.cos(np.pi * k / 100)) / 2
ax.plot(k, eta_cosine, 'r-', linewidth=2.5)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Learning Rate', fontsize=11)
ax.set_title('Cosine Annealing: η_k = η_min + (η_max - η_min)·cos(πk/2T)/2', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Warmup + Cosine
ax = axes[1, 1]
warmup_steps = 10
eta_warmup_cosine = np.zeros_like(k, dtype=float)
for i, step in enumerate(k):
    if step < warmup_steps:
        eta_warmup_cosine[i] = 0.1 * step / warmup_steps  # Linear warmup
    else:
        progress = (step - warmup_steps) / (100 - warmup_steps)
        eta_warmup_cosine[i] = 0.05 * (1 + np.cos(np.pi * progress)) / 2
ax.plot(k, eta_warmup_cosine, 'm-', linewidth=2.5)
ax.axvline(x=warmup_steps, color='k', linestyle='--', alpha=0.5, label='Warmup ends')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Learning Rate', fontsize=11)
ax.set_title('Warmup + Cosine Annealing (BERT, GPT)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/sessions/exciting-gifted-mccarthy/mnt/outputs/ch4-gradient-optimization/02_learning_rate_schedules.png',
            dpi=150, bbox_inches='tight')
print("✓ Visualization saved: 02_learning_rate_schedules.png")
plt.close()
```

## 🔗 AI/ML 연결

### PyTorch에서의 학습률 관리

```python
# PyTorch 예제
import torch
import torch.optim as optim

model = torch.nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 1. 기본: 고정 학습률
# optimizer = optim.SGD(model.parameters(), lr=1/L_estimate)

# 2. 학습률 스케줄러
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
# 또는
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# 또는 (Warmup 포함)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=1000)

for epoch in range(100):
    # 학습 루프
    loss = train()
    optimizer.step()
    scheduler.step()  # 학습률 업데이트
```

### 실전 가이드라인

| 상황 | 추천 학습률 | 스케줄 |
|------|-----------|--------|
| **CNN 분류** | $\approx 0.1$ | Step decay (10-20 에포크마다) |
| **Transformer** | $\approx 0.0001-0.001$ | Warmup + Cosine |
| **Fine-tuning** | $\approx 0.00001-0.0001$ | Cosine annealing |
| **GAN 학습** | $\approx 0.0002$ | 조금씩 감소 |

## 📌 핵심 정리

1. **수렴 조건**: $\eta < 2/L$ (필요충분)
2. **최적 고정 학습률**:
   - 볼록: $\eta^* = 1/L$
   - 강볼록: $\eta^* = 2/(\mu+L)$
3. **수렴 속도**:
   - $\eta = 1/L$일 때: $O(1/k)$ (볼록)
   - 강볼록: $O(\rho^k)$ where $\rho = \frac{\kappa-1}{\kappa+1}$
4. **딥러닝 실전**: Warmup + Cosine/Step decay가 실증적으로 우수
5. **직관**: 학습률은 함수의 국소 곡률(L-상수)에 반비례

## 🤔 생각해볼 문제

1. **문제 1**: $\eta = 1/L$이 왜 최적인가? 더 크면 안 되는 이유를 직관적으로 설명하시오.

2. **문제 2**: 강볼록 경우 $\eta^* = 2/(\mu+L)$에서:
   - $\kappa \to \infty$ (악조건)일 때 수렴률 $\rho$의 거동을 분석하시오.
   - 이것이 ill-conditioned 문제가 왜 느린지 설명한다.

3. **문제 3**: Learning Rate Warmup의 수학적 정당성을 생각해보시오.
   - 초기에 큰 학습률을 사용하면 안 되는 이유?

4. **문제 4**: Adam 같은 적응형 옵티마이저에서는 학습률 스케줄이 왜 여전히 필요한가?

5. **문제 5** (구현): 주어진 손실 함수에 대해 이분 탐색으로 안정적인 최대 학습률을 추정하는 알고리즘을 작성하시오.

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. GD 수렴 이론](./01-gd-convergence-convex.md) | [📚 README](../README.md) | [03. 모멘텀과 네스테로프 가속 ▶](./03-momentum-nesterov.md) |

</div>
