# 07. Loss Landscape의 기하학

## 🎯 핵심 질문
- 왜 딥러닝은 비볼록 함수에서도 수렴하는가?
- "Sharp minima"와 "flat minima"는 일반화와 어떤 관계가 있는가?
- Loss landscape의 기하학이 옵티마이저 선택에 어떻게 영향을 미치는가?
- Mode connectivity와 sharpness awareness minimization이 의미하는 바는?

## 🔍 왜 이 개념이 AI에서 중요한가

**실전 딥러닝의 역설**:
- 이론: 비볼록 최적화는 불가능 (모든 국소최소값이 최적)
- 현실: 거대 신경망이 놀랍도록 잘 학습됨
- 이유: Loss landscape의 특수한 기하학 구조

**현대 최적화의 핵심**:
- Sharpness-Aware Minimization (SAM): SOTA 성능
- Mode connectivity: 모델 앙상블의 이론적 기초
- Catapult phase: 학습 초기의 비직관적 동작
- Neural Tangent Kernel: 무한폭 극한에서의 선형 동학

## 📐 수학적 선행 조건

- **헤시안 고유값**: 손실함수의 local curvature
- **Condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$
- **Riemannian geometry**: 곡면 위에서의 거리와 측지선
- **Hessian spectrum**: 손실 지형의 주요 특성

## ✏️ 정의와 핵심 도구

### 정의 1: Sharp vs Flat Minima

점 $x^*$에서의 **Hessian spectrum**:
$$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$$

**Sharp minimum**: $\lambda_{\max}$ 크고 양수 (큰 곡률)
**Flat minimum**: $\lambda_{\max}$ 작음 (작은 곡률)

**정량적 정의**:
$$\text{Sharpness}(x^*, \rho) = \max_{\|u\|=1, \|u\| \leq \rho} f(x^* + u) - f(x^*)$$

### 정의 2: Sharpness-Aware Minimization (SAM)

**목표**:
$$\min_w \max_{\|\epsilon\| \leq \rho} f(w + \epsilon)$$

**SAM 알고리즘**:
$$\epsilon^* = \arg\max_{\|\epsilon\| \leq \rho} f(w + \epsilon) \approx \rho \frac{\nabla f(w)}{\|\nabla f(w)\|}$$

$$w_{t+1} = w_t - \alpha \nabla f(w_t + \epsilon^*)$$

### 정의 3: Mode Connectivity

두 수렴된 모델 $w_1$, $w_2$에 대해:
**선형 모드 연결성**:
$$\exists \gamma \in [0,1]: f(\gamma w_1 + (1-\gamma)w_2) \leq \max(f(w_1), f(w_2))$$

또는 더 일반적으로, Bezier 곡선을 통한 연결성.

### 정의 4: Neural Tangent Kernel (NTK)

**커널 함수** (width $\to \infty$):
$$K(x, x') = \nabla_\theta f(x)^\top \nabla_\theta f(x')$$

**동역학** (NTK regime):
$$\frac{d}{dt}f(x, \theta(t)) = -K(x, x'(t)) (f(x, \theta) - y(x))$$

## 🔬 정리와 증명

### 정리 1: Sharp Minima와 Generalization Gap

**경험적 관찰** (Keskar et al., 2016):
- Sharp minima → 큰 generalization gap
- Flat minima → 작은 generalization gap

**이론적 설명** (PAC-Bayes):
일반화 오차의 상한:
$$\text{Gen Error} \lesssim \sqrt{\frac{1}{m}(d \log(1 + \text{Sharpness}) + \log(1/\delta))}$$

여기서:
- $m$: 데이터 샘플 수
- $d$: 파라미터 차원
- Sharpness: $\lambda_{\max}(H)$ 또는 $\max_{\|\delta\|=\rho} f(w+\delta) - f(w)$

**증명 스케치:**

PAC-Bayes 프레임워크에서, 훈련 데이터 $S$에 대해:
$$\mathbb{P}_{w \sim \mathcal{N}(w^*, \sigma^2)} [\text{loss}_S(w) > \text{loss}_S(w^*) + \epsilon] \leq \ldots$$

$w^*$ 주변의 "신뢰도"가 높을수록 (flat), generalization이 좋음.

Flat minimum에서는 작은 섭동 $\delta$에도 손실이 거의 변하지 않으므로:
$$f(w^* + \delta) \approx f(w^*)$$

따라서 테스트 샘플도 비슷하게 동작할 확률이 높음.

반면 sharp minimum은:
$$f(w^* + \delta) \gg f(w^*) \text{ for small } \delta$$

테스트 데이터가 약간 다르면 손실이 급증 가능.

$\square$

### 정리 2: Sharpness-Aware Minimization의 수렴성

**설정:**
$$L(w) = \mathbb{E}_{(x,y)} \ell(f(w, x), y)$$

SAM 목표:
$$\min_w \max_{\|\epsilon\| \leq \rho} L(w + \epsilon)$$

**정리:**
gradient descent를 다음과 같이 수정하면:
$$\epsilon_t = \arg\max_{\|\epsilon\| \leq \rho} L(w_t + \epsilon)$$
$$w_{t+1} = w_t - \alpha \nabla L(w_t + \epsilon_t)$$

$\rho$가 적절하면:
$$\mathbb{E}[\|\nabla L(w_T)\|^2] = O(1/\sqrt{T}) \quad \text{(비볼록)}$$

그리고 동시에 **sharpness도 제어**됨:
$$\lambda_{\max}(H_L(w_T)) = O(\text{regularization effect})$$

**증명 아이디어:**

$\epsilon^* \approx \rho \frac{\nabla L(w)}{\|\nabla L(w)\|}$일 때:
$$\nabla L(w + \epsilon^*) \approx \nabla L(w) + H_L(w) \epsilon^*$$

따라서:
$$\nabla L(w + \epsilon^*) \approx \nabla L(w) + \rho H_L(w) \frac{\nabla L(w)}{\|\nabla L(w)\|}$$

이를 사용하여 업데이트하면:
$$w_{t+1} = w_t - \alpha \left( \nabla L(w_t) + \rho H_L(w_t) \frac{\nabla L(w_t)}{\|\nabla L(w_t)\|} \right)$$

두 번째 항이 **Hessian eigenvalues를 제한**:

높은 고유값에서는:
$$\text{step} \propto \lambda_{\max} \cdot \text{sign}(g) \to \text{큰 스텝이 고유값을 활용}$$

$\rho$ 선택이 이를 조절함.

$\square$

### 정리 3: Mode Connectivity와 손실 표면 구조

**정리** (Garipov et al., 2018):
SGD로 수렴된 두 모델 $w_1, w_2$에 대해, 대부분의 경우:
$$\exists \gamma \in [0,1]: f(\gamma w_1 + (1-\gamma)w_2) \approx f(w_1) \approx f(w_2)$$

즉, 두 최소값이 **선형적으로** 연결 가능 (permutation 후).

**증명 스케치:**

1. Permutation invariance: 신경망 뉴런은 순서가 없음
   - 같은 수용력의 모델들은 같은 손실 도달 가능
   
2. Loss landscape의 고차원 특성:
   - 거대 차원 $d$에서, 두 점 사이의 "직선"은 거의 대부분:
   $$f(\gamma w_1 + (1-\gamma)w_2) \approx \frac{1}{2}(f(w_1) + f(w_2))$$
   
3. 특히 flat region (큰 신경망):
   - $\lambda_{\max}(H) \approx 0$인 곡선 방향 존재
   - 그 방향으로는 손실 변화 없음

따라서 mode connectivity 가능.

$\square$

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# 1D Loss landscape 함수들
def loss_landscape_1d(x, style='multimodal'):
    """여러 스타일의 손실 지형"""
    if style == 'multimodal':
        return np.sin(x) + 0.1*x**2
    elif style == 'sharp':
        return 100 * (x**2 - 0.1)**2 + (x - 0.1)**2
    elif style == 'flat':
        return 0.01 * np.sin(5*x) + 0.001*x**2
    elif style == 'multi_minima':
        return (x**2 - 1)**2 * np.cos(5*x) + 0.1*x**2

# 2D Loss landscape (Rosenbrock)
def loss_2d(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

# Hessian 계산 (수치)
def hessian_numerical(x, loss_fn, eps=1e-5):
    """수치 헤시안"""
    d = len(x)
    H = np.zeros((d, d))
    
    for i in range(d):
        for j in range(d):
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            
            x_pm = x.copy()
            x_pm[i] += eps
            x_pm[j] -= eps
            
            x_mp = x.copy()
            x_mp[i] -= eps
            x_mp[j] += eps
            
            x_mm = x.copy()
            x_mm[i] -= eps
            x_mm[j] -= eps
            
            H[i, j] = (loss_2d(x_pp[0], x_pp[1]) 
                      - loss_2d(x_pm[0], x_pm[1])
                      - loss_2d(x_mp[0], x_mp[1]) 
                      + loss_2d(x_mm[0], x_mm[1])) / (4 * eps**2)
    
    return H

# Sharpness 계산
def compute_sharpness(x, loss_fn, rho=0.1, n_samples=20):
    """
    Sharpness(x, ρ) = max_||u||=1, ||u||≤ρ [f(x+u) - f(x)]
    """
    f_x = loss_2d(x[0], x[1])
    max_loss = f_x
    
    # 랜덤 방향 샘플링
    for _ in range(n_samples):
        u = np.random.randn(2)
        u = u / np.linalg.norm(u) * rho
        f_xu = loss_2d(x[0] + u[0], x[1] + u[1])
        max_loss = max(max_loss, f_xu)
    
    return max_loss - f_x

# SAM (Sharpness Aware Minimization) 구현
def sam_step(x, loss_fn, rho=0.01, alpha=0.01, eps=1e-5):
    """SAM 한 스텝"""
    # 1. 그래디언트 계산
    grad = np.zeros(2)
    for i in range(2):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (loss_2d(x_plus[0], x_plus[1]) - loss_2d(x_minus[0], x_minus[1])) / (2*eps)
    
    # 2. Adversarial perturbation
    if np.linalg.norm(grad) > 0:
        epsilon = rho * grad / np.linalg.norm(grad)
    else:
        return x
    
    # 3. SAM 업데이트 (adversarial point에서의 그래디언트)
    x_adv = x + epsilon
    grad_adv = np.zeros(2)
    for i in range(2):
        x_plus = x_adv.copy()
        x_plus[i] += eps
        x_minus = x_adv.copy()
        x_minus[i] -= eps
        grad_adv[i] = (loss_2d(x_plus[0], x_plus[1]) - loss_2d(x_minus[0], x_minus[1])) / (2*eps)
    
    return x - alpha * grad_adv

# GD 스텝
def gd_step(x, loss_fn, alpha=0.01, eps=1e-5):
    """일반적 경사하강 스텝"""
    grad = np.zeros(2)
    for i in range(2):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (loss_2d(x_plus[0], x_plus[1]) - loss_2d(x_minus[0], x_minus[1])) / (2*eps)
    
    return x - alpha * grad

# Mode connectivity 시뮬레이션
def check_mode_connectivity(w1, w2, loss_fn, n_points=100):
    """두 점 사이의 선형 모드 연결성"""
    gamma_range = np.linspace(0, 1, n_points)
    losses = []
    
    for gamma in gamma_range:
        w_interp = gamma * w1 + (1 - gamma) * w2
        loss = loss_2d(w_interp[0], w_interp[1])
        losses.append(loss)
    
    return gamma_range, np.array(losses)

print("="*70)
print("LOSS LANDSCAPE GEOMETRY ANALYSIS")
print("="*70)

# 1D 손실 지형 분석
print("\n1D Loss Landscape Sharpness")
x_range = np.linspace(-2, 2, 1000)

for style in ['sharp', 'flat', 'multimodal']:
    losses = loss_landscape_1d(x_range, style=style)
    
    # 최소값 찾기
    min_idx = np.argmin(losses)
    x_min = x_range[min_idx]
    
    # 수치 2차 미분 (sharpness)
    eps = 0.01
    f_pp = (loss_landscape_1d(x_min + eps, style) 
           - 2*loss_landscape_1d(x_min, style) 
           + loss_landscape_1d(x_min - eps, style)) / (eps**2)
    
    print(f"  {style:<15}: x*={x_min:7.4f}, f(x*)={losses[min_idx]:8.5f}, f''(x*)={f_pp:8.5f}")

print()

# 2D 모드 연결성 실험
print("Mode Connectivity Test (Rosenbrock)")

# 두 개의 다른 초기점에서 GD로 수렴
x0_1 = np.array([0.5, 0.5])
x0_2 = np.array([-1.0, 1.5])

# GD로 수렴
w1 = x0_1.copy()
for _ in range(500):
    w1 = gd_step(w1, loss_2d, alpha=0.001)

w2 = x0_2.copy()
for _ in range(500):
    w2 = gd_step(w2, loss_2d, alpha=0.001)

print(f"  w1 = {w1}, f(w1) = {loss_2d(w1[0], w1[1]):.6e}")
print(f"  w2 = {w2}, f(w2) = {loss_2d(w2[0], w2[1]):.6e}")

gamma_range, conn_losses = check_mode_connectivity(w1, w2, loss_2d, n_points=50)
max_loss_on_line = np.max(conn_losses)
print(f"  Max loss on line: {max_loss_on_line:.6e}")
print(f"  Mode connectivity ratio: {max_loss_on_line / max(loss_2d(w1[0], w1[1]), loss_2d(w2[0], w2[1])):.4f}")

print()

# SAM vs GD 비교
print("SAM vs GD Sharpness Comparison")

x_init = np.array([-0.8, 0.5])

# GD
x_gd = x_init.copy()
sharp_history_gd = []
for k in range(200):
    sharp_history_gd.append(compute_sharpness(x_gd, loss_2d, rho=0.1))
    x_gd = gd_step(x_gd, loss_2d, alpha=0.001)

# SAM
x_sam = x_init.copy()
sharp_history_sam = []
for k in range(200):
    sharp_history_sam.append(compute_sharpness(x_sam, loss_2d, rho=0.1))
    x_sam = sam_step(x_sam, loss_2d, rho=0.01, alpha=0.001)

print(f"  Final sharpness (GD): {sharp_history_gd[-1]:.6f}")
print(f"  Final sharpness (SAM): {sharp_history_sam[-1]:.6f}")
print(f"  Sharpness reduction: {sharp_history_gd[-1]/sharp_history_sam[-1]:.2f}x")

print()

# 시각화
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. 1D Loss Landscapes
ax1 = fig.add_subplot(gs[0, :])
x_range_plot = np.linspace(-2, 2, 300)
styles = ['sharp', 'flat', 'multimodal']
colors = ['red', 'blue', 'green']

for style, color in zip(styles, colors):
    losses = loss_landscape_1d(x_range_plot, style=style)
    ax1.plot(x_range_plot, losses, color=color, linewidth=2.5, label=style)

ax1.set_xlabel('Parameter', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('1D Loss Landscapes: Comparing Sharpness', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 2. 2D Loss contour
ax2 = fig.add_subplot(gs[1, 0])
x_2d = np.linspace(-2, 3, 100)
y_2d = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_2d, y_2d)
Z = loss_2d(X, Y)

contours = ax2.contour(X, Y, Z, levels=20, alpha=0.6, cmap='viridis')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.plot(1, 1, 'r*', markersize=15, label='Optimum')
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)
ax2.set_title('Loss Landscape Contour', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. 3D Loss surface
ax3 = fig.add_subplot(gs[1, 1], projection='3d')
x_3d = np.linspace(-1.5, 2.5, 50)
y_3d = np.linspace(-0.5, 3.5, 50)
X3, Y3 = np.meshgrid(x_3d, y_3d)
Z3 = np.log(loss_2d(X3, Y3) + 1)

surf = ax3.plot_surface(X3, Y3, Z3, cmap='viridis', alpha=0.8, edgecolor='none')
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y', fontsize=10)
ax3.set_zlabel('log(Loss)', fontsize=10)
ax3.set_title('3D Loss Surface (log scale)', fontsize=12, fontweight='bold')

# 4. Hessian eigenvalues along path
ax4 = fig.add_subplot(gs[1, 2])
x_path = x_init.copy()
eigenvalues_max = []
eigenvalues_min = []

for k in range(100):
    H = hessian_numerical(x_path)
    eigs = np.linalg.eigvalsh(H)
    eigenvalues_max.append(np.max(eigs))
    eigenvalues_min.append(np.min(eigs))
    x_path = gd_step(x_path, loss_2d, alpha=0.001)

ax4.semilogy(eigenvalues_max, 'r-', linewidth=2.5, label='λ_max', marker='o', markersize=4, markevery=10)
ax4.semilogy(eigenvalues_min, 'b-', linewidth=2.5, label='λ_min', marker='^', markersize=4, markevery=10)
ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('Eigenvalue', fontsize=11)
ax4.set_title('Hessian Spectrum Evolution (GD)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')

# 5. Mode Connectivity
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(gamma_range, conn_losses, 'b-', linewidth=2.5, marker='o', markersize=6)
ax5.axhline(y=loss_2d(w1[0], w1[1]), color='r', linestyle='--', alpha=0.7, linewidth=2, label='f(w₁)')
ax5.axhline(y=loss_2d(w2[0], w2[1]), color='g', linestyle='--', alpha=0.7, linewidth=2, label='f(w₂)')
ax5.fill_between(gamma_range, 
                 np.minimum(loss_2d(w1[0], w1[1]), loss_2d(w2[0], w2[1])),
                 np.maximum(loss_2d(w1[0], w1[1]), loss_2d(w2[0], w2[1])),
                 alpha=0.2, color='gray')
ax5.set_xlabel('Interpolation γ', fontsize=11)
ax5.set_ylabel('Loss', fontsize=11)
ax5.set_title('Mode Connectivity (Linear)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. GD trajectory (sharpness 기반 색상)
ax6 = fig.add_subplot(gs[2, 1])
x_gd_traj = x_init.copy()
traj_gd_full = [x_gd_traj.copy()]
sharpness_vals = []

for k in range(100):
    sharp = compute_sharpness(x_gd_traj, loss_2d, rho=0.1)
    sharpness_vals.append(sharp)
    traj_gd_full.append(x_gd_traj.copy())
    x_gd_traj = gd_step(x_gd_traj, loss_2d, alpha=0.001)

traj_gd_full = np.array(traj_gd_full)
scatter = ax6.scatter(traj_gd_full[:, 0], traj_gd_full[:, 1], 
                     c=sharpness_vals + [sharpness_vals[-1]], cmap='RdYlBu_r', 
                     s=50, alpha=0.7, edgecolor='black', linewidth=0.5)
ax6.plot(traj_gd_full[:, 0], traj_gd_full[:, 1], 'k--', alpha=0.3, linewidth=1)
ax6.plot(1, 1, 'g*', markersize=15, label='Optimum')
plt.colorbar(scatter, ax=ax6, label='Sharpness')
ax6.set_xlabel('x', fontsize=11)
ax6.set_ylabel('y', fontsize=11)
ax6.set_title('GD Trajectory (colored by sharpness)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

# 7. SAM trajectory
ax7 = fig.add_subplot(gs[2, 2])
x_sam_traj = x_init.copy()
traj_sam_full = [x_sam_traj.copy()]
sharpness_sam = []

for k in range(100):
    sharp = compute_sharpness(x_sam_traj, loss_2d, rho=0.1)
    sharpness_sam.append(sharp)
    traj_sam_full.append(x_sam_traj.copy())
    x_sam_traj = sam_step(x_sam_traj, loss_2d, rho=0.01, alpha=0.001)

traj_sam_full = np.array(traj_sam_full)
scatter_sam = ax7.scatter(traj_sam_full[:, 0], traj_sam_full[:, 1], 
                         c=sharpness_sam + [sharpness_sam[-1]], cmap='RdYlBu_r', 
                         s=50, alpha=0.7, edgecolor='black', linewidth=0.5)
ax7.plot(traj_sam_full[:, 0], traj_sam_full[:, 1], 'k--', alpha=0.3, linewidth=1)
ax7.plot(1, 1, 'g*', markersize=15, label='Optimum')
plt.colorbar(scatter_sam, ax=ax7, label='Sharpness')
ax7.set_xlabel('x', fontsize=11)
ax7.set_ylabel('y', fontsize=11)
ax7.set_title('SAM Trajectory (colored by sharpness)', fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3)

# 8. Sharpness 비교
ax8 = fig.add_subplot(gs[3, 0])
ax8.semilogy(sharp_history_gd, 'r-', linewidth=2.5, label='GD', marker='o', markersize=4, markevery=20)
ax8.semilogy(sharp_history_sam, 'b-', linewidth=2.5, label='SAM', marker='^', markersize=4, markevery=20)
ax8.set_xlabel('Iteration', fontsize=11)
ax8.set_ylabel('Sharpness', fontsize=11)
ax8.set_title('Sharpness Evolution: GD vs SAM', fontsize=12, fontweight='bold')
ax8.legend(fontsize=11)
ax8.grid(True, alpha=0.3, which='both')

# 9. Loss 비교
ax9 = fig.add_subplot(gs[3, 1])
loss_gd_hist = []
loss_sam_hist = []
for traj_point in traj_gd_full:
    loss_gd_hist.append(loss_2d(traj_point[0], traj_point[1]))
for traj_point in traj_sam_full:
    loss_sam_hist.append(loss_2d(traj_point[0], traj_point[1]))

ax9.semilogy(loss_gd_hist, 'r-', linewidth=2.5, label='GD', marker='o', markersize=4, markevery=20)
ax9.semilogy(loss_sam_hist, 'b-', linewidth=2.5, label='SAM', marker='^', markersize=4, markevery=20)
ax9.set_xlabel('Iteration', fontsize=11)
ax9.set_ylabel('Loss', fontsize=11)
ax9.set_title('Loss Evolution: GD vs SAM', fontsize=12, fontweight='bold')
ax9.legend(fontsize=11)
ax9.grid(True, alpha=0.3, which='both')

# 10. Condition number evolution
ax10 = fig.add_subplot(gs[3, 2])
condition_numbers = []
x_cond = x_init.copy()
for k in range(100):
    H = hessian_numerical(x_cond)
    eigs = np.linalg.eigvalsh(H)
    if np.min(np.abs(eigs)) > 1e-10:
        cond = np.max(np.abs(eigs)) / np.max(1e-10, np.min(np.abs(eigs)))
        condition_numbers.append(cond)
    x_cond = gd_step(x_cond, loss_2d, alpha=0.001)

ax10.semilogy(condition_numbers, 'g-', linewidth=2.5, marker='s', markersize=5, markevery=10)
ax10.set_xlabel('Iteration', fontsize=11)
ax10.set_ylabel('Condition Number κ', fontsize=11)
ax10.set_title('Condition Number Evolution', fontsize=12, fontweight='bold')
ax10.grid(True, alpha=0.3, which='both')

plt.savefig('/sessions/exciting-gifted-mccarthy/mnt/outputs/ch4-gradient-optimization/07_loss_landscape_geometry.png',
            dpi=150, bbox_inches='tight')
print("✓ Visualization saved: 07_loss_landscape_geometry.png")
plt.close()
```

## 🔗 AI/ML 연결

### PyTorch SAM 구현
```python
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.rho = rho
    
    def step(self, closure):
        # 1단계: 현재 그래디언트 계산
        loss = closure()
        loss.backward()
        
        # 2단계: Adversarial perturbation
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                p.data_grad_norm = torch.norm(p.grad)
        
        # 3단계: SAM 업데이트
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm = torch.norm(p.grad)
                    if grad_norm > 0:
                        p.data += self.rho * p.grad / grad_norm
        
        # 4단계: Perturbed point에서 그래디언트 재계산
        loss = closure()
        loss.backward()
        
        # 5단계: 원래 위치로 복원 및 업데이트
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                p.data -= self.rho * p.grad / (torch.norm(p.grad) + 1e-12)
        
        self.base_optimizer.step()
```

## 📌 핵심 정리

1. **Sharp vs Flat Minima**:
   - Sharp: $\lambda_{\max}(H) \gg 0$ → 나쁜 일반화
   - Flat: $\lambda_{\max}(H) \approx 0$ → 좋은 일반화
   - PAC-Bayes 이론으로 설명 가능

2. **SAM (Sharpness-Aware Minimization)**:
   - Adversarial perturbation: $\epsilon^* = \rho \nabla f / \|\nabla f\|$
   - Flat minima 찾음 → 더 나은 일반화
   - 추가 계산 비용: 약 2배 (2x forward/backward)

3. **Mode Connectivity**:
   - 신경망의 loss landscape는 고차원에서 "convex"에 가까움
   - SGD로 찾은 여러 해들이 선형적으로 연결 가능
   - 모델 앙상블의 기초 제공

4. **Neural Tangent Kernel (NTK)**:
   - 무한폭 신경망은 "선형" 동역학
   - Practical networks (finite width)는 NTK와 유사하게 행동
   - 수렴 분석에 유용

## 🤔 생각해볼 문제

1. **문제 1**: Sharp minima가 일반화 성능이 나쁜 이유를 직관적으로 설명하시오.

2. **문제 2**: SAM이 flat minima를 찾는 원리는?
   - Adversarial perturbation이 왜 이를 보장하는가?

3. **문제 3**: Mode connectivity는 왜 존재하는가?
   - 고차원의 특수한 성질이 있는가?

4. **문제 4**: NTK가 실제 신경망과 왜 비슷할까?
   - 언제까지 유효한 근사인가?

5. **문제 5** (구현): 주어진 손실 함수에서 sharpness와 일반화 오차의 상관관계를 검증하시오.

---

<div align="center">

| | | |
|---|---|---|
| [◀ 06. 뉴턴법과 준뉴턴법](./06-newton-quasi-newton.md) | [📚 README](../README.md) | [Ch5-01. 계산 그래프와 자동미분 ▶](../ch5-backprop-autograd/01-computational-graph-ad.md) |

</div>
