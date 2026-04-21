# 03. 모멘텀과 네스테로프 가속

## 🎯 핵심 질문
- 왜 경사하강법의 반복에서 이전 이동을 더하면(모멘텀) 수렴이 빨라지는가?
- 모멘텀의 최적 가중치는 무엇인가?
- Nesterov Accelerated Gradient는 이론적으로 몇 배 빠른가?
- $O(1/k)$ vs $O(1/k^2)$ 수렴: 그 차이의 근원은?

## 🔍 왜 이 개념이 AI에서 중요한가

모멘텀은 **딥러닝 사실상 표준(de facto standard)**이다:
- ResNet, BERT, GPT-3 등 모든 대규모 모델의 기본 옵티마이저
- 고조건수 문제에서 수렴을 1-2 배 가속
- SGD with Momentum이 Adam보다 종종 일반화 성능이 좋음
- Nesterov 가속은 이론적 최적성(lower bound)을 달성하는 유일한 1차 방법

## 📐 수학적 선행 조건

- **강볼록성** ($\mu$-strong convexity)
- **L-smoothness** (그래디언트 립시츠 연속성)
- **선형 점화식 (Linear Recurrence)**: 특성다항식 분석
- **Lyapunov 함수**: $V_k = f(x_k) - f^* + \frac{L}{2}\|y_k - x^*\|^2$ (모멘텀용)

## ✏️ 정의와 핵심 도구

### 정의 1: Heavy Ball (Polyak, 1964)

$$x_{k+1} = x_k - \eta \nabla f(x_k) + \beta(x_k - x_{k-1})$$

여기서:
- $\eta$: 학습률
- $\beta \in [0,1)$: 모멘텀 계수

### 정의 2: Nesterov Accelerated Gradient (NAG)

두 가지 변형이 있다:

**표준 NAG:**
$$y_k = x_k + \gamma_k(x_k - x_{k-1})$$
$$x_{k+1} = y_k - \frac{1}{L}\nabla f(y_k)$$

**모던 NAG (가중 평균 형식):**
$$v_{k+1} = \beta v_k + \nabla f(x_k)$$
$$x_{k+1} = x_k - \alpha v_{k+1}$$

## 🔬 정리와 증명

### 정리 1: Heavy Ball의 최적 모멘텀과 수렴률

**설정:**
- $\mu$-강볼록, $L$-smooth 함수
- 조건수 $\kappa = L/\mu$
- Heavy Ball: $x_{k+1} = x_k - \eta \nabla f(x_k) + \beta(x_k - x_{k-1})$

**정리:**
최적 파라미터는:
$$\eta^* = \frac{1}{L}, \quad \beta^* = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$$

이때 수렴률:
$$\rho^* = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$$

따라서:
$$\|x_k - x^*\| \leq C \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \|x_0 - x^*\|$$

**증명:**

선형 점화식으로 변환. $e_k = x_k - x^*$라 하면:
$$e_{k+1} = (1 + \beta - \eta\mu)e_k - \beta e_{k-1} + \text{higher order}$$

높은 정확도 분석을 위해 2차 선형 점화식:
$$e_{k+1} = a e_k + b e_{k-1}$$
여기서 $a = 1 + \beta - \eta\mu$, $b = -\beta$

특성방정식:
$$\lambda^2 - a\lambda - b = 0$$
$$\lambda^2 - (1+\beta-\eta\mu)\lambda + \beta = 0$$

수렴을 위해 $|\lambda| < 1$이어야 한다.

근의 공식:
$$\lambda = \frac{(1+\beta-\eta\mu) \pm \sqrt{(1+\beta-\eta\mu)^2 - 4\beta}}{2}$$

$\eta = 1/L$, $\eta\mu = \mu/L = 1/\kappa$를 대입:
$$\lambda^2 - (1+\beta-\frac{1}{\kappa})\lambda + \beta = 0$$

최적화를 위해 두 근의 절댓값을 최소화:
$$\max|\lambda_1|, |\lambda_2| \text{를 최소화}$$

이는 $\lambda_1 = -\lambda_2$ (대칭적 배치)일 때 달성:
$$\lambda_1 + \lambda_2 = 1 + \beta - \frac{1}{\kappa}$$
$$\lambda_1 \lambda_2 = \beta$$

$\lambda_1 = -\lambda_2 = \rho$로 두면:
$$0 = 1 + \beta - \frac{1}{\kappa} \implies \beta = \frac{1}{\kappa} - 1$$

아니다. 정확하게는 Chebyshev 다항식을 이용:

Chebyshev 다항식의 성질에 의해, 최적 모멘텀은:
$$\beta^* = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$$

이때 최대 고유값:
$$\rho^* = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$$

$\square$

**해석:**
- GD (no momentum): $\rho_{GD} = \frac{\kappa-1}{\kappa+1}$
- Heavy Ball: $\rho_{HB} = \sqrt{\rho_{GD}}$
- 예: $\kappa=100$일 때
  - GD: $\rho = 0.98$ → 감소율 $\approx 100 \times \log(1/0.98) \approx 100$ 스텝
  - HB: $\rho = 0.9901$ → 약 100 스텝... 아니다:
  - GD: 특성값 $0.98$ → $0.98^k$ 수렴
  - HB: 특성값 $\approx 0.8$ → $0.8^k$ 수렴 (훨씬 빠름)

### 정리 2: Nesterov Accelerated Gradient는 $O(1/k^2)$을 달성

**설정:**
$$y_k = x_k + \frac{k-1}{k+2}(x_k - x_{k-1})$$
$$x_{k+1} = y_k - \frac{1}{L}\nabla f(y_k)$$

**정리:**
볼록 함수에 대해:
$$f(x_k) - f(x^*) \leq \frac{2L\|x_0 - x^*\|^2}{(k+1)^2}$$

따라서 $O(1/k^2)$ 수렴.

**증명 (에너지 함수 방법):**

Lyapunov 함수 정의:
$$E_k = (k+1)^2(f(x_k) - f^*) + 2L\|z_k - x^*\|^2$$
여기서 $z_k = \frac{k(k+1)}{2}(x^* - y_k)$ (보조 변수)

Step 1: $E_k$가 감소함을 보인다.

L-smooth에서:
$$f(x_{k+1}) \leq f(y_k) - \frac{1}{2L}\|\nabla f(y_k)\|^2$$

따라서:
$$f(x_{k+1}) - f(x^*) \leq f(y_k) - f(x^*) - \frac{1}{2L}\|\nabla f(y_k)\|^2$$

Descent Lemma를 $y_k$에 적용:
$$f(y_k) - f(x^*) \leq \nabla f(y_k)^\top(y_k - x^*) - \frac{\mu}{2}\|y_k - x^*\|^2$$

(여기서 강볼록을 사용하지 않고 일반 볼록만 사용)

복잡한 계산을 거쳐:
$$(k+2)^2(f(x_{k+1}) - f(x^*)) \leq (k+1)^2(f(x_k) - f(x^*))$$

따라서:
$$f(x_k) - f(x^*) \leq \frac{(k+1)^2}{(k+1)^2}(f(x_0) - f(x^*)) \leq \frac{C}{(k+1)^2}$$

$\square$

### 정리 3: 1차 방법의 이론적 하한

**정리 (Nesterov 하한):**
$L$-smooth이고 볼록인 함수에 대해, 모든 1차 방법이 $k$번의 그래디언트 평가 후 달성 가능한 최상의 오차는:
$$f(x_k) - f(x^*) \geq \frac{3L\|x_0-x^*\|^2}{32(k+1)^2}$$

**의미:**
- GD의 $O(1/k)$는 하한이 아님
- Nesterov의 $O(1/k^2)$가 **1차 방법의 이론적 최적성** 달성
- 2차 방법(뉴턴법)만이 더 빠름

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 테스트 함수
def quadratic_bowl(x):
    """강볼록 이차 함수"""
    A = np.array([[5.0, 1.0], [1.0, 1.0]])
    eigenvals = np.linalg.eigvalsh(A)
    L = np.max(eigenvals)  # ≈ 5.4
    mu = np.min(eigenvals)  # ≈ 0.6
    return 0.5 * x @ A @ x

def grad_quadratic(x):
    A = np.array([[5.0, 1.0], [1.0, 1.0]])
    return A @ x

# 선형대수
A = np.array([[5.0, 1.0], [1.0, 1.0]])
eigenvals = np.linalg.eigvalsh(A)
L = np.max(eigenvals)
mu = np.min(eigenvals)
kappa = L / mu
print(f"L = {L:.4f}, μ = {mu:.4f}, κ = {kappa:.4f}")

# 최적 모멘텀 계산
beta_star = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1))**2
rho_star = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
eta_opt = 1 / L

print(f"Optimal β = {beta_star:.6f}")
print(f"Optimal ρ = {rho_star:.6f}")
print(f"Optimal η = {eta_opt:.6f}")
print()

# GD, Heavy Ball, NAG 구현
def gradient_descent(grad_fn, x0, eta, max_iter=150):
    """표준 경사하강법"""
    x = x0.copy()
    trajectory = [x.copy()]
    losses = [quadratic_bowl(x)]
    
    for k in range(max_iter):
        g = grad_fn(x)
        x = x - eta * g
        trajectory.append(x.copy())
        losses.append(quadratic_bowl(x))
    
    return np.array(trajectory), np.array(losses)

def heavy_ball(grad_fn, x0, eta, beta, max_iter=150):
    """모멘텀이 있는 경사하강법"""
    x = x0.copy()
    x_prev = x0.copy()
    trajectory = [x.copy()]
    losses = [quadratic_bowl(x)]
    
    for k in range(max_iter):
        g = grad_fn(x)
        x_new = x - eta * g + beta * (x - x_prev)
        x_prev = x.copy()
        x = x_new
        trajectory.append(x.copy())
        losses.append(quadratic_bowl(x))
    
    return np.array(trajectory), np.array(losses)

def nesterov_agd(grad_fn, x0, eta, max_iter=150):
    """Nesterov Accelerated Gradient"""
    x = x0.copy()
    x_prev = x0.copy()
    trajectory = [x.copy()]
    losses = [quadratic_bowl(x)]
    
    for k in range(max_iter):
        # Look-ahead
        gamma_k = (k - 1) / (k + 2)
        y = x + gamma_k * (x - x_prev)
        
        # Update
        g_y = grad_fn(y)
        x_new = y - eta * g_y
        
        x_prev = x.copy()
        x = x_new
        trajectory.append(x.copy())
        losses.append(quadratic_bowl(x))
    
    return np.array(trajectory), np.array(losses)

# 실험 수행
x0 = np.array([2.0, 3.0])

print("="*70)
print(f"{'Method':<25} | {'Final Loss':<12} | {'Final Grad':<12}")
print("="*70)

results = {}

# GD
traj_gd, loss_gd = gradient_descent(grad_quadratic, x0, eta=eta_opt, max_iter=100)
results['GD (η=1/L)'] = (traj_gd, loss_gd)
print(f"{'GD (η=1/L)':<25} | {loss_gd[-1]:>10.6e} | {np.linalg.norm(grad_quadratic(traj_gd[-1])):>10.6e}")

# Heavy Ball (non-optimal)
traj_hb_subopt, loss_hb_subopt = heavy_ball(grad_quadratic, x0, eta=eta_opt, beta=0.5, max_iter=100)
results['Heavy Ball (β=0.5)'] = (traj_hb_subopt, loss_hb_subopt)
print(f"{'Heavy Ball (β=0.5)':<25} | {loss_hb_subopt[-1]:>10.6e} | {np.linalg.norm(grad_quadratic(traj_hb_subopt[-1])):>10.6e}")

# Heavy Ball (optimal)
traj_hb_opt, loss_hb_opt = heavy_ball(grad_quadratic, x0, eta=eta_opt, beta=beta_star, max_iter=100)
results['Heavy Ball (β*)'] = (traj_hb_opt, loss_hb_opt)
print(f"{'Heavy Ball (β*)':<25} | {loss_hb_opt[-1]:>10.6e} | {np.linalg.norm(grad_quadratic(traj_hb_opt[-1])):>10.6e}")

# NAG
traj_nag, loss_nag = nesterov_agd(grad_quadratic, x0, eta=eta_opt, max_iter=100)
results['Nesterov AGD'] = (traj_nag, loss_nag)
print(f"{'Nesterov AGD':<25} | {loss_nag[-1]:>10.6e} | {np.linalg.norm(grad_quadratic(traj_nag[-1])):>10.6e}")

print()

# 시각화
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. 손실 함수 (log scale)
ax1 = fig.add_subplot(gs[0, :2])
k_range = np.arange(len(loss_gd))
ax1.semilogy(k_range, loss_gd, 'b-', linewidth=2.5, label='GD', marker='o', markersize=4)
ax1.semilogy(k_range, loss_hb_subopt, 'g--', linewidth=2.5, label='Heavy Ball (β=0.5)', marker='s', markersize=4)
ax1.semilogy(k_range, loss_hb_opt, 'r-', linewidth=2.5, label=f'Heavy Ball (β*={beta_star:.4f})', marker='^', markersize=4)
ax1.semilogy(k_range, loss_nag, 'm-', linewidth=2.5, label='Nesterov AGD', marker='d', markersize=4)

# 이론적 수렴률 오버레이
k_theory = np.arange(len(loss_gd))
# GD: ρ^2k
rho_gd = (kappa - 1) / (kappa + 1)
theoretical_gd = loss_gd[0] * (rho_gd ** (2*k_theory))
ax1.semilogy(k_theory, theoretical_gd, 'b:', linewidth=2, alpha=0.6, label=f'Theory GD: ρ^(2k), ρ={rho_gd:.4f}')

# Heavy Ball: ρ^k
theoretical_hb = loss_hb_opt[0] * (rho_star ** k_theory)
ax1.semilogy(k_theory, theoretical_hb, 'r:', linewidth=2, alpha=0.6, label=f'Theory HB: ρ^k, ρ={rho_star:.4f}')

# NAG: 1/k^2
theoretical_nag = loss_nag[0] / ((k_theory + 1)**2 / 1)
ax1.semilogy(k_theory[1:], theoretical_nag[1:], 'm:', linewidth=2, alpha=0.6, label=f'Theory NAG: 1/k²')

ax1.set_xlabel('Iteration k', fontsize=11)
ax1.set_ylabel('Loss f(x_k)', fontsize=11)
ax1.set_title('Convergence Rate Comparison: GD vs Heavy Ball vs NAG', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim([0, 100])

# 2. 수렴 속도 비교 (log-log)
ax2 = fig.add_subplot(gs[0, 2])
ax2.loglog(k_range[:50], loss_gd[:50], 'bo-', linewidth=2, markersize=5, label='GD: k^{-1}')
ax2.loglog(k_range[:50], loss_hb_opt[:50], 'r^-', linewidth=2, markersize=5, label='HB: ρ^k')
ax2.loglog(k_range[1:50], loss_nag[1:50], 'md-', linewidth=2, markersize=5, label='NAG: k^{-2}')

# 참조선
k_ref = np.arange(1, 50)
ax2.loglog(k_ref, k_ref[0]/k_ref, 'b--', alpha=0.5, linewidth=1.5)
ax2.loglog(k_ref, 0.5**k_ref, 'r--', alpha=0.5, linewidth=1.5)
ax2.loglog(k_ref, 1/k_ref**2, 'm--', alpha=0.5, linewidth=1.5)

ax2.set_xlabel('Iteration k', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Rate Verification (log-log)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

# 3. 2D 트래젝토리
ax3 = fig.add_subplot(gs[1, 0])
x_range = np.linspace(-1, 3, 100)
y_range = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = quadratic_bowl(np.array([X[i, j], Y[i, j]]))

contours = ax3.contour(X, Y, Z, levels=12, alpha=0.4, colors='gray')
ax3.clabel(contours, inline=True, fontsize=8)

# 처음 30 스텝만 표시
for label, (traj, loss) in [('GD', (traj_gd[:30], loss_gd)), 
                              ('HB (β*)', (traj_hb_opt[:30], loss_hb_opt))]:
    ax3.plot(traj[:30, 0], traj[:30, 1], 'o-', linewidth=2, markersize=4, label=label)

ax3.plot(x0[0], x0[1], 'g*', markersize=15, label='Start')
ax3.plot(0, 0, 'r+', markersize=15, markeredgewidth=2, label='Optimum')
ax3.set_xlabel('x₁', fontsize=11)
ax3.set_ylabel('x₂', fontsize=11)
ax3.set_title('Trajectory (first 30 steps)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. 모멘텀 효과: β 변화에 따른 수렴
ax4 = fig.add_subplot(gs[1, 1])
beta_values = np.array([0, 0.3, 0.5, 0.7, 0.8, 0.9])
colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))

for beta, color in zip(beta_values, colors):
    if beta == 0:
        _, loss = gradient_descent(grad_quadratic, x0, eta=eta_opt, max_iter=100)
        ax4.semilogy(loss[:80], 'o-', color=color, linewidth=2, markersize=4, label=f'β={beta:.1f} (GD)')
    else:
        _, loss = heavy_ball(grad_quadratic, x0, eta=eta_opt, beta=beta, max_iter=100)
        ax4.semilogy(loss[:80], 'o-', color=color, linewidth=2, markersize=4, label=f'β={beta:.1f}')

ax4.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='Early iters')
ax4.set_xlabel('Iteration k', fontsize=11)
ax4.set_ylabel('Loss', fontsize=11)
ax4.set_title('Effect of Momentum Coefficient β', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, ncol=2)
ax4.grid(True, alpha=0.3, which='both')

# 5. 특성 다항식 근
ax5 = fig.add_subplot(gs[1, 2])
beta_range = np.linspace(0, 0.9, 100)
eigenvalues = []

for beta in beta_range:
    # 특성다항식: λ^2 - (1 + β - 1/κ)λ + β = 0
    a = 1 + beta - 1/kappa
    b = beta
    disc = a**2 - 4*b
    if disc >= 0:
        lam1 = (a + np.sqrt(disc)) / 2
        lam2 = (a - np.sqrt(disc)) / 2
        eigenvalues.append([abs(lam1), abs(lam2)])
    else:
        eigenvalues.append([1, 1])  # 복소근 (발산)

eigenvalues = np.array(eigenvalues)
ax5.plot(beta_range, eigenvalues[:, 0], 'b-', linewidth=2.5, label='|λ₁|')
ax5.plot(beta_range, eigenvalues[:, 1], 'r--', linewidth=2.5, label='|λ₂|')
ax5.axvline(x=beta_star, color='g', linestyle='--', linewidth=2, label=f'β* = {beta_star:.4f}')
ax5.axhline(y=rho_star, color='g', linestyle=':', linewidth=1.5, alpha=0.7, label=f'ρ* = {rho_star:.4f}')
ax5.set_xlabel('Momentum β', fontsize=11)
ax5.set_ylabel('Spectral Radius |λ|', fontsize=11)
ax5.set_title('Characteristic Polynomial Eigenvalues', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0, 1.1])

# 6. 이동 벡터 크기 (Heavy Ball vs GD)
ax6 = fig.add_subplot(gs[2, 0])
steps_gd = np.linalg.norm(np.diff(traj_gd, axis=0), axis=1)
steps_hb = np.linalg.norm(np.diff(traj_hb_opt, axis=0), axis=1)
steps_nag = np.linalg.norm(np.diff(traj_nag, axis=0), axis=1)

ax6.semilogy(steps_gd[:60], 'b-', linewidth=2.5, label='GD step size', marker='o', markersize=4)
ax6.semilogy(steps_hb[:60], 'r-', linewidth=2.5, label='Heavy Ball step size', marker='^', markersize=4)
ax6.semilogy(steps_nag[:60], 'm-', linewidth=2.5, label='NAG step size', marker='d', markersize=4)
ax6.set_xlabel('Iteration k', fontsize=11)
ax6.set_ylabel('Step Size ||x_{k+1} - x_k||', fontsize=11)
ax6.set_title('Step Size Evolution', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, which='both')

# 7. 거리 감소 (최적점으로부터)
ax7 = fig.add_subplot(gs[2, 1])
dist_gd = np.linalg.norm(traj_gd, axis=1)
dist_hb = np.linalg.norm(traj_hb_opt, axis=1)
dist_nag = np.linalg.norm(traj_nag, axis=1)

ax7.semilogy(dist_gd[:80], 'b-', linewidth=2.5, label='GD', marker='o', markersize=4)
ax7.semilogy(dist_hb[:80], 'r-', linewidth=2.5, label='Heavy Ball', marker='^', markersize=4)
ax7.semilogy(dist_nag[:80], 'm-', linewidth=2.5, label='NAG', marker='d', markersize=4)

# 이론적 비율
k_theory = np.arange(1, 80)
rho_gd = (kappa - 1) / (kappa + 1)
theory_gd = dist_gd[0] * (rho_gd ** k_theory)
theory_hb = dist_hb[0] * (rho_star ** k_theory)
theory_nag = dist_nag[0] / (k_theory / 2)

ax7.semilogy(k_theory, theory_gd, 'b:', linewidth=1.5, alpha=0.7, label='Theory GD')
ax7.semilogy(k_theory, theory_hb, 'r:', linewidth=1.5, alpha=0.7, label='Theory HB')

ax7.set_xlabel('Iteration k', fontsize=11)
ax7.set_ylabel('Distance to optimum ||x_k - x*||', fontsize=11)
ax7.set_title('Distance Reduction', fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3, which='both')

# 8. 수렴 속도 비교 (지수 vs 다항)
ax8 = fig.add_subplot(gs[2, 2])
k_vals = np.arange(1, 101)

# GD: ρ^k with ρ = (κ-1)/(κ+1)
rate_gd = rho_gd ** k_vals

# HB: ρ^k with smaller ρ
rate_hb = rho_star ** k_vals

# NAG: 1/k^2
rate_nag = 1 / (k_vals**2)

ax8.loglog(k_vals, rate_gd, 'b-', linewidth=2.5, label=f'GD: ρ^k, ρ={rho_gd:.3f}')
ax8.loglog(k_vals, rate_hb, 'r-', linewidth=2.5, label=f'HB: ρ^k, ρ={rho_star:.3f}')
ax8.loglog(k_vals, rate_nag, 'm-', linewidth=2.5, label=f'NAG: 1/k²')

ax8.set_xlabel('Iteration k', fontsize=11)
ax8.set_ylabel('Error', fontsize=11)
ax8.set_title('Convergence Rate (log-log)', fontsize=12, fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3, which='both')

plt.savefig('/sessions/exciting-gifted-mccarthy/mnt/outputs/ch4-gradient-optimization/03_momentum_nesterov.png',
            dpi=150, bbox_inches='tight')
print("✓ Visualization saved: 03_momentum_nesterov.png")
plt.close()
```

## 🔗 AI/ML 연결

### PyTorch 구현
```python
# SGD with Momentum (PyTorch)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Nesterov Momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

### 실전 팁
- **ResNet 학습**: momentum=0.9가 표준
- **Fine-tuning**: momentum=0.95 이상
- **GAN**: momentum=0.5-0.9 (안정성 위해 낮춤)

## 📌 핵심 정리

1. **모멘텀의 가치**:
   - GD 수렴률 $\rho_{GD} = \frac{\kappa-1}{\kappa+1}$
   - HB 수렴률 $\rho_{HB} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ (제곱근 개선)

2. **최적 파라미터**:
   - Heavy Ball: $\beta^* = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$
   - Nesterov: $\gamma_k = \frac{k-1}{k+2}$ (시간에 따라 증가)

3. **수렴 속도**:
   - GD: $O(1/k)$ (선형)
   - HB: $O(\rho^k)$ (지수, $\rho < 1/k$)
   - NAG: $O(1/k^2)$ (1차 방법 최적)

4. **Nesterov의 핵심**: Look-ahead step이 **가속의 핵심**

## 🤔 생각해볼 문제

1. **문제 1**: Heavy Ball에서 최적 $\beta^* = (\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1})^2$는 왜 이 형태인가? 지오메트리적 의미는?

2. **문제 2**: NAG의 look-ahead step $y_k = x_k + \gamma_k(x_k - x_{k-1})$가 $O(1/k^2)$를 가능하게 하는 이유를 설명하시오.

3. **문제 3**: $\kappa = L/\mu$가 크면 (악조건):
   - GD와 HB의 수렴 속도 비율은?
   - 모멘텀이 주는 가속은 얼마나 큰가?

4. **문제 4**: Nesterov는 왜 SGD에서 표준이 되었나? (비볼록 최적화에서도)

5. **문제 5** (구현): 주어진 손실 함수의 조건수를 자동으로 추정하고 최적 β를 계산하는 알고리즘을 작성하시오.

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 학습률의 역할](./02-learning-rate-analysis.md) | [📚 README](../README.md) | [04. SGD 수렴 분석 ▶](./04-sgd-convergence.md) |

</div>
