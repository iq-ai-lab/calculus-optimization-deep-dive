# 06. 뉴턴법과 준뉴턴법

## 🎯 핵심 질문
- 왜 뉴턴법은 2차 수렴하지만 실제로는 거의 안 쓰이는가?
- BFGS와 L-BFGS는 헤시안을 어떻게 근사하는가?
- 준뉴턴법의 "Secant 조건"이 왜 중요한가?
- 딥러닝에서 뉴턴법이 작동 안 하는 이유는?

## 🔍 왜 이 개념이 AI에서 중요한가

**2차 방법의 매력과 현실**:
- 뉴턴법: 이차 수렴 (매우 빠름) - but 비용이 너무 높음
- BFGS: 준뉴턴 근사 - 구조화된 문제에서 효과적
- 딥러닝: Gradient 기반 최적화만 가능 (Hessian 계산 불가)
- 최근 트렌드: Hessian-vector product를 근사하는 2차 방법들이 등장

## 📐 수학적 선행 조건

- **헤시안 행렬**: $H_f(x) = \nabla^2 f(x)$
- **Positive definite 행렬**: $v^\top H v > 0$ for all $v \neq 0$
- **특이값 분해 (SVD)**: 행렬의 역행렬과 수치 안정성
- **Secant 방정식**: $B(x_{k+1}-x_k) = \nabla f(x_{k+1}) - \nabla f(x_k)$

## ✏️ 정의와 핵심 도구

### 정의 1: 뉴턴 방법

$f(x)$를 2차 테일러 근사로 전개:
$$m_k(s) = f(x_k) + \nabla f(x_k)^\top s + \frac{1}{2}s^\top H_f(x_k) s$$

이를 최소화:
$$s_k^* = -H_f(x_k)^{-1}\nabla f(x_k)$$

따라서 **뉴턴 스텝**:
$$x_{k+1} = x_k - H_f(x_k)^{-1}\nabla f(x_k)$$

### 정의 2: BFGS 업데이트

헤시안을 근사하는 행렬 $B_k$ 유지:

**Secant 조건**:
$$B_{k+1}(x_{k+1} - x_k) = \nabla f(x_{k+1}) - \nabla f(x_k)$$

또는 간단히:
$$B_{k+1} s_k = y_k$$
여기서 $s_k = x_{k+1} - x_k$, $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$

**BFGS 공식** (대칭 Rank-2 업데이트):
$$B_{k+1} = B_k - \frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \frac{y_k y_k^\top}{y_k^\top s_k}$$

### 정의 3: L-BFGS (Limited Memory)

최근 $m$개의 $(s_k, y_k)$ 쌍만 저장하고, 역행렬 $H_{k+1}^{-1}$를 재귀적으로 계산:

**2-loop recursion**:
```
function H_times_v(v):
    q = v
    for i = k-m to k:
        α_i = ρ_i s_i^T q
        q = q - α_i y_i
    r = H_0^{-1} q  // 스칼라 배수 (예: I * γ)
    for i = k-m to k:
        β_i = ρ_i y_i^T r
        r = r + (α_i - β_i) s_i
    return r
```

여기서 $\rho_i = 1/(y_i^\top s_i)$

## 🔬 정리와 증명

### 정리 1: 뉴턴법의 이차 수렴

**설정:**
- $\mu$-강볼록, $L$-smooth 함수 ($\mu > 0$)
- $H_f$가 Lipschitz 연속 (bounded 2nd derivative)
- 초기점이 최적점 근처: $\|x_0 - x^*\| < \epsilon_0$

**정리 (Newton's Quadratic Convergence):**
충분히 작은 $\|x_0 - x^*\|$에 대해:
$$\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^2$$

따라서:
$$\|x_k - x^*\| \leq \left(\frac{1}{2C}\right)^{2^k - 1} \|x_0 - x^*\|$$

(기하급수적 수렴, 반복마다 자릿수 배증)

**증명:**

2차 테일러 전개:
$$f(x^*) = f(x_k) + \nabla f(x_k)^\top(x^* - x_k) + \int_0^1 (1-t) (x^* - x_k)^\top H_f(x_k + t(x^*-x_k)) (x^* - x_k) dt$$

$\nabla f(x^*) = 0$이므로:
$$0 = f(x_k) + \nabla f(x_k)^\top(x^* - x_k) + \text{2nd order term}$$

뉴턴 스텝 후:
$$x_{k+1} = x_k - H_f(x_k)^{-1}\nabla f(x_k)$$

$$x_{k+1} - x^* = (x_k - x^*) - H_f(x_k)^{-1}\nabla f(x_k)$$

강볼록성과 Taylor 전개에서:
$$H_f(x_k)^{-1}\nabla f(x_k) = H_f(x_k)^{-1}[H_f(x^*)(x_k-x^*) + O(\|x_k-x^*\|^2)]$$

따라서:
$$x_{k+1} - x^* = H_f(x_k)^{-1}[H_f(x^*) - H_f(x_k)](x_k-x^*) + O(\|x_k-x^*\|^2)$$

$H_f$가 Lipschitz이면:
$$\|H_f(x^*) - H_f(x_k)\| \leq M\|x_k - x^*\|$$

따라서:
$$\|x_{k+1} - x^*\| \leq \|H_f(x_k)^{-1}\| \cdot M\|x_k-x^*\|^2$$

강볼록이면 $\|H_f(x_k)^{-1}\| \leq 1/\mu$이므로:
$$\|x_{k+1} - x^*\| \leq \frac{M}{\mu}\|x_k - x^*\|^2$$

이차 수렴. $\square$

### 정리 2: BFGS의 초선형 수렴

**정리:**
적절한 line search와 함께 사용하면:
$$\lim_{k \to \infty} \frac{\|x_{k+1} - x^*\|}{\|x_k - x^*\|} = 0$$

강볼록이면 **초선형** (superlinear):
$$\|x_k - x^*\| = o(\rho^k) \text{ for any } \rho < 1$$

"Dennis-Moré" 조건을 만족하면 **superlinear convergence**도 가능.

**증명 스케치:**

Secant 조건 $B_{k+1}s_k = y_k$에서 BFGS 행렬 $B_k$는 진정한 헤시안 방향을 추적함.

$E_k = B_k - H_f(x_k)$ (근사 오차)라 하면:

BFGS 업데이트는:
$$E_{k+1} = \text{(특정 형식 - 복잡)} = O(\|x_k - x^*\|^2)$$

이는 $B_k \to H_f(x^*)$를 의미하고, 따라서 Newton 방향으로 수렴.

$\square$

### 정리 3: BFGS의 양정치성 보존

**정리:**
- $B_0 \succ 0$ (양정치)
- $y_k^\top s_k > 0$ (curvature 조건)

이면 모든 $B_k \succ 0$ (양정치 유지)

**증명:**

BFGS 공식:
$$B_{k+1} = B_k - \frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \frac{y_k y_k^\top}{y_k^\top s_k}$$

우변의 각 항을 분석:
- 첫 번째 항: Rank-1 감소 (negative semidefinite)
- 두 번째 항: Rank-1 증가 (positive semidefinite)

Sherman-Morrison 정리의 inertia 성질에 의해, $y_k^\top s_k > 0$이면 전체 양정치성 유지.

$\square$

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 테스트 함수들
def quadratic(x):
    """강볼록 이차 함수"""
    A = np.array([[10.0, 1.0], [1.0, 2.0]])
    return 0.5 * x @ A @ x

def quadratic_grad(x):
    A = np.array([[10.0, 1.0], [1.0, 2.0]])
    return A @ x

def quadratic_hess(x):
    return np.array([[10.0, 1.0], [1.0, 2.0]])

def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def rosenbrock_grad(x):
    dfdx0 = 400*x[0]*(x[0]**2 - x[1]) + 2*(x[0]-1)
    dfdx1 = 200*(x[1] - x[0]**2)
    return np.array([dfdx0, dfdx1])

def rosenbrock_hess(x):
    dfdx0x0 = 400*(3*x[0]**2 - x[1]) + 2
    dfdx0x1 = -400*x[0]
    dfdx1x0 = -400*x[0]
    dfdx1x1 = 200
    return np.array([[dfdx0x0, dfdx0x1], [dfdx1x0, dfdx1x1]])

# 옵티마이저 구현

def newton_method(grad_fn, hess_fn, x0, max_iter=100, tol=1e-8):
    """Newton's method"""
    x = x0.copy()
    trajectory = [x.copy()]
    losses = []
    
    for k in range(max_iter):
        g = grad_fn(x)
        H = hess_fn(x)
        
        try:
            # H^{-1} g 계산
            newton_step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        
        x = x - newton_step
        trajectory.append(x.copy())
        loss = rosenbrock(x)
        losses.append(loss)
        
        if np.linalg.norm(g) < tol:
            break
    
    return np.array(trajectory), np.array(losses)

def bfgs(grad_fn, x0, max_iter=500, tol=1e-8, c1=1e-4, c2=0.9):
    """BFGS with Backtracking Line Search"""
    x = x0.copy()
    n = len(x0)
    B = np.eye(n)  # 초기 근사: 단위행렬
    
    trajectory = [x.copy()]
    losses = [rosenbrock(x)]
    
    for k in range(max_iter):
        g = grad_fn(x)
        
        if np.linalg.norm(g) < tol:
            break
        
        # BFGS 방향 (대략 Newton 방향)
        try:
            p = -np.linalg.solve(B, g)  # B는 H 근사
        except np.linalg.LinAlgError:
            p = -g  # Fallback to GD
        
        # Backtracking line search
        alpha = 1.0
        loss_old = rosenbrock(x)
        
        for _ in range(20):
            x_new = x + alpha * p
            loss_new = rosenbrock(x_new)
            
            if loss_new <= loss_old + c1 * alpha * g @ p:
                break
            alpha *= 0.5
        
        s = alpha * p  # s_k
        x_new = x + s
        g_new = grad_fn(x_new)
        y = g_new - g  # y_k
        
        # BFGS 업데이트
        if y @ s > 0:  # Secant 조건 확인
            # B_{k+1} = B_k - (B_k s s^T B_k)/(s^T B_k s) + (y y^T)/(y^T s)
            Bs = B @ s
            denom1 = s @ Bs
            denom2 = y @ s
            
            if abs(denom1) > 1e-10 and abs(denom2) > 1e-10:
                B = B - np.outer(Bs, Bs) / denom1 + np.outer(y, y) / denom2
        
        x = x_new
        trajectory.append(x.copy())
        losses.append(rosenbrock(x))
    
    return np.array(trajectory), np.array(losses)

def lbfgs(grad_fn, x0, max_iter=500, m=10, tol=1e-8):
    """L-BFGS"""
    x = x0.copy()
    n = len(x0)
    
    trajectory = [x.copy()]
    losses = [rosenbrock(x)]
    
    s_list = []  # 최근 s_k들
    y_list = []  # 최근 y_k들
    
    for k in range(max_iter):
        g = grad_fn(x)
        
        if np.linalg.norm(g) < tol:
            break
        
        # 2-loop recursion으로 H^{-1}g 계산
        q = g.copy()
        
        # Forward loop
        alphas = []
        for i in range(len(s_list)-1, -1, -1):
            s_i, y_i = s_list[i], y_list[i]
            rho_i = 1.0 / (y_i @ s_i)
            alpha_i = rho_i * (s_i @ q)
            alphas.append(alpha_i)
            q = q - alpha_i * y_i
        
        # 초기 H_0 (스칼라 배수)
        if len(s_list) > 0:
            gamma = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
        else:
            gamma = 1.0
        r = gamma * q
        
        # Backward loop
        alphas.reverse()
        for i in range(len(s_list)):
            s_i, y_i = s_list[i], y_list[i]
            rho_i = 1.0 / (y_i @ s_i)
            beta_i = rho_i * (y_i @ r)
            r = r + s_i * (alphas[i] - beta_i)
        
        # 스텝
        alpha = 0.1  # 고정 스텝 (간단함)
        x_new = x - alpha * r
        g_new = grad_fn(x_new)
        
        s = x_new - x
        y = g_new - g
        
        # 리스트 업데이트 (최대 m개 유지)
        if y @ s > 0:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = x_new
        trajectory.append(x.copy())
        losses.append(rosenbrock(x))
    
    return np.array(trajectory), np.array(losses)

def gradient_descent(grad_fn, x0, eta, max_iter=1000):
    """기본 경사하강법"""
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

# 초기점
x0 = np.array([-1.5, 2.5])

print("="*70)
print("SECOND-ORDER OPTIMIZATION METHODS")
print("="*70)

# 실험 (Rosenbrock)
print("\nRosenbrock Function (Non-convex)")

# Newton
print("\nNewton's Method (quadratic function - for comparison)")
traj_newton, loss_newton = newton_method(quadratic_grad, quadratic_hess, x0, max_iter=20)
print(f"  Iterations: {len(loss_newton)}, Final loss: {loss_newton[-1]:.6e}")

# BFGS
print("\nBFGS")
traj_bfgs, loss_bfgs = bfgs(rosenbrock_grad, x0, max_iter=500)
print(f"  Iterations: {len(loss_bfgs)}, Final loss: {loss_bfgs[-1]:.6e}")

# L-BFGS (m=5)
print("\nL-BFGS (m=5)")
traj_lbfgs5, loss_lbfgs5 = lbfgs(rosenbrock_grad, x0, max_iter=500, m=5)
print(f"  Iterations: {len(loss_lbfgs5)}, Final loss: {loss_lbfgs5[-1]:.6e}")

# L-BFGS (m=10)
print("\nL-BFGS (m=10)")
traj_lbfgs10, loss_lbfgs10 = lbfgs(rosenbrock_grad, x0, max_iter=500, m=10)
print(f"  Iterations: {len(loss_lbfgs10)}, Final loss: {loss_lbfgs10[-1]:.6e}")

# GD (for comparison)
print("\nGradient Descent (η=0.001)")
traj_gd, loss_gd = gradient_descent(rosenbrock_grad, x0, eta=0.001, max_iter=3000)
print(f"  Iterations: {len(loss_gd)}, Final loss: {loss_gd[-1]:.6e}")

print()

# 시각화
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. 손실 곡선 비교
ax1 = fig.add_subplot(gs[0, :2])
ax1.semilogy(loss_newton, 'b-', linewidth=2.5, label='Newton (quadratic)', marker='o', markersize=5)
ax1.semilogy(loss_bfgs, 'r-', linewidth=2.5, label='BFGS', marker='^', markersize=4)
ax1.semilogy(loss_lbfgs5, 'g--', linewidth=2.5, label='L-BFGS (m=5)', marker='s', markersize=4)
ax1.semilogy(loss_lbfgs10, 'purple', linewidth=2.5, label='L-BFGS (m=10)', marker='d', markersize=4)
ax1.semilogy(loss_gd[:500], 'orange', linewidth=2, alpha=0.7, label='GD', marker='v', markersize=3, markevery=50)

ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Loss (log)', fontsize=11)
ax1.set_title('Convergence Comparison: Second-Order Methods', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim([0, 150])

# 2. 초기 수렴 (zoom)
ax2 = fig.add_subplot(gs[0, 2])
ax2.semilogy(loss_bfgs[:50], 'r-', linewidth=2.5, label='BFGS', marker='^', markersize=5)
ax2.semilogy(loss_lbfgs10[:50], 'purple', linewidth=2.5, label='L-BFGS (m=10)', marker='d', markersize=5)
ax2.semilogy(loss_gd[:100], 'orange', linewidth=2, alpha=0.7, label='GD', marker='v', markersize=4, markevery=5)

ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Early Convergence (zoom)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# 3. 2D 트래젝토리
ax3 = fig.add_subplot(gs[1, 0])

x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = 100 * (Y - X**2)**2 + (1 - X)**2

contours = ax3.contour(X, Y, Z, levels=20, alpha=0.3, colors='gray')
ax3.clabel(contours, inline=True, fontsize=7)

colors = {'Newton': 'blue', 'BFGS': 'red', 'L-BFGS (m=10)': 'purple', 'GD': 'orange'}

# Newton (limited to first few steps due to convergence)
ax3.plot(traj_newton[:min(8, len(traj_newton)), 0], traj_newton[:min(8, len(traj_newton)), 1], 
         'o-', color='blue', label='Newton', linewidth=2, markersize=5, alpha=0.8)

ax3.plot(traj_bfgs[:min(50, len(traj_bfgs)), 0], traj_bfgs[:min(50, len(traj_bfgs)), 1], 
         'o-', color='red', label='BFGS', linewidth=2, markersize=4, alpha=0.7)

ax3.plot(traj_lbfgs10[:min(80, len(traj_lbfgs10)), 0], traj_lbfgs10[:min(80, len(traj_lbfgs10)), 1], 
         '^-', color='purple', label='L-BFGS (m=10)', linewidth=2, markersize=4, alpha=0.6)

ax3.plot(traj_gd[:min(200, len(traj_gd)), 0], traj_gd[:min(200, len(traj_gd)), 1], 
         'v-', color='orange', label='GD', linewidth=1.5, markersize=3, alpha=0.5, markevery=20)

ax3.plot(x0[0], x0[1], 'g*', markersize=15, label='Start')
ax3.plot(1, 1, 'r+', markersize=15, markeredgewidth=2, label='Optimum')
ax3.set_xlabel('x₁', fontsize=11)
ax3.set_ylabel('x₂', fontsize=11)
ax3.set_title('2D Trajectories', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. Secant 조건 확인
ax4 = fig.add_subplot(gs[1, 1])

# 작은 이차 함수에서 Secant 조건 검증
x = x0.copy()
B = np.eye(2)
secant_errors = []

for k in range(50):
    g = quadratic_grad(x)
    p = -np.linalg.solve(B, g)
    alpha = 0.1
    s = alpha * p
    x_new = x + s
    g_new = quadratic_grad(x_new)
    y = g_new - g
    
    # Secant 조건 오차: ||B_{k+1}s_k - y_k||
    Bs = B @ s
    if y @ s > 0:
        denom1 = s @ Bs
        denom2 = y @ s
        B = B - np.outer(Bs, Bs) / denom1 + np.outer(y, y) / denom2
    
    Bs_new = B @ s
    secant_error = np.linalg.norm(Bs_new - y)
    secant_errors.append(secant_error)
    
    x = x_new

ax4.semilogy(secant_errors, 'b-', linewidth=2.5, marker='o', markersize=5)
ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('||Bs - y|| (Secant Condition Error)', fontsize=11)
ax4.set_title('BFGS: Secant Condition', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, which='both')

# 5. 메모리 비용 비교
ax5 = fig.add_subplot(gs[1, 2])

m_values = np.arange(1, 20)
memory_lbfgs = m_values * 2  # 각 (s, y) 쌍이 2개 벡터
memory_bfgs = np.ones_like(m_values) * (20**2)  # 전체 n×n 행렬 (n=20 가정)

ax5.plot(m_values, memory_lbfgs, 'b-', linewidth=2.5, marker='o', markersize=8, label='L-BFGS (vectors)')
ax5.axhline(y=400, color='r', linestyle='--', linewidth=2, alpha=0.6, label='BFGS (full matrix)')
ax5.fill_between(m_values, memory_lbfgs, 400, alpha=0.2, color='blue')
ax5.set_xlabel('Memory Parameter m', fontsize=11)
ax5.set_ylabel('Memory Usage (units)', fontsize=11)
ax5.set_title('L-BFGS vs BFGS: Memory', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. 수렴 속도 분석 (log-log)
ax6 = fig.add_subplot(gs[2, 0])

k_range = np.arange(1, min(100, len(loss_bfgs)))
ax6.loglog(k_range, loss_bfgs[1:100], 'r-', linewidth=2.5, label='BFGS', marker='^', markersize=5, markevery=10)
ax6.loglog(k_range, loss_gd[1:100], 'orange', linewidth=2, alpha=0.7, label='GD', marker='v', markersize=4, markevery=10)

# 참조선
k_ref = np.arange(5, 100)
ax6.loglog(k_ref, loss_bfgs[0] / k_ref**2, 'r--', linewidth=1.5, alpha=0.5, label='1/k² (superlinear)')
ax6.loglog(k_ref, loss_gd[0] / k_ref, 'orange', linewidth=1.5, alpha=0.5, linestyle='--', label='1/k (GD)')

ax6.set_xlabel('Iteration k (log)', fontsize=11)
ax6.set_ylabel('Loss (log)', fontsize=11)
ax6.set_title('Convergence Rate (log-log)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, which='both')

# 7. 스텝 크기 진화
ax7 = fig.add_subplot(gs[2, 1])

step_sizes_bfgs = np.linalg.norm(np.diff(traj_bfgs, axis=0), axis=1)
step_sizes_lbfgs = np.linalg.norm(np.diff(traj_lbfgs10, axis=0), axis=1)
step_sizes_gd = np.linalg.norm(np.diff(traj_gd, axis=0), axis=1)

ax7.semilogy(step_sizes_bfgs[:100], 'r-', linewidth=2.5, label='BFGS', marker='^', markersize=4, markevery=10)
ax7.semilogy(step_sizes_lbfgs[:100], 'purple', linewidth=2.5, label='L-BFGS', marker='d', markersize=4, markevery=10)
ax7.semilogy(step_sizes_gd[:200], 'orange', linewidth=2, alpha=0.7, label='GD', marker='v', markersize=3, markevery=20)

ax7.set_xlabel('Iteration', fontsize=11)
ax7.set_ylabel('Step Size', fontsize=11)
ax7.set_title('Step Size Evolution', fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3, which='both')

# 8. Newton vs BFGS (이차 함수)
ax8 = fig.add_subplot(gs[2, 2])

# 이차 함수에서 비교
print("\nQuadratic Function (Newton should be perfect)")
x0_quad = np.array([2.0, 3.0])
traj_newton_quad, loss_newton_quad = newton_method(quadratic_grad, quadratic_hess, x0_quad, max_iter=10)
traj_bfgs_quad, loss_bfgs_quad = bfgs(quadratic_grad, x0_quad, max_iter=20)
traj_gd_quad, loss_gd_quad = gradient_descent(quadratic_grad, x0_quad, eta=0.05, max_iter=100)

ax8.semilogy(loss_newton_quad, 'b-', linewidth=2.5, label='Newton (quadratic)', marker='o', markersize=6)
ax8.semilogy(loss_bfgs_quad, 'r-', linewidth=2.5, label='BFGS (quadratic)', marker='^', markersize=5)
ax8.semilogy(loss_gd_quad[:50], 'orange', linewidth=2, alpha=0.7, label='GD (quadratic)', marker='v', markersize=4, markevery=5)

ax8.set_xlabel('Iteration', fontsize=11)
ax8.set_ylabel('Loss', fontsize=11)
ax8.set_title('Newton on Quadratic (理想 케이스)', fontsize=12, fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3, which='both')

plt.savefig('/sessions/exciting-gifted-mccarthy/mnt/outputs/ch4-gradient-optimization/06_newton_quasi_newton.png',
            dpi=150, bbox_inches='tight')
print("✓ Visualization saved: 06_newton_quasi_newton.png")
plt.close()
```

## 🔗 AI/ML 연결

### scipy 구현 (실전)
```python
from scipy.optimize import minimize

# BFGS
result = minimize(loss_fn, x0, method='BFGS', jac=grad_fn)

# L-BFGS-B
result = minimize(loss_fn, x0, method='L-BFGS-B', jac=grad_fn)

# PyTorch에서는? → 불가능
# (Hessian 계산이 매우 비쌈)
```

## 📌 핵심 정리

| 메서드 | 수렴 속도 | 메모리 | 실제 비용 | 사용처 |
|--------|---------|--------|---------|--------|
| GD | $O(1/k)$ | 낮음 | 낮음 | 딥러닝 표준 |
| Newton | Quadratic | 높음 | 매우 높음 | 작은 문제 |
| BFGS | Superlinear | 높음 | 높음 | 구조화된 최적화 |
| L-BFGS | Superlinear | 낮음 | 중간 | 대규모 구조화 최적화 |

1. **뉴턴법**:
   - 이차 수렴 (매우 빠름)
   - 비용: $O(n^3)$ (Hessian 역행렬)
   - 초기점에 민감 (국소성)

2. **BFGS**:
   - Secant 조건으로 헤시안 근사
   - 초선형 수렴
   - 메모리: $O(n^2)$

3. **L-BFGS**:
   - 최근 m개 벡터만 저장
   - 메모리: $O(mn)$
   - 실제로는 거의 이차적 수렴

4. **딥러닝에서 못 쓰는 이유**:
   - Hessian 계산 불가능 ($O(n^2)$ 추가 계산)
   - Stochastic 그래디언트와 호환 안 됨

## 🤔 생각해볼 문제

1. **문제 1**: 뉴턴법이 이차 수렴하는 이유를 직관적으로 설명하시오.

2. **문제 2**: BFGS의 Secant 조건 $B_{k+1}s_k = y_k$는 뭘 의미하는가?

3. **문제 3**: L-BFGS에서 왜 최근 m개만 저장해도 "거의" 같은 수렴이 나오는가?

4. **문제 4**: 딥러닝에서 Hessian-vector product를 근사할 수 있다면, 2차 방법을 쓸 수 있을까?

5. **문제 5** (구현): 주어진 손실 함수에서 Hessian을 수치적으로 근사하고, Newton 방향과 BFGS 방향을 비교하시오.

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. 적응형 학습률](./05-adam-rmsprop-adagrad.md) | [📚 README](../README.md) | [07. Loss Landscape 기하 ▶](./07-loss-landscape-geometry.md) |

</div>
