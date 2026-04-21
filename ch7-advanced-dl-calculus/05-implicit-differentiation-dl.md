# 05. 딥러닝에서의 암묵적 미분

## 🎯 핵심 질문

- 암묵적 미분(Implicit Differentiation)이란 무엇인가?
- Deep Equilibrium Model은 무한 깊이 네트워크를 어떻게 다루는가?
- Neural ODE는 활성화를 저장하지 않고 어떻게 역전파하는가?
- 이러한 기법들이 메모리 효율성과 계산 효율성을 어떻게 개선하는가?

## 🔍 왜 이 개념이 AI에서 중요한가

암묵적 미분은 현대 딥러닝의 메모리 병목을 해결합니다:

1. **메모리 효율성**: 중간 활성화를 저장하지 않음 (Deep Equilibrium: $O(n)$ 메모리)
2. **무한 깊이**: DEQ, Neural ODE로 "깊이"의 개념 재정의
3. **동역학계 모델링**: 미분 방정식으로 신경망 표현
4. **하이퍼파라미터 최적화**: 이중 최적화 문제의 효율적 해결

## 📐 수학적 선행 조건

- 암묵적 함수 정리(Implicit Function Theorem)
- 선형계 풀이 (Conjugate Gradient 등)
- 상미분방정식(ODE)의 수치 해석
- 벡터-야코비안 곱(VJP)

## ✏️ 정의와 핵심 도구

### 암묵적 미분의 일반 공식

암묵적 방정식 $F(x^*, \theta) = 0$이 주어졌을 때:

$$\frac{\partial x^*}{\partial \theta} = -\left[\frac{\partial F}{\partial x^*}\right]^{-1} \frac{\partial F}{\partial \theta}$$

손실 함수 $\mathcal{L}(x^*(\theta), \theta)$에 대한 그래디언트:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial x^*} \frac{\partial x^*}{\partial \theta} + \frac{\partial \mathcal{L}}{\partial \theta}\bigg|_{x^*}$$

### Deep Equilibrium Model (DEQ)

고정점 방정식:
$$z^* = f_\theta(z^*, x)$$

이를 반복 해법으로 풀기:
$$z_{k+1} = f_\theta(z_k, x)$$

## 🔬 정리와 증명

### 정리 1: 암묵적 함수 정리와 미분

**명제**: $F(x^*(\theta), \theta) = 0$이고 $\nabla_x F$가 가역이면:

$$\frac{d\mathcal{L}}{d\theta} = -\left(\frac{\partial \mathcal{L}}{\partial x^*}\right)^\top \left[\frac{\partial F}{\partial x^*}\right]^{-1} \frac{\partial F}{\partial \theta}$$

**증명** (Implicit Function Theorem):

$F(x^*(\theta), \theta) = 0$을 $\theta$로 미분하면:

$$\frac{\partial F}{\partial x^*} \frac{\partial x^*}{\partial \theta} + \frac{\partial F}{\partial \theta} = 0$$

따라서:
$$\frac{\partial x^*}{\partial \theta} = -\left[\frac{\partial F}{\partial x^*}\right]^{-1} \frac{\partial F}{\partial \theta}$$

손실의 기울기:
$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial x^*} \frac{\partial x^*}{\partial \theta} + \frac{\partial \mathcal{L}}{\partial \theta}\bigg|_{x^*}$$

$$= -\frac{\partial \mathcal{L}}{\partial x^*} \left[\frac{\partial F}{\partial x^*}\right]^{-1} \frac{\partial F}{\partial \theta} + \frac{\partial \mathcal{L}}{\partial \theta}\bigg|_{x^*}$$

VJP로 계산하려면, $u = \left[\frac{\partial F}{\partial x^*}\right]^{-T} \left(\frac{\partial \mathcal{L}}{\partial x^*}\right)^\top$를 풀어서:

$$\frac{\partial \mathcal{L}}{\partial \theta} = -u^\top \frac{\partial F}{\partial \theta}$$

$\checkmark$

### 정리 2: Deep Equilibrium Model의 역전파

**명제**: DEQ에서 순전파는 $z^* = \lim_{k \to \infty} f^{(k)}(0, x)$, 역전파는:

$$\left(I - J_f\right)^\top u = v$$

를 풀어서 수행합니다. 여기서 $J_f = \nabla_z f_\theta(z^*, x)$는 고정점에서의 야코비안입니다.

**증명**:

DEQ에서 $z^* = f_\theta(z^*, x)$이므로:

$$F(z, \theta, x) = z - f_\theta(z, x) = 0$$

암묵적 미분으로:
$$\frac{\partial z^*}{\partial \theta} = -\left[\frac{\partial F}{\partial z}\right]^{-1} \frac{\partial F}{\partial \theta} = -(I - J_f)^{-1} \left(-\frac{\partial f}{\partial \theta}\right) = (I - J_f)^{-1} \frac{\partial f}{\partial \theta}$$

VJP:
$$v^\top (I - J_f)^{-1} \frac{\partial f}{\partial \theta} = u^\top \frac{\partial f}{\partial \theta}$$

여기서:
$$(I - J_f)^\top u = v$$

이 선형계를 Conjugate Gradient로 풀면 $O(n \cdot k)$ 시간이 소요됩니다 (k: 반복 횟수).

$\checkmark$

### 정리 3: DEQ의 메모리 효율성

**명제**: 표준 신경망은 $O(d \cdot L)$ 메모리 (L: 깊이, d: 특징 차원), DEQ는 $O(d)$ 메모리만 필요합니다.

**증명**:

표준 심층망:
- Forward: 모든 중간 활성화 $[h_1, \ldots, h_L]$ 저장 → $O(L \cdot d)$
- Backward: 저장된 활성화로부터 기울기 계산

DEQ:
- Forward: 최종 $z^*$ 만 저장 → $O(d)$
- Backward: 필요하면 재계산 (CG로 해결) → O(d) 메모리

깊이 L과 무관하게 메모리 일정.

$\checkmark$

### 정리 4: Neural ODE와 Adjoint 방법

**명제**: Neural ODE $\frac{dz}{dt} = f_\theta(z(t), t)$의 손실 기울기는:

$$\frac{\partial \mathcal{L}}{\partial \theta} = -\int_{t_1}^{t_0} a(t)^\top \frac{\partial f}{\partial \theta}(z(t), t) dt$$

여기서 adjoint $a(t)$는 역시간 ODE를 만족합니다:

$$\frac{da}{dt} = -a(t)^\top \frac{\partial f}{\partial z}(z(t), t)$$

초기조건: $a(t_1) = \frac{\partial \mathcal{L}}{\partial z(t_1)}$

**증명**:

손실을 시간에 대해 미분하면:

$$\frac{d\mathcal{L}}{dt} = \frac{\partial \mathcal{L}}{\partial z}^\top \frac{dz}{dt} + \frac{\partial \mathcal{L}}{\partial t}$$

$$= \frac{\partial \mathcal{L}}{\partial z}^\top f_\theta(z, t) + \frac{\partial \mathcal{L}}{\partial t}$$

Lagrange 승수 $a(t)$를 도입하여:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \int_{t_0}^{t_1} \left[\frac{\partial \mathcal{L}}{\partial \theta} + a(t)^\top \frac{\partial f}{\partial \theta}\right] dt + [a(t)^\top z(t)]_{t_0}^{t_1}$$

변분 계산으로 $a(t)$의 미분방정식:

$$\frac{da}{dt} = -a(t)^\top \frac{\partial f}{\partial z}$$

이를 $t_1$에서 $t_0$로 적분하면:

$$\frac{\partial \mathcal{L}}{\partial \theta} = -\int_{t_1}^{t_0} a(t)^\top \frac{\partial f}{\partial \theta} dt$$

$\checkmark$

### 정리 5: Log-determinant Jacobian

**명제**: ODE의 궤적이 부피를 어떻게 변화시키는지 추적하려면:

$$\frac{d}{dt}\log|\det J| = \text{Tr}\left(\frac{\partial f}{\partial z}\right)$$

**증명**:

$z_{k+1} = z_k + \epsilon f(z_k)$의 확장 인수(expansion factor):

$$\left|\det\left(\frac{\partial z_{k+1}}{\partial z_k}\right)\right| = \left|\det(I + \epsilon \frac{\partial f}{\partial z})\right| \approx 1 + \epsilon \text{Tr}\left(\frac{\partial f}{\partial z}\right)$$

연속 극한:
$$\frac{d}{dt}\log|\det J| = \text{Tr}\left(\frac{\partial f}{\partial z}\right)$$

흐름 기반(Flow-based) 생성 모델에서 사용 가능.

$\checkmark$

### 정리 6: 하이퍼파라미터 최적화의 암묵적 미분

**명제**: 검증 손실을 통한 하이퍼파라미터 최적화:

$$\lambda^* = \text{argmin}_\lambda \mathcal{L}_{\text{val}}(w^*(\lambda))$$

에서 $w^*(\lambda)$는 훈련 손실의 최적점:

$$w^*(\lambda) = \text{argmin}_w \mathcal{L}_{\text{train}}(w, \lambda)$$

Neumann 급수 근사:

$$\frac{d\mathcal{L}_{\text{val}}}{d\lambda} \approx \sum_{j=0}^J (I - H^{-1})^j \frac{\partial \mathcal{L}_{\text{val}}}{\partial w}$$

**증명** (Lorraine et al., 2020):

최적성 조건: $\nabla_w \mathcal{L}_{\text{train}}(w^*, \lambda) = 0$

$\lambda$로 미분:
$$\left[\nabla_w^2 \mathcal{L}_{\text{train}}\right] \frac{dw^*}{d\lambda} + \nabla_{\lambda w} \mathcal{L}_{\text{train}} = 0$$

따라서:
$$\frac{dw^*}{d\lambda} = -H^{-1} \nabla_{\lambda w} \mathcal{L}_{\text{train}}$$

validation gradient:
$$\frac{d\mathcal{L}_{\text{val}}}{d\lambda} = \nabla_w \mathcal{L}_{\text{val}} \frac{dw^*}{d\lambda} + \nabla_\lambda \mathcal{L}_{\text{val}}$$

$$\approx -\nabla_w \mathcal{L}_{\text{val}} H^{-1} \nabla_{\lambda w} \mathcal{L}_{\text{train}}$$

Neumann 급수로 $H^{-1} v \approx \sum_j (I - H)^j v$로 계산 (H가 작은 고유값에서 PD):

$$\frac{d\mathcal{L}_{\text{val}}}{d\lambda} \approx \sum_{j=0}^J (I - H)^j \nabla_w \mathcal{L}_{\text{val}} \nabla_{\lambda w} \mathcal{L}_{\text{train}}$$

$\checkmark$

## 💻 NumPy/PyTorch 구현으로 검증

### Deep Equilibrium Model 역전파

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import root

class DEQBlock(nn.Module):
    """Deep Equilibrium Block"""
    
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, z, x):
        """f(z, x) = σ(fc1([z; x])) 맵핑 후 fc2"""
        return torch.relu(self.fc1(z)) + z

class DEQ(nn.Module):
    """Deep Equilibrium Model"""
    
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.block = DEQBlock(feature_dim, hidden_dim)
        self.tol = 1e-3
        self.max_iter = 100
    
    def forward(self, x):
        """고정점 찾기"""
        # 초기값
        z = torch.zeros(x.shape[0], self.feature_dim, device=x.device)
        
        # 반복: z_{k+1} = f(z_k, x)
        for iter_idx in range(self.max_iter):
            z_prev = z.clone().detach()
            z = self.block(z, x)
            
            # 수렴 확인
            residual = torch.norm(z - z_prev) / (torch.norm(z) + 1e-8)
            if residual < self.tol:
                print(f"  Converged in {iter_idx+1} iterations")
                break
        
        return z

print("=" * 70)
print("Deep Equilibrium Model (DEQ) 역전파")
print("=" * 70)

torch.manual_seed(42)

# 모델 및 데이터
feature_dim = 5
hidden_dim = 10
batch_size = 3

model = DEQ(feature_dim, hidden_dim)
x = torch.randn(batch_size, feature_dim, requires_grad=True)
y_true = torch.randn(batch_size, feature_dim)

print(f"Feature dimension: {feature_dim}")
print(f"Batch size: {batch_size}\n")

# Forward pass
print("Forward pass:")
z_star = model(x)
print(f"Equilibrium z* shape: {z_star.shape}")
print(f"z* value (first sample): {z_star[0]}\n")

# Loss and backward
loss = F.mse_loss(z_star, y_true)
print("Backward pass:")
print(f"Loss: {loss.item():.6f}")

loss.backward()

print(f"x.grad norm: {torch.norm(x.grad):.6f}")
print(f"fc1.weight.grad norm: {torch.norm(model.block.fc1.weight.grad):.6f}")
```

### Neural ODE 구현 및 Adjoint 역전파

```python
from scipy.integrate import solve_ivp

class NeuralODE(nn.Module):
    """Neural ODE: dz/dt = f_θ(z, t)"""
    
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim + 1, hidden_dim)  # +1 for time
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
    
    def forward_ode(self, t, z):
        """ODE 함수: dz/dt"""
        z = torch.tensor(z, dtype=torch.float32, requires_grad=True)
        t_tensor = torch.tensor([t], dtype=torch.float32)
        
        zt = torch.cat([z, t_tensor])
        h = torch.relu(self.fc1(zt))
        dz = self.fc2(h)
        
        return dz.detach().numpy()
    
    def solve_ode(self, z0, t_eval):
        """ODE 풀이 (RK45)"""
        sol = solve_ivp(
            self.forward_ode,
            [t_eval[0], t_eval[-1]],
            z0,
            t_eval=t_eval,
            method='RK45',
            dense_output=True
        )
        return sol.y[:, -1]  # 최종 시간에서의 상태
    
    def forward(self, z0, t_span=(0.0, 1.0), n_points=10):
        """Neural ODE를 풀고 최종 상태 반환"""
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        z_final = self.solve_ode(z0.detach().numpy()[0], t_eval)
        return torch.tensor(z_final, dtype=torch.float32)

print("\n" + "=" * 70)
print("Neural ODE")
print("=" * 70)

torch.manual_seed(42)

ode_model = NeuralODE(feature_dim=3, hidden_dim=10)
z0 = torch.randn(1, 3)
t_span = (0.0, 1.0)

print(f"Initial state z0: {z0}")
print(f"Time span: {t_span}\n")

print("Solving ODE:")
z_final = ode_model(z0, t_span, n_points=5)
print(f"Final state z(T): {z_final}")
print(f"State change: {torch.norm(z_final - z0[0]):.6f}")
```

### Implicit Function Theorem을 이용한 검증

```python
def implicit_differentiation_example():
    """암묵적 미분 수치 검증"""
    
    print("\n" + "=" * 70)
    print("암묵적 미분 (Implicit Differentiation) 검증")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 간단한 예제: F(x, θ) = x² + θx - 1 = 0
    # 암묵적으로: x*(θ) = (-θ + sqrt(θ² + 4))/2
    # dx*/dθ = (x* - 1)/(2x* + θ)
    
    theta = torch.tensor(0.5, requires_grad=True)
    
    # x*를 수치적으로 찾기 (비선형 방정식)
    def equation(x_val):
        return x_val**2 + theta.item() * x_val - 1.0
    
    from scipy.optimize import fsolve
    x_star = torch.tensor(fsolve(equation, 0.5)[0], requires_grad=False)
    
    print(f"θ = {theta.item():.4f}")
    print(f"x*(θ) = {x_star.item():.6f}")
    
    # 손실: L = x*²
    L = x_star**2
    
    # 해석적 미분: dL/dθ = 2x* dx*/dθ
    # 여기서 dx*/dθ = -x / (2x + θ) (암묵적 미분)
    with torch.no_grad():
        dx_star_dtheta = -x_star / (2*x_star + theta)
        dL_dtheta_analytical = 2 * x_star * dx_star_dtheta
    
    print(f"\nAnalytical dL/dθ (by implicit differentiation): {dL_dtheta_analytical:.6f}")
    
    # 수치 미분으로 검증
    eps = 1e-5
    theta_plus = torch.tensor(theta.item() + eps)
    
    def equation_plus(x_val):
        return x_val**2 + theta_plus.item() * x_val - 1.0
    
    x_star_plus = fsolve(equation_plus, 0.5)[0]
    L_plus = x_star_plus**2
    
    dL_dtheta_numerical = (L_plus - L.item()) / eps
    
    print(f"Numerical dL/dθ: {dL_dtheta_numerical:.6f}")
    print(f"Difference: {abs(dL_dtheta_analytical - dL_dtheta_numerical):.2e}")

implicit_differentiation_example()
```

### 하이퍼파라미터 최적화 (Neumann 급수)

```python
def hyperparameter_optimization_neumann():
    """Neumann 급수를 이용한 하이퍼파라미터 최적화"""
    
    print("\n" + "=" * 70)
    print("하이퍼파라미터 최적화 (Neumann Series)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 간단한 선형 회귀: min_w (y - Xw)² + λ||w||²
    # 최적 w*: (X^T X + 2λI)^{-1} X^T y
    
    n_features = 5
    n_train = 20
    n_val = 5
    
    X_train = torch.randn(n_train, n_features)
    y_train = torch.randn(n_train)
    X_val = torch.randn(n_val, n_features)
    y_val = torch.randn(n_val)
    
    def optimal_w(lam):
        """최적 가중치 (정규화 계수 lam)"""
        H = X_train.T @ X_train + 2*lam*torch.eye(n_features)
        return torch.linalg.solve(H, X_train.T @ y_train)
    
    def val_loss(lam):
        """검증 손실"""
        w_opt = optimal_w(lam)
        pred = X_val @ w_opt
        return F.mse_loss(pred, y_val).item()
    
    # 최적 λ 찾기 (그리드 서치)
    lambdas = torch.logspace(-3, 1, 20)
    val_losses = [val_loss(l.item()) for l in lambdas]
    
    best_lambda = lambdas[np.argmin(val_losses)]
    
    print(f"Number of features: {n_features}")
    print(f"Training samples: {n_train}, Validation samples: {n_val}\n")
    
    print(f"Optimal λ: {best_lambda.item():.6f}")
    print(f"Validation loss: {val_loss(best_lambda.item()):.6f}")
    
    # Neumann 급수 근사 (느린 계산이므로 간단한 버전)
    print("\nNeumann series approximation:")
    w_opt = optimal_w(best_lambda)
    
    # 검증 gradient: dL_val/dw
    pred_val = X_val @ w_opt
    dL_val_dw = 2 * X_val.T @ (pred_val - y_val) / n_val
    
    # 헤시안: H = X^T X + 2λI
    H = X_train.T @ X_train + 2*best_lambda*torch.eye(n_features)
    
    # H^{-1} dL_val/dw (정확한 계산)
    grad_exact = torch.linalg.solve(H, dL_val_dw)
    
    # Neumann 급수: H^{-1} v ≈ sum_{j=0}^{K} (I - H)^j v
    # 먼저 H를 정규화: H -> H / ||H||
    H_norm = H / torch.norm(H)
    
    grad_neumann = torch.zeros_like(dL_val_dw)
    for k in range(10):
        if k == 0:
            term = dL_val_dw
        else:
            term = (torch.eye(n_features) - H_norm) @ term
        grad_neumann += term
    
    print(f"Exact H^{{-1}} dL_val/dw norm: {torch.norm(grad_exact):.6f}")
    print(f"Neumann (K=10) approximation norm: {torch.norm(grad_neumann):.6f}")

hyperparameter_optimization_neumann()
```

### DEQ vs 표준 깊은 네트워크의 메모리 비교

```python
def memory_comparison():
    """메모리 사용량 비교"""
    
    print("\n" + "=" * 70)
    print("DEQ vs 표준 네트워크: 메모리 비교")
    print("=" * 70)
    
    import sys
    
    torch.manual_seed(42)
    
    feature_dim = 64
    hidden_dim = 128
    batch_size = 32
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Batch size: {batch_size}\n")
    
    # 표준 신경망 (깊이 L)
    depths = [10, 20, 50, 100]
    
    print(f"{'Depth':<8} | {'Standard NN Memory (MB)':<25} | {'DEQ Memory (MB)':<20}")
    print("-" * 55)
    
    for L in depths:
        # 표준 네트워크: 모든 중간 활성화 저장
        standard_memory_bytes = feature_dim * batch_size * L * 4  # float32
        standard_memory_mb = standard_memory_bytes / (1024**2)
        
        # DEQ: 최종 상태만 저장
        deq_memory_bytes = feature_dim * batch_size * 4
        deq_memory_mb = deq_memory_bytes / (1024**2)
        
        print(f"{L:<8} | {standard_memory_mb:<25.2f} | {deq_memory_mb:<20.2f}")
    
    print("\n메모리 절감 비율:")
    print(f"깊이 L=100일 때: {standard_memory_mb / deq_memory_mb:.0f}배")

memory_comparison()
```

## 🔗 AI/ML 연결

### 1. 정규화 문제에서의 암묵적 미분

- 라그랑주 승수 최적화
- 제약 조건부 신경망

### 2. 생성 모델

- Neural ODE: 연속 시간 생성 모델
- Flow-based 생성 모델 (Glow, Neural Spline Flows)

### 3. 효율적 깊은 네트워크

- Deep Equilibrium: 메모리 $O(d)$ 독립적 깊이
- 역동역학 계산 ODE

### 4. 미분 방정식 기반 모델링

- Physics-informed Neural Networks (PINNs)
- 보존 법칙 준수

## 📌 핵심 정리

| 개념 | 공식 | 역할 |
|------|------|------|
| **암묵적 미분** | $\frac{dx^*}{d\theta} = -F_x^{-1} F_\theta$ | 고정점의 기울기 |
| **DEQ Forward** | $z^* = f_\theta(z^*, x)$ (고정점) | 무한 깊이 네트워크 |
| **DEQ Backward** | $(I - J_f)^\top u = v$ (선형계) | 메모리 효율적 역전파 |
| **Neural ODE** | $\frac{dz}{dt} = f_\theta(z, t)$ | 연속 시간 모델링 |
| **Adjoint Method** | $\frac{da}{dt} = -a^\top \frac{\partial f}{\partial z}$ | ODE 역전파 |
| **Neumann Series** | $H^{-1}v \approx \sum (I-H)^j v$ | 근사 계산 |

## 🤔 생각해볼 문제

1. **DEQ의 수렴성**: 고정점이 항상 존재하는가? 어떤 조건에서 유일한가?

2. **Stability**: DEQ의 $J_f$ 고유값이 크면 어떻게 되는가? (수렴 속도에 미치는 영향)

3. **Neural ODE와 깊이**: Neural ODE의 "깊이"는 시간 적분 스텝이다. 이것이 전통적 깊이와 어떻게 다른가?

4. **메모리-계산 트레이드오프**: DEQ는 메모리는 절감하지만 CG 반복이 필요하다. 언제 DEQ가 더 효율적인가?

5. **수치 안정성**: 
   - DEQ: 고정점을 너무 정밀하게 푸는 것은 비효율적일까?
   - Neural ODE: ODE 솔버의 오차가 역전파에 미치는 영향?

<div align="center">

| | |
|---|---|
| [◀ 04. MAML과 고차 미분](./04-maml-higher-order.md) | [📚 README](../README.md) |

</div>
