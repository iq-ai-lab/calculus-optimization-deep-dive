# 03. Neural Tangent Kernel과 무한 폭 극한

## 🎯 핵심 질문

- Neural Tangent Kernel이란 정확히 무엇인가?
- 무한 폭 극한에서 신경망은 어떻게 동작하는가?
- 왜 "lazy training"이 나타나는가?
- 실제 딥러닝과 NTK 이론의 괴리는 무엇인가?

## 🔍 왜 이 개념이 AI에서 중요한가

Neural Tangent Kernel은 신경망의 고차원 기하학을 이해하는 핵심 도구입니다:

1. **수렴 보장**: 무한 폭에서 그라디언트 하강이 선형 ODE로 축소됨
2. **Kernel 방법의 재발견**: 신경망 = 커널 방법의 특수한 경우
3. **이론과 실제의 간극**: Feature learning (유한 폭) vs kernel regime (무한 폭)
4. **메타 학습 이론**: MAML, Few-shot 학습의 수학적 기초

## 📐 수학적 선행 조건

- 신경망의 파라미터화
- 전미분과 그래디언트
- 행렬 고유값/고유벡터
- 상미분방정식(ODE)의 기본
- 확률 이론 (중심극한정리)

## ✏️ 정의와 핵심 도구

### Neural Tangent Kernel 정의

주어진 신경망 $f_\theta(x): \mathbb{R}^d \to \mathbb{R}$에서:

**NTK (Neural Tangent Kernel)**:
$$\Theta(x, x') = \nabla_\theta f(x, \theta)^\top \nabla_\theta f(x', \theta) \in \mathbb{R}$$

또는 샘플 쌍 $(x_i, x_j)$에 대해:
$$[\Theta]_{ij} = \nabla_\theta f(x_i, \theta)^\top \nabla_\theta f(x_j, \theta)$$

**해석**: 파라미터 공간에서의 그래디언트 내적 = 함수 공간에서의 "거리" 측도

### Kernel Gradient Descent

손실 함수 $\mathcal{L}(f) = \frac{1}{2}\sum_i (f(x_i) - y_i)^2$에서 그라디언트 하강:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L} = \theta_t - \eta \sum_i (f(x_i, \theta_t) - y_i) \nabla_\theta f(x_i, \theta_t)$$

함수 공간에서:
$$\frac{\partial f(x, \theta_t)}{\partial t} = -\sum_j (f(x_j, \theta_t) - y_j) \cdot \Theta(x, x_j)$$

## 🔬 정리와 증명

### 정리 1: Kernel Regime ODE

**명제**: NTK이 훈련 중 거의 상수인 경우, 예측 함수는 다음 선형 ODE를 따릅니다:

$$\frac{d(f_t - y)}{dt} = -\Theta(f_t - y)$$

여기서 $f_t = [f(x_1, \theta_t), \ldots, f(x_n, \theta_t)]^\top$

**증명**:

신경망이 스칼라 손실로 훈련되는 경우:

$$\frac{d\theta}{dt} = -\nabla_\theta \mathcal{L} = -\sum_i (f_i - y_i) \nabla_\theta f_i$$

함수 공간에서:
$$\frac{df(x, \theta)}{dt} = \nabla_\theta f(x, \theta)^\top \frac{d\theta}{dt}$$

$$= -\sum_j (f(x_j) - y_j) \cdot \underbrace{\nabla_\theta f(x, \theta)^\top \nabla_\theta f(x_j, \theta)}_{\Theta(x, x_j)}$$

벡터로 쓰면:
$$\frac{df}{dt} = -\Theta (f - y)$$

이는 선형 ODE입니다. $\checkmark$

### 정리 2: Kernel Gradient Descent의 수렴성

**명제**: $\Theta$가 양정부호(positive definite)이면, $f_t \to y$로 지수 수렴합니다.

**증명**:

선형 ODE의 해:
$$f_t - y = e^{-\Theta t}(f_0 - y)$$

$\Theta$가 PD이므로 모든 고유값이 양수: $\lambda_i > 0$

$$\|f_t - y\|_2 \le \|f_0 - y\|_2 \cdot e^{-\lambda_{\min} t}$$

따라서 지수 속도로 수렴합니다. $\checkmark$

### 정리 3: 무한 폭 극한에서 NTK의 수렴

**명제**: 파라미터 수 $n \to \infty$인 경우, NTK는 확률적 수렴으로 결정론적 극한 $\Theta^\infty$로 수렴합니다.

$$\Theta^{(n)}(x, x') \xrightarrow{p} \Theta^\infty(x, x')$$

**증명 개요** (2층 네트워크의 경우):

네트워크를 다음과 같이 파라미터화:
$$f(x, \theta) = \frac{1}{\sqrt{n}} \sum_{k=1}^n a_k \sigma(w_k^\top x)$$

여기서 $a_k \in \mathbb{R}$, $w_k \in \mathbb{R}^d$, $\theta = (a, w)$

그래디언트:
$$\nabla_\theta f = \left(\sigma(w^\top x), \frac{a}{\sqrt{n}}\sigma'(w^\top x) \otimes x\right)^\top$$

NTK:
$$\Theta(x, x') = \frac{1}{n}\sum_k \sigma(w_k^\top x)\sigma(w_k^\top x') + \frac{a_k^2}{\sqrt{n}}\sigma'(w_k^\top x)\sigma'(w_k^\top x') (x^\top x')$$

$n \to \infty$일 때:

**첫 번째 항**: 대수의 법칙(LLN)
$$\frac{1}{n}\sum_k \sigma(w_k^\top x)\sigma(w_k^\top x') \xrightarrow{LLN} \mathbb{E}_{w \sim \mathcal{N}(0,I)}[\sigma(w^\top x)\sigma(w^\top x')]$$

**두 번째 항**: $O(n^{-1/2})$ 수렴, 극한에서 무시됨

따라서:
$$\Theta^\infty(x, x') = \mathbb{E}_w[\sigma(w^\top x)\sigma(w^\top x')] + (x^\top x')\mathbb{E}_w[\sigma'(w^\top x)\sigma'(w^\top x')]$$

$\checkmark$

### 정리 4: 2층 네트워크의 명시적 NTK

**명제**: 2층 ReLU 네트워크
$$f(x, \theta) = \frac{1}{\sqrt{n}} \sum_{k=1}^n a_k \max(w_k^\top x, 0) + b$$

의 무한 폭 극한 NTK는:
$$\Theta^\infty(x, x') = \mathbb{E}_{w \sim \mathcal{N}(0,I)}[\max(w^\top x, 0) \cdot \max(w^\top x', 0)] + (x^\top x') \cdot \mathbb{P}(w^\top x > 0 \land w^\top x' > 0)$$

**증명**:

ReLU: $\sigma(z) = \max(z, 0)$, $\sigma'(z) = \mathbb{1}_{z > 0}$

$w \sim \mathcal{N}(0, I^d)$일 때, $(w^\top x, w^\top x')$는 이변량 정규분포:

$$\begin{pmatrix} w^\top x \\ w^\top x' \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{pmatrix} \|x\|^2 & x^\top x' \\ x^\top x' & \|x'\|^2 \end{pmatrix}\right)$$

$\mathbb{E}[\max(w^\top x, 0) \cdot \max(w^\top x', 0)]$는 다변량 정규분포의 잘린 모멘트(truncated moment)로 계산 가능:

$$\mathbb{E}[z_1 z_2 | z_1 > 0, z_2 > 0] = \frac{\rho}{\pi} + \arcsin(\rho) \cdot \frac{\sqrt{\rho^2(1-\rho^2)}}{2\pi}$$

여기서 $\rho$는 $(w^\top x)$와 $(w^\top x')$의 상관계수 $\checkmark$

### 정리 5: Lazy Training (게으른 훈련)

**명제**: 무한 폭 극한에서 파라미터는 거의 움직이지 않으며, 신경망은 커널 방법처럼 동작합니다.

**증명**:

동역학:
$$\frac{d\theta}{dt} = -\sum_i (f_i - y_i) \nabla_\theta f_i$$

$n \to \infty$에서 스케일링: $\theta = \theta_0 + \frac{1}{\sqrt{n}}\delta\theta$

따라서:
$$\|\delta\theta\| = O(\sqrt{n})$$

상대 변화:
$$\frac{\|\theta_t - \theta_0\|}{\|\theta_0\|} = \frac{O(\sqrt{n})}{O(\sqrt{n})} = O(1)$$

하지만 그래디언트:
$$\nabla_\theta f = O(1)$$

이므로 파라미터의 상대 변화는 무시할 수 있습니다. $\checkmark$

### 정리 6: NTK vs Feature Learning

**명제**: 유한 폭 신경망에서 헤시안 $H_f = \nabla^2_\theta f$가 0이 아니면, 파라미터가 유의미하게 변하고 특성 학습이 발생합니다.

**증명**:

2차 Taylor 전개:
$$f_t(x) \approx f_0(x) + \nabla_\theta f_0 \cdot \Delta\theta_t + \frac{1}{2}(\Delta\theta_t)^\top H_f \Delta\theta_t$$

커널 방법 (NTK): 첫 두 항만 유지 (헤시안 무시)

Feature learning: 헤시안 항이 중요해지는 영역에서 발생

$\checkmark$

## 💻 NumPy/PyTorch 구현으로 검증

### 2층 네트워크의 NTK 계산

```python
import numpy as np
import matplotlib.pyplot as plt

def two_layer_network(x, W, a, b=0):
    """2층 네트워크: f(x) = 1/sqrt(n) * sum(a_k * ReLU(w_k^T x)) + b"""
    n = W.shape[0]
    hidden = np.maximum(W @ x, 0)  # (n,)
    output = np.dot(a, hidden) / np.sqrt(n) + b
    return output

def ntk_gradient(x, W, a):
    """NTK 그래디언트 계산"""
    n = W.shape[0]
    hidden = np.maximum(W @ x, 0)
    mask = (W @ x > 0).astype(float)  # ReLU 미분
    
    # df/da: ReLU(w_k^T x) / sqrt(n)
    grad_a = hidden / np.sqrt(n)
    
    # df/dw_k: a_k * mask * x / sqrt(n)
    grad_W = (a[:, None] * mask[:, None] * x[None, :]) / np.sqrt(n)
    
    return np.concatenate([grad_a, grad_W.flatten()])

def compute_ntk_matrix(X, W, a):
    """NTK 행렬 계산: K_ij = grad(x_i)^T @ grad(x_j)"""
    n_samples = X.shape[0]
    gradients = []
    
    for i in range(n_samples):
        grad = ntk_gradient(X[i], W, a)
        gradients.append(grad)
    
    gradients = np.array(gradients)
    K = gradients @ gradients.T
    
    return K

# 검증
print("=" * 70)
print("2층 ReLU 네트워크의 Neural Tangent Kernel")
print("=" * 70)

np.random.seed(42)

# 네트워크 파라미터
d = 3  # 입력 차원
n_widths = [10, 50, 100, 500, 1000]  # 다양한 폭
n_samples = 5

# 학습 데이터
X = np.random.randn(n_samples, d)
y = np.random.randn(n_samples)

print(f"\nInput dimension: {d}")
print(f"Number of samples: {n_samples}")
print(f"Data X shape: {X.shape}\n")

# 폭에 따른 NTK 수렴 확인
ntk_matrices = []
condition_numbers = []

for n in n_widths:
    # 가우시안 가중치 초기화
    W = np.random.randn(n, d)
    a = np.random.randn(n)
    
    K = compute_ntk_matrix(X, W, a)
    ntk_matrices.append(K)
    
    # 조건수 계산
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 1e-10)
    cond = np.max(eigvals) / np.min(eigvals)
    condition_numbers.append(cond)
    
    print(f"Width n={n:5d} | NTK shape: {K.shape} | Condition number: {cond:.4f}")

# NTK 수렴: 폭이 증가하면서 안정화
print("\n폭에 따른 NTK 고유값 변화:")
print(f"{'Width':<8} | {' Eigenvalues':<50}")
print("-" * 65)
for n, K in zip(n_widths, ntk_matrices):
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.sort(eigvals)[::-1]
    eigvals_str = ", ".join([f"{e:.4f}" for e in eigvals])
    print(f"{n:<8} | {eigvals_str}")
```

### Kernel Gradient Descent 시뮬레이션

```python
def kernel_gd_dynamics(y_true, K, eta=0.01, n_steps=100):
    """Kernel GD의 동역학"""
    n_samples = len(y_true)
    f_init = np.zeros(n_samples)  # 초기 예측
    
    f_trajectory = [f_init.copy()]
    loss_trajectory = [0.5 * np.sum((f_init - y_true)**2)]
    
    f = f_init.copy()
    
    for step in range(n_steps):
        # 기울기: K @ (f - y)
        grad = K @ (f - y_true)
        
        # 그래디언트 하강
        f = f - eta * grad
        
        f_trajectory.append(f.copy())
        loss = 0.5 * np.sum((f - y_true)**2)
        loss_trajectory.append(loss)
    
    return np.array(f_trajectory), np.array(loss_trajectory)

print("\n" + "=" * 70)
print("Kernel Gradient Descent 동역학")
print("=" * 70)

np.random.seed(42)
n_samples = 4
y_true = np.array([1.0, -1.0, 0.5, -0.5])

# NTK 행렬 (가우시안 커널로 근사)
X_train = np.random.randn(n_samples, 2)
sigma = 1.0
D_sq = np.sum(X_train**2, axis=1, keepdims=True) + np.sum(X_train**2, axis=1) - 2*X_train@X_train.T
K = np.exp(-D_sq / (2*sigma**2))

print(f"\nNTK matrix K (shape {K.shape}):")
print(K)
print(f"\nEigenvalues: {np.linalg.eigvalsh(K)[::-1]}")

# GD 동역학
f_traj, loss_traj = kernel_gd_dynamics(y_true, K, eta=0.01, n_steps=200)

print(f"\nTraining trajectory:")
print(f"Initial f: {f_traj[0]}")
print(f"Final f:   {f_traj[-1]}")
print(f"True y:    {y_true}")
print(f"\nInitial loss: {loss_traj[0]:.6f}")
print(f"Final loss:   {loss_traj[-1]:.6f}")

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 손실 곡선
ax1.semilogy(loss_traj, linewidth=2)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Loss (log scale)', fontsize=12)
ax1.set_title('Kernel GD Loss Trajectory', fontsize=13)
ax1.grid(True, alpha=0.3)

# 예측 수렴
steps = [0, len(f_traj)//4, len(f_traj)//2, len(f_traj)-1]
x_pos = np.arange(n_samples)
width = 0.2

for i, step in enumerate(steps):
    ax2.bar(x_pos + i*width, f_traj[step], width, 
            label=f'Step {step}', alpha=0.7)

ax2.plot(x_pos + 1.5*width, y_true, 'k*', markersize=15, label='True y')
ax2.set_xlabel('Sample index', fontsize=12)
ax2.set_ylabel('Prediction value', fontsize=12)
ax2.set_title('Prediction Convergence', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/kernel_gd.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장: /tmp/kernel_gd.png")
```

### 유한 폭에서의 Feature Learning

```python
def train_finite_width_network(X, y, n_width, n_steps=100, lr=0.01):
    """유한 폭 2층 네트워크 훈련"""
    np.random.seed(42)
    d = X.shape[1]
    
    # 파라미터 초기화
    W = np.random.randn(n_width, d) * 0.1
    a = np.random.randn(n_width) * 0.1
    
    W_trajectory = [W.copy()]
    a_trajectory = [a.copy()]
    loss_trajectory = []
    
    for step in range(n_steps):
        # Forward
        hidden = np.maximum(W @ X.T, 0)  # (n_width, n_samples)
        f = np.dot(a, hidden) / np.sqrt(n_width)
        
        # Loss
        loss = 0.5 * np.sum((f - y)**2)
        loss_trajectory.append(loss)
        
        # Backward
        df = f - y  # (n_samples,)
        
        # Gradient for a
        da = (hidden @ df) / np.sqrt(n_width)
        
        # Gradient for W
        mask = (W @ X.T > 0).astype(float)
        dW = (a[:, None] * mask @ X) / np.sqrt(n_width)
        
        # Update
        W = W - lr * dW
        a = a - lr * da
        
        W_trajectory.append(W.copy())
        a_trajectory.append(a.copy())
    
    return W_trajectory, a_trajectory, loss_trajectory

print("\n" + "=" * 70)
print("유한 폭 네트워크의 Feature Learning")
print("=" * 70)

np.random.seed(42)
n_samples, d = 5, 2
X = np.random.randn(n_samples, d)
y = np.random.randn(n_samples)

print(f"\nTraining data shape: {X.shape}")
print(f"Target shape: {y.shape}\n")

# 다양한 폭에서 훈련
widths = [5, 20, 100]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, width in enumerate(widths):
    W_traj, a_traj, loss_traj = train_finite_width_network(X, y, width, n_steps=100, lr=0.01)
    
    ax = axes[idx]
    ax.semilogy(loss_traj, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_title(f'Width n={width}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 파라미터 변화 측정
    W_init = np.linalg.norm(W_traj[0], 'fro')
    W_final = np.linalg.norm(W_traj[-1], 'fro')
    a_init = np.linalg.norm(a_traj[0])
    a_final = np.linalg.norm(a_traj[-1])
    
    print(f"Width {width}:")
    print(f"  ||W|| change: {W_init:.4f} → {W_final:.4f} (Δ = {abs(W_final-W_init):.4f})")
    print(f"  ||a|| change: {a_init:.4f} → {a_final:.4f} (Δ = {abs(a_final-a_init):.4f})")
    print()

plt.tight_layout()
plt.savefig('/tmp/feature_learning.png', dpi=150, bbox_inches='tight')
print("그래프 저장: /tmp/feature_learning.png")

print("해석: 폭이 작을수록 파라미터가 더 크게 변하며 feature learning 발생")
```

### PyTorch에서 NTK 계산

```python
import torch
import torch.nn as nn

def pytorch_ntk_gradient(model, x):
    """PyTorch 모델의 NTK 그래디언트"""
    x = torch.tensor(x, dtype=torch.float32, requires_grad=False)
    
    # forward
    output = model(x.unsqueeze(0))
    
    # 파라미터 수 계산
    n_params = sum(p.numel() for p in model.parameters())
    
    # 각 파라미터에 대한 그래디언트
    grads = []
    for p in model.parameters():
        if output.grad is not None:
            p.grad = None
        
        # output에 대한 해당 파라미터의 그래디언트
        grad = torch.autograd.grad(
            output, p, create_graph=False, retain_graph=True
        )[0]
        grads.append(grad.detach().cpu().numpy().flatten())
    
    return np.concatenate(grads)

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

print("\n" + "=" * 70)
print("PyTorch NTK 계산")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

d = 3
hidden_dims = [10, 50, 100]
n_samples = 4

X = np.random.randn(n_samples, d)

for hidden_dim in hidden_dims:
    model = TwoLayerNet(d, hidden_dim)
    
    # NTK 계산
    gradients = []
    for i in range(n_samples):
        with torch.enable_grad():
            x = torch.tensor(X[i], dtype=torch.float32, requires_grad=True)
            output = model(x.unsqueeze(0))
            
            # 각 파라미터에 대한 그래디언트
            grads = torch.autograd.grad(
                output, list(model.parameters()), create_graph=False,
                retain_graph=True, allow_unused=True
            )
            grad_vec = torch.cat([g.flatten() for g in grads])
            gradients.append(grad_vec.detach().numpy())
    
    # NTK 행렬
    gradients = np.array(gradients)
    K = gradients @ gradients.T
    
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.sort(eigvals)[::-1]
    
    print(f"\nHidden dimension: {hidden_dim}")
    print(f"NTK eigenvalues: {eigvals}")
```

## 🔗 AI/ML 연결

### 1. 신경망 이론의 통일적 이해

- **Kernel 관점**: 신경망 = 무한 차원 feature expansion + 선형 회귀
- **Optimization**: 그라디언트 하강이 선형 ODE 수렴

### 2. Overparameterization의 역할

- 폭이 증가하면서 NTK가 안정화
- 손실 랜드스케이프가 기하급수적으로 단순해짐

### 3. Few-shot Learning

- NTK 관점에서 MAML은 inner loop를 통해 effective kernel 학습
- Meta-learning = kernel 공간에서의 적응

### 4. 트랜스포머 분석

- Self-attention의 NTK 구조
- Scaling law와 NTK 이론의 연결

## 📌 핵심 정리

| 개념 | 공식 | 의미 |
|------|------|------|
| **NTK** | $\Theta(x,x') = \nabla f^\top \nabla f$ | 파라미터 공간의 내적 |
| **Kernel Regime** | $\frac{d(f-y)}{dt} = -\Theta(f-y)$ | 선형 ODE |
| **무한 폭 극한** | $\Theta^{(n)} \to \Theta^\infty$ | 결정론적 수렴 |
| **Lazy Training** | $\|\theta_t - \theta_0\| = o(1)$ | 파라미터 거의 변화 없음 |
| **2층 NTK** | $\Theta = \mathbb{E}[\sigma \sigma] + (x^\top x')\mathbb{E}[\sigma'\sigma']$ | 명시적 형태 |
| **Feature Learning** | $\frac{\partial f}{\partial W} \neq 0$ 중요 | 유한 폭에서 발생 |

## 🤔 생각해볼 문제

1. **NTK의 특이성**: NTK가 singular matrix인 경우는? 어떤 의미인가?

2. **Lazy vs Rich Training**: 
   - Lazy: NTK regime, 커널 방법처럼 동작
   - Rich: feature learning, 파라미터 크게 변함
   
   어떤 조건에서 각각 발생하는가?

3. **수렴 속도**: $\|f_t - y\|_2 \le \|f_0 - y\|_2 e^{-\lambda_{\min} t}$에서 $\lambda_{\min}$을 최대화하려면?

4. **3층 이상 네트워크**: NTK 이론이 깊은 네트워크에서도 성립하는가?

5. **실제 신경망**: ResNet, Transformer 같은 실제 아키텍처에서 NTK 이론이 얼마나 정확한가? 왜 gap이 있는가?

<div align="center">

| | | |
|---|---|---|
| [◀ 02. BN/LN 기울기](./02-batch-layer-norm-gradient.md) | [📚 README](../README.md) | [04. MAML과 고차 미분 ▶](./04-maml-higher-order.md) |

</div>
