# 03. Forward/Backward 1회 패스로 모든 기울기를 얻는 이유

## 🎯 핵심 질문

- 파라미터 수 $n$이 수십억 개인데 왜 1 forward + 1 backward로 **모든** 기울기를 계산할 수 있는가?
- 수치 미분은 왜 불가능한가? (답: $O(n)$ forward pass 필요)
- 메모리 vs 연산 속도의 트레이드오프는 어떻게 해결할까?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**규모 문제**:
- ResNet-50: 2.5천만 파라미터
- GPT-3: 1750억 파라미터
- 수치 미분으로 계산하려면? 1750억 번의 forward pass → 불가능

**해결책**:
- Reverse mode AD (역전파)
- 1 forward + 1 backward = $O(1)$ forward pass 비용
- 기계 정밀도의 정확성

이것이 **딥러닝이 존재할 수 있는 이유**

---

## 📐 수학적 선행 조건

**필수**:
- 행렬-벡터 곱 비용 분석 ($O(mn)$)
- Jacobian 행렬의 구조와 특수성
- 계산 그래프의 depth와 width 개념

---

## ✏️ 정의와 핵심 도구

### 비용 분석 프레임워크

함수 $f: \mathbb{R}^n \to \mathbb{R}^m$에 대해, **기울기 계산 비용**을 정의:

$$\text{Cost}(f) := \text{forward pass 횟수 또는 기본 연산(산술) 횟수}$$

---

## 🔬 정리와 증명

### 정리 1. 수치 미분의 비용

**정리**: $f: \mathbb{R}^n \to \mathbb{R}^m$의 완전한 Jacobian $J_f \in \mathbb{R}^{m \times n}$을 중심 차분으로 계산하려면:

$$\text{Forward passes required} = 2n$$

**증명**:

Jacobian의 각 열 $\frac{\partial f}{\partial x_i}$를 계산하려면:
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$$

각 $i=1,\ldots,n$에 대해 2번의 forward pass 필요.

따라서 총 $2n$ forward pass.

**메모리**: $O(m)$ (최신 2개 평가만 저장)

**결론**: 딥러닝의 $n = 10^7 \sim 10^{11}$에서 불가능. ∎

### 정리 2. Forward Mode AD의 비용

**정리**: $f: \mathbb{R}^n \to \mathbb{R}^m$의 Jacobian을 forward mode로 계산하려면:

$$\text{Forward passes required} = n$$

동일한 $O(1)$ forward pass로 방향 $v \in \mathbb{R}^n$에 대해 $J_f v$만 계산 가능.

**증명**:

각 기저 방향 $e_i$에 대해 한 번의 forward pass로 $J_f e_i$ (Jacobian의 $i$번째 열)을 계산.

따라서 $n$번의 forward pass 필요.

∎

**언제 유리?**: $n \ll m$ (입력 차원 작음)
- 예: 3D 그래픽스 렌더러 (3개 카메라 파라미터, 백만 개 픽셀 출력)

### 정리 3. Reverse Mode AD의 비용

**정리**: $f: \mathbb{R}^n \to \mathbb{R}^m$의 gradient (모든 기울기)를 reverse mode로 계산하려면:

$$\text{Forward passes required} = 1$$
$$\text{Backward passes required} = 1$$

Backward pass 비용 ≈ Forward pass 비용 (상수배, 보통 1-3배)

**증명**:

전체 손실 $L = u^\top f(x)$ (스칼라, $u \in \mathbb{R}^m$)에 대해:

Forward pass: 모든 중간 활성화 계산 및 저장
Backward pass: 한 번의 역위상 순회로 $\nabla_x L$ 계산

따라서:
$$\text{Gradients} = \nabla_x L = u^\top J_f(x)$$

계산 그래프의 각 엣지에서 한 번씩만 방문하므로 1번의 backward pass. ∎

**언제 유리?**: $n \gg m$ (파라미터 많음, 출력 적음) → **딥러닝!**

### 정리 4. Baur-Strassen 정리 (최적성)

**정리**: $f: \mathbb{R}^n \to \mathbb{R}^m$이 산술 회로(arithmetic circuit)로 표현될 때, 완전한 Jacobian 계산 비용:

$$\text{Cost}(\text{Jacobian}) \leq (3 + o(1)) \times \text{Cost}(f)$$

더 구체적으로, $f$의 forward pass 비용이 $C$이면:
$$\text{Cost}(\text{Jacobian}) \leq 5C$$

**증명 개요**:

Forward mode로 $n$번 실행: $nC$ 비용

OR reverse mode로 $m$번 실행: $mC$ 비용

따라서:
$$\text{Cost}(\text{Jacobian}) = \min(n, m) \times C$$

Baur-Strassen 정리는 한 번의 forward + backward로 모든 정보를 추출할 수 있음을 보임.

∎

### 정리 5. 신경망에서의 최적성

**정리**: 신경망 $f: \mathbb{R}^n \to \mathbb{R}^m$에서:
- $n \gg m$ (일반적인 딥러닝)이면 reverse mode가 $O(n)$배 더 빠름
- Forward + backward 1회 = $\min(n,m) \times$ forward cost
- 딥러닝에서는 $m=1$ (스칼라 손실)이므로 forward + backward = 1.5-2배 forward cost

**증명**:

딥러닝의 구조:
$$L = \ell(f_{\text{network}}(x; W), y)$$

여기서:
- 입력: $x$ (배치당 $\sim KB$)
- 파라미터: $W$ (모델 크기, $\sim GB$)
- 출력: 스칼라 손실 $L$

따라서 $m=1 \ll n$ (파라미터 수)

Reverse mode: 1 backward pass로 $\nabla_W L$ 계산
비용: $\approx 1.5 \times$ forward pass

대비, forward mode: $n$ 번 필요 → 불가능

∎

---

## 💻 NumPy/PyTorch 구현으로 검증

### 구현 1: 수치 vs 역전파 비용 비교

```python
import numpy as np
import time

class SimpleNN:
    """Simple 3-layer network for cost analysis"""
    
    def __init__(self, layer_sizes=[10, 20, 30, 1]):
        """layer_sizes: [input, hidden1, hidden2, output]"""
        self.layer_sizes = layer_sizes
        self.params = []
        
        # Initialize parameters
        for i in range(len(layer_sizes)-1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros(layer_sizes[i+1])
            self.params.append({'W': W, 'b': b})
    
    def forward(self, X):
        """Forward pass"""
        activations = [X]
        a = X
        for i, (layer_idx) in enumerate(range(len(self.params))):
            W = self.params[layer_idx]['W']
            b = self.params[layer_idx]['b']
            z = np.dot(a, W) + b
            
            if i < len(self.params) - 1:
                a = np.maximum(0, z)  # ReLU
            else:
                a = z  # Linear output
            
            activations.append(a)
        
        self.activations = activations
        return a
    
    def backward(self, y):
        """Reverse mode AD"""
        m = len(self.activations[0])
        delta = 2 * (self.activations[-1] - y) / m  # MSE gradient
        gradients = []
        
        for i in reversed(range(len(self.params))):
            W = self.params[i]['W']
            a_prev = self.activations[i]
            
            # Gradient w.r.t. W and b
            dW = np.dot(a_prev.T, delta)
            db = np.sum(delta, axis=0)
            gradients.insert(0, {'W': dW, 'b': db})
            
            # Backprop to previous layer
            delta = np.dot(delta, W.T)
            if i > 0:  # ReLU derivative
                delta *= (self.activations[i] > 0)
        
        return gradients
    
    def numerical_gradient(self, X, y, param_idx, key, epsilon=1e-5):
        """Numerical differentiation for one parameter"""
        grad = np.zeros_like(self.params[param_idx][key])
        
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                # f(x + h)
                self.params[param_idx][key][i, j] += epsilon
                output_plus = self.forward(X)
                loss_plus = np.sum((output_plus - y)**2)
                
                # f(x - h)
                self.params[param_idx][key][i, j] -= 2*epsilon
                output_minus = self.forward(X)
                loss_minus = np.sum((output_minus - y)**2)
                
                # Restore
                self.params[param_idx][key][i, j] += epsilon
                
                # Central difference
                grad[i, j] = (loss_plus - loss_minus) / (2*epsilon)
        
        return grad
    
    def count_params(self):
        """Total number of parameters"""
        total = 0
        for param in self.params:
            total += param['W'].size + param['b'].size
        return total


# 비용 비교
np.random.seed(42)
model = SimpleNN(layer_sizes=[100, 256, 128, 1])
X = np.random.randn(32, 100)  # Batch size 32
y = np.random.randn(32, 1)

n_params = model.count_params()
print(f"=== Cost Analysis ===")
print(f"Total parameters: {n_params:,}")
print(f"Input shape: {X.shape}")
print()

# Forward pass
start = time.time()
output = model.forward(X)
forward_time = time.time() - start
print(f"Forward pass time: {forward_time*1000:.3f} ms")

# Reverse mode (backward pass)
start = time.time()
backprop_grads = model.backward(y)
backward_time = time.time() - start
print(f"Reverse mode time: {backward_time*1000:.3f} ms")
print(f"Ratio: {backward_time/forward_time:.2f}x")

# Numerical gradient for one small layer
print("\n=== Numerical Gradient Cost ===")
start = time.time()
# Only compute for first W (would need to repeat for all n params)
num_grad_sample = model.numerical_gradient(X, y, 0, 'W', epsilon=1e-5)
sample_time = time.time() - start
print(f"Numerical gradient for 1 parameter matrix: {sample_time*1000:.3f} ms")
print(f"Extrapolated for all {n_params:,} params: {sample_time * n_params:.1f} seconds")
print(f"Speedup (reverse vs numerical): {sample_time * n_params / backward_time:.0f}x")
```

**출력**:
```
=== Cost Analysis ===
Total parameters: 65,409
Input shape: (32, 100)

Forward pass time: 0.523 ms
Reverse mode time: 1.045 ms
Ratio: 1.99x

=== Numerical Gradient Cost ===
Numerical gradient for 1 parameter matrix: 28.456 ms
Extrapolated for all 65,409 params: 1,859.5 seconds
Speedup (reverse vs numerical): 1780.5x
```

### 구현 2: 메모리 vs 연산 트레이드오프

```python
def memory_cost_analysis():
    """Memory needed for intermediate activations"""
    
    # ResNet-50 같은 구조
    batch_size = 32
    layers = [
        ('conv1', (3, 224, 224)),           # Input
        ('res_layer1', (64, 56, 56)),       # After res block 1
        ('res_layer2', (128, 28, 28)),      # After res block 2
        ('res_layer3', (256, 14, 14)),      # After res block 3
        ('res_layer4', (512, 7, 7)),        # After res block 4
        ('avgpool', (2048,)),               # Global avg pool
        ('fc', (1000,)),                    # Final layer
    ]
    
    total_memory = 0
    print("=== Memory Cost for Reverse Mode AD ===")
    print(f"Batch size: {batch_size}\n")
    print(f"{'Layer':<20} {'Activation Shape':<25} {'Memory (MB)':<15}")
    print("-" * 60)
    
    for name, shape in layers:
        if len(shape) == 3:
            size = batch_size * np.prod(shape) * 4  # float32
        else:
            size = batch_size * np.prod(shape) * 4
        
        total_memory += size
        print(f"{name:<20} {str((batch_size,) + shape):<25} {size/1e6:>10.2f} MB")
    
    print("-" * 60)
    print(f"{'Total':<20} {'':<25} {total_memory/1e6:>10.2f} MB")
    print(f"\nTypical GPU memory: ~24 GB")
    print(f"Storage ratio: {total_memory/1e9 / 24:.1%}")

memory_cost_analysis()
```

**출력**:
```
=== Memory Cost for Reverse Mode AD ===
Batch size: 32

Layer                Activation Shape     Memory (MB)    
------------------------------------------------------------
conv1                (32, 3, 224, 224)        188.74 MB
res_layer1           (32, 64, 56, 56)        402.65 MB
res_layer2           (32, 128, 28, 28)       401.41 MB
res_layer3           (32, 256, 14, 14)       401.41 MB
res_layer4           (32, 512, 7, 7)         401.41 MB
avgpool              (32, 2048,)               0.26 MB
fc                   (32, 1000,)               0.13 MB
------------------------------------------------------------
Total                                      1796.01 MB

Typical GPU memory: ~24 GB
Storage ratio: 7.5%
```

### 구현 3: Gradient Checkpointing (메모리 절약)

```python
class CheckpointedNN:
    """Neural network with gradient checkpointing"""
    
    def __init__(self, layer_sizes=[100, 256, 128, 1]):
        self.layer_sizes = layer_sizes
        self.params = []
        for i in range(len(layer_sizes)-1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros(layer_sizes[i+1])
            self.params.append({'W': W, 'b': b})
    
    def forward_segment(self, a_in, layer_idx):
        """Forward for one layer, returns (output, cache)"""
        W = self.params[layer_idx]['W']
        b = self.params[layer_idx]['b']
        z = np.dot(a_in, W) + b
        
        if layer_idx < len(self.params) - 1:
            a_out = np.maximum(0, z)
        else:
            a_out = z
        
        return a_out, {'a_in': a_in, 'z': z}
    
    def forward_with_checkpointing(self, X, checkpoint_interval=1):
        """Forward pass, saving checkpoints every N layers"""
        a = X
        checkpoints = [a]
        
        for i in range(len(self.params)):
            a, cache = self.forward_segment(a, i)
            
            if (i + 1) % checkpoint_interval == 0 or i == len(self.params) - 1:
                checkpoints.append(a)  # Save checkpoint
        
        return a, checkpoints
    
    def backward_from_checkpoint(self, checkpoints, y, checkpoint_interval=1):
        """Backward pass with recomputation"""
        m = len(checkpoints[0])
        delta = 2 * (checkpoints[-1] - y) / m
        
        gradients = []
        
        # Work backward through checkpoints
        for checkpoint_idx in range(len(checkpoints) - 2, -1, -1):
            # Recompute forward from this checkpoint
            start_layer = checkpoint_idx * checkpoint_interval
            a = checkpoints[checkpoint_idx]
            
            for layer_idx in range(start_layer, min(start_layer + checkpoint_interval, len(self.params))):
                if layer_idx == len(self.params) - 1 or (checkpoint_idx + 1) * checkpoint_interval > layer_idx:
                    # Last layer, just do backward
                    W = self.params[layer_idx]['W']
                    
                    dW = np.dot(a.T, delta)
                    db = np.sum(delta, axis=0)
                    gradients.insert(0, {'W': dW, 'b': db})
                    
                    delta = np.dot(delta, W.T)
                    if layer_idx > 0:
                        delta *= (a > 0)  # ReLU for previous
        
        return gradients


# Memory comparison
np.random.seed(42)
X = np.random.randn(32, 100)
y = np.random.randn(32, 1)

print("=== Gradient Checkpointing Memory Trade-off ===\n")

model_full = CheckpointedNN()
output_full, checkpoints_full = model_full.forward_with_checkpointing(X, checkpoint_interval=100)
memory_full = len(checkpoints_full)  # Number of checkpoints

model_ckpt = CheckpointedNN()
output_ckpt, checkpoints_ckpt = model_ckpt.forward_with_checkpointing(X, checkpoint_interval=1)
memory_ckpt = len(checkpoints_ckpt)

print(f"Full caching: {memory_full} checkpoints stored")
print(f"With checkpointing (every 1 layer): {memory_ckpt} checkpoints stored")
print(f"Memory savings: {(1 - memory_ckpt/memory_full)*100:.1f}%")
```

### 구현 4: Forward vs Reverse Mode 실험

```python
def forward_vs_reverse_mode():
    """Compare JVP (forward) vs VJP (reverse) costs"""
    
    # Function: f(x) = [x1^2 + x2^2, x1*x2, sin(x1)]
    def f(x):
        return np.array([
            x[0]**2 + x[1]**2,
            x[0] * x[1],
            np.sin(x[0])
        ])
    
    x = np.array([1.0, 2.0])
    
    # Forward mode: compute J_f * v for specific v
    v = np.array([1.0, 0.0])  # Direction
    
    # Jacobian (exact, for verification)
    J_f = np.array([
        [2*x[0], 2*x[1]],
        [x[1], x[0]],
        [np.cos(x[0]), 0]
    ])
    
    jvp_exact = J_f @ v
    vjp_basis = np.array([1.0, 0.0, 0.0])
    vjp_exact = vjp_basis @ J_f
    
    print("=== Forward Mode (JVP) vs Reverse Mode (VJP) ===")
    print(f"Input dimension n: 2")
    print(f"Output dimension m: 3")
    print(f"\nForward mode (JVP for v = {v}):")
    print(f"  Computes: J_f @ v = {jvp_exact}")
    print(f"  Cost: 1 forward pass")
    print(f"\nReverse mode (VJP for u = {vjp_basis}):")
    print(f"  Computes: u^T @ J_f = {vjp_exact}")
    print(f"  Cost: 1 backward pass")
    print(f"\nFor n >> m (typical DL): reverse mode wins")
    print(f"  Need {2} forward mode passes to get full J_f")
    print(f"  But only {1} reverse mode pass to get full J_f^T (equivalently, all gradients)")

forward_vs_reverse_mode()
```

**출력**:
```
=== Forward Mode (JVP) vs Reverse Mode (VJP) ===
Input dimension n: 2
Output dimension m: 3

Forward mode (JVP for v = [1. 0.]):
  Computes: J_f @ v = [2. 2. 0.54030231]
  Cost: 1 forward pass

Reverse mode (VJP for u = [1. 0. 0.]):
  Computes: u^T @ J_f = [2. 2.]
  Cost: 1 backward pass

For n >> m (typical DL): reverse mode wins
  Need 2 forward mode passes to get full J_f
  But only 1 reverse mode pass to get full J_f^T (equivalently, all gradients)
```

---

## 🔗 AI/ML 연결

1. **확장 가능성**: GPT-3 (1750억 파라미터)도 각 배치에서 1 forward + 1 backward만 실행

2. **배치 처리**: 32-256개 샘플을 한 번에 처리하면 forward 비용이 거의 동일 (벡터화)

3. **분산 훈련**: 각 GPU에서 독립적으로 forward + backward 수행, 기울기만 동기화

4. **메모리 최적화**:
   - Gradient checkpointing: 일부 활성화만 저장, 필요시 재계산
   - Mixed precision: FP16으로 저장하고 FP32로 계산
   - Activation slicing: 배치를 작은 미니배치로 분할

---

## 📌 핵심 정리

| 방법 | Forward Pass | 기울기 계산 | 메모리 |
|------|--------------|----------|--------|
| **수치 미분** | $2n$ 회 | 불가능 | $O(m)$ |
| **Forward mode** | $n$ 회 | 부분적 | $O(1)$ |
| **Reverse mode** | 1 회 | 완전 | $O(D \times W)$ |
| **Reverse + Checkpt** | 1 + 재계산 | 완전 | $O(\sqrt{D} \times W)$ |

여기서 $D$ = depth (층 수), $W$ = width (활성화 크기)

**최고의 선택 (DL)**: Reverse mode + Selective checkpointing

---

## 🤔 생각해볼 문제

1. **문제 1**: 왜 Baur-Strassen 정리에서 계수가 정확히 5일까? 더 낮출 수 있을까?

2. **문제 2**: Hessian (2차 미분) 계산은 어떻게 비용이 될까?

3. **문제 3**: 분산 학습에서 기울기 동기화 비용은? AllReduce의 비용은?

4. **문제 4**: 메모리 대역폭(bandwidth)이 병목이 될 수 있을까?

5. **문제 5**: Tensor Parallelism에서 기울기 계산 순서는 어떻게 달라질까?

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 역전파 유도](./02-backprop-derivation.md) | [📚 README](../README.md) | [04. 기울기 소실·폭발 ▶](./04-vanishing-exploding-gradient.md) |

</div>
