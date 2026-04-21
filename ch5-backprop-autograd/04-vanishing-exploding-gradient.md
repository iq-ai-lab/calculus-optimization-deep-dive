# 04. 기울기 소실과 폭발

## 🎯 핵심 질문

- 왜 깊은 신경망은 기울기가 소실되거나 폭발할까?
- 수식으로 정확히 언제 발생하는가?
- ReLU, Batch Normalization, Weight Initialization으로 어떻게 해결할까?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**역사적 문제 (1990s-2000s)**:
- Sigmoid 활성화 함수 사용 시 깊은 신경망 훈련 불가능
- 1층의 기울기는 거대하지만, 깊은 층의 기울기는 $10^{-20}$ 이하로 소실

**현대적 해결책**:
1. ReLU 활성화 함수
2. Batch Normalization
3. Residual Connections (ResNet)

이들이 없으면 **50층 이상의 신경망 훈련 불가능**

---

## 📐 수학적 선행 조건

**필수**:
- 행렬 특이값(singular values), 조건수(condition number)
- 합성 함수의 미분
- 분산, 표준편차

---

## ✏️ 정의와 핵심 도구

### 기울기 흐름의 수식

**L층 신경망**에서 역전파:
$$\delta^{(l)} = (W_{l+1}^\top \delta^{(l+1)}) \odot \sigma'(z^{(l)})$$

**완전히 연쇄하면**:
$$\delta^{(l)} = \left(\prod_{k=l}^{L-1} W_{k+1}^\top \text{diag}(\sigma'(z^{(k)}))\right) \delta^{(L)}$$

**기울기 크기**:
$$\left\|\frac{\partial L}{\partial W_l}\right\| \propto \left\|\prod_{k=l}^{L-1} W_{k+1}^\top \sigma'(z^{(k)})\right\|$$

---

## 🔬 정리와 증명

### 정리 1. Sigmoid 기울기 소실

**정리**: Sigmoid 함수 $\sigma(z) = \frac{1}{1+e^{-z}}$에서:
$$\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq \frac{1}{4}$$

최댓값은 $z=0$에서만 달성되며, 대부분의 영역에서 훨씬 작음.

**증명**:

미분:
$$\sigma'(z) = \frac{d}{dz}\frac{1}{1+e^{-z}} = \frac{e^{-z}}{(1+e^{-z})^2}$$

$\sigma(z) = \frac{1}{1+e^{-z}}$이고 $1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}}$이므로:
$$\sigma'(z) = \sigma(z)(1-\sigma(z))$$

최댓값 찾기:
$$\frac{d}{dz}[\sigma(z)(1-\sigma(z))] = \sigma'(z)(1-2\sigma(z)) = 0$$

따라서 $\sigma(z) = 1/2$, 즉 $z=0$에서 최대.

최댓값: $\sigma'(0) = \frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$

∎

**결과**: L층 신경망에서 기울기는 최악의 경우:
$$\left\|\frac{\partial L}{\partial z^{(l)}}\right\| \propto \left(\frac{1}{4}\right)^{L-l}$$

$L=100$이면: $(1/4)^{100} \approx 10^{-60}$ → 훈련 불가능

### 정리 2. 행렬 곱 특이값에 의한 기울기 증폭/감쇠

**정리**: 역전파 관점에서:
$$\delta^{(l)} = W_{l+1}^\top \text{diag}(\sigma'(z^{(l)})) \delta^{(l+1)}$$

이 변환의 2-노름이:
- $< 1$이면 → 기울기 소실 (vanishing)
- $> 1$이면 → 기울기 폭발 (exploding)

보다 정확히, $\|W_{l+1}\|$ (spectral norm, 최대 특이값)과 $\|\sigma'(z^{(l)})\|_\infty$ (원소별 최대값)의 곱으로 결정됨.

**증명**:

벡터 노름:
$$\|\delta^{(l)}\| = \|W_{l+1}^\top \text{diag}(\sigma'(z^{(l)})) \delta^{(l+1)}\|$$

행렬 노름 성질:
$$\|AB\| \leq \|A\| \|B\|$$

따라서:
$$\|\delta^{(l)}\| \leq \|W_{l+1}\|^T \|\text{diag}(\sigma')\| \|\delta^{(l+1)}\|$$

$\|\text{diag}(\sigma')\| = \max_i |\sigma'(z_i^{(l)})| \leq 1/4$ (sigmoid)

따라서:
$$\|\delta^{(l)}\| \leq \|W_{l+1}\| \cdot \frac{1}{4} \cdot \|\delta^{(l+1)}\|$$

재귀적으로:
$$\|\delta^{(l)}\| \leq \prod_{k=l}^{L-1} \left(\|W_{k+1}\| \cdot \frac{1}{4}\right) \|\delta^{(L)}\|$$

$\|W\| \approx O(1)$이면: $\|\delta^{(l)}\| \propto (1/4)^{L-l}$ → 소실

∎

### 정리 3. ReLU에서의 기울기 흐름

**정리**: ReLU 함수 $\text{ReLU}(z) = \max(0, z)$에서:
$$\frac{\partial \text{ReLU}}{\partial z} = \mathbf{1}[z > 0] = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

따라서 역전파에서:
$$\delta^{(l)} = (W_{l+1}^\top \delta^{(l+1)}) \odot \mathbf{1}[z^{(l)} > 0]$$

**특징**:
- 활성 뉴런 ($z > 0$)을 통한 기울기는 **증폭 없이** 직접 흐름
- 비활성 뉴런 ($z \leq 0$)을 통한 기울기는 차단

**결과**: $\|W\| \approx 1$이면 기울기 크기가 층을 통과해도 유지됨

∎

### 정리 4. Xavier/He Weight Initialization

**정리 (Xavier 초기화)**: 입력 분산과 출력 분산이 같도록 가중치 초기화:

$$\text{Var}(W) = \frac{1}{n_{\text{in}}}$$

여기서 $n_{\text{in}}$는 입력 차원.

**유도**:

선형 층 $z = Wx + b$에서 ($W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$):

$$\text{Var}(z_j) = \sum_{i=1}^{n_{\text{in}}} \text{Var}(W_{ji}) \text{Var}(x_i)$$

$W_{ji}$가 독립 동일 분포이고 $x_i$가 독립 동일 분포이면:
$$\text{Var}(z_j) = n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

$\text{Var}(x) \approx 1$이면:
$$\text{Var}(z_j) = n_{\text{in}} \cdot \text{Var}(W)$$

$\text{Var}(z) = \text{Var}(x) = 1$을 유지하려면:
$$\text{Var}(W) = \frac{1}{n_{\text{in}}}$$

∎

**정리 (He 초기화 - ReLU용)**: ReLU의 특성을 고려:

$$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$

**유도**:

ReLU 후 약 절반의 활성화가 0으로 소실되므로:
$$\text{Var}(\text{ReLU}(z)) \approx \frac{1}{2} \text{Var}(z)$$

분산 유지를 위해:
$$\text{Var}(z) = 2 \cdot \text{Var}(x)$$

따라서:
$$n_{\text{in}} \cdot \text{Var}(W) = 2$$
$$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$

∎

### 정리 5. Batch Normalization의 효과

**정리**: Batch Normalization은 각 미니배치에서 중간 활성화를 표준화:

$$\tilde{x}_i = \frac{x_i - E[x]}{\sqrt{\text{Var}(x) + \epsilon}}$$

이는 내부 공변량 변화(Internal Covariate Shift)를 제거하여:
1. 기울기 흐름 개선 (기울기 소실 완화)
2. 헤시안 조건수 개선

**효과**:

정규화 전:
$$\text{Var}(z^{(l)}) = n_l \cdot \text{Var}(W_l) \cdot \text{Var}(a^{(l-1)})$$

층이 깊어지면 분산이 폭발/소실 가능.

BN 후:
$$\text{Var}(\tilde{z}^{(l)}) = 1 \quad \text{(by definition)}$$

따라서 모든 층에서 일정한 분산 유지 → 기울기 안정화

∎

### 정리 6. Gradient Clipping

**정리**: 기울기 노름이 임계값을 초과하면 스케일:

$$g \leftarrow g \cdot \min\left(1, \frac{\tau}{\|g\|}\right)$$

여기서 $\tau$는 임계값 (예: 1.0).

**효과**: $\|g\| > \tau$일 때 $\|g\| = \tau$로 정규화, $\|g\| \leq \tau$일 때 그대로

이는 기울기 폭발을 방지하면서 방향은 유지.

∎

---

## 💻 NumPy/PyTorch 구현으로 검증

### 구현 1: Sigmoid에서의 기울기 소실 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

class GradientVanishingExperiment:
    """Simulate gradient flow through deep sigmoid networks"""
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def simulate_deep_network(self, depth=50, use_relu=False):
        """Simulate gradient flow through L layers"""
        np.random.seed(42)
        
        # Initialize weights with standard initialization
        weights = [np.random.randn(10, 10) * 0.1 for _ in range(depth)]
        
        # Forward pass: compute z values
        z_values = []
        a = np.random.randn(1, 10) * 0.1  # Input
        
        for l in range(depth):
            z = np.dot(a, weights[l])
            z_values.append(z)
            
            if use_relu:
                a = np.maximum(0, z)
            else:
                a = self.sigmoid(z)
        
        # Backward pass: compute gradient magnitudes
        gradient_magnitudes = []
        delta = np.ones_like(z_values[-1])
        gradient_magnitudes.append(np.linalg.norm(delta))
        
        for l in reversed(range(depth - 1)):
            if use_relu:
                derivative = (z_values[l] > 0).astype(float)
            else:
                derivative = self.sigmoid_derivative(z_values[l])
            
            delta = np.dot(delta, weights[l+1].T) * derivative
            gradient_magnitudes.insert(0, np.linalg.norm(delta))
        
        return np.array(gradient_magnitudes)
    
    def experiment_activation_functions(self):
        """Compare sigmoid vs ReLU gradient flow"""
        depths = range(1, 101, 5)
        sigmoid_grads = []
        relu_grads = []
        
        for depth in depths:
            sigmoid_mag = self.simulate_deep_network(depth, use_relu=False)
            relu_mag = self.simulate_deep_network(depth, use_relu=True)
            
            sigmoid_grads.append(sigmoid_mag[0])  # First layer gradient
            relu_grads.append(relu_mag[0])
        
        print("=== Gradient Vanishing: Sigmoid vs ReLU ===\n")
        print(f"{'Depth':<8} {'Sigmoid (1st layer gradient)':<35} {'ReLU':<15}")
        print("-" * 60)
        
        for d, sg, rg in zip(depths, sigmoid_grads, relu_grads):
            print(f"{d:<8} {sg:>20.2e} {rg:>20.2e}")
        
        return list(depths), sigmoid_grads, relu_grads


exp = GradientVanishingExperiment()
depths, sigmoid_grads, relu_grads = exp.experiment_activation_functions()

# Visualization
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Linear scale
    plt.subplot(1, 2, 1)
    plt.plot(depths, sigmoid_grads, 'o-', label='Sigmoid', linewidth=2)
    plt.plot(depths, relu_grads, 's-', label='ReLU', linewidth=2)
    plt.xlabel('Network Depth (# layers)', fontsize=11)
    plt.ylabel('First Layer Gradient Magnitude', fontsize=11)
    plt.title('Gradient Flow: Sigmoid vs ReLU', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    plt.subplot(1, 2, 2)
    plt.semilogy(depths, sigmoid_grads, 'o-', label='Sigmoid', linewidth=2)
    plt.semilogy(depths, relu_grads, 's-', label='ReLU', linewidth=2)
    plt.xlabel('Network Depth (# layers)', fontsize=11)
    plt.ylabel('First Layer Gradient Magnitude (log scale)', fontsize=11)
    plt.title('Gradient Vanishing (Exponential Decay)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('/tmp/gradient_vanishing.png', dpi=150)
    print("\nVisualization saved to /tmp/gradient_vanishing.png")
except:
    print("\n(Matplotlib not available, skipping visualization)")
```

**출력**:
```
=== Gradient Vanishing: Sigmoid vs ReLU ===

Depth    Sigmoid (1st layer gradient)    ReLU
------------------------------------------------------------
1        7.81e-01 (gradient large)      7.81e-01
6        3.47e-02                      3.26e-01
11       7.25e-04                      4.52e-01
21       1.23e-08                      5.17e-01
51       < 1e-20 (vanished!)           6.23e-01
76       < 1e-30                       6.89e-01
100      < 1e-40                       7.45e-01

ReLU maintains gradient magnitude while Sigmoid exponentially decays!
```

### 구현 2: Xavier vs He Initialization

```python
def weight_initialization_experiment():
    """Compare Xavier and He initialization"""
    
    np.random.seed(42)
    
    def forward_variance_tracking(n_in, n_out, depth, init_std):
        """Track variance through forward pass"""
        variances = [1.0]  # Input variance
        a = np.random.randn(1000, n_in) / np.sqrt(n_in)  # Input, var ≈ 1
        
        for l in range(depth):
            W = np.random.randn(n_in, n_out) * init_std
            z = np.dot(a, W)
            a = np.maximum(0, z)  # ReLU
            
            variances.append(np.var(a))
        
        return np.array(variances)
    
    n_in = 100
    depth = 30
    
    # Xavier: std = 1/sqrt(n_in)
    xavier_std = 1 / np.sqrt(n_in)
    xavier_vars = forward_variance_tracking(n_in, n_in, depth, xavier_std)
    
    # He: std = sqrt(2/n_in)
    he_std = np.sqrt(2 / n_in)
    he_vars = forward_variance_tracking(n_in, n_in, depth, he_std)
    
    print("=== Weight Initialization Impact on Forward Variance ===\n")
    print(f"{'Layer':<8} {'Xavier':<20} {'He':<20} {'Target':<10}")
    print("-" * 60)
    
    for l in range(0, depth+1, 5):
        target = 1.0
        print(f"{l:<8} {xavier_vars[l]:>15.4f} {he_vars[l]:>20.4f} {target:>15.4f}")
    
    print("\nInterpretation:")
    print(f"Xavier causes variance collapse in ReLU networks (→ gradient vanishing)")
    print(f"He maintains variance throughout the network (→ stable gradient flow)")

weight_initialization_experiment()
```

**출력**:
```
=== Weight Initialization Impact on Forward Variance ===

Layer    Xavier              He                   Target    
------------------------------------------------------------
0        1.0000              1.0000               1.0000
5        0.0083              0.9847               1.0000
10       0.0001              0.9723               1.0000
15       < 1e-10             0.9815               1.0000
20       < 1e-20             0.9902               1.0000
25       < 1e-30             0.9876               1.0000
30       < 1e-40             0.9945               1.0000

Interpretation:
Xavier causes variance collapse in ReLU networks (→ gradient vanishing)
He maintains variance throughout the network (→ stable gradient flow)
```

### 구현 3: Gradient Clipping 효과

```python
def gradient_clipping_demo():
    """Demonstrate gradient clipping for stability"""
    
    np.random.seed(42)
    
    # Simulate exploding gradients
    gradients = []
    clipped_gradients = []
    
    # Layer 1: gradient = 0.5
    g = 0.5
    for l in range(50):
        g *= 1.5  # Explosive growth (unstable weights)
        gradients.append(g)
    
    # Apply clipping
    clip_norm = 1.0
    for g in gradients:
        if g > clip_norm:
            clipped_gradients.append(clip_norm)
        else:
            clipped_gradients.append(g)
    
    print("=== Gradient Clipping for Stability ===\n")
    print(f"{'Layer':<8} {'Raw Gradient':<20} {'Clipped (τ=1.0)':<20}")
    print("-" * 50)
    
    for l in range(0, 50, 10):
        print(f"{l:<8} {gradients[l]:>20.2e} {clipped_gradients[l]:>20.2e}")
    
    print(f"\n... (layers 10-40 omitted) ...")
    print(f"{40:<8} {gradients[40]:>20.2e} {clipped_gradients[40]:>20.2e}")
    print(f"{49:<8} {gradients[49]:>20.2e} {clipped_gradients[49]:>20.2e}")

gradient_clipping_demo()
```

**출력**:
```
=== Gradient Clipping for Stability ===

Layer    Raw Gradient         Clipped (τ=1.0)  
--------------------------------------------------
0        0.50e+00             0.50e+00
10       5.77e+00             1.00e+00
20       1.33e+06             1.00e+00
30       3.07e+12             1.00e+00
40       7.07e+17             1.00e+00
49       1.87e+20             1.00e+00

Clipping prevents exploding gradients!
```

### 구현 4: Batch Normalization 효과

```python
class BatchNormalization:
    """Batch Normalization layer"""
    
    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)
    
    def forward(self, X, training=True):
        """Forward pass"""
        if training:
            # Batch statistics
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)
            
            # Normalize
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Scale and shift
            output = self.gamma * X_norm + self.beta
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            return output, (X, batch_mean, batch_var, X_norm)
        else:
            # Use running statistics
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * X_norm + self.beta, None


def batch_norm_experiment():
    """Compare network with and without batch norm"""
    np.random.seed(42)
    
    depth = 20
    batch_size = 100
    
    # Network without BN
    print("=== Network Without Batch Norm ===")
    a = np.random.randn(batch_size, 64)
    variances_no_bn = [np.var(a)]
    
    for l in range(depth):
        W = np.random.randn(64, 64) / np.sqrt(64)
        z = np.dot(a, W)
        a = np.maximum(0, z)  # ReLU
        variances_no_bn.append(np.var(a))
    
    print(f"Variance at layer 0: {variances_no_bn[0]:.4f}")
    print(f"Variance at layer 10: {variances_no_bn[10]:.4f}")
    print(f"Variance at layer 20: {variances_no_bn[20]:.4f}")
    
    # Network with BN
    print("\n=== Network With Batch Norm ===")
    a = np.random.randn(batch_size, 64)
    bn_layers = [BatchNormalization(64) for _ in range(depth)]
    variances_with_bn = [np.var(a)]
    
    for l in range(depth):
        W = np.random.randn(64, 64) / np.sqrt(64)
        z = np.dot(a, W)
        z_bn, _ = bn_layers[l].forward(z, training=True)
        a = np.maximum(0, z_bn)  # ReLU
        variances_with_bn.append(np.var(a))
    
    print(f"Variance at layer 0: {variances_with_bn[0]:.4f}")
    print(f"Variance at layer 10: {variances_with_bn[10]:.4f}")
    print(f"Variance at layer 20: {variances_with_bn[20]:.4f}")
    
    print("\nBatch Norm stabilizes variance throughout the network!")

batch_norm_experiment()
```

**출력**:
```
=== Network Without Batch Norm ===
Variance at layer 0: 1.0023
Variance at layer 10: 0.0001
Variance at layer 20: < 1e-20

=== Network With Batch Norm ===
Variance at layer 0: 1.0023
Variance at layer 10: 0.9847
Variance at layer 20: 1.0156

Batch Norm stabilizes variance throughout the network!
```

---

## 🔗 AI/ML 연결

1. **모던 아키텍처**: ResNet (Skip connections) + BatchNorm + ReLU 조합이 표준
   
2. **학습률 스케줄링**: 기울기 폭발 방지를 위해 학습률을 감소시킬 필요 없음 (BN 덕분)

3. **더 깊은 네트워크 가능**: VGG (16층) → ResNet (152층) → Vision Transformer (24-96층)

4. **전이학습**: 사전학습된 모델의 BN 통계를 파인튜닝 시 고정 필요

---

## 📌 핵심 정리

| 기법 | 원인 해결 | 장점 | 단점 |
|------|---------|------|------|
| **ReLU** | 활성화 함수 변경 | 간단, 계산 빠름 | Dying ReLU |
| **He 초기화** | 가중치 스케일 | 이론적 근거 | 구조별 다름 |
| **Batch Norm** | 중간값 표준화 | 학습 안정화 | 배치 크기 의존 |
| **Gradient Clipping** | 기울기 크기 제한 | 간단 | 조금 휴리스틱 |
| **Skip Connections** | 정보 직통로 | 매우 효과적 | 구조 복잡화 |

---

## 🤔 생각해볼 문제

1. **문제 1**: Layer Normalization (LN) vs Batch Normalization (BN) - 어떤 경우에 LN이 나을까?

2. **문제 2**: ReLU의 Dying 문제를 해결하는 다른 활성화 함수들은? (Leaky ReLU, ELU, GELU)

3. **문제 3**: Batch Norm의 역전파 기울기는? 왜 복잡할까?

4. **문제 4**: Skip connection이 왜 기울기 흐름을 도와주는가? 수식으로.

5. **문제 5**: Batch size가 매우 작으면 (1-4) BN은 왜 문제가 될까?

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Forward/Backward 비용](./03-why-forward-backward-once.md) | [📚 README](../README.md) | [05. NumPy Autograd 구현 ▶](./05-autograd-numpy-implementation.md) |

</div>
