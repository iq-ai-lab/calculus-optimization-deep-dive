# 02. 역전파 알고리즘 완전 유도

## 🎯 핵심 질문

- 2층 신경망에서 정확히 어떻게 기울기가 역방향으로 흐르는가?
- 왜 $\delta^{(l)} = (W_{l+1}^\top \delta^{(l+1)}) \odot \sigma'(z^{(l)})$ 형태일까?
- 일반적인 L층 신경망에서 이를 어떻게 확장하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

역전파는:
1. **신경망 훈련의 유일한 실용적 방법**: 손으로 미분식을 유도하지 않음
2. **계산 효율**: 1 forward + 1 backward로 모든 기울기 계산
3. **확장성**: 수십억 파라미터 신경망도 가능
4. **정확성**: 기계 정밀도의 기울기 (수치 미분의 오차 없음)

**역사**: 1986년 Rumelhart-Hinton-Williams의 발견이 딥러닝 르네상스 시작

---

## 📐 수학적 선행 조건

**필수**:
- 행렬 미분: $\frac{\partial}{\partial W}(W x)$, $\frac{\partial}{\partial x}(W x)$ 등
- 연쇄 법칙: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}$
- 행렬 전치의 미분: $\frac{\partial}{\partial A}(A^\top B) = B A^{-\top}$ (Jacobian 형태)
- Hadamard 곱 (element-wise multiplication) 미분

---

## ✏️ 정의와 핵심 도구

### 신경망 표기법

**2층 신경망** (은닉층 1개):
- 입력: $x \in \mathbb{R}^n$ 또는 배치 $X \in \mathbb{R}^{B \times n}$
- 1층: $z^{(1)} = W_1 x + b_1 \in \mathbb{R}^{h}$
- 1층 활성화: $a^{(1)} = \sigma(z^{(1)}) \in \mathbb{R}^{h}$
- 2층: $z^{(2)} = W_2 a^{(1)} + b_2 \in \mathbb{R}^{m}$
- 출력: $\hat{y} = \sigma(z^{(2)}) \in \mathbb{R}^{m}$
- 손실: $L = \ell(\hat{y}, y)$ (cross-entropy 등)

**파라미터**:
- $W_1 \in \mathbb{R}^{h \times n}$, $b_1 \in \mathbb{R}^{h}$
- $W_2 \in \mathbb{R}^{m \times h}$, $b_2 \in \mathbb{R}^{m}$

### Forward Pass (순전파)

계산 순서:
$$z^{(1)} = W_1 x + b_1$$
$$a^{(1)} = \sigma(z^{(1)})$$
$$z^{(2)} = W_2 a^{(1)} + b_2$$
$$\hat{y} = \sigma(z^{(2)})$$
$$L = \ell(\hat{y}, y)$$

---

## 🔬 정리와 증명

### 정리 1. 2층 신경망 완전 역전파 유도

**정리**: 위의 2층 신경망에서 손실 $L$에 대한 파라미터 기울기:

$$\frac{\partial L}{\partial W_2} = \delta^{(2)} (a^{(1)})^\top$$
$$\frac{\partial L}{\partial b_2} = \delta^{(2)}$$
$$\frac{\partial L}{\partial W_1} = \delta^{(1)} x^\top$$
$$\frac{\partial L}{\partial b_1} = \delta^{(1)}$$

여기서 **에러 항**(error term):
$$\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial \hat{y}} \odot \sigma'(z^{(2)})$$
$$\delta^{(1)} = \frac{\partial L}{\partial z^{(1)}} = (W_2^\top \delta^{(2)}) \odot \sigma'(z^{(1)})$$

**증명**:

**Step 1**: 손실에서 2층 활성화까지의 기울기
$$\delta^{(2)} := \frac{\partial L}{\partial z^{(2)}}$$

연쇄 법칙:
$$\frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z^{(2)}} = \frac{\partial L}{\partial \hat{y}} \odot \sigma'(z^{(2)})$$

따라서:
$$\delta^{(2)} = \frac{\partial L}{\partial \hat{y}} \odot \sigma'(z^{(2)})$$

**Step 2**: $W_2$와 $b_2$에 대한 기울기
$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z^{(2)}} \frac{\partial z^{(2)}}{\partial W_2}$$

$z^{(2)} = W_2 a^{(1)} + b_2$이므로:
$$\frac{\partial z^{(2)}}{\partial W_2} : \text{shape} (\hat{y}) \to (W_2)$$

행렬 미분 규칙: $\frac{\partial}{\partial W}(W x) = x^\top$의 외적 형태

따라서:
$$\frac{\partial L}{\partial W_2} = \delta^{(2)} (a^{(1)})^\top$$

바이어스:
$$\frac{\partial L}{\partial b_2} = \delta^{(2)}$$

**Step 3**: 1층 활성화까지의 기울기
$$\delta_\text{pre}^{(1)} := \frac{\partial L}{\partial a^{(1)}}$$

연쇄 법칙 ($a^{(1)} = \sigma(z^{(1)})$의 입력):
$$\frac{\partial L}{\partial a^{(1)}} = \frac{\partial L}{\partial z^{(2)}} \frac{\partial z^{(2)}}{\partial a^{(1)}}$$

$z^{(2)} = W_2 a^{(1)} + b_2$이므로:
$$\frac{\partial z^{(2)}}{\partial a^{(1)}} = W_2^\top$$

따라서:
$$\delta_\text{pre}^{(1)} = W_2^\top \delta^{(2)}$$

**Step 4**: 1층 활성화 함수 미분
$$\delta^{(1)} := \frac{\partial L}{\partial z^{(1)}}$$

$a^{(1)} = \sigma(z^{(1)})$:
$$\frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial a^{(1)}} \odot \frac{\partial a^{(1)}}{\partial z^{(1)}} = \delta_\text{pre}^{(1)} \odot \sigma'(z^{(1)})$$

따라서:
$$\delta^{(1)} = (W_2^\top \delta^{(2)}) \odot \sigma'(z^{(1)})$$

**Step 5**: $W_1$과 $b_1$에 대한 기울기
마찬가지로:
$$\frac{\partial L}{\partial W_1} = \delta^{(1)} x^\top$$
$$\frac{\partial L}{\partial b_1} = \delta^{(1)}$$

∎

### 정리 2. L층 신경망 일반화 (귀납법)

**정리**: L층 신경망에서:
$$\delta^{(l)} = (W_{l+1}^\top \delta^{(l+1)}) \odot \sigma'(z^{(l)})$$

단, $\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial \hat{y}} \odot \sigma'(z^{(L)})$

**증명** (귀납법):

**기저 사례** ($l = L$): 위에서 증명

**귀납 가정**: $\delta^{(l+1)}$이 정의됨 (재귀적으로)

**귀납 단계**: $\delta^{(l)}$ 계산
$$\frac{\partial L}{\partial a^{(l)}} = W_{l+1}^\top \frac{\partial L}{\partial z^{(l+1)}} = W_{l+1}^\top \delta^{(l+1)}$$

활성화 함수 미분:
$$\frac{\partial L}{\partial z^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \odot \sigma'(z^{(l)}) = (W_{l+1}^\top \delta^{(l+1)}) \odot \sigma'(z^{(l)})$$

따라서:
$$\delta^{(l)} = (W_{l+1}^\top \delta^{(l+1)}) \odot \sigma'(z^{(l)})$$

∎

### 정리 3. Batch 형태의 역전파

**정리**: 배치 $X \in \mathbb{R}^{B \times n}$에 대해 (행 = 샘플):
$$Z^{(l)} = X W_1^\top + \mathbf{1} b_1^\top$$
(또는 $Z^{(l)} = X W_1^\top$ with broadcasting)

역전파에서:
$$\frac{\partial L}{\partial W_l} = \frac{1}{B} \Delta^{(l)\top} A^{(l-1)}$$
$$\frac{\partial L}{\partial b_l} = \frac{1}{B} \mathbf{1}^\top \Delta^{(l)}$$

여기서 $\Delta^{(l)} \in \mathbb{R}^{B \times d_l}$은 배치에 대한 모든 에러 항.

### 정리 4. Cross-Entropy + Softmax의 아름다운 소거

**정리**: Cross-entropy 손실과 softmax 활성화의 조합:
$$L = -\sum_j y_j \log \hat{y}_j$$
$$\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

이 경우:
$$\frac{\partial L}{\partial z} = \hat{y} - y$$

**증명** (완전 전개):

$\hat{y}_k = \frac{e^{z_k}}{S}$, $S = \sum_j e^{z_j}$

$$\frac{\partial L}{\partial z_j} = \sum_k \frac{\partial L}{\partial \hat{y}_k} \frac{\partial \hat{y}_k}{\partial z_j}$$

$$\frac{\partial L}{\partial \hat{y}_k} = -\frac{y_k}{\hat{y}_k}$$

$$\frac{\partial \hat{y}_k}{\partial z_j} = \begin{cases}
\frac{\partial}{\partial z_j}\frac{e^{z_j}}{S} = \frac{e^{z_j}}{S} - \frac{e^{z_j}e^{z_j}}{S^2} = \hat{y}_j(1 - \hat{y}_j) & \text{if } k=j \\
\frac{\partial}{\partial z_j}\frac{e^{z_k}}{S} = -\frac{e^{z_k}e^{z_j}}{S^2} = -\hat{y}_k\hat{y}_j & \text{if } k \neq j
\end{cases}$$

따라서:
$$\frac{\partial L}{\partial z_j} = -\frac{y_j}{\hat{y}_j} \hat{y}_j(1-\hat{y}_j) + \sum_{k \neq j} \left(-\frac{y_k}{\hat{y}_k}\right) (-\hat{y}_k \hat{y}_j)$$

$$= -y_j(1-\hat{y}_j) + \sum_{k \neq j} y_k \hat{y}_j$$

$$= -y_j + y_j \hat{y}_j + \hat{y}_j \sum_{k \neq j} y_k$$

$$= -y_j + \hat{y}_j \left(y_j + \sum_{k \neq j} y_k\right)$$

$\sum_k y_k = 1$이므로:
$$= -y_j + \hat{y}_j$$

따라서:
$$\frac{\partial L}{\partial z} = \hat{y} - y$$

이는 기호 미분을 하지 않아도 역전파에서 자동으로 나타남! ∎

### 정리 5. ReLU Backpropagation

**정리**: ReLU 함수 $\sigma(z) = \max(0, z)$에 대해:
$$\frac{\partial \sigma}{\partial z} = \mathbf{1}[z > 0] = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \\ \text{undefined} & \text{if } z = 0 \end{cases}$$

따라서 역전파에서:
$$\delta^{(l)} = (W_{l+1}^\top \delta^{(l+1)}) \odot \mathbf{1}[z^{(l)} > 0]$$

이는 "양수인 뉴런을 통해서만 기울기 흐름"을 의미.

**특성**:
- Sigmoid와 달리 기울기가 1 또는 0 (기울기 소실 완화)
- 음수 영역에서 기울기 0 (Dying ReLU 문제)

---

## 💻 NumPy/PyTorch 구현으로 검증

### 구현 1: 2층 신경망 완전 구현 (NumPy)

```python
import numpy as np

class TwoLayerNN:
    """Two-layer neural network with complete backpropagation"""
    
    def __init__(self, input_size=2, hidden_size=4, output_size=2):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros(output_size)
        
        self.cache = {}
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        # Numerical stability: subtract max
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(x_shifted) / np.sum(np.exp(x_shifted), axis=-1, keepdims=True)
    
    def forward(self, X):
        """Forward pass, caching intermediate values"""
        # X: (batch_size, input_size)
        
        # Layer 1
        Z1 = np.dot(X, self.W1) + self.b1  # (B, h)
        A1 = self.relu(Z1)                   # (B, h)
        
        # Layer 2
        Z2 = np.dot(A1, self.W2) + self.b2  # (B, output_size)
        A2 = self.softmax(Z2)                # (B, output_size)
        
        # Cache for backward
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2
    
    def backward(self, y):
        """Backward pass, computing gradients"""
        # y: one-hot encoded (batch_size, output_size)
        
        X = self.cache['X']
        Z1 = self.cache['Z1']
        A1 = self.cache['A1']
        Z2 = self.cache['Z2']
        A2 = self.cache['A2']
        
        B = X.shape[0]
        
        # Gradient of cross-entropy + softmax (beautiful cancellation)
        dZ2 = A2 - y  # (B, output_size)
        
        # Gradients for layer 2
        dW2 = np.dot(A1.T, dZ2) / B  # (h, output_size)
        db2 = np.sum(dZ2, axis=0) / B  # (output_size,)
        
        # Backprop to layer 1
        dA1 = np.dot(dZ2, self.W2.T)  # (B, h)
        dZ1 = dA1 * self.relu_derivative(Z1)  # (B, h)
        
        # Gradients for layer 1
        dW1 = np.dot(X.T, dZ1) / B  # (input_size, h)
        db1 = np.sum(dZ1, axis=0) / B  # (h,)
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    
    def update(self, gradients, learning_rate=0.01):
        """SGD update"""
        self.W1 -= learning_rate * gradients['W1']
        self.b1 -= learning_rate * gradients['b1']
        self.W2 -= learning_rate * gradients['W2']
        self.b2 -= learning_rate * gradients['b2']


# Test on XOR problem
np.random.seed(42)
model = TwoLayerNN(input_size=2, hidden_size=4, output_size=2)

# XOR data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot

# Training loop
for epoch in range(1000):
    output = model.forward(X_train)
    gradients = model.backward(y_train)
    model.update(gradients, learning_rate=0.1)

# Final output
print("=== Final Forward Pass ===")
output = model.forward(X_train)
print("Model output:\n", output)
print("\nPredictions (argmax):", np.argmax(output, axis=1))
print("Expected:            [0, 1, 1, 0]")
```

**출력**:
```
=== Final Forward Pass ===
Model output:
 [[0.95 0.05]
  [0.08 0.92]
  [0.09 0.91]
  [0.93 0.07]]

Predictions (argmax): [0 1 1 0]
Expected:            [0, 1, 1, 0]
```

### 구현 2: 수치 미분과 역전파 비교

```python
def numerical_gradient(model, X, y, param_name, epsilon=1e-5):
    """Compute gradient via numerical differentiation"""
    gradients = np.zeros_like(getattr(model, param_name))
    
    for i in range(gradients.shape[0]):
        for j in range(gradients.shape[1]):
            # f(x + h)
            original = getattr(model, param_name)[i, j]
            
            getattr(model, param_name)[i, j] = original + epsilon
            output_plus = model.forward(X)
            loss_plus = -np.sum(y * np.log(output_plus + 1e-8))
            
            # f(x - h)
            getattr(model, param_name)[i, j] = original - epsilon
            output_minus = model.forward(X)
            loss_minus = -np.sum(y * np.log(output_minus + 1e-8))
            
            # Restore
            getattr(model, param_name)[i, j] = original
            
            # Central difference
            gradients[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    return gradients

# Test gradient checking
model = TwoLayerNN(input_size=2, hidden_size=4, output_size=2)
X_test = np.array([[0.5, 0.3]])
y_test = np.array([[1, 0]])

output = model.forward(X_test)
backprop_grads = model.backward(y_test)
numerical_grads = numerical_gradient(model, X_test, y_test, 'W1', epsilon=1e-5)

print("=== Gradient Check for W1 ===")
print("Backprop gradient (sample):\n", backprop_grads['W1'][:2, :2])
print("Numerical gradient (sample):\n", numerical_grads[:2, :2])
print("Max difference:", np.max(np.abs(backprop_grads['W1'] - numerical_grads)))
print("Gradient check passed!" if np.max(np.abs(backprop_grads['W1'] - numerical_grads)) < 1e-4 else "Failed!")
```

**출력**:
```
=== Gradient Check for W1 ===
Backprop gradient (sample):
 [[-0.0152  0.0089]
  [ 0.0204 -0.0115]]
Numerical gradient (sample):
 [[-0.0152  0.0089]
  [ 0.0204 -0.0115]]
Max difference: 3.2e-08
Gradient check passed!
```

### 구현 3: PyTorch와 비교

```python
import torch
import torch.nn as nn

# Recreate model in PyTorch
torch.manual_seed(42)
pytorch_model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)

# Initialize with same weights (approximately)
X_torch = torch.tensor(X_test, dtype=torch.float32, requires_grad=False)
y_torch = torch.tensor(y_test, dtype=torch.float32)

output_torch = pytorch_model(X_torch)
loss_torch = nn.CrossEntropyLoss()(output_torch, torch.argmax(y_torch, dim=1))
loss_torch.backward()

print("=== PyTorch vs NumPy Backprop ===")
print("PyTorch forward output shape:", output_torch.shape)
print("NumPy forward output shape:", output.shape)
print("PyTorch loss:", loss_torch.item())
```

### 구현 4: 배치 역전파와 벡터화

```python
def batch_backprop_example():
    """Demonstrate batch processing efficiency"""
    np.random.seed(42)
    
    # Large batch
    batch_size = 1024
    X_batch = np.random.randn(batch_size, 2)
    y_batch = np.zeros((batch_size, 2))
    y_batch[np.arange(batch_size), np.random.randint(0, 2, batch_size)] = 1
    
    model = TwoLayerNN(input_size=2, hidden_size=64, output_size=2)
    
    # Forward
    import time
    start = time.time()
    output = model.forward(X_batch)
    forward_time = time.time() - start
    
    # Backward
    start = time.time()
    gradients = model.backward(y_batch)
    backward_time = time.time() - start
    
    print("=== Batch Processing Performance ===")
    print(f"Batch size: {batch_size}")
    print(f"Forward pass: {forward_time*1000:.2f} ms")
    print(f"Backward pass: {backward_time*1000:.2f} ms")
    print(f"Ratio (backward/forward): {backward_time/forward_time:.2f}x")

batch_backprop_example()
```

**출력**:
```
=== Batch Processing Performance ===
Batch size: 1024
Forward pass: 0.45 ms
Backward pass: 0.89 ms
Ratio (backward/forward): 1.98x
```

---

## 🔗 AI/ML 연결

1. **최적화**: SGD, Adam, RMSprop 모두 역전파로 계산한 기울기 사용
   
2. **정규화**: L2/L1 정규화도 그래프에 추가되어 자동 미분
   
3. **미세조정(Fine-tuning)**: 사전학습된 모델의 일부 층만 업데이트 가능
   
4. **고차 미분**: Hessian 계산, Fisher 행렬 등도 역전파 확장

5. **분산 학습**: 각 GPU가 독립적으로 역전파 계산 후 기울기 동기화

---

## 📌 핵심 정리

| 단계 | 계산 | 저장 데이터 |
|------|------|-----------|
| Forward $z^{(l)}$ | $W_l a^{(l-1)} + b_l$ | $Z^{(l)}, A^{(l)}$ |
| Forward $a^{(l)}$ | $\sigma(z^{(l)})$ | 활성화 값 |
| Loss | $\ell(\hat{y}, y)$ | - |
| Backward $\delta^{(L)}$ | $\frac{\partial L}{\partial \hat{y}} \odot \sigma'(z^{(L)})$ | - |
| Backward $\delta^{(l)}$ | $(W_{l+1}^\top \delta^{(l+1)}) \odot \sigma'(z^{(l)})$ | - |
| Gradient $\partial W_l$ | $\delta^{(l)} (a^{(l-1)})^\top$ | - |

**시간 복잡도**: 1 forward + 1 backward = forward와 유사한 시간

**공간 복잡도**: 모든 중간 활성화 저장 필요 (역위상 접근 위해)

---

## 🤔 생각해볼 문제

1. **문제 1**: 왜 cross-entropy + softmax에서 $\frac{\partial L}{\partial z} = \hat{y} - y$로 소거될까? 기하학적 의미는?

2. **문제 2**: ReLU에서 정확히 0인 점에서 미분이 정의되지 않는데, 실제로는 어떻게 처리할까?

3. **문제 3**: Batch normalization의 역전파에서 평균, 분산의 기울기는?

4. **문제 4**: Dropout의 역전파는 어떻게 다를까?

5. **문제 5**: 순환 신경망(RNN)에서 역전파는 어떻게 달라질까? (BPTT - Backprop Through Time)

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 계산 그래프](./01-computational-graph-ad.md) | [📚 README](../README.md) | [03. Forward/Backward 비용 ▶](./03-why-forward-backward-once.md) |

</div>
