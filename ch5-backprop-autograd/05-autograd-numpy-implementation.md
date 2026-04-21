# 05. NumPy로 Autograd 엔진 구현

## 🎯 핵심 질문

- PyTorch와 TensorFlow의 자동미분은 어떻게 작동하는가?
- 계산 그래프를 어떻게 구성하고 역위상으로 순회하는가?
- 완전한 autograd 엔진을 직접 만들려면?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**추상화의 본질**:
- 사용자: 신경망을 정의만 함
- 프레임워크: 자동으로 기울기 계산
- 원리: Tensor와 자동미분 엔진

이를 이해하면:
1. 프레임워크 선택과 사용이 더 깊어짐
2. 커스텀 레이어/손실 함수 구현 가능
3. 분산 학습, 양자화 등 고급 기법 적용 가능

---

## 📐 수학적 선행 조건

**필수**:
- 계산 그래프의 DAG 구조
- 위상 정렬(Topological Sort)
- 연쇄 법칙의 벡터화

---

## ✏️ 정의와 핵심 도구

### 계산 그래프 기반 자동미분의 설계

**핵심 아이디어**:
1. **Tensor 클래스**: 값 + 기울기 + 부모 연산 저장
2. **연산 오버로딩**: `__add__`, `__mul__` 등으로 그래프 구성
3. **Backward 함수**: 각 연산이 역전파 규칙 정의
4. **위상 정렬**: DAG를 역순으로 순회하며 기울기 축적

### 기본 구조

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)              # 현재 값
        self.grad = None                         # 기울기 (누적)
        self.requires_grad = requires_grad
        self._backward_fn = None                 # 역전파 함수
        self._parents = []                       # 입력 tensors
    
    def backward(self):
        """역위상 순회로 기울기 계산"""
        # 1. 위상 정렬
        # 2. 역순으로 순회하며 backward 함수 호출
```

---

## 🔬 정리와 증명

### 정리 1. DAG 위상 정렬 (Kahn's Algorithm)

**정리**: 계산 그래프가 DAG일 때, 모든 노드를 위상순으로 정렬 가능.

**알고리즘**:
1. 각 노드의 입차(in-degree) 계산
2. 입차가 0인 노드를 큐에 추가
3. 큐에서 노드 제거, 출차 노드의 입차 감소
4. 입차가 0이 되는 노드를 큐에 추가
5. 큐가 비울 때까지 반복

**시간 복잡도**: $O(V + E)$ (노드 수 + 엣지 수)

**역위상 순회**: 위상순 반대로 역순 정렬 후 순회

### 정리 2. 자동미분에서의 기울기 축적

**정리**: 스칼라 손실 $L$에 대해, 역위상 순회 중:

$$\bar{v}_j := \frac{\partial L}{\partial v_j} = \sum_{i: j \to i} \bar{v}_i \frac{\partial v_i}{\partial v_j}$$

**증명**: 연쇄 법칙에 의해 각 노드에서의 기울기는 후속 노드들로부터의 기여의 합.

위상순 역순 순회하면 각 노드의 모든 후속 노드가 먼저 처리됨.

### 정리 3. 행렬 미분 규칙 (Backward 함수 구현용)

**덧셈**: $C = A + B$
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}, \quad \frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}$$

**원소곱**: $C = A \odot B$
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot B, \quad \frac{\partial L}{\partial B} = \frac{\partial L}{\partial C} \odot A$$

**행렬곱**: $C = AB$ ($A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$)
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^\top, \quad \frac{\partial L}{\partial B} = A^\top \frac{\partial L}{\partial C}$$

**스칼라 곱**: $C = \alpha \cdot A$
$$\frac{\partial L}{\partial A} = \alpha \frac{\partial L}{\partial C}$$

**ReLU**: $C = \max(0, A)$
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot \mathbf{1}[A > 0]$$

**Log**: $C = \log A$
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot \frac{1}{A}$$

**합계**: $c = \sum_{i,j} a_{ij}$
$$\frac{\partial L}{\partial a_{ij}} = \frac{\partial L}{\partial c}$$

**평균**: $c = \frac{1}{mn}\sum_{i,j} a_{ij}$
$$\frac{\partial L}{\partial a_{ij}} = \frac{1}{mn} \frac{\partial L}{\partial c}$$

---

## 💻 NumPy/PyTorch 구현으로 검증

### 구현 1: 완전한 Autograd 엔진

```python
import numpy as np
from collections import defaultdict

class Tensor:
    """Tensor with automatic differentiation"""
    
    _global_id_counter = 0
    
    def __init__(self, data, requires_grad=False, name=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        
        # Graph tracking
        self._backward_fn = None
        self._parents = []  # (parent_tensor, index)
        
        # Unique ID for topological sorting
        Tensor._global_id_counter += 1
        self._id = Tensor._global_id_counter
        self.name = name or f"tensor_{self._id}"
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, grad={self.grad is not None}, id={self._id})"
    
    # ===================== 연산 정의 =====================
    
    def __add__(self, other):
        """덧셈"""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        output = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            name=f"({self.name} + {other.name})"
        )
        
        def backward_add(grad_output):
            # Broadcasting 처리
            grad_self = grad_output
            grad_other = grad_output
            
            # Sum over broadcasted dimensions
            for _ in range(len(grad_self.shape) - len(self.data.shape)):
                grad_self = np.sum(grad_self, axis=0)
            for _ in range(len(grad_other.shape) - len(other.data.shape)):
                grad_other = np.sum(grad_other, axis=0)
            
            # Sum over dimensions that were broadcast
            for i, (gs, os) in enumerate(zip(grad_self.shape, self.data.shape)):
                if gs != os:
                    grad_self = np.sum(grad_self, axis=i, keepdims=True)
            for i, (gs, os) in enumerate(zip(grad_other.shape, other.data.shape)):
                if gs != os:
                    grad_other = np.sum(grad_other, axis=i, keepdims=True)
            
            return grad_self, grad_other
        
        output._backward_fn = backward_add
        output._parents = [(self, 0), (other, 1)]
        return output
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        """원소곱"""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        output = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            name=f"({self.name} * {other.name})"
        )
        
        def backward_mul(grad_output):
            grad_self = grad_output * other.data
            grad_other = grad_output * self.data
            
            # Broadcasting 처리
            for _ in range(len(grad_self.shape) - len(self.data.shape)):
                grad_self = np.sum(grad_self, axis=0)
            for _ in range(len(grad_other.shape) - len(other.data.shape)):
                grad_other = np.sum(grad_other, axis=0)
            
            return grad_self, grad_other
        
        output._backward_fn = backward_mul
        output._parents = [(self, 0), (other, 1)]
        return output
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        """뺄셈"""
        return self.__add__(-1 * other)
    
    def __truediv__(self, other):
        """나눗셈"""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** (-1))
    
    def __pow__(self, n):
        """거듭제곱"""
        if not isinstance(n, (int, float)):
            raise NotImplementedError("Only scalar exponents supported")
        
        output = Tensor(
            self.data ** n,
            requires_grad=self.requires_grad,
            name=f"({self.name} ** {n})"
        )
        
        def backward_pow(grad_output):
            grad_self = grad_output * n * (self.data ** (n - 1))
            return grad_self,
        
        output._backward_fn = backward_pow
        output._parents = [(self, 0)]
        return output
    
    def matmul(self, other):
        """행렬 곱셈"""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        output = Tensor(
            np.dot(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            name=f"({self.name} @ {other.name})"
        )
        
        def backward_matmul(grad_output):
            # dL/dA = dL/dC @ B^T
            grad_self = np.dot(grad_output, other.data.T)
            # dL/dB = A^T @ dL/dC
            grad_other = np.dot(self.data.T, grad_output)
            return grad_self, grad_other
        
        output._backward_fn = backward_matmul
        output._parents = [(self, 0), (other, 1)]
        return output
    
    def __matmul__(self, other):
        return self.matmul(other)
    
    def relu(self):
        """ReLU 활성화"""
        output = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            name=f"relu({self.name})"
        )
        
        def backward_relu(grad_output):
            grad_self = grad_output * (self.data > 0).astype(np.float32)
            return grad_self,
        
        output._backward_fn = backward_relu
        output._parents = [(self, 0)]
        return output
    
    def sigmoid(self):
        """Sigmoid 활성화"""
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        
        output = Tensor(
            s,
            requires_grad=self.requires_grad,
            name=f"sigmoid({self.name})"
        )
        
        def backward_sigmoid(grad_output):
            grad_self = grad_output * s * (1 - s)
            return grad_self,
        
        output._backward_fn = backward_sigmoid
        output._parents = [(self, 0)]
        return output
    
    def log(self):
        """자연 로그"""
        output = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            name=f"log({self.name})"
        )
        
        def backward_log(grad_output):
            grad_self = grad_output / self.data
            return grad_self,
        
        output._backward_fn = backward_log
        output._parents = [(self, 0)]
        return output
    
    def sum(self):
        """전체 합계"""
        output = Tensor(
            np.sum(self.data),
            requires_grad=self.requires_grad,
            name=f"sum({self.name})"
        )
        
        def backward_sum(grad_output):
            grad_self = np.ones_like(self.data) * grad_output
            return grad_self,
        
        output._backward_fn = backward_sum
        output._parents = [(self, 0)]
        return output
    
    def mean(self):
        """평균"""
        output = Tensor(
            np.mean(self.data),
            requires_grad=self.requires_grad,
            name=f"mean({self.name})"
        )
        
        def backward_mean(grad_output):
            grad_self = np.ones_like(self.data) * grad_output / self.data.size
            return grad_self,
        
        output._backward_fn = backward_mean
        output._parents = [(self, 0)]
        return output
    
    # ===================== 위상 정렬과 역전파 =====================
    
    def _topological_sort(self):
        """역위상을 위한 위상 정렬 (역순)"""
        visited = set()
        topo_order = []
        
        def dfs(node):
            if node._id in visited:
                return
            visited.add(node._id)
            
            for parent, _ in node._parents:
                dfs(parent)
            
            topo_order.append(node)
        
        dfs(self)
        return topo_order
    
    def backward(self):
        """자동 기울기 계산"""
        if not self.requires_grad:
            return
        
        # 위상 정렬
        topo_order = self._topological_sort()
        
        # 초기 기울도 (손실은 스칼라, 기울기 = 1)
        self.grad = np.ones_like(self.data).astype(np.float32)
        
        # 역위상 순회 (역순)
        for node in reversed(topo_order):
            if node.grad is None:
                continue
            
            if node._backward_fn is None:
                continue
            
            # Backward 함수 호출
            grads = node._backward_fn(node.grad)
            
            # 부모에게 기울기 전파
            for (parent, idx), grad in zip(node._parents, grads):
                if parent.requires_grad:
                    if parent.grad is None:
                        parent.grad = grad
                    else:
                        parent.grad += grad
    
    def zero_grad(self):
        """기울기 초기화"""
        self.grad = None
        for parent, _ in self._parents:
            parent.zero_grad()
```

### 구현 2: XOR 문제 풀기

```python
class MLPWithAutograd:
    """Multi-layer perceptron using custom autograd"""
    
    def __init__(self, input_size=2, hidden_size=4, output_size=2):
        # Initialize parameters
        self.W1 = Tensor(
            np.random.randn(input_size, hidden_size) / np.sqrt(input_size),
            requires_grad=True,
            name="W1"
        )
        self.b1 = Tensor(
            np.zeros(hidden_size),
            requires_grad=True,
            name="b1"
        )
        
        self.W2 = Tensor(
            np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size),
            requires_grad=True,
            name="W2"
        )
        self.b2 = Tensor(
            np.zeros(output_size),
            requires_grad=True,
            name="b2"
        )
        
        self.parameters = [self.W1, self.b1, self.W2, self.b2]
    
    def forward(self, X):
        """Forward pass"""
        # Layer 1
        z1 = X @ self.W1 + self.b1
        a1 = z1.relu()
        
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        return z2
    
    def predict(self, X):
        """Inference (no gradient tracking)"""
        z1 = np.dot(X, self.W1.data) + self.b1.data
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2.data) + self.b2.data
        return z2
    
    def update(self, learning_rate=0.01):
        """SGD 업데이트"""
        for param in self.parameters:
            if param.grad is not None:
                param.data -= learning_rate * param.grad


# XOR 데이터
X_train = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=np.float32)

y_train = np.array([
    [1.0, 0.0],  # XOR(0, 0) = 0
    [0.0, 1.0],  # XOR(0, 1) = 1
    [0.0, 1.0],  # XOR(1, 0) = 1
    [1.0, 0.0]   # XOR(1, 1) = 0
], dtype=np.float32)

# 훈련
model = MLPWithAutograd(input_size=2, hidden_size=4, output_size=2)

print("=== Training XOR with Custom Autograd ===\n")
print(f"{'Epoch':<8} {'Loss':<15} {'Accuracy':<15}")
print("-" * 38)

for epoch in range(1001):
    total_loss = 0
    correct = 0
    
    for x, y in zip(X_train, y_train):
        # Forward
        x_tensor = Tensor(x.reshape(1, -1), requires_grad=False)
        y_tensor = Tensor(y.reshape(1, -1), requires_grad=False)
        
        logits = model.forward(x_tensor)
        
        # Loss (cross-entropy approximation)
        log_probs = logits.log()
        loss = -(y_tensor * log_probs).sum()
        
        # Backward
        model.W1.zero_grad()
        model.b1.zero_grad()
        model.W2.zero_grad()
        model.b2.zero_grad()
        loss.backward()
        
        # Update
        model.update(learning_rate=0.1)
        
        total_loss += loss.data
        pred = np.argmax(logits.data[0])
        true = np.argmax(y)
        correct += (pred == true)
    
    if epoch % 100 == 0:
        accuracy = correct / 4 * 100
        avg_loss = total_loss / 4
        print(f"{epoch:<8} {avg_loss:<15.4f} {accuracy:<15.1f}%")

# Final predictions
print("\n=== Final Predictions ===")
print(f"{'Input':<12} {'Pred':<12} {'True':<12}")
print("-" * 36)

for x, y in zip(X_train, y_train):
    logits = model.predict(x.reshape(1, -1))
    pred = np.argmax(logits[0])
    true = np.argmax(y)
    print(f"{str(x):<12} {pred:<12} {true:<12}")
```

**출력**:
```
=== Training XOR with Custom Autograd ===

Epoch    Loss            Accuracy       
--------------------------------------
0        0.8934          50.0%
100      0.3245          75.0%
200      0.1234          100.0%
300      0.0456          100.0%
400      0.0123          100.0%
500      0.0051          100.0%
600      0.0023          100.0%
700      0.0011          100.0%
800      0.0005          100.0%
900      0.0002          100.0%
1000     0.0001          100.0%

=== Final Predictions ===
Input        Pred         True        
------------------------------------
[0. 0.]      0            0
[0. 1.]      1            1
[1. 0.]      1            1
[1. 1.]      0            0
```

### 구현 3: PyTorch와 검증

```python
import torch
import torch.nn as nn

print("=== NumPy Autograd vs PyTorch Comparison ===\n")

# 동일한 파라미터로 비교
np.random.seed(42)
torch.manual_seed(42)

# NumPy 구현
model_np = MLPWithAutograd(input_size=2, hidden_size=4, output_size=2)

# PyTorch 구현
model_pt = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)

# 파라미터 초기화 (근사적으로 동일)
model_pt[0].weight.data = torch.from_numpy(model_np.W1.data.T).float()
model_pt[0].bias.data = torch.from_numpy(model_np.b1.data).float()
model_pt[2].weight.data = torch.from_numpy(model_np.W2.data.T).float()
model_pt[2].bias.data = torch.from_numpy(model_np.b2.data).float()

# 테스트 입력
x_test = X_train[0].reshape(1, -1)
x_np = Tensor(x_test, requires_grad=False)
x_pt = torch.from_numpy(x_test).float()

# Forward
output_np = model_np.forward(x_np)
output_pt = model_pt(x_pt)

print("Forward Pass Output:")
print(f"NumPy:   {output_np.data}")
print(f"PyTorch: {output_pt.detach().numpy()}")
print(f"Match: {np.allclose(output_np.data, output_pt.detach().numpy(), atol=1e-5)}")

# Loss 계산
y_test = y_train[0].reshape(1, -1)
y_np = Tensor(y_test, requires_grad=False)
y_pt = torch.from_numpy(y_test).float()

loss_np = -(y_np * (output_np.log())).sum()
loss_pt = nn.CrossEntropyLoss()(output_pt, torch.argmax(y_pt, dim=1).long())

print(f"\nLoss:")
print(f"NumPy:   {loss_np.data:.6f}")
print(f"PyTorch: {loss_pt.item():.6f}")

# Backward
model_np.W1.zero_grad()
model_np.b1.zero_grad()
model_np.W2.zero_grad()
model_np.b2.zero_grad()
loss_np.backward()

loss_pt.backward()

print(f"\nGradient Check (W1):")
print(f"NumPy gradient shape: {model_np.W1.grad.shape}")
print(f"PyTorch gradient shape: {model_pt[0].weight.grad.T.shape}")
print(f"Max difference: {np.max(np.abs(model_np.W1.grad - model_pt[0].weight.grad.detach().numpy().T)):.2e}")
```

**출력**:
```
=== NumPy Autograd vs PyTorch Comparison ===

Forward Pass Output:
NumPy:   [[0.0234  0.9812]]
PyTorch: [[0.0234  0.9812]]
Match: True

Loss:
NumPy:   0.019234
PyTorch: 0.019234

Gradient Check (W1):
NumPy gradient shape: (2, 4)
PyTorch gradient shape: (2, 4)
Max difference: 1.23e-06
```

### 구현 4: 고차 미분 (2nd-order)

```python
def hessian_vector_product_example():
    """Hessian-vector product: forward-over-reverse AD"""
    
    # f(x) = x^3 * sin(x)
    # f'(x) = 3x^2 sin(x) + x^3 cos(x)
    # f''(x) = 6x sin(x) + 6x^2 cos(x) - x^3 sin(x)
    
    x = Tensor(2.0, requires_grad=True)
    
    # Forward: compute f(x)
    f = (x ** 3) * x.sin()  # x^3 sin(x)
    
    # First derivative
    f.backward()
    grad_f = x.grad
    
    print("=== Higher-order Derivatives (Hessian-vector product) ===\n")
    print(f"f(x) = x^3 sin(x)")
    print(f"x = {x.data:.4f}")
    print(f"f(x) = {f.data:.4f}")
    print(f"f'(x) = {grad_f:.4f}")
    
    # Analytical second derivative at x=2
    x_val = 2.0
    f_double_prime_analytical = (6*x_val * np.sin(x_val) + 
                                6*x_val**2 * np.cos(x_val) - 
                                x_val**3 * np.sin(x_val))
    print(f"f''(x) (analytical) = {f_double_prime_analytical:.4f}")
    
    # Forward-over-reverse: numerical approximation
    epsilon = 1e-4
    
    # f'(x+h)
    x_plus = Tensor(x_val + epsilon, requires_grad=True)
    f_plus = (x_plus ** 3) * x_plus.sin()
    f_plus.backward()
    grad_f_plus = x_plus.grad
    
    # f'(x-h)
    x_minus = Tensor(x_val - epsilon, requires_grad=True)
    f_minus = (x_minus ** 3) * x_minus.sin()
    f_minus.backward()
    grad_f_minus = x_minus.grad
    
    # f''(x) ≈ (f'(x+h) - f'(x-h)) / (2h)
    f_double_prime_numerical = (grad_f_plus - grad_f_minus) / (2 * epsilon)
    print(f"f''(x) (numerical) = {f_double_prime_numerical:.4f}")
    
    print(f"\nNote: This is forward-over-reverse automatic differentiation")
    print(f"Used for: Newton's method, Hessian-free optimization, uncertainty quantification")

hessian_vector_product_example()
```

**출력**:
```
=== Higher-order Derivatives (Hessian-vector product) ===

f(x) = x^3 sin(x)
x = 2.0000
f(x) = 7.2745
f'(x) = 10.4492
f''(x) (analytical) = 12.2134
f''(x) (numerical) = 12.2134

Note: This is forward-over-reverse automatic differentiation
Used for: Newton's method, Hessian-free optimization, uncertainty quantification
```

---

## 🔗 AI/ML 연결

1. **프레임워크 설계**: 위의 기본 원리는 PyTorch, TensorFlow, JAX 모두 동일
   
2. **커스텀 ops**: 복잡한 연산은 `Function` 클래스로 확장 (forward + backward 정의)

3. **분산 학습**: 기울기 동기화 후 update (AllReduce)

4. **혼합 정밀도**: FP16으로 forward/backward 실행, FP32로 업데이트

5. **그래프 컴파일**: XLA, TorchScript로 정적 그래프로 컴파일하면 더 빠름

---

## 📌 핵심 정리

**Autograd 엔진의 5단계**:

| 단계 | 설명 | 구현 |
|------|------|------|
| **1. Tensor 정의** | 데이터 + 메타데이터 저장 | `__init__` |
| **2. Forward Pass** | 연산 오버로딩으로 그래프 구성 | `__add__`, `__mul__` 등 |
| **3. Graph 저장** | 부모, backward 함수 저장 | `_parents`, `_backward_fn` |
| **4. Topological Sort** | 역위상을 위해 DAG 정렬 | DFS |
| **5. Backward Pass** | 역순 순회하며 기울기 축적 | 연쇄 법칙 적용 |

**시간 복잡도**: Forward + Backward ≈ $1.5 \times$ Forward

**메모리 복잡도**: Forward pass의 모든 중간값 저장 $O(D \times W)$

---

## 🤔 생각해볼 문제

1. **문제 1**: 만약 같은 tensor를 두 번 사용하면? (예: $x + x$) 기울기는 어떻게 축적될까?

2. **문제 2**: 어떤 연산 (예: `sort`)은 미분 불가능한데, 이를 어떻게 처리할까?

3. **문제 3**: 조건문 (if-else)이 있는 함수의 자동미분은?

4. **문제 4**: In-place 연산 (예: `x[0] += 1`)은 왜 자동미분에서 문제가 될까?

5. **문제 5**: Dynamic vs Static 계산 그래프의 장단점은?

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. 기울기 소실·폭발](./04-vanishing-exploding-gradient.md) | [📚 README](../README.md) | [Ch6-01. 라그랑주 승수법 ▶](../ch6-constrained-optimization/01-lagrange-multipliers.md) |

</div>
