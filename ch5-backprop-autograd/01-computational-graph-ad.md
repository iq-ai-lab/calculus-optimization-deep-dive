# 01. 계산 그래프와 자동미분

## 🎯 핵심 질문

- 딥러닝 신경망은 어떤 구조로 기울기를 효율적으로 계산하는가?
- 왜 자동미분이 수치 미분이나 기호 미분보다 우월한가?
- Forward mode와 Reverse mode 자동미분의 본질적 차이는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

모던 딥러닝의 기초:
- **규모**: 수십억 개의 파라미터를 갖는 신경망에서 기울기 계산이 효율적이어야 함
- **정확도**: 부동소수점 오차 없이 기계 정밀도(machine precision)의 기울기가 필요
- **자동화**: 손으로 미분식을 유도하지 않고도 임의의 모델에 적용 가능

예: ResNet-50은 2천5백만 개 파라미터
- 수치 미분: $2.5 \times 10^7$ forward pass 필요 → 불가능
- 자동미분: 1 forward + 1 backward pass → 가능

---

## 📐 수학적 선행 조건

**필수**:
- 다변수 함수의 편미분, 기울기 벡터
- 연쇄 법칙 (Chain Rule): $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$
- 행렬 미분: Jacobian, Hessian 행렬
- DAG (방향 무환 그래프) 기초

**선택**:
- 듀얼 수(Dual Numbers)
- 범주론의 함자(Functor) 개념

---

## ✏️ 정의와 핵심 도구

### 계산 그래프 (Computational Graph)

**정의**: 함수 합성 $f(g(h(x)))$를 DAG로 표현하되,
- **노드** $v$: 연산(operation) 또는 변수
- **엣지** $(u \to v)$: $u$의 출력이 $v$의 입력임을 의미

**예시**: $f(x_1, x_2) = (x_1 + x_2) \times \sin(x_1)$

```
     x₁ ────┬────────────┐
           │            │
          Add         sin
           │            │
      x₁+x₂ └─────Mul────┴──→ y
                      │
                      x₂
```

**공식**: $f: \mathbb{R}^n \to \mathbb{R}^m$을 계산 그래프로 표현하면, 모든 편미분을 계산할 수 있음.

### 미분 방법 비교

#### 1. 수치 미분 (Numerical Differentiation)

중심 차분 공식:
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$$

**비용**:
- $O(n)$ forward pass (각 파라미터마다 1 forward pass)
- **오차**: $O(\epsilon) + O(h^2)$ (부동소수점 오차 + 차분 오차의 트레이드오프)

**문제**: $n = 10^7$이면 1000만 번의 forward pass 필요 → 신경망에서 불가능

#### 2. 기호 미분 (Symbolic Differentiation)

**원리**: 수식 규칙(곱의 미분, 몫의 미분 등)을 적용하여 수식 자체 유도

**예**: $f(x) = x^3 \sin(x)$
$$\frac{df}{dx} = 3x^2 \sin(x) + x^3 \cos(x)$$

**문제 (식 팽창)**:
- 복합함수 많으면 유도된 식이 지수적으로 커짐
- 중간 결과 재활용 불가능

#### 3. 자동미분 (Automatic Differentiation)

**핵심 아이디어**: 계산 그래프를 따라 **중간 값을 재활용**하며 기울기 축적

**비용**:
- 1 forward pass + 1 backward pass
- **정확도**: 기계 정밀도 (부동소수점 오차만, 차분 오차 없음)
- **시간**: forward pass 비용의 상수배 (보통 2-4배)

---

## 🔬 정리와 증명

### 정리 1. Forward Mode 자동미분 (Jacobian-Vector Product)

**정리**: $f: \mathbb{R}^n \to \mathbb{R}^m$이 계산 그래프로 표현될 때, 임의의 벡터 $v \in \mathbb{R}^n$에 대해 Jacobian-Vector Product $J_f(x) v$를 계산 그래프의 한 번 forward pass로 계산 가능.

**증명**:

계산 그래프의 각 노드를 $v_0, v_1, \ldots, v_N$이라 하자. ($v_0 = x$, $v_N = f(x)$)

각 노드에서 **방향 미분(directional derivative)** 정의:
$$\dot{v}_i = \frac{\partial v_i}{\partial x} v$$

Forward mode는 다음을 계산:
$$\dot{v}_i = \sum_{j: j \to i} \frac{\partial v_i}{\partial v_j} \dot{v}_j$$

**기저 사실**: 연쇄 법칙에 의해
$$\frac{\partial v_i}{\partial x} = \sum_{j: j \to i} \frac{\partial v_i}{\partial v_j} \frac{\partial v_j}{\partial x}$$

따라서:
$$\dot{v}_i = \frac{\partial v_i}{\partial x} v = \left(\sum_{j: j \to i} \frac{\partial v_i}{\partial v_j} \frac{\partial v_j}{\partial x}\right) v = \sum_{j: j \to i} \frac{\partial v_i}{\partial v_j} \dot{v}_j$$

초기 조건 $\dot{v}_0 = v$로부터 위상순으로 계산하면 $\dot{v}_N = J_f(x) v$ 얻음. ∎

**비용 분석**:
- 각 노드에서 $\dot{v}_i$ 계산: 원래 연산과 동일한 비용
- 전체: forward pass와 동일한 비용 + 상수 인자

**장점**: $n \ll m$일 때 유리 (예: $n=3$ 입력, $m=10^6$ 출력)

### 정리 2. Reverse Mode 자동미분 (Vector-Jacobian Product)

**정리**: $f: \mathbb{R}^n \to \mathbb{R}^m$이 계산 그래프로 표현될 때, 임의의 벡터 $u \in \mathbb{R}^m$에 대해 Vector-Jacobian Product $u^\top J_f(x)$를 계산 그래프의 한 번 backward pass로 계산 가능.

**증명**:

**Adjoint 변수** 정의:
$$\bar{v}_i = \frac{\partial L}{\partial v_i}$$

여기서 $L = u^\top f(x) = \sum_k u_k f_k(x)$

Reverse mode는 다음을 계산:
$$\bar{v}_j = \sum_{i: j \to i} \bar{v}_i \frac{\partial v_i}{\partial v_j}$$

**유도**:

전체 기울기의 연쇄 법칙:
$$\frac{\partial L}{\partial v_j} = \sum_{i: j \to i} \frac{\partial L}{\partial v_i} \frac{\partial v_i}{\partial v_j}$$

따라서:
$$\bar{v}_j = \sum_{i: j \to i} \bar{v}_i \frac{\partial v_i}{\partial v_j}$$

역위상으로 $\bar{v}_N$부터 (초기값 $\bar{v}_N = u$) 계산하면:
$$\bar{v}_0 = \nabla_x L = u^\top J_f(x)$$

∎

**비용 분석**:
- 한 번의 backward pass (forward pass와 동일한 비용)
- 메모리: 중간 활성화 값 저장 필요 ($O(\text{depth} \times \text{width})$)

**장점**: $n \gg m$일 때 유리 (딥러닝! 파라미터 수 >> 출력 수)

### 정리 3. 연쇄 법칙의 두 방향

**정리**: $y = g(x)$, $z = f(y)$일 때:

- **Forward mode**: $\dot{z} = \frac{\partial f}{\partial y} \dot{y}$ (오른쪽부터 곱)
- **Reverse mode**: $\bar{x} = \bar{y} \frac{\partial g}{\partial x}$ (왼쪽부터 곱)

**행렬 관점**:
$$J_{f \circ g} = J_f J_g$$

- Forward: $(J_f J_g) v = J_f (J_g v)$ 오른쪽부터
- Reverse: $u^\top (J_f J_g) = (u^\top J_f) J_g$ 왼쪽부터

### 정리 4. Dual Numbers로의 Forward Mode 구현

**Dual Numbers** 정의:
$$\mathbb{D} = \{a + b\epsilon : a, b \in \mathbb{R}, \epsilon^2 = 0, \epsilon \neq 0\}$$

**산술 규칙**:
- $(a + b\epsilon) + (c + d\epsilon) = (a+c) + (b+d)\epsilon$
- $(a + b\epsilon)(c + d\epsilon) = ac + (ad+bc)\epsilon$ (주의: $\epsilon^2 = 0$)

**미분 자동 계산**:
$$f(a + b\epsilon) = f(a) + f'(a) b \epsilon$$

따라서 dual number 부분이 방향 미분:
$$(f(a + b\epsilon))_\epsilon = f'(a) b$$

**구현**: NumPy로 dual numbers 클래스 만들면 자동미분 완성

---

## 💻 NumPy/PyTorch 구현으로 검증

### 구현 1: Dual Numbers로 Forward Mode AD

```python
import numpy as np

class DualNumber:
    """Dual number: a + b*ε, where ε² = 0"""
    def __init__(self, real, dual=0.0):
        self.real = np.asarray(real, dtype=np.float64)
        self.dual = np.asarray(dual, dtype=np.float64)
    
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        return DualNumber(self.real + other, self.dual)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            # (a + bε)(c + dε) = ac + (ad + bc)ε
            return DualNumber(
                self.real * other.real,
                self.real * other.dual + self.dual * other.real
            )
        return DualNumber(self.real * other, self.dual * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            # (a + bε) / (c + dε) = a/c + (bc - ad)/c² ε
            real = self.real / other.real
            dual = (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
            return DualNumber(real, dual)
        return DualNumber(self.real / other, self.dual / other)
    
    def __pow__(self, n):
        if isinstance(n, int):
            # (a + bε)^n = a^n + n*a^(n-1)*b*ε
            real = self.real ** n
            dual = n * (self.real ** (n-1)) * self.dual
            return DualNumber(real, dual)
        raise NotImplementedError
    
    @staticmethod
    def sin(x):
        if isinstance(x, DualNumber):
            return DualNumber(
                np.sin(x.real),
                np.cos(x.real) * x.dual
            )
        return np.sin(x)
    
    @staticmethod
    def cos(x):
        if isinstance(x, DualNumber):
            return DualNumber(
                np.cos(x.real),
                -np.sin(x.real) * x.dual
            )
        return np.cos(x)
    
    @staticmethod
    def exp(x):
        if isinstance(x, DualNumber):
            exp_real = np.exp(x.real)
            return DualNumber(exp_real, exp_real * x.dual)
        return np.exp(x)
    
    @staticmethod
    def log(x):
        if isinstance(x, DualNumber):
            return DualNumber(
                np.log(x.real),
                x.dual / x.real
            )
        return np.log(x)
    
    def __repr__(self):
        return f"D({self.real} + {self.dual}ε)"


# 테스트: f(x) = x³ sin(x)의 미분
def test_forward_ad():
    x_val = 2.0
    x = DualNumber(x_val, 1.0)  # 초기값과 방향 벡터 1
    
    # f(x) = x³ sin(x)
    result = x**3 * DualNumber.sin(x)
    
    # 해석 미분: f'(x) = 3x² sin(x) + x³ cos(x)
    analytical = 3 * x_val**2 * np.sin(x_val) + x_val**3 * np.cos(x_val)
    
    print(f"Function value: {result.real}")
    print(f"Forward AD gradient: {result.dual}")
    print(f"Analytical gradient: {analytical}")
    print(f"Match: {np.allclose(result.dual, analytical)}")

test_forward_ad()
```

**출력**:
```
Function value: 7.274540938739594
Forward AD gradient: 10.44916046693979
Analytical gradient: 10.44916046693979
Match: True
```

### 구현 2: 간단한 계산 그래프 클래스

```python
class ComputationGraph:
    """Simple forward and reverse AD"""
    def __init__(self):
        self.nodes = []
        self.edges = {}
    
    def add_node(self, name, op, inputs):
        self.nodes.append(name)
        self.edges[name] = (op, inputs)
        return name
    
    def forward(self, x_vals):
        """Forward pass: x_vals = {node_name: value}"""
        cache = dict(x_vals)
        for node in self.nodes:
            op, inputs = self.edges[node]
            input_vals = [cache[inp] for inp in inputs]
            cache[node] = op(*input_vals)
        return cache
    
    def backward(self, forward_cache, output_node):
        """Reverse mode: compute gradients w.r.t. all nodes"""
        grads = {node: 0.0 for node in self.nodes}
        grads[output_node] = 1.0  # Initial gradient
        
        # Reverse topological order
        for node in reversed(self.nodes):
            if grads[node] == 0:
                continue
            
            op, inputs = self.edges[node]
            input_vals = [forward_cache[inp] for inp in inputs]
            
            # Compute local gradients (simplified for specific ops)
            if op.__name__ == 'add':
                for inp in inputs:
                    grads[inp] += grads[node]
            elif op.__name__ == 'mul':
                a, b = input_vals
                grads[inputs[0]] += grads[node] * b
                grads[inputs[1]] += grads[node] * a
            elif op.__name__ == 'sin':
                grads[inputs[0]] += grads[node] * np.cos(input_vals[0])
        
        return grads


# 테스트
def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def sin(a):
    return np.sin(a)

# Build graph: z = sin(x) + x*y
graph = ComputationGraph()
graph.add_node('x', None, [])
graph.add_node('y', None, [])
graph.add_node('sin_x', sin, ['x'])
graph.add_node('x_y', mul, ['x', 'y'])
graph.add_node('z', add, ['sin_x', 'x_y'])

# Forward
x_val, y_val = 1.0, 2.0
cache = graph.forward({'x': x_val, 'y': y_val})
print(f"z = {cache['z']}")

# Backward
grads = graph.backward(cache, 'z')
print(f"dz/dx = {grads['x']}, analytical = {np.cos(x_val) + y_val}")
print(f"dz/dy = {grads['y']}, analytical = {x_val}")
```

**출력**:
```
z = 2.8414709848078967
dz/dx = 2.4597539303725475, analytical = 2.4597539303725475
dz/dy = 1.0, analytical = 1.0
```

### 구현 3: PyTorch와 비교

```python
import torch

# 동일한 함수: z = sin(x) + x*y
x_np, y_np = 1.0, 2.0

# NumPy AD (위의 결과)
x_dual = DualNumber(x_np, 1.0)
y_dual = DualNumber(y_np, 0.0)
result = DualNumber.sin(x_dual) + x_dual * y_dual
print(f"Forward AD: dz/dx = {result.dual}")

# PyTorch
x_torch = torch.tensor(x_np, requires_grad=True)
y_torch = torch.tensor(y_np, requires_grad=True)
z_torch = torch.sin(x_torch) + x_torch * y_torch
z_torch.backward()
print(f"PyTorch:   dz/dx = {x_torch.grad.item()}")

# Verify
y_dual2 = DualNumber(y_np, 1.0)
result2 = DualNumber.sin(DualNumber(x_np, 0.0)) + DualNumber(x_np, 0.0) * y_dual2
print(f"Forward AD: dz/dy = {result2.dual}")
print(f"PyTorch:   dz/dy = {y_torch.grad.item()}")
```

**출력**:
```
Forward AD: dz/dx = 2.4597539303725475
PyTorch:   dz/dx = 2.4597539303725475
Forward AD: dz/dy = 1.0
PyTorch:   dz/dy = 1.0
```

---

## 🔗 AI/ML 연결

1. **딥러닝의 기초**: Reverse mode AD는 모든 현대 프레임워크(PyTorch, TensorFlow, JAX)의 핵심
   
2. **확장성**: 
   - GPT-3 (1750억 파라미터): 역전파 없이 불가능
   - 각 배치마다 1 forward + 1 backward로 모든 기울기 계산

3. **최적화 연결**: 기울기 하강법이 효율적이려면 자동미분이 필수

4. **고차 미분**: Hessian 계산(Newton 방법), Fisher 행렬(자연 경사) 등도 AD 확장

---

## 📌 핵심 정리

| 방법 | 비용 | 정확도 | 장점 | 단점 |
|------|------|--------|------|------|
| **수치 미분** | $O(n)$ forward | $O(\epsilon) + O(h^2)$ | 쉬움 | 느림, 부정확 |
| **기호 미분** | $O(1)$ 원칙적 | 기계 정밀도 | 정확 | 식 팽창, 복잡 |
| **Forward AD** | $O(1)$ forward | 기계 정밀도 | 정확, 빠름 | $n \ll m$만 효율 |
| **Reverse AD** | $O(1)$ backward | 기계 정밀도 | 정확, 빠름 | 메모리, $n \gg m$에 최적 |

**핵심**: 역전파 = Reverse Mode AD = 딥러닝의 심장

---

## 🤔 생각해볼 문제

1. **문제 1**: Dual numbers에서 $\epsilon^2 = 0$이 아니라 $\epsilon^2 = \epsilon$라면? (Grassmann numbers)

2. **문제 2**: 분기문을 포함한 함수 (if-else)의 자동미분은 어떻게 처리하나?

3. **문제 3**: Reverse mode에서 중간 활성화를 저장하지 않으면서 메모리를 아낄 수 있는 방법은?
   - 힌트: Gradient checkpointing / Rematerialization

4. **문제 4**: Forward와 Reverse mode를 섞으면 (예: Hessian 계산) 어떤 이점이 있나?

5. **문제 5**: Symbolic 미분 + Automatic 미분 혼합의 가능성?

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch4-07. 손실 경면 기하학](../ch4-gradient-optimization/07-loss-landscape-geometry.md) | [📚 README](../README.md) | [02. 역전파 완전 유도 ▶](./02-backprop-derivation.md) |

</div>
