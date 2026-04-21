# 04. MAML과 고차 미분

## 🎯 핵심 질문

- Meta-Learning의 수학적 구조는?
- MAML 기울기 계산에서 헤시안이 왜 등장하는가?
- Hessian-Vector Product를 효율적으로 계산하는 방법은?
- FOMAML과 Reptile의 관계는?

## 🔍 왜 이 개념이 AI에서 중요한가

Meta-Learning은 적응형 AI의 핵심입니다:

1. **Few-shot 학습**: 소수 샘플로 새로운 작업 학습
2. **2차 미분 필요성**: Meta-gradient는 내부 루프를 미분 → 헤시안 필요
3. **효율적 계산**: HVP(Hessian-Vector Product)로 $O(n^3)$ 계산 회피
4. **일반화 이론**: 파라미터 업데이트가 일반화 에러를 직접 최적화

## 📐 수학적 선행 조건

- 1계 미분 및 역전파
- 헤시안 행렬
- 암묵적 미분 정리
- 연쇄법칙과 고차 미분
- 선형 시스템 풀기

## ✏️ 정의와 핵심 도구

### Meta-Learning 문제 설정

주어진 작업 분포 $p(\mathcal{T})$에서:

**작업**: $\mathcal{T}_i = \{(\mathbf{x}^{\text{sup}}, \mathbf{y}^{\text{sup}}), (\mathbf{x}^{\text{query}}, \mathbf{y}^{\text{query}})\}$

**목표**:
$$\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\text{query}}^i(\theta - \alpha \nabla_\theta \mathcal{L}_{\text{support}}^i(\theta))$$

이는 지원 집합(support set)으로 내부 루프 업데이트 후, 쿼리 집합(query set)에서의 손실을 최소화합니다.

### MAML Gradient 유도

**Inner Loop (작업 $i$)**:
$$\phi_i = \theta - \alpha \nabla_\theta \mathcal{L}_i^{\text{sup}}(\theta)$$

**Outer Loop (Meta-update)**:
$$\nabla_\theta \mathcal{L}_{\text{meta}} = \sum_i \nabla_\theta \mathcal{L}_i^{\text{query}}(\phi_i)$$

## 🔬 정리와 증명

### 정리 1: MAML Meta-Gradient의 헤시안 형태

**명제**: MAML의 meta-gradient는:

$$\nabla_\theta \mathcal{L}_{\text{meta}} = \sum_i (I - \alpha H_i)^\top \nabla_{\phi_i} \mathcal{L}_i^{\text{query}}$$

여기서 $H_i = \nabla_\theta^2 \mathcal{L}_i^{\text{sup}}(\theta)$는 지원 손실의 헤시안입니다.

**증명**:

연쇄법칙으로:
$$\frac{d \mathcal{L}_i^{\text{query}}(\phi_i)}{d\theta} = \frac{\partial \mathcal{L}_i^{\text{query}}}{\partial \phi_i} \cdot \frac{\partial \phi_i}{\partial \theta}$$

$\phi_i = \theta - \alpha \nabla_\theta \mathcal{L}_i^{\text{sup}}(\theta)$이므로:

$$\frac{\partial \phi_i}{\partial \theta} = \frac{\partial}{\partial \theta}\left(\theta - \alpha \nabla_\theta \mathcal{L}_i^{\text{sup}}\right)$$

$$= I - \alpha \frac{\partial}{\partial \theta}(\nabla_\theta \mathcal{L}_i^{\text{sup}})$$

$$= I - \alpha \nabla_\theta^2 \mathcal{L}_i^{\text{sup}}(\theta)$$

$$= I - \alpha H_i$$

따라서:
$$\nabla_\theta \mathcal{L}_{\text{meta}} = \sum_i \nabla_{\phi_i} \mathcal{L}_i^{\text{query}} \cdot (I - \alpha H_i)^\top$$

벡터화하면:
$$\nabla_\theta \mathcal{L}_{\text{meta}} = \sum_i (I - \alpha H_i)^\top \nabla_{\phi_i} \mathcal{L}_i^{\text{query}} \quad \checkmark$$

### 정리 2: Hessian-Vector Product (HVP)

**명제**: 벡터 $v$에 대한 헤시안-벡터 곱은:

$$Hv = \nabla_\theta(\nabla_\theta \mathcal{L} \cdot v)$$

이는 Pearlmutter trick으로 $O(n)$ 시간에 계산 가능합니다.

**증명**:

헤시안의 정의:
$$H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}$$

벡터 곱:
$$(Hv)_i = \sum_j H_{ij} v_j = \frac{\partial}{\partial \theta_i}\left(\sum_j \frac{\partial \mathcal{L}}{\partial \theta_j} v_j\right) = \frac{\partial}{\partial \theta_i}(\nabla \mathcal{L} \cdot v)$$

따라서:
$$Hv = \nabla_\theta(\nabla_\theta \mathcal{L} \cdot v)$$

$\checkmark$

**계산 효율성**:
- 직접 헤시안: $O(n^2)$ 메모리, $O(n^3)$ 시간
- HVP: forward + reverse mode AD로 $O(n)$ 시간, $O(n)$ 메모리

### 정리 3: HVP의 Automatic Differentiation 구현

**명제**: 다음 알고리즘으로 HVP를 계산할 수 있습니다:

```
1. Forward pass: loss = L(θ)
2. grad = ∇L(θ)  [1차 미분]
3. dot_prod = grad · v  [내적]
4. hvp = ∇_θ(dot_prod)  [2차 미분]
```

**증명**:

1단계에서 손실 계산
2단계에서 첫 번째 역전파로 기울기 $g = \nabla_\theta \mathcal{L}$ 획득
3단계에서 $g \cdot v = \sum_i g_i v_i$ 계산
4단계에서 이를 다시 $\theta$에 대해 미분

이는 reverse-over-forward 자동 미분이며, PyTorch의 `torch.autograd.grad`를 2번 호출하면 됩니다.

### 정리 4: First-Order MAML (FOMAML)

**명제**: 헤시안을 무시하면:

$$\nabla_\theta \mathcal{L}_{\text{FOMAML}} = \sum_i \nabla_{\phi_i} \mathcal{L}_i^{\text{query}}$$

이는 실제로는 더 복잡하지만, 계산 효율적입니다.

**증명**:

$(I - \alpha H)^\top \approx I$ (작은 $\alpha$ 또는 약한 곡률 가정):

$$\nabla_\theta \mathcal{L}_{\text{FOMAML}} \approx \sum_i \nabla_{\phi_i} \mathcal{L}_i^{\text{query}}$$

실제로는 다음과 같이 해석됩니다:
$$\frac{d\mathcal{L}_{\text{query}}}{d\phi} \cdot \frac{d\phi}{d\theta} \approx \frac{d\mathcal{L}_{\text{query}}}{d\phi}$$

즉, inner loop를 통한 파라미터 경로를 무시하는 것입니다. $\checkmark$

### 정리 5: Reptile의 기울기 근사

**명제**: Reptile 업데이트:
$$\theta_{t+1} = \theta_t + \beta(\phi_i - \theta_t)$$

는 다음 형태로 분석할 수 있습니다:

$$\theta_{t+1} \approx \theta_t + \beta \left[-\alpha \nabla_{\theta} \mathcal{L}_i^{\text{sup}} + \frac{\alpha^2}{2}H_i \nabla_{\phi} \mathcal{L}_i^{\text{query}} + O(\alpha^3)\right]$$

**증명**:

Taylor 전개:
$$\phi_i = \theta - \alpha \nabla_\theta \mathcal{L}_i^{\text{sup}} + O(\alpha^2)$$

$$\phi_i - \theta = -\alpha \nabla_\theta \mathcal{L}_i^{\text{sup}} + O(\alpha^2)$$

그런데 실제로는 $\phi_i$에서의 손실 기울기도 포함되므로, 전체 동역학은:

$$\phi_i - \theta = -\alpha \nabla_\theta \mathcal{L}_i^{\text{sup}} + \alpha^2 H_i^{-1} \nabla_{\phi_i} \mathcal{L}_i^{\text{query}} + O(\alpha^3)$$

따라서 Reptile은 MAML과 같은 1차 항을 공유하면서 다른 2차 항을 추가합니다.

$\checkmark$

### 정리 6: Implicit MAML (iMAML)

**명제**: Meta-gradient를 암묵적으로 계산할 수 있습니다:

$$\nabla_\theta \mathcal{L}_{\text{meta}} = -\left(\nabla_{\phi}^2 \mathcal{L}^{\text{query}} + \lambda I\right)^{-1} \nabla_{\theta \phi} \mathcal{L}^{\text{query}}$$

여기서 $\phi^*$는 inner optimization의 최적점입니다.

**증명** (Implicit Function Theorem):

Inner 최적화의 KKT 조건:
$$F(\phi^*, \theta) := \nabla_{\phi} \mathcal{L}^{\text{support}}(\phi^*, \theta) = 0$$

이를 $\theta$로 미분하면:
$$\frac{\partial F}{\partial \phi^*}\frac{d\phi^*}{d\theta} + \frac{\partial F}{\partial \theta} = 0$$

따라서:
$$\frac{d\phi^*}{d\theta} = -\left[\frac{\partial^2 \mathcal{L}}{\partial \phi^2}\right]^{-1} \frac{\partial^2 \mathcal{L}}{\partial \phi \partial \theta}$$

Meta-gradient:
$$\nabla_\theta \mathcal{L}_{\text{meta}} = \nabla_{\phi} \mathcal{L}^{\text{query}} \cdot \frac{d\phi^*}{d\theta}$$

$$= -\nabla_{\phi} \mathcal{L}^{\text{query}} \left[\nabla_{\phi}^2 \mathcal{L}^{\text{support}}\right]^{-1} \frac{\partial^2 \mathcal{L}^{\text{support}}}{\partial \phi \partial \theta}$$

또는 벡터 형태로 Neumann 급수 근사:

$$\left[\nabla_{\phi}^2 \mathcal{L}\right]^{-1} u \approx \sum_{j=0}^{K} (I - \nabla_{\phi}^2 \mathcal{L})^j u$$

$\checkmark$

## 💻 NumPy/PyTorch 구현으로 검증

### MAML 기본 구현

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class MAMLMeta:
    """MAML 메타 학습기"""
    
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
    
    def inner_update(self, support_x, support_y, num_steps=1):
        """Inner loop: 지원 집합에서 몇 단계 업데이트"""
        # 파라미터의 복사본으로 작업
        fast_model = deepcopy(self.model)
        fast_model.train()
        
        for _ in range(num_steps):
            # Forward pass
            logits = fast_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # Backward + update
            loss.backward()
            with torch.no_grad():
                for param in fast_model.parameters():
                    if param.grad is not None:
                        param.data -= self.inner_lr * param.grad
                    param.grad = None
        
        return fast_model
    
    def meta_update(self, tasks, num_inner_steps=1):
        """Meta-level 업데이트"""
        meta_loss = 0
        meta_gradients = None
        
        self.model.zero_grad()
        
        for task_idx, (support_x, support_y, query_x, query_y) in enumerate(tasks):
            # Inner loop
            fast_model = self.inner_update(support_x, support_y, num_inner_steps)
            
            # Query loss with fast model
            query_logits = fast_model(query_x)
            query_loss = F.cross_entropy(query_logits, query_y)
            meta_loss += query_loss / len(tasks)
        
        # Meta gradient
        meta_loss.backward()
        
        return meta_loss.item()

# 검증
print("=" * 70)
print("MAML 구현 검증")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

# 간단한 모델
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleMLP(input_dim=10, hidden_dim=20, output_dim=5)
maml = MAMLMeta(model, inner_lr=0.01, outer_lr=0.001)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Inner LR: {maml.inner_lr}, Outer LR: {maml.outer_lr}\n")

# 더미 작업 생성
def generate_dummy_task(input_dim=10, num_support=5, num_query=3, num_classes=5):
    """더미 분류 작업"""
    support_x = torch.randn(num_support, input_dim)
    support_y = torch.randint(0, num_classes, (num_support,))
    query_x = torch.randn(num_query, input_dim)
    query_y = torch.randint(0, num_classes, (num_query,))
    return support_x, support_y, query_x, query_y

# 메타 훈련
print("Meta-training:")
for meta_step in range(3):
    # 배치 작업
    tasks = [generate_dummy_task() for _ in range(4)]
    meta_loss = maml.meta_update(tasks, num_inner_steps=1)
    print(f"  Meta-step {meta_step}: Loss = {meta_loss:.6f}")
```

### Hessian-Vector Product 계산

```python
def hessian_vector_product(loss, params, vector):
    """
    Pearlmutter trick으로 HVP 계산
    hvp = ∇(∇L · v)
    """
    # 첫 번째 역전파: 기울기 계산
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=True
    )
    
    # 기울기와 벡터의 내적
    grad_vec = torch.cat([g.flatten() for g in grads if g is not None])
    dot_product = torch.sum(grad_vec * vector)
    
    # 두 번째 역전파: 헤시안-벡터 곱
    hvp = torch.autograd.grad(
        dot_product, params, retain_graph=True, allow_unused=True
    )
    
    hvp = torch.cat([h.flatten() for h in hvp if h is not None])
    
    return hvp

print("\n" + "=" * 70)
print("Hessian-Vector Product (HVP) 계산")
print("=" * 70)

torch.manual_seed(42)

# 간단한 예제: f(x) = x1^2 + 2*x2^2
# ∇f = [2*x1, 4*x2]
# H = [[2, 0], [0, 4]]
# H @ [1, 1] = [2, 4]

x = torch.tensor([1.0, 2.0], requires_grad=True)
loss = x[0]**2 + 2*x[1]**2

v = torch.tensor([1.0, 1.0])
hvp = hessian_vector_product(loss, [x], v)

print(f"\nInput x: {x.data}")
print(f"Loss: x₁² + 2x₂²")
print(f"∇L = {torch.autograd.grad(loss, x)[0]}")
print(f"\nVector v: {v}")
print(f"HVP result: {hvp}")
print(f"Expected:   tensor([2., 4.])")
print(f"Match: {torch.allclose(hvp, torch.tensor([2., 4.]))}")

# 더 복잡한 예제: 신경망
print("\n" + "-" * 70)
print("신경망의 HVP")
print("-" * 70)

model = SimpleMLP(input_dim=5, hidden_dim=10, output_dim=2)
x = torch.randn(1, 5, requires_grad=False)
y = torch.tensor([1])

output = model(x)
loss = F.cross_entropy(output, y)

# 파라미터 모음
params = list(model.parameters())
n_params = sum(p.numel() for p in params)

# 무작위 벡터
vector = torch.randn(n_params)

# HVP 계산
hvp = hessian_vector_product(loss, params, vector)

print(f"Number of parameters: {n_params}")
print(f"Vector shape: {vector.shape}")
print(f"HVP shape: {hvp.shape}")
print(f"HVP norm: {torch.norm(hvp):.6f}")
```

### First-Order MAML vs Full MAML

```python
def compute_maml_gradient(model, support_x, support_y, query_x, query_y, 
                         inner_lr=0.01, first_order=False):
    """MAML 기울기 계산"""
    
    # Inner loop
    fast_model = deepcopy(model)
    support_logits = fast_model(support_x)
    support_loss = F.cross_entropy(support_logits, support_y)
    
    # Inner 기울기
    inner_grads = torch.autograd.grad(
        support_loss, fast_model.parameters(), 
        create_graph=(not first_order), retain_graph=True
    )
    
    # Inner update
    with torch.no_grad():
        for param, grad in zip(fast_model.parameters(), inner_grads):
            param.data -= inner_lr * grad
    
    # Query loss
    query_logits = fast_model(query_x)
    query_loss = F.cross_entropy(query_logits, query_y)
    
    # Meta gradient
    meta_grads = torch.autograd.grad(
        query_loss, model.parameters(), create_graph=True
    )
    
    return query_loss, meta_grads

print("\n" + "=" * 70)
print("First-Order MAML vs Full MAML")
print("=" * 70)

torch.manual_seed(42)

model_full = SimpleMLP(input_dim=10, hidden_dim=20, output_dim=5)
model_fo = deepcopy(model_full)

support_x = torch.randn(5, 10)
support_y = torch.randint(0, 5, (5,))
query_x = torch.randn(3, 10)
query_y = torch.randint(0, 5, (3,))

# Full MAML
print("\nFull MAML (with Hessian):")
loss_full, grads_full = compute_maml_gradient(
    model_full, support_x, support_y, query_x, query_y, 
    inner_lr=0.01, first_order=False
)
grad_norm_full = torch.norm(torch.cat([g.flatten() for g in grads_full]))
print(f"  Query loss: {loss_full:.6f}")
print(f"  Gradient norm: {grad_norm_full:.6f}")

# First-Order MAML
print("\nFirst-Order MAML (without Hessian):")
loss_fo, grads_fo = compute_maml_gradient(
    model_fo, support_x, support_y, query_x, query_y, 
    inner_lr=0.01, first_order=True
)
grad_norm_fo = torch.norm(torch.cat([g.flatten() for g in grads_fo if g is not None]))
print(f"  Query loss: {loss_fo:.6f}")
print(f"  Gradient norm: {grad_norm_fo:.6f}")

print(f"\nGradient difference: {torch.norm(torch.cat([g.flatten() if g is not None else torch.tensor(0.) for g in grads_full]) - torch.cat([g.flatten() if g is not None else torch.tensor(0.) for g in grads_fo])):.6f}")
print("(Note: FOMAML은 빠르지만 2차 항 손실)")
```

### Reptile 알고리즘

```python
class Reptile:
    """Reptile 메타 학습기"""
    
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.initial_params = deepcopy(self.model.state_dict())
    
    def inner_update(self, support_x, support_y, num_steps=1):
        """Inner loop"""
        fast_model = deepcopy(self.model)
        fast_model.train()
        
        for _ in range(num_steps):
            logits = fast_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            
            with torch.no_grad():
                for param in fast_model.parameters():
                    if param.grad is not None:
                        param.data -= self.inner_lr * param.grad
                    param.grad = None
        
        return fast_model
    
    def meta_update(self, tasks, num_inner_steps=1):
        """Reptile meta-update: θ ← θ + β(φ - θ)"""
        
        total_update = None
        
        for support_x, support_y, _, _ in tasks:
            fast_model = self.inner_update(support_x, support_y, num_inner_steps)
            
            # φ - θ
            for p_model, p_fast in zip(self.model.parameters(), fast_model.parameters()):
                if total_update is None:
                    total_update = p_fast.data - p_model.data
                else:
                    total_update = total_update + (p_fast.data - p_model.data)
        
        # Meta update
        with torch.no_grad():
            for param in self.model.parameters():
                param.data += (self.outer_lr / len(tasks)) * total_update

print("\n" + "=" * 70)
print("Reptile 알고리즘")
print("=" * 70)

model_reptile = SimpleMLP(input_dim=10, hidden_dim=20, output_dim=5)
reptile = Reptile(model_reptile, inner_lr=0.01, outer_lr=0.001)

print(f"Model parameters: {sum(p.numel() for p in model_reptile.parameters())}")
print(f"Inner LR: {reptile.inner_lr}, Outer LR: {reptile.outer_lr}\n")

print("Reptile meta-training:")
for meta_step in range(3):
    tasks = [generate_dummy_task() for _ in range(4)]
    reptile.meta_update(tasks, num_inner_steps=1)
    
    # Evaluate
    test_task = generate_dummy_task()
    support_x, support_y, query_x, query_y = test_task
    test_model = reptile.inner_update(support_x, support_y, num_inner_steps=1)
    with torch.no_grad():
        query_logits = test_model(query_x)
        test_loss = F.cross_entropy(query_logits, query_y)
    
    print(f"  Meta-step {meta_step}: Test loss = {test_loss:.6f}")
```

## 🔗 AI/ML 연결

### 1. Few-Shot Learning

- Prototypical Networks, Matching Networks 등의 메타 학습 기초
- Task distribution p(T)에서 효율적 적응

### 2. Hyperparameter Optimization

- 메타 learning rate 학습
- 데이터 증강 정책 자동 설계

### 3. Multi-Task Learning

- 공유 표현 학습
- 작업 간 지식 이전

### 4. Continual Learning

- 이전 작업을 잊지 않으면서 새 작업 학습
- Meta-plasticity

## 📌 핵심 정리

| 개념 | 공식 | 역할 |
|------|------|------|
| **Meta-Objective** | $\sum_i L_i^{\text{query}}(\phi_i)$ | 어댑팅 후 손실 |
| **Inner Update** | $\phi_i = \theta - \alpha \nabla L_i^{\text{sup}}$ | 빠른 적응 |
| **MAML Gradient** | $(I - \alpha H_i)^\top \nabla L_i^{\text{query}}$ | 헤시안 포함 |
| **HVP** | $Hv = \nabla(\nabla L \cdot v)$ | 효율적 계산 |
| **FOMAML** | $\nabla L_i^{\text{query}}$ | 빠르지만 근사 |
| **Reptile** | $\theta + \beta(\phi_i - \theta)$ | 평균화 기반 |

## 🤔 생각해볼 문제

1. **헤시안의 필요성**: 왜 FOMAML의 근사가 작동하는가? 언제 실패하는가?

2. **계산 복잡도**: 
   - Full MAML: O(n²) 헤시안 계산
   - HVP 사용: O(n) 계산
   - 메모리 vs 시간 trade-off는?

3. **Reptile vs MAML**: 왜 Reptile이 MAML과 비슷한 성능을 내는가?

4. **고차 미분의 수치 안정성**: 3차, 4차 미분은 왜 어려운가?

5. **Implicit MAML**: Neumann 급수 근사의 수렴 조건은? 
   $$\sum_{j=0}^{\infty} (I - H)^j = H^{-1}$$

<div align="center">

| | | |
|---|---|---|
| [◀ 03. NTK와 무한 폭 네트워크](./03-ntk-infinite-width.md) | [📚 README](../README.md) | [05. 딥러닝의 암묵적 미분 ▶](./05-implicit-differentiation-dl.md) |

</div>
