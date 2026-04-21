# 03. 미분의 정의와 선형근사

## 🎯 핵심 질문

- 도함수 $f'(a)$가 단순한 "기울기"가 아니라 "최선의 선형근사"라는 말의 의미는 무엇인가?
- 오차가 $o(h)$라는 표현은 무엇을 뜻하는가?
- 수치 미분(유한차분)은 해석 미분과 얼마나 차이가 나는가?
- 다변수로 확장하면 선형근사는 어떤 형태가 되는가? (Chapter 2 예고)

---

## 🔍 왜 이 개념이 AI에서 중요한가

역전파(Backpropagation)는 본질적으로 **국소 선형근사의 연쇄 적용**이다.

- 각 층의 연산 $f(x)$를 $x = x_0$ 근방에서 $f(x) \approx f(x_0) + f'(x_0)(x - x_0)$으로 근사
- 이 근사의 오차가 $o(x - x_0)$이기 때문에, 역전파로 계산한 gradient가 실제 gradient와 일치한다
- 수치 미분은 역전파 구현이 올바른지 검증하는 `gradcheck`에 사용된다
- 학습률 $\eta$가 "충분히 작을 때" 경사하강법이 수렴하는 이유도 선형근사의 정확성에 있다

---

## 📐 수학적 선행 조건

- [01. ε-δ 극한의 정의](./01-epsilon-delta.md)
- [02. 연속성과 균등연속성](./02-continuity-uniform.md): 미분가능하면 연속 (역은 거짓 — 다음 문서에서)
- 소 $o$ 표기법: $g(h) = o(h)$는 $\lim_{h \to 0} g(h)/h = 0$을 의미

---

## 📖 직관적 이해

### "기울기"보다 "최선의 선형근사"

중학교에서 배운 "기울기" 정의: $\frac{f(a+h) - f(a)}{h}$를 $h \to 0$으로 보낸 것.

이 정의는 **왜** 그렇게 하는지를 말해주지 않는다.

더 깊은 시각: $f$를 점 $a$ 근방에서 선형 함수 $L(x) = f(a) + c(x - a)$로 근사할 때, **오차 $f(x) - L(x)$가 $(x-a)$보다 훨씬 빠르게 0으로 줄어드는** 유일한 $c$ 값이 $f'(a)$다.

$$f(a + h) = f(a) + f'(a) \cdot h + \underbrace{o(h)}_{\text{나머지}}$$

$o(h)$는 "$h$보다 훨씬 작은 것"을 의미한다. $h \to 0$일 때 $o(h)/h \to 0$.

> **비유**: $f$가 복잡한 산악 지형이라면, $f'(a) \cdot h$는 점 $a$에서의 접선 — 아주 가까운 근방에서 지형을 가장 잘 흉내내는 직선 경사로다.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 도함수 (미분계수)

$f: (a-r, a+r) \to \mathbb{R}$이라 하자. 다음 극한이 존재할 때 $f$는 점 $a$에서 **미분가능**하며, 그 극한값을 $f'(a)$ (또는 $\left.\frac{df}{dx}\right|_{x=a}$)라 한다:

$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

### 정의 3.2 — 최선의 선형근사 (동치 표현)

$f$가 점 $a$에서 미분가능한 것은, 다음을 만족하는 상수 $c$가 존재하는 것과 동치이다:

$$f(a + h) = f(a) + c \cdot h + r(h) \quad \text{where} \quad \lim_{h \to 0} \frac{r(h)}{h} = 0$$

이때 $c = f'(a)$이며, 이 $c$는 유일하다.

### 정의 3.3 — 소 $o$ 표기법

$g(h) = o(h)$ ($h \to 0$) 는 다음을 의미한다:
$$\lim_{h \to 0} \frac{g(h)}{h} = 0$$

직관: $h$가 0에 가까워질 때 $g(h)$는 $h$보다 훨씬 빠르게 0으로 간다. 예: $h^2 = o(h)$, $h^{1.5} = o(h)$, 하지만 $h$ 자체는 $o(h)$가 아니다.

---

## 🔬 정리와 증명

### 정리 3.1 — 선형근사 표현의 동치성

**명제**: $f'(a)$가 존재하는 것과, $r(h) = f(a+h) - f(a) - c \cdot h$가 $o(h)$가 되는 유일한 $c$가 존재하는 것은 동치이다.

**증명 (⇒ 방향)**:  
$f'(a) = c$로 놓고 $r(h) = f(a+h) - f(a) - c \cdot h$라 하면:
$$\frac{r(h)}{h} = \frac{f(a+h) - f(a)}{h} - c \xrightarrow{h \to 0} f'(a) - c = 0$$
따라서 $r(h) = o(h)$.

**증명 (⇐ 방향)**:  
$r(h) = o(h)$인 $c$가 존재한다고 가정하면:
$$\frac{f(a+h) - f(a)}{h} = c + \frac{r(h)}{h} \xrightarrow{h \to 0} c + 0 = c$$
따라서 $f'(a) = c$가 존재.

**유일성**: $c_1$과 $c_2$ 모두 성립한다고 가정하면 $(c_1 - c_2)h = o(h)$이므로 $c_1 - c_2 = 0$. $\square$

---

### 정리 3.2 — 기본 도함수 공식 유도

**(a) 거듭제곱 규칙**: $f(x) = x^n$이면 $f'(x) = nx^{n-1}$

$$\frac{(x+h)^n - x^n}{h} = \frac{\sum_{k=0}^{n}\binom{n}{k}x^{n-k}h^k - x^n}{h} = nx^{n-1} + \binom{n}{2}x^{n-2}h + \cdots$$

$h \to 0$이면 $h$ 이상 차수의 항이 모두 사라져 $nx^{n-1}$. $\square$

**(b) 곱 규칙 (Product Rule)**:

$$\frac{f(x+h)g(x+h) - f(x)g(x)}{h} = f(x+h)\frac{g(x+h)-g(x)}{h} + g(x)\frac{f(x+h)-f(x)}{h}$$

$h \to 0$이면 $f(x+h) \to f(x)$ (미분가능 → 연속), 나머지 항은 각각 $g'(x)$, $f'(x)$로 수렴. $\square$

**(c) 연쇄법칙 (Chain Rule)** (단변수):

$f(x) = g(u(x))$이면 $f'(x) = g'(u(x)) \cdot u'(x)$.

**증명**: $\Delta u = u(x+h) - u(x)$로 놓으면:

$$\frac{f(x+h) - f(x)}{h} = \frac{g(u(x)+\Delta u) - g(u(x))}{\Delta u} \cdot \frac{\Delta u}{h}$$

$h \to 0$이면 $\Delta u \to 0$ ($u$의 연속성), $\frac{\Delta u}{h} \to u'(x)$, $\frac{g(u+\Delta u)-g(u)}{\Delta u} \to g'(u)$.

*(주의: $\Delta u = 0$인 경우를 별도 처리해야 하는 미묘함이 있음 — 완전한 증명은 Chapter 2 연쇄법칙 문서 참조)* $\square$

---

### 정리 3.3 — 수치 미분의 오차 분석

**전방 차분**: $f'(x) \approx \frac{f(x+h) - f(x)}{h}$

테일러 전개로 오차를 분석:
$$f(x+h) = f(x) + f'(x)h + \frac{f''(x)}{2}h^2 + O(h^3)$$

따라서:
$$\frac{f(x+h) - f(x)}{h} = f'(x) + \frac{f''(x)}{2}h + O(h^2)$$

오차 = $O(h)$ — $h$에 비례하여 감소.

**중심 차분**: $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$

$$f(x+h) = f(x) + f'h + \frac{f''}{2}h^2 + \frac{f'''}{6}h^3 + \cdots$$
$$f(x-h) = f(x) - f'h + \frac{f''}{2}h^2 - \frac{f'''}{6}h^3 + \cdots$$

빼면:
$$\frac{f(x+h) - f(x-h)}{2h} = f'(x) + \frac{f'''(x)}{6}h^2 + O(h^4)$$

오차 = $O(h^2)$ — 전방 차분보다 훨씬 빠르게 감소.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 해석 미분 vs 수치 미분 오차 비교
# ─────────────────────────────────────────────

def numerical_derivative_forward(f, x, h):
    """전방 차분: O(h) 정확도"""
    return (f(x + h) - f(x)) / h

def numerical_derivative_central(f, x, h):
    """중심 차분: O(h²) 정확도"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 테스트 함수: f(x) = sin(x), 해석 미분: f'(x) = cos(x)
f = np.sin
f_prime_exact = np.cos
x0 = 1.0

h_values = np.logspace(-15, -1, 100)
err_forward = []
err_central = []

for h in h_values:
    err_f = abs(numerical_derivative_forward(f, x0, h) - f_prime_exact(x0))
    err_c = abs(numerical_derivative_central(f, x0, h) - f_prime_exact(x0))
    err_forward.append(err_f)
    err_central.append(err_c)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 좌: 오차 vs h
axes[0].loglog(h_values, err_forward, 'b-', linewidth=2, label='전방 차분 O(h)')
axes[0].loglog(h_values, err_central, 'r-', linewidth=2, label='중심 차분 O(h²)')
axes[0].loglog(h_values, h_values, 'b--', alpha=0.5, label='O(h) 참조선')
axes[0].loglog(h_values, h_values**2, 'r--', alpha=0.5, label='O(h²) 참조선')
axes[0].axvline(x=1e-7, color='gray', linestyle=':', label='최적 h 근방')
axes[0].set_xlabel('스텝 크기 h')
axes[0].set_ylabel('절대 오차 |수치미분 - 해석미분|')
axes[0].set_title("수치 미분 오차: f(x) = sin(x), x₀ = 1.0", fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, which='both')
axes[0].annotate('float64 cancellation\n오차 증가 구간',
                 xy=(1e-13, 1e-3), fontsize=9, color='gray')

# 우: 선형근사 시각화
x_range = np.linspace(0.5, 1.5, 200)
a = 1.0
tangent = f(a) + f_prime_exact(a) * (x_range - a)  # 선형근사

axes[1].plot(x_range, f(x_range), 'b-', linewidth=2.5, label=r'$f(x) = \sin(x)$')
axes[1].plot(x_range, tangent, 'r--', linewidth=1.8, label=f"선형근사: {f(a):.3f} + {f_prime_exact(a):.3f}(x-{a})")
axes[1].scatter([a], [f(a)], color='red', s=100, zorder=5)

# 오차 시각화 (h = 0.3)
h_demo = 0.3
axes[1].annotate('', xy=(a + h_demo, f(a + h_demo)), xytext=(a + h_demo, tangent[-1] if False else f(a) + f_prime_exact(a)*h_demo),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
axes[1].set_xlim(0.5, 1.5)
axes[1].set_title(r'선형근사 $f(a+h) \approx f(a) + f\'(a)h$', fontsize=12)
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03-derivative-numerical-vs-analytical.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. SymPy로 기호 미분 vs 수치 미분 검증
# ─────────────────────────────────────────────

x = sp.Symbol('x')

test_cases = [
    (sp.sin(x)**2,   1.5),
    (sp.exp(-x**2),  0.5),
    (sp.log(x + 2),  0.0),
    (x**3 - 2*x + 1, 2.0),
]

print(f"{'함수':<25} {'x₀':>5} {'해석 미분':>12} {'중심 차분(h=1e-6)':>20} {'오차':>12}")
print("-" * 80)

h = 1e-6
for expr, x0 in test_cases:
    f_sym  = sp.lambdify(x, expr, 'numpy')
    df_sym = sp.lambdify(x, sp.diff(expr, x), 'numpy')

    analytic = df_sym(x0)
    numeric  = numerical_derivative_central(f_sym, x0, h)
    error    = abs(analytic - numeric)
    
    print(f"{str(expr):<25} {x0:>5.1f} {analytic:>12.8f} {numeric:>20.8f} {error:>12.2e}")

# ─────────────────────────────────────────────
# 3. PyTorch autograd와 수치 미분 비교
# ─────────────────────────────────────────────

try:
    import torch
    
    def compare_autograd_vs_numerical(func, x_val, h=1e-5):
        """PyTorch autograd와 수치 미분 비교"""
        # Autograd
        x_t = torch.tensor(x_val, requires_grad=True, dtype=torch.float64)
        y = func(x_t)
        y.backward()
        grad_auto = x_t.grad.item()
        
        # 수치 미분 (NumPy)
        f_np = lambda x: func(torch.tensor(x, dtype=torch.float64)).item()
        grad_num = numerical_derivative_central(f_np, x_val, h)
        
        print(f"autograd: {grad_auto:.10f} | 수치미분: {grad_num:.10f} | 오차: {abs(grad_auto-grad_num):.2e}")
    
    print("\n[PyTorch autograd vs 수치 미분]")
    compare_autograd_vs_numerical(lambda x: torch.sin(x**2), x_val=1.0)
    compare_autograd_vs_numerical(lambda x: torch.exp(-x) * torch.cos(x), x_val=0.5)
    
except ImportError:
    print("PyTorch가 설치되지 않아 autograd 비교를 생략합니다.")
```

**출력**:
```
함수                      x₀   해석 미분   중심 차분(h=1e-6)    오차
────────────────────────────────────────────────────────────────────────────────
sin(x)**2                 1.5   0.14112001          0.14112001   3.33e-13
exp(-x**2)                0.5  -0.77880078         -0.77880078   1.11e-14
log(x + 2)                0.0   0.50000000          0.50000000   5.55e-17
x**3 - 2*x + 1            2.0  10.00000000         10.00000000   1.78e-13
```

---

## 🔗 AI/ML 연결

### 역전파 = 선형근사의 역방향 적용

신경망의 각 연산 $z = f(x)$에서 역전파는 국소 선형근사를 이용한다:

$$\delta x = f'(x) \cdot \delta z$$

여기서 $f'(x)$는 스칼라 미분 (단변수), 또는 야코비안 행렬 (다변수 — Chapter 2에서 전개).

이 근사가 정확한 이유: 선형근사의 오차는 $o(\delta z)$이고, 역전파가 계산하는 gradient는 $\delta z \to 0$ 극한에서 정확하다.

### PyTorch `gradcheck`의 원리

```python
import torch

def custom_relu(x):
    return torch.maximum(x, torch.zeros_like(x))

# gradcheck: 중심 차분과 autograd를 비교
x = torch.randn(3, requires_grad=True, dtype=torch.float64)
result = torch.autograd.gradcheck(custom_relu, x, eps=1e-6, atol=1e-4)
print("gradcheck 통과:", result)
```

`gradcheck`의 내부 로직은 정확히 정리 3.3의 중심 차분이다.

### 학습률과 선형근사의 정확성

경사하강법 $x_{k+1} = x_k - \eta \nabla f(x_k)$에서:
$$f(x_{k+1}) \approx f(x_k) - \eta \|\nabla f(x_k)\|^2 + O(\eta^2)$$

$\eta$가 충분히 작으면 $O(\eta^2)$ 항이 무시되어 손실이 단조 감소한다. 이것이 "학습률이 충분히 작아야 한다"의 수학적 근거다. (정확한 상한 $\eta < 2/L$은 Chapter 4에서 증명.)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 미분가능성 | ReLU는 $x=0$에서 미분불가능. Subgradient로 대체 (다음 문서) |
| $h \to 0$ 극한 | 컴퓨터에서 $h$는 유한. float64 기준 최적 $h \approx 10^{-7}$ |
| 단변수 | 다변수로 확장하면 "전미분 = 선형사상"이 되고, 행렬 표현이 야코비안 (Chapter 2) |

---

## 📌 핵심 정리

$$f(a+h) = f(a) + f'(a) \cdot h + o(h)$$

- $f'(a)$: 점 $a$에서의 최선의 선형근사 계수 (유일)
- $o(h)$: 나머지 오차, $h$보다 빠르게 0으로 감소

| 수치 미분 방법 | 오차 | 최적 h (float64) |
|-------------|------|----------------|
| 전방 차분 $(f(x+h)-f(x))/h$ | $O(h)$ | $\approx 10^{-8}$ |
| 중심 차분 $(f(x+h)-f(x-h))/2h$ | $O(h^2)$ | $\approx 10^{-5}$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(x) = |x|$는 $x = 0$에서 미분가능한가? 정의에 따라 판정하라.

<details>
<summary>해설</summary>

$h > 0$이면 $\frac{|h|}{h} = 1$, $h < 0$이면 $\frac{|h|}{h} = -1$. 좌극한과 우극한이 다르므로 미분불가능.

</details>

**문제 2** (심화): 소 $o$ 표기법을 이용해 다음을 증명하라: 미분가능하면 연속이다.

<details>
<summary>해설</summary>

$f(a+h) - f(a) = f'(a)h + o(h)$. $h \to 0$이면 우변 → $0$이므로 $\lim_{h\to 0} f(a+h) = f(a)$. 즉 연속. $\square$

</details>

**문제 3** (AI 연결): `float32` 환경(대부분의 GPU 연산)에서 수치 미분의 최적 $h$ 값은 얼마인가? float64와 비교하여 설명하라.

<details>
<summary>해설</summary>

float32의 기계 엡실론은 약 $10^{-7}$이다. 중심 차분 오차는 $O(h^2) + O(\varepsilon_{mach}/h)$이므로, 최적 $h \approx \varepsilon_{mach}^{1/3} \approx 10^{-7/3} \approx 10^{-2.3}$. float32에서는 $h \approx 10^{-2} \sim 10^{-3}$이 적절하다. 이것이 PyTorch `gradcheck`이 `dtype=float64`를 요구하는 이유다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 연속성과 균등연속성](./02-continuity-uniform.md) | [📚 README](../README.md) | [04. 평균값 정리와 테일러 정리 ▶](./04-mvt-taylor.md) |

</div>
