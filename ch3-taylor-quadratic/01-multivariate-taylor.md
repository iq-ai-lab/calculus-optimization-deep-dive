# 01. 다변수 테일러 정리 완전 유도

## 🎯 핵심 질문

- 단변수 테일러 정리를 다변수로 어떻게 일반화하는가?
- $f(x+h) = f(x) + \nabla f^\top h + \frac{1}{2}h^\top H h + o(\|h\|^2)$를 엄밀히 유도할 수 있는가?
- 나머지 항을 어떻게 추정하는가?
- 이 전개가 경사하강법과 뉴턴 방법에 어떻게 직접 연결되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

다변수 테일러 전개는 최적화 알고리즘의 이론적 핵심이다:
- **1차 근사** → 경사하강법: $f(x-\eta\nabla f) \approx f(x) - \eta\|\nabla f\|^2$
- **2차 근사** → 뉴턴 방법: 2차 항을 최소화하면 $x \leftarrow x - H^{-1}\nabla f$
- **L-smooth 조건**: 2차 여분항의 상한 $\frac{L}{2}\|h\|^2$이 수렴 조건을 결정

---

## 📐 수학적 선행 조건

- [Ch1-04. 단변수 테일러 정리](../ch1-analysis-foundations/04-mvt-taylor.md)
- [Ch2-04. 야코비안과 헤시안](../ch2-multivariable-calculus/04-jacobian-hessian-geometry.md)
- [Ch2-05. 다변수 연쇄법칙](../ch2-multivariable-calculus/05-chain-rule-general.md)

---

## ✏️ 엄밀한 정의와 정리

### 정리 3.1 — 2차 다변수 테일러 정리

**명제**: $f: \mathbb{R}^n \to \mathbb{R}$이 $C^2$이면, 점 $x$ 근방의 임의의 $h \in \mathbb{R}^n$에 대해:

$$f(x+h) = f(x) + \nabla f(x)^\top h + \frac{1}{2} h^\top H_f(x) h + R_2(x, h)$$

여기서 나머지 항: $\lim_{\|h\|\to 0} \frac{R_2(x,h)}{\|h\|^2} = 0$

**증명**:  
$g(t) = f(x + th)$로 놓으면 $g: [0,1] \to \mathbb{R}$.

**1차 도함수**: 연쇄법칙에 의해
$$g'(t) = \nabla f(x+th)^\top h$$

**2차 도함수**:
$$g''(t) = h^\top H_f(x+th) h$$

단변수 테일러 정리 ($g$에 적용, 전개점 $t=0$):
$$g(1) = g(0) + g'(0) \cdot 1 + \frac{1}{2}g''(0) \cdot 1^2 + R$$

대입:
- $g(1) = f(x+h)$
- $g(0) = f(x)$
- $g'(0) = \nabla f(x)^\top h$
- $g''(0) = h^\top H_f(x) h$

따라서:
$$f(x+h) = f(x) + \nabla f(x)^\top h + \frac{1}{2} h^\top H_f(x) h + R_2$$

**나머지 항 추정** ($f \in C^2$):
$$R_2 = \frac{1}{2}\int_0^1 (1-t) h^\top [H_f(x+th) - H_f(x)] h \, dt$$

$H_f$의 연속성과 $\|H_f(x+th) - H_f(x)\| = o(1)$으로부터:
$$|R_2| \leq \frac{\|h\|^2}{2} \sup_{t\in[0,1]} \|H_f(x+th) - H_f(x)\| = o(\|h\|^2) \quad \square$$

---

### 정리 3.2 — 고차 다변수 테일러

$f \in C^k$이면:
$$f(x+h) = \sum_{|\alpha| \leq k} \frac{1}{\alpha!} (D^\alpha f)(x) h^\alpha + O(\|h\|^{k+1})$$

여기서 $\alpha = (\alpha_1, \ldots, \alpha_n)$은 다중지수, $D^\alpha = \frac{\partial^{|\alpha|}}{\partial x_1^{\alpha_1}\cdots\partial x_n^{\alpha_n}}$.

**주요 항 (n=2, k=2)**:
$$f(x_0+h_1, y_0+h_2) \approx f + f_x h_1 + f_y h_2 + \frac{1}{2}(f_{xx}h_1^2 + 2f_{xy}h_1h_2 + f_{yy}h_2^2)$$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 2차 테일러 근사와 실제 함수 비교
# ─────────────────────────────────────────────

def taylor_approx_2nd(f, grad_f, hess_f, x0, h):
    """2차 테일러 근사"""
    return f(x0) + grad_f(x0) @ h + 0.5 * h @ hess_f(x0) @ h

def taylor_approx_1st(f, grad_f, x0, h):
    """1차 테일러 근사"""
    return f(x0) + grad_f(x0) @ h

# 테스트: f(x,y) = exp(x) * cos(y)
def f(x):      return np.exp(x[0]) * np.cos(x[1])
def grad_f(x): return np.array([np.exp(x[0])*np.cos(x[1]), -np.exp(x[0])*np.sin(x[1])])
def hess_f(x): return np.exp(x[0]) * np.array([[np.cos(x[1]), -np.sin(x[1])],
                                                 [-np.sin(x[1]), -np.cos(x[1])]])

x0 = np.array([0.0, 0.0])
h_norms = np.logspace(-2, 0, 30)

err_1st = []; err_2nd = []
for hn in h_norms:
    h = hn * np.array([1.0, 0.5]) / np.linalg.norm([1.0, 0.5])
    true_val = f(x0 + h)
    err_1st.append(abs(true_val - taylor_approx_1st(f, grad_f, x0, h)))
    err_2nd.append(abs(true_val - taylor_approx_2nd(f, grad_f, hess_f, x0, h)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].loglog(h_norms, err_1st, 'b-o', markersize=4, label='1차 근사 오차 O(h²)')
axes[0].loglog(h_norms, err_2nd, 'r-s', markersize=4, label='2차 근사 오차 O(h³)')
axes[0].loglog(h_norms, h_norms**2, 'b--', alpha=0.5, label='h² 참조')
axes[0].loglog(h_norms, h_norms**3, 'r--', alpha=0.5, label='h³ 참조')
axes[0].set_xlabel('‖h‖'); axes[0].set_ylabel('|f(x+h) - 근사|')
axes[0].set_title(r"테일러 근사 오차: $f=e^x\cos y$", fontsize=11)
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, which='both')

# 2차 근사 등위선 vs 실제 등위선
x_r = np.linspace(-1.5, 1.5, 200)
y_r = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x_r, y_r)
Z_true = np.exp(X) * np.cos(Y)

# 2차 테일러 at (0,0)
g0 = grad_f(x0); H0 = hess_f(x0); f0 = f(x0)
Z_taylor = f0 + g0[0]*X + g0[1]*Y + 0.5*(H0[0,0]*X**2 + 2*H0[0,1]*X*Y + H0[1,1]*Y**2)

levels = np.linspace(-1.5, 2.5, 15)
axes[1].contour(X, Y, Z_true,   levels=levels, colors='blue',  linestyles='-',  linewidths=1, alpha=0.7)
axes[1].contour(X, Y, Z_taylor, levels=levels, colors='red',   linestyles='--', linewidths=1, alpha=0.7)
axes[1].scatter(*x0, s=80, color='black', zorder=5)
from matplotlib.lines import Line2D
h_lines = [Line2D([0],[0],color='blue',lw=2,label='실제 f'),
           Line2D([0],[0],color='red',lw=2,ls='--',label='2차 근사')]
axes[1].legend(handles=h_lines, fontsize=9)
axes[1].set_title('실제 등위선 vs 2차 테일러 등위선', fontsize=11)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch3-01-multivariate-taylor.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. SymPy로 기호 테일러 전개
# ─────────────────────────────────────────────

x1, x2 = sp.symbols('x1 x2')
h1, h2  = sp.symbols('h1 h2')

f_sym = sp.exp(x1) * sp.cos(x2)

# x0 = (0,0)에서 2차 테일러 전개
f0_val   = f_sym.subs([(x1, 0), (x2, 0)])
df_dx1   = sp.diff(f_sym, x1).subs([(x1,0),(x2,0)])
df_dx2   = sp.diff(f_sym, x2).subs([(x1,0),(x2,0)])
d2f_dx1  = sp.diff(f_sym, x1, 2).subs([(x1,0),(x2,0)])
d2f_dx12 = sp.diff(f_sym, x1, x2).subs([(x1,0),(x2,0)])
d2f_dx2  = sp.diff(f_sym, x2, 2).subs([(x1,0),(x2,0)])

taylor_2nd = (f0_val + df_dx1*h1 + df_dx2*h2 
              + sp.Rational(1,2)*(d2f_dx1*h1**2 + 2*d2f_dx12*h1*h2 + d2f_dx2*h2**2))

print("[SymPy 2차 테일러 전개 at (0,0)]")
print(f"f(0+h) ≈ {sp.expand(taylor_2nd)}")
```

---

## 🔗 AI/ML 연결

### 경사하강법 한 스텝의 2차 근사

$y = x - \eta \nabla f(x)$로 놓으면 $h = -\eta\nabla f(x)$:

$$f(x - \eta\nabla f) \approx f(x) - \eta\|\nabla f\|^2 + \frac{\eta^2}{2} (\nabla f)^\top H (\nabla f)$$

$H$의 최대 고유값 $\leq L$이면 $(\nabla f)^\top H (\nabla f) \leq L\|\nabla f\|^2$:

$$f(x - \eta\nabla f) \leq f(x) - \eta(1 - \frac{\eta L}{2})\|\nabla f\|^2$$

$\eta < 2/L$이면 손실 감소 보장.

### 뉴턴 방법의 2차 근사 최소화

$$f(x+h) \approx f(x) + \nabla f^\top h + \frac{1}{2}h^\top H h$$

$\nabla_h (\text{우변}) = \nabla f + Hh = 0 \implies h = -H^{-1}\nabla f$

즉 뉴턴 방법은 2차 테일러 근사의 최솟점으로 이동.

---

## 📌 핵심 정리

$$f(x+h) = f(x) + \nabla f(x)^\top h + \frac{1}{2}h^\top H_f(x) h + o(\|h\|^2)$$

| 근사 차수 | 오차 | 이용하는 알고리즘 |
|---------|------|----------------|
| 0차: $f(x)$ | $O(\|h\|)$ | — |
| 1차: $+\nabla f^\top h$ | $O(\|h\|^2)$ | 경사하강법 |
| 2차: $+\frac{1}{2}h^\top Hh$ | $O(\|h\|^3)$ | 뉴턴 방법, L-BFGS |

---

## 🤔 생각해볼 문제

**문제 1**: $f(x,y) = x^3 + y^3 - 3xy$의 점 $(1,1)$에서 2차 테일러 전개를 구하라.

<details><summary>해설</summary>
$\nabla f = (3x^2-3y, 3y^2-3x)|_{(1,1)} = (0,0)$. $H = \begin{pmatrix}6x&-3\\-3&6y\end{pmatrix}|_{(1,1)} = \begin{pmatrix}6&-3\\-3&6\end{pmatrix}$.  
$f(1+h_1, 1+h_2) \approx -1 + 3h_1^2 - 3h_1h_2 + 3h_2^2$.
</details>

**문제 2 (AI 연결)**: L-BFGS는 왜 2차 테일러 근사를 이용하면서도 헤시안을 직접 계산하지 않는가?

<details><summary>해설</summary>
$n$차원에서 헤시안 저장은 $O(n^2)$, 역행렬 계산은 $O(n^3)$. L-BFGS는 최근 $m$개의 gradient 차이 벡터 쌍으로 $H^{-1}$을 암묵적으로 근사하여 $O(mn)$ 비용으로 VJP를 계산.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch2-06. 음함수 정리](../ch2-multivariable-calculus/06-implicit-function-theorem.md) | [📚 README](../README.md) | [02. 헤시안의 고유값과 국소 기하 ▶](./02-hessian-eigenvalues-geometry.md) |

</div>
