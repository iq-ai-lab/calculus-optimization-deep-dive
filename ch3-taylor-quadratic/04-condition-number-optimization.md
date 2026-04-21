# 04. 조건수와 최적화 속도

## 🎯 핵심 질문

- 헤시안 조건수 $\kappa = \lambda_{\max}/\lambda_{\min}$이 등고선을 타원으로 만드는 이유는?
- GD의 수렴 속도가 조건수에 어떻게 의존하는가?
- Preconditioning과 Batch Normalization은 조건수를 어떻게 개선하는가?
- 뉴턴 방법은 왜 조건수에 무관하게 2차 수렴하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **학습 불안정의 핵심 원인**: 조건수 $\kappa \gg 1$이면 GD의 수렴이 진동하거나 매우 느려진다. 이것이 딥러닝에서 학습률 조정이 까다로운 근본 이유다.
- **Batch Normalization의 이론적 근거**: BN이 훈련을 안정화하는 한 가지 이유는 각 층의 활성화 분포를 정규화하여 손실 함수의 헤시안 조건수를 감소시키기 때문이다.
- **Adam vs SGD**: Adam의 좌표별 적응 학습률은 대각 Preconditioning으로 볼 수 있다 — 조건수를 암묵적으로 개선.

---

## 📐 수학적 선행 조건

- [Ch3-01. 다변수 테일러 정리](./01-multivariate-taylor.md)
- [Ch3-02. 헤시안의 고유값과 국소 기하](./02-hessian-eigenvalues-geometry.md)
- 선형대수: 조건수, 스펙트럼 반경

---

## 🔬 정리와 증명

### 정의 3.2 — 조건수

행렬 $A$의 **조건수(condition number)**:
$$\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

대칭 양정부호 행렬(헤시안)에서 특이값 = 고유값이므로:
$$\kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$$

---

### 정리 3.8 — 조건수와 등고선 타원

**명제**: $f(x) = \frac{1}{2}x^\top H x$ ($H$ PD)의 등위집합 $\{x: f(x) = c\}$는 타원이며, 반축 길이 비가 $\sqrt{\lambda_{\max}/\lambda_{\min}} = \sqrt{\kappa(H)}$이다.

**증명**:  
Spectral Theorem: $H = Q\Lambda Q^\top$. $y = Q^\top x$로 변수 변환하면:

$$f(x) = \frac{1}{2}x^\top Q\Lambda Q^\top x = \frac{1}{2}y^\top \Lambda y = \frac{1}{2}\sum_i \lambda_i y_i^2$$

등위 집합: $\sum_i \lambda_i y_i^2 = 2c \iff \sum_i \frac{y_i^2}{2c/\lambda_i} = 1$

이것은 반축 길이 $r_i = \sqrt{2c/\lambda_i}$인 타원. 반축 비 = $\sqrt{\lambda_{\max}/\lambda_{\min}} = \sqrt{\kappa}$. $\square$

---

### 정리 3.9 — GD 수렴률과 조건수 (강볼록 경우)

**명제**: $f$가 $\mu$-강볼록, $L$-smooth이면, 학습률 $\eta = 2/(\mu + L)$에서 GD의 수렴:

$$f(x_k) - f^* \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2k} (f(x_0) - f^*)$$

여기서 $\kappa = L/\mu$.

**증명 스케치**:  
최적 학습률 $\eta^* = 2/(\mu+L)$에서 각 스텝의 진행:

$$\|x_{k+1} - x^*\|^2 \leq \left(1 - \frac{2\mu L}{(\mu+L)^2}\right) \|x_k - x^*\|^2 = \left(\frac{L-\mu}{L+\mu}\right)^2 \|x_k - x^*\|^2$$

수렴 계수 $\rho = \left(\frac{\kappa-1}{\kappa+1}\right)^2$.

- $\kappa = 1$ (등방성): $\rho = 0$, 한 스텝에 수렴
- $\kappa = 10$: $\rho \approx 0.67$, 느린 수렴
- $\kappa = 100$: $\rho \approx 0.96$, 매우 느린 수렴 $\square$

---

### 정리 3.10 — 뉴턴 방법은 조건수에 무관

**명제**: 뉴턴 방법 $x_{k+1} = x_k - H_k^{-1}\nabla f(x_k)$는 $f$가 2차 함수 $f(x) = \frac{1}{2}x^\top H x + b^\top x$이면 단 **1번 스텝에 수렴**한다.

**증명**:  
$\nabla f(x) = Hx + b$. 뉴턴 스텝: $x_1 = x_0 - H^{-1}(Hx_0 + b) = -H^{-1}b = x^*$. 조건수에 무관. $\square$

**직관**: 뉴턴 방법은 헤시안으로 입력 공간을 "재스케일"하여 조건수를 1로 만드는 Preconditioning과 동치이다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 조건수 vs GD 수렴 속도 실험
# ─────────────────────────────────────────────

def run_gd(H, b, x0, lr, n_steps=500):
    """f(x) = 0.5 xᵀHx + bᵀx 에서 GD"""
    x = x0.copy()
    losses = []
    x_star = -np.linalg.solve(H, b)  # 정확한 최솟값
    f_star = 0.5 * x_star @ H @ x_star + b @ x_star
    
    for _ in range(n_steps):
        f_val = 0.5 * x @ H @ x + b @ x
        losses.append(f_val - f_star)
        grad = H @ x + b
        x -= lr * grad
    return losses

# 다양한 조건수
kappas = [1, 5, 20, 100]
colors  = ['green', 'blue', 'orange', 'red']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for kappa, color in zip(kappas, colors):
    n = 50
    mu = 1.0
    L  = kappa * mu
    
    # H = 대각 행렬, 고유값이 [mu, ..., L]에 균등 분포
    eigvals = np.linspace(mu, L, n)
    H = np.diag(eigvals)
    b = np.zeros(n)
    x0 = np.ones(n)
    
    lr = 2 / (mu + L)  # 최적 학습률
    losses = run_gd(H, b, x0, lr)
    axes[0].semilogy(losses[:200], color=color, linewidth=1.8, label=f'κ={kappa}')

axes[0].set_xlabel('반복 횟수')
axes[0].set_ylabel('f(x_k) - f*')
axes[0].set_title('조건수에 따른 GD 수렴 속도', fontsize=11)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, which='both')

# ─────────────────────────────────────────────
# 2. 등고선 타원 시각화: 조건수 = 축 비
# ─────────────────────────────────────────────

x_r = np.linspace(-3, 3, 200)
y_r = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_r, y_r)

cases = [(1, 1, "κ=1 (원)"), (1, 4, "κ=4"), (1, 16, "κ=16")]
for (lam1, lam2, label), style in zip(cases, ['solid','dashed','dotted']):
    Z = 0.5 * (lam1*X**2 + lam2*Y**2)
    axes[1].contour(X, Y, Z, levels=[0.5, 1, 2, 4], linestyles=style,
                    linewidths=1.5, colors=[color for color in ['blue','green','red']])
    # 더미 선 (legend용)
    axes[1].plot([], [], linestyle=style, color='black', linewidth=1.5, label=label)

axes[1].set_aspect('equal')
axes[1].set_title('조건수와 등위 타원 형태', fontsize=11)
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

plt.tight_layout()
plt.savefig('ch3-04-condition-number.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. GD vs 뉴턴 방법 비교
# ─────────────────────────────────────────────

def run_newton(H, b, x0, n_steps=20):
    x = x0.copy()
    x_star = -np.linalg.solve(H, b)
    f_star = 0.5*x_star@H@x_star + b@x_star
    losses = []
    for _ in range(n_steps):
        f_val = 0.5*x@H@x + b@x
        losses.append(f_val - f_star)
        grad = H@x + b
        x -= np.linalg.solve(H, grad)
    return losses

n = 20; kappa_test = 50
eigvals = np.linspace(1.0, kappa_test, n)
H = np.diag(eigvals)
b = np.ones(n)
x0 = np.zeros(n)

losses_gd  = run_gd(H, b, x0, lr=2/(1+kappa_test), n_steps=500)
losses_nwt = run_newton(H, b, x0, n_steps=5)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(losses_gd[:100], 'b-', linewidth=2, label=f'GD (κ={kappa_test})')
ax.semilogy(range(len(losses_nwt)), losses_nwt, 'r-o', linewidth=2,
            markersize=8, label='뉴턴 방법 (단 1스텝)')
ax.set_xlabel('반복 횟수')
ax.set_ylabel('f(x_k) - f*')
ax.set_title(f'GD vs 뉴턴 방법 (κ={kappa_test})', fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('ch3-04-gd-vs-newton.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 4. Preconditioning 효과
# ─────────────────────────────────────────────

# Preconditioned GD: x ← x - P⁻¹∇f (대각 스케일링)
def run_precond_gd(H, b, x0, lr, P_inv, n_steps=200):
    x = x0.copy()
    x_star = -np.linalg.solve(H, b)
    f_star = 0.5*x_star@H@x_star + b@x_star
    losses = []
    for _ in range(n_steps):
        losses.append(0.5*x@H@x + b@x - f_star)
        grad = H@x + b
        x -= lr * (P_inv * grad)  # 대각 P의 경우
    return losses

kappa_p = 100
L_p = kappa_p; mu_p = 1.0
eigvals_p = np.linspace(mu_p, L_p, n)
H_p = np.diag(eigvals_p)
b_p = np.ones(n)
x0_p = np.zeros(n)

# 표준 GD
losses_std = run_gd(H_p, b_p, x0_p, lr=2/(mu_p+L_p))
# 대각 Preconditioning (완벽: P = H의 대각 성분)
P_inv_diag = 1.0 / eigvals_p  # 이상적인 대각 Preconditioner
losses_pre = run_precond_gd(H_p, b_p, x0_p, lr=1.0, P_inv=P_inv_diag)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(losses_std[:200], 'b-', linewidth=2, label=f'표준 GD (κ={kappa_p})')
ax.semilogy(losses_pre[:200], 'r-', linewidth=2, label='대각 Preconditioning (κ→1)')
ax.set_xlabel('반복 횟수'); ax.set_ylabel('손실 갭')
ax.set_title('Preconditioning의 효과 (Adam의 직관)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('ch3-04-preconditioning.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"조건수 κ={kappa_p}에서:")
print(f"  이론적 GD 수렴 계수: ρ = {((kappa_p-1)/(kappa_p+1))**2:.4f}")
print(f"  Preconditioning 후: κ=1 → ρ=0 (1스텝)")
```

---

## 🔗 AI/ML 연결

### Batch Normalization과 조건수

BN이 훈련 안정화에 기여하는 메커니즘 중 하나:
1. 각 층의 활성화를 평균 0, 분산 1로 정규화
2. 이에 따라 손실 함수의 헤시안 고유값 분포가 더 균일해짐
3. 조건수 $\kappa$ 감소 → GD의 수렴 속도 개선

Santurkar et al. 2018 "How Does Batch Normalization Help Optimization?"에서 BN이 Loss Landscape를 더 매끄럽게 (L이 작아짐, $\kappa$도 감소) 만든다는 것을 보였다.

### Adam = 좌표별 Preconditioning

Adam의 업데이트: $\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

$\sqrt{\hat{v}_t}$는 gradient의 RMS — 헤시안의 대각 성분 제곱근 근사.

따라서 Adam은 $P^{-1} \approx \text{diag}(H)^{-1/2}$인 대각 Preconditioning이다. 이것이 Adam이 조건수가 큰 문제에서 SGD보다 빠른 이유다.

---

## 📌 핵심 정리

$$\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}: \text{수렴 속도 결정}$$

| 조건수 $\kappa$ | GD 수렴 계수 $\rho$ | 실제 수렴 |
|---------------|------------------|---------|
| 1 (완벽) | 0 | 1스텝 |
| 10 | 0.67 | 느림 |
| 100 | 0.96 | 매우 느림 |
| ∞ | 1 | 수렴 안 함 |

**해결책**:
- Preconditioning: $P^{-1}H$의 조건수 감소 (Adam, Natural Gradient)
- 뉴턴 방법: $H^{-1}H = I$ → $\kappa = 1$
- Batch Normalization: 암묵적 조건수 감소

---

## 🤔 생각해볼 문제

**문제 1**: $f(x,y) = 0.5x^2 + 50y^2$에서 GD의 최적 학습률과 수렴 계수를 계산하라.

<details><summary>해설</summary>
$\mu=1$, $L=100$, $\kappa=100$. $\eta^* = 2/(1+100) \approx 0.0198$. $\rho = ((100-1)/(100+1))^2 \approx 0.961$. 1% 감소에 수십 스텝 필요.
</details>

**문제 2 (AI 연결)**: 왜 Transformer 훈련에서 Learning Rate Warmup이 필요한가? 조건수 관점에서 설명하라.

<details><summary>해설</summary>
초기화 직후 헤시안의 $\lambda_{\max}$가 매우 클 수 있다 (파라미터가 임의 초기화되어 손실 표면이 날카로움). 큰 학습률이 $\eta > 1/L$을 위반하면 발산. Warmup은 $L$이 큰 초기에 $\eta$를 작게 유지하고, 학습이 진행됨에 따라 (Loss가 평탄해지면서 $L$이 감소) $\eta$를 높이는 전략.
</details>

---

## 🔁 Chapter 3 요약 및 Chapter 4 예고

이 챕터에서 배운 것들:

| 문서 | 핵심 도구 | AI/ML 연결 |
|------|---------|-----------|
| 01. 다변수 테일러 | $f(x+h) \approx f+\nabla f^\top h+\frac{1}{2}h^\top Hh$ | GD/뉴턴의 이론적 기반 |
| 02. 헤시안과 고유값 | PD/ND/Indefinite 판정 | Flat Minima, 안장점 |
| 03. 안장점·볼록성 | $D = f_{xx}f_{yy}-f_{xy}^2$ | SGD 안장점 탈출 |
| 04. 조건수 | $\kappa = \lambda_{\max}/\lambda_{\min}$ | BN, Adam, Preconditioning |

**Chapter 4 예고**: 테일러 전개가 "학습률 상한 $\eta < 2/L$"을 도출하는 것을 보았다. Chapter 4에서는 이를 이용해 GD의 수렴률 $O(1/k)$를 엄밀히 증명하고, Momentum, SGD, Adam의 수렴까지 분석한다.

---

<div align="center">

| | |
|---|---|
| [◀ 03. 안장점과 볼록성 판정](./03-saddle-points-convexity.md) | [📚 README](../README.md) |

</div>

<div align="center">

🔜 **다음 챕터**: [Chapter 4: 기울기 기반 최적화 — 경사하강법 완전 분석 →](../ch4-gradient-optimization/01-gd-convergence-convex.md)

</div>
