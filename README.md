<div align="center">

# 🧮 Calculus & Optimization Deep Dive

### `loss.backward()` 를 **호출하는 것** 과,

### 역전파가 **야코비안의 연쇄 행렬곱**

$$\frac{\partial L}{\partial \theta_1} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdots \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial \theta_1}$$

### 이라는 **기하학적 본질** 을 아는 것은 **다르다.**

<br/>

> *Adam 옵티마이저를 **쓰는 것** 과, **Adam 은 볼록이 아닌 문제에서 수렴을 보장하지 않는다** 는 사실을 Reddi 2018 의 반례로 알 수 있는 것은 다르다.*
>
> *$\nabla$ 기호를 **외우는 것** 과, 방향도함수가 기울기와 어떻게 연결되는지 — **코시–슈바르츠 부등식**으로*
>
> $$\nabla_u f = \langle \nabla f, u \rangle \;\leq\; \|\nabla f\|$$
>
> *증명할 수 있는 것은 다르다.*

<br/>

**다루는 정리 (시간순)**

Cauchy 1821 *ε–δ 정의 + 코시–슈바르츠* · Newton–Leibniz 1687 *미적분 기본정리* · Lagrange 1797 *Lagrange 승수* · Cauchy 1847 *Steepest descent* · Werbos 1974 / Rumelhart 1986 *Backpropagation* · Robbins–Monro 1951 *확률근사* · Nesterov 1983 *$O(1/T^2)$ 가속* · Kingma 2014 *Adam* · Reddi 2018 *Adam 반례*

<br/>

**핵심 질문**

> **왜 이렇게 작동하는가** — ε–δ 정의부터 시작해 **야코비안 행렬곱으로서의 역전파** · 옵티마이저의 수렴 증명 · NumPy Autograd 구현까지, 신경망 학습의 수학적 기반을 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.12-3B5526?style=flat-square)](https://www.sympy.org/)
[![Docs](https://img.shields.io/badge/Docs-37개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

미적분과 최적화에 관한 자료는 넘쳐납니다. 하지만 대부분은 **"어떻게 계산하나"** 에서 멈춥니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "체인룰을 쓰면 역전파가 됩니다" | 다변수 연쇄법칙이 야코비안 행렬곱 $J_{f\circ g} = J_f \cdot J_g$임을 증명하고, 역방향 누적이 왜 forward 1회 + backward 1회로 완료되는지 Baur-Strassen 정리로 설명 |
| "Adam은 잘 수렴합니다" | Adam의 수렴이 볼록 문제에서조차 보장되지 않는 반례 (Reddi 2018), `AMSGrad`가 이를 어떻게 수정하는지 |
| "Gradient는 기울기입니다" | 방향도함수가 $\nabla f \cdot v$로 쓰이는 이유, 코시-슈바르츠로 "Gradient 방향이 최대 증가"임을 엄밀히 증명 |
| "헤시안은 2차 미분입니다" | 헤시안의 고유값 부호가 국소 극솟값/극댓값/안장점을 결정하는 Spectral Theorem 기반 충분조건 증명 |
| "Vanishing Gradient가 문제입니다" | 층별 야코비안의 곱 $\prod_l J_l$에서 스펙트럼 반경 $\rho < 1$이면 기하급수적 소실, $\rho > 1$이면 폭발하는 수치적 근거 |
| "라그랑주 승수법으로 풀면 됩니다" | $\lambda$가 왜 "그림자 가격"인지 — 제약을 한 단위 완화했을 때 목적함수가 $\lambda$만큼 바뀐다는 감도 분석 증명 |
| 이론 나열 | NumPy/SymPy로 직접 검증하는 실험 + Autograd 엔진 구현 + matplotlib 시각화 |

---

## 📌 선행 레포 & 후속 레포

```
[Linear Algebra Deep Dive]  ──►  이 레포  ──►  [Convex Optimization Deep Dive]
  야코비안, 헤시안은 행렬이다          미적분·최적화         볼록성, 쌍대성, Interior Point
  고유값·스펙트럼 이해 필수            의 수학적 기반         SVM, LP/QP, ADMM
```

> ⚠️ **선행 학습**: 야코비안 / 헤시안은 행렬입니다. 고유값과 스펙트럼 분해를 모른다면 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)를 먼저 학습하세요.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-ε--δ_극한의_엄밀한_정의-4A90D9?style=for-the-badge)](./ch1-analysis-foundations/01-epsilon-delta.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-편미분과_방향도함수-4A90D9?style=for-the-badge)](./ch2-multivariable-calculus/01-partial-directional-derivative.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-다변수_테일러_전개-4A90D9?style=for-the-badge)](./ch3-taylor-quadratic/01-multivariate-taylor.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-경사하강법_수렴_증명-4A90D9?style=for-the-badge)](./ch4-gradient-optimization/01-gd-convergence-convex.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-역전파_수학적_유도-4A90D9?style=for-the-badge)](./ch5-backpropagation/01-computational-graph-ad.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-라그랑주_승수법-4A90D9?style=for-the-badge)](./ch6-constrained-optimization/01-lagrange-multipliers.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Softmax_야코비안_유도-4A90D9?style=for-the-badge)](./ch7-ml-applications/01-softmax-jacobian.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 해석학의 기초 — 극한·연속·미분의 엄밀한 정의

> **핵심 질문:** "가까워진다"는 직관이 왜 수학적으로 부족한가? ε-δ 언어로 극한을 정의했을 때 무엇이 엄밀해지는가? 미분이 단순한 "기울기"가 아니라 "최선의 선형근사"라는 말의 의미는 무엇인가?

<details>
<summary><b>ε-δ 정의부터 Weierstrass 반례까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. ε-δ 언어 — 극한의 엄밀한 정의](./ch1-analysis-foundations/01-epsilon-delta.md) | $\lim_{x\to a} f(x) = L$의 ε-δ 정의, 직관적 "가까워진다"로는 부족한 이유(발산 함수 반례), 수열 극한과 함수 극한의 관계, 수렴 증명 예제 5개 (등비수열, $\sin x / x$, 복합 함수 등) |
| [02. 연속성과 균등연속성](./ch1-analysis-foundations/02-continuity-uniform.md) | 각 점에서의 연속($\delta$가 $x$에 의존) vs 균등연속($\delta$가 $x$에 무관), 하이네-보렐 정리와 콤팩트 집합 위의 연속 함수는 균등연속임을 증명, 최적화 문제에서 콤팩트 도메인이 왜 중요한가 |
| [03. 미분의 정의와 선형근사](./ch1-analysis-foundations/03-derivative-linear-approx.md) | $f'(a) = \lim_{h\to 0}\frac{f(a+h)-f(a)}{h}$가 "점 $a$에서의 최선의 선형근사"임을 엄밀히 증명, 선형근사 오차가 $o(h)$임을 보이는 방법, 수치 미분(유한차분)과 해석 미분의 오차 비교 실험 |
| [04. 평균값 정리와 테일러 정리](./ch1-analysis-foundations/04-mvt-taylor.md) | Rolle → MVT → 테일러 정리 순서로 완전 유도, Lagrange / Cauchy / 적분형 여분항(Remainder) 비교, Taylor 전개의 수렴 반경과 최적화에서의 역할($f(x+h) \approx f(x) + f'h + \frac{1}{2}f''h^2$) |
| [05. 미분가능 함수의 성질과 반례](./ch1-analysis-foundations/05-differentiability-properties.md) | 미분가능 ⇒ 연속의 증명, 역이 거짓인 반례 (Weierstrass 함수: 어디서도 미분불가능한 연속함수), SymPy로 Weierstrass 함수 부분합 시각화, 딥러닝의 ReLU가 미분불가능점을 가지면서도 Subgradient로 학습 가능한 이유 |

</details>

<br/>

### 🔹 Chapter 2: 다변수 미적분 — AI가 사는 고차원 공간

> **핵심 질문:** 편미분이 있어도 전미분이 없는 함수가 있는가? Gradient는 왜 "최대 증가 방향"인가? 야코비안과 헤시안은 기하학적으로 무엇을 뜻하는가?

<details>
<summary><b>편미분부터 음함수 정리까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 편미분과 방향도함수](./ch2-multivariable-calculus/01-partial-directional-derivative.md) | 편미분이 특정 방향 도함수의 특수 경우임을 정의로 증명, 모든 방향 미분이 존재해도 전미분이 존재하지 않는 반례($f(x,y)=\frac{xy^2}{x^2+y^4}$ 같은 함수), NumPy로 방향도함수 수치 계산 실험 |
| [02. 전미분과 야코비안](./ch2-multivariable-calculus/02-total-derivative-jacobian.md) | $f: \mathbb{R}^n \to \mathbb{R}^m$의 미분이 선형사상임을 정의, 행렬 표현이 야코비안인 이유, 편미분 존재 vs 전미분 존재의 차이를 ε-δ로 구분, PyTorch `.jacobian()`과 수치 야코비안 비교 |
| [03. Gradient와 코시-슈바르츠](./ch2-multivariable-calculus/03-gradient-cauchy-schwarz.md) | $D_v f = \nabla f \cdot v$의 유도, $\nabla f \cdot v \leq \|\nabla f\| \|v\|$ (코시-슈바르츠), 등호 조건이 $v \parallel \nabla f$임을 증명 → "Gradient는 최대 증가 방향"의 엄밀한 증명 완성, 경사하강법이 이 성질을 사용하는 방식 |
| [04. 야코비안과 헤시안의 기하](./ch2-multivariable-calculus/04-jacobian-hessian-geometry.md) | 야코비안: 1차 선형근사(국소 부피 변환율), 헤시안: 2차 곡률 (주곡률과 고유벡터 방향), $f: \mathbb{R}^n \to \mathbb{R}$의 2차 테일러 전개 완전 유도, SymPy로 헤시안 기호 계산 및 시각화 |
| [05. 다변수 연쇄법칙](./ch2-multivariable-calculus/05-chain-rule-general.md) | 다변수 연쇄법칙 $J_{f\circ g}(x) = J_f(g(x)) \cdot J_g(x)$의 엄밀한 증명 (전미분의 합성), 스칼라·벡터·행렬 각각의 연쇄법칙 예제, 역전파가 이 연쇄법칙을 역순으로 적용하는 것임을 예고 |
| [06. 음함수 정리](./ch2-multivariable-calculus/06-implicit-function-theorem.md) | $g(x, y) = 0$ 조건에서 야코비안 $\frac{\partial g}{\partial y}$가 가역이면 $y$를 $x$의 함수로 국소 표현 가능함을 증명, 제약 최적화에서의 역할(제약 곡면 위의 접평면 = $\ker(Dg)$), Deep Equilibrium Models에서의 응용 |

</details>

<br/>

### 🔹 Chapter 3: 다변수 테일러 전개와 2차 근사의 기하

> **핵심 질문:** 헤시안의 고유값 부호가 왜 극소/극대/안장점을 결정하는가? 딥러닝 Loss Landscape에 왜 안장점이 넘쳐나는가? 조건수가 최적화 속도에 어떤 영향을 주는가?

<details>
<summary><b>다변수 테일러 전개부터 Loss Landscape까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 다변수 테일러 정리 완전 유도](./ch3-taylor-quadratic/01-multivariate-taylor.md) | $f(x+h) = f(x) + \nabla f^\top h + \frac{1}{2} h^\top H h + o(\|h\|^2)$ 엄밀한 유도 (단변수 MVT에서 다변수로 귀납), 나머지 항 추정, SymPy로 기호 계산 vs 수치 계산 비교 |
| [02. 헤시안의 고유값과 국소 기하](./ch3-taylor-quadratic/02-hessian-eigenvalues-geometry.md) | 양의 정부호(PD) / 반양정부호(PSD) / 부정부호 헤시안과 극소·극대·안장점의 관계, Spectral Theorem으로 헤시안을 고유벡터 기저로 분해, 2차 충분조건의 엄밀한 증명, NumPy로 3D Loss Surface 시각화 |
| [03. 안장점과 볼록성 판정](./ch3-taylor-quadratic/03-saddle-points-convexity.md) | 제2계 충분조건 (2×2 판별식 $D = f_{xx}f_{yy} - f_{xy}^2$의 일반화), 딥러닝에서 극소값이 아닌 안장점이 지배적인 이유 (Dauphin 2014: 고차원에서 모든 고유값이 양수일 확률은 지수적으로 작음), matplotlib으로 안장점 궤적 시뮬레이션 |
| [04. 조건수와 최적화 속도](./ch3-taylor-quadratic/04-condition-number-optimization.md) | 헤시안 조건수 $\kappa(H) = \lambda_{\max}/\lambda_{\min}$가 등고선을 타원으로 만드는 이유, GD의 수렴 속도가 $\kappa$에 의존하는 증명, Preconditioning의 직관, Batch Norm이 조건수를 개선하는 원리 |

</details>

<br/>

### 🔹 Chapter 4: 기울기 기반 최적화 — 경사하강법 완전 분석

> **핵심 질문:** GD의 수렴률이 정확히 $O(1/k)$인 이유는 무엇인가? 학습률 상한 $\eta < 2/L$은 어떻게 유도되는가? Adam은 왜 수렴을 보장하지 못하는가?

<details>
<summary><b>볼록 GD 수렴 증명부터 뉴턴 방법까지 (7개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 경사하강법 수렴 — 볼록·매끄러운 경우](./ch4-gradient-optimization/01-gd-convergence-convex.md) | $L$-smooth 가정 하의 $f(x_k) - f^* \leq O(1/k)$ 증명 (Nesterov Thm 2.1.5), 강볼록(strongly convex) 경우의 선형 수렴 $O((1-\mu/L)^k)$, 각 가정이 깨질 때 수렴이 실패하는 반례 시뮬레이션 |
| [02. 학습률의 역할 — 수렴 조건 유도](./ch4-gradient-optimization/02-learning-rate-analysis.md) | $L$-smooth에서 $f(x - \eta\nabla f) \leq f(x) - \eta(1 - \frac{\eta L}{2})\|\nabla f\|^2$ 유도, $\eta < 2/L$이 수렴 조건인 이유, 학습률이 너무 크면 발산하는 수치 실험 (Lyapunov 함수 접근) |
| [03. 모멘텀과 Nesterov 가속](./ch4-gradient-optimization/03-momentum-nesterov.md) | Heavy-Ball 모멘텀의 직관 (물리적 유추), Nesterov가 $O(1/k^2)$를 달성하는 "미리 한 스텝 앞에서 gradient"의 수학적 이유, $\sqrt{\kappa}$ 의존성 개선 증명, Momentum vs Nesterov 등고선 궤적 비교 |
| [04. SGD의 수렴 — 확률적 분석](./ch4-gradient-optimization/04-sgd-convergence.md) | 기댓값 기반 $\mathbb{E}[f(x_k)] - f^* \leq O(1/\sqrt{k})$ 증명, 분산이 수렴에 미치는 영향, 학습률 스케줄링 ($O(1/k)$, cosine annealing)의 이론적 근거, Mini-batch의 분산 감소 효과 |
| [05. Adam, RMSProp, AdaGrad의 수학](./ch4-gradient-optimization/05-adam-rmsprop-adagrad.md) | AdaGrad → RMSProp → Adam 순서로 업데이트 식 유도, Adam의 편향 보정(bias correction) 이유, Reddi 2018 반례: 단순한 볼록 문제에서 Adam이 발산할 수 있음, AMSGrad가 이를 수정하는 방식 |
| [06. 뉴턴 방법과 Quasi-Newton](./ch4-gradient-optimization/06-newton-quasi-newton.md) | 뉴턴 방법 $x_{k+1} = x_k - H^{-1}\nabla f$의 2차 수렴성 증명 (수렴 반경 내에서), 헤시안 역행렬 계산 비용 $O(n^3)$의 문제, L-BFGS가 헤시안을 근사하는 두 벡터 쌍 재귀 갱신 원리, 소규모 vs 대규모 문제 적용 기준 |
| [07. Loss Landscape의 기하](./ch4-gradient-optimization/07-loss-landscape-geometry.md) | 딥러닝에서 국소 극솟값이 "대부분 같은 품질"인 이유 (과매개변수화 이론), Mode Connectivity 현상 (두 극솟값을 잇는 저손실 경로의 존재), Sharp vs Flat Minima의 일반화 능력 차이, matplotlib으로 2D Loss Landscape 시각화 |

</details>

<br/>

### 🔹 Chapter 5: Backpropagation — 미적분의 걸작

> **핵심 질문:** 역전파가 체인룰의 단순 적용과 어떻게 다른가? 왜 forward 1회 + backward 1회만으로 모든 파라미터의 gradient를 얻는가? Vanishing Gradient의 수치적 근거는 무엇인가?

<details>
<summary><b>계산 그래프부터 NumPy Autograd 구현까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 계산 그래프와 자동미분](./ch5-backpropagation/01-computational-graph-ad.md) | 계산 그래프의 정의, 순방향 자동미분(forward-mode AD): Dual Number로 미분 전파, 역방향 자동미분(reverse-mode AD): gradient 누적, 출력이 스칼라일 때 역방향이 효율적인 이유 (입력 $n$개 → forward-mode $O(n)$ vs reverse-mode $O(1)$) |
| [02. 역전파의 수학적 유도](./ch5-backpropagation/02-backprop-derivation.md) | 역전파 = VJP(Vector-Jacobian Product) 연속 계산임을 증명, 각 연산 노드의 VJP 도출 (덧셈, 곱셈, ReLU, Softmax), 행렬 연산의 역전파 (선형 레이어 $W \cdot x$에서 $\partial L/\partial W = \delta x^\top$의 유도) |
| [03. 왜 Forward 1회 + Backward 1회인가](./ch5-backpropagation/03-why-forward-backward-once.md) | Baur-Strassen 정리: 계산량이 forward의 상수 배로 제한되는 이유, Checkpoint 없이 역전파 시 중간 활성값이 메모리에 남아야 하는 이유, Gradient Checkpointing으로 메모리-시간 트레이드오프를 다루는 방법 |
| [04. Vanishing/Exploding Gradient의 수학](./ch5-backpropagation/04-vanishing-exploding-gradient.md) | $\partial L/\partial x = \prod_l J_l$ (층별 야코비안의 곱), 스펙트럼 반경 $\rho < 1$이면 지수적 소실, $\rho > 1$이면 지수적 폭발을 수치 실험으로 확인, 해결책의 수학: Residual Connection (항등 야코비안), He/Xavier 초기화 (스펙트럼 반경 = 1 목표), Gradient Clipping |
| [05. NumPy로 Autograd 엔진 구현](./ch5-backpropagation/05-autograd-numpy-implementation.md) | 스칼라 Autograd 구현 (Value 클래스, 연산별 backward 정의), 계산 그래프 위상 정렬로 역전파 순서 결정, 벡터 Autograd로 확장, PyTorch `.backward()` 결과와 일치 검증, 2층 MLP를 직접 구현한 Autograd로 학습 |

</details>

<br/>

### 🔹 Chapter 6: 제약 최적화 — 라그랑주와 KKT

> **핵심 질문:** $\nabla f = \lambda \nabla g$라는 조건은 어떻게 기하학적으로 유도되는가? $\lambda$가 왜 "그림자 가격"인가? KKT 조건의 4가지 요소는 각각 무엇을 의미하는가?

<details>
<summary><b>라그랑주 승수법부터 AI 응용까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 등식 제약과 라그랑주 승수법](./ch6-constrained-optimization/01-lagrange-multipliers.md) | $\nabla f = \lambda \nabla g$의 기하학적 유도: 극점에서 등위선과 제약 곡선이 접함, $\lambda$가 "그림자 가격"인 이유 (감도 분석: $\frac{\partial f^*}{\partial b} = -\lambda$), SymPy로 기호 풀이 + 수치 검증 |
| [02. 부등식 제약과 KKT 조건](./ch6-constrained-optimization/02-inequality-kkt.md) | KKT 4개 조건: 정류성(stationarity) / 원시 가능성(primal feasibility) / 쌍대 가능성(dual feasibility) / 상보적 이완(complementary slackness)의 직관적 의미와 수학적 도출, 활성/비활성 제약의 구분 |
| [03. 라그랑주 쌍대성 맛보기](./ch6-constrained-optimization/03-lagrangian-duality.md) | 원시 문제 vs 쌍대 문제 정의, 약쌍대성 $g(\lambda) \leq f^*$ 증명, 강쌍대성이 성립하는 조건 (Slater's condition), 쌍대성 갭(duality gap)의 의미, Convex Optimization 레포에서 완전 전개 |
| [04. 음함수 정리와 제약 최적화의 연결](./ch6-constrained-optimization/04-implicit-function-constrained.md) | 제약 곡면 $g(x) = 0$ 위의 접평면이 $\ker(Dg(x^*))$임을 음함수 정리로 증명, 라그랑주 조건이 접평면 위의 방향 도함수를 0으로 만드는 조건과 동치임을 보임, 기하학적 시각화 |
| [05. AI 응용 — PINN과 Lagrangian Neural Network](./ch6-constrained-optimization/05-constrained-ai-applications.md) | Physics-Informed Neural Network (PINN): 편미분방정식을 제약으로 Loss에 추가하는 방식, Lagrangian Neural Network (LNN): 라그랑지안을 직접 학습해 물리 법칙을 보존, Constrained GAN에서 KKT 조건의 역할 |

</details>

<br/>

### 🔹 Chapter 7: AI/ML에서의 미적분·최적화

> **핵심 질문:** 교차 엔트로피 + Softmax의 gradient가 왜 깔끔하게 $(p - y)$가 되는가? Batch Norm의 역전파에 어떤 항들이 추가되는가? MAML이 왜 2차 미분을 필요로 하는가?

<details>
<summary><b>Softmax 야코비안부터 음함수 정리 응용까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Softmax의 야코비안과 교차 엔트로피 역전파](./ch7-ml-applications/01-softmax-jacobian.md) | Softmax의 야코비안 $J_{ij} = p_i(\delta_{ij} - p_j)$ 유도, 교차 엔트로피와 합성 시 gradient가 $(p - y)$로 단순화되는 이유 (log-sum-exp 트릭 포함), NumPy로 직접 구현 후 PyTorch autograd와 비교 |
| [02. Batch/Layer Normalization의 미분](./ch7-ml-applications/02-batch-layer-norm-gradient.md) | Batch Norm 순전파 ($\hat{x} = (x - \mu)/\sigma$), 역전파 시 평균·분산을 통과하는 gradient 흐름의 완전 유도 (입력 $N$개가 서로 gradient를 공유하는 이유), Layer Norm과의 차이, 실험: BN 있을 때/없을 때 조건수 비교 |
| [03. Neural Tangent Kernel (NTK) 맛보기](./ch7-ml-applications/03-ntk-infinite-width.md) | 무한 폭 신경망 극한에서 야코비안 $J_\theta f(x)$가 학습 중 고정되는 이유, NTK $K(x, x') = J_\theta f(x) \cdot J_\theta f(x')^\top$로 학습 동역학이 선형 회귀로 환원됨을 스케치, 유한 폭 네트워크에서의 한계 |
| [04. Meta-Learning과 고차 미분 — MAML](./ch7-ml-applications/04-maml-higher-order.md) | MAML의 "gradient of gradient": $\theta - \alpha \nabla_\theta \mathcal{L}_{task}$를 다시 $\theta$로 미분하면 2차 미분이 등장하는 이유, 계산 비용 $O(n^2)$과 first-order 근사(FOMAML), JAX로 2차 미분 구현 예시 |
| [05. Implicit Differentiation in Deep Learning](./ch7-ml-applications/05-implicit-differentiation-dl.md) | Deep Equilibrium Model (DEQ): 무한 깊이 네트워크의 고정점 $z^* = f(z^*, \theta)$에서 음함수 정리로 gradient를 구하는 방법, Optimization as a Layer: 내부 최적화의 gradient를 KKT 조건 미분으로 얻는 방식 (CVXPY Layers) |

</details>

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
sympy==1.12           # 기호 미분·적분·방정식 풀이
jax==0.4.20           # 자동미분 비교 및 고차 미분
torch==2.1.0          # PyTorch autograd 결과 검증
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            sympy==1.12 jax==0.4.20 torch==2.1.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 예시 — Vanishing/Exploding Gradient 재현
import numpy as np
import matplotlib.pyplot as plt

def simulate_depth_gradient(depth, sigma_w=1.0, n=100):
    """층별 야코비안을 곱해가며 스펙트럼 반경을 관찰"""
    J = np.eye(n)
    spectral_norms = []
    for _ in range(depth):
        W = np.random.randn(n, n) * sigma_w / np.sqrt(n)
        J = W @ J
        spectral_norms.append(np.linalg.norm(J, 2))
    return spectral_norms

plt.figure(figsize=(10, 5))
for s, label in [(0.5, 'σ_w=0.5 (소실)'), (1.0, 'σ_w=1.0 (안정)'), (2.0, 'σ_w=2.0 (폭발)')]:
    norms = simulate_depth_gradient(depth=100, sigma_w=s)
    plt.plot(norms, label=label)

plt.yscale('log')
plt.xlabel('Layer Depth')
plt.ylabel('Spectral Norm of Accumulated Jacobian')
plt.title('Vanishing & Exploding Gradient: 야코비안 스펙트럼 반경 변화')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# σ_w ≈ 1.0 → 스펙트럼 반경 ≈ 1 (He/Xavier 초기화의 수학적 목표)
```

---

## 📖 각 문서 구성 방식

모든 문서는 동일한 구조로 작성됩니다.

| 섹션 | 설명 |
|------|------|
| 🎯 **핵심 질문** | 이 문서를 읽고 나면 답할 수 있는 질문 |
| 🔍 **왜 이 정리가 AI에서 중요한가** | `.backward()`, Adam, BN 등 실제 구현과의 연결 |
| 📐 **수학적 선행 조건** | Linear Algebra 레포 참조 링크 포함 |
| 📖 **직관적 이해** | 그림 + 물리적·기하학적 비유 |
| ✏️ **엄밀한 정의** | ε-δ 수준의 정형적 정의 |
| 🔬 **정리와 증명** | 보조정리부터 차근차근, "자명하다" 생략 없음 |
| 💻 **NumPy / SymPy 구현으로 검증** | 기호 미분과 수치 미분 비교, Autograd 구현 |
| 🔗 **AI/ML 연결** | 역전파·옵티마이저·정규화 등 구체 사례 |
| ⚖️ **가정과 한계** | 수렴 정리의 가정이 깨지는 순간, 수치 불안정성 조건 |
| 📌 **핵심 정리** | 한 화면 요약 |
| 🤔 **생각해볼 문제** | 개념 심화 질문 + 해설 |

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "역전파가 어떻게 동작하는지 수학적으로 설명 못한다" — 역전파 집중 (3일)</b></summary>

<br/>

```
Day 1  Ch2-05  다변수 연쇄법칙 → 야코비안 행렬곱의 의미
Day 2  Ch5-01  계산 그래프와 자동미분 → forward vs reverse mode
       Ch5-02  역전파의 수학적 유도 → VJP 연속 계산
Day 3  Ch5-05  NumPy Autograd 구현 → 직접 만들어서 PyTorch와 결과 비교
```

</details>

<details>
<summary><b>🟡 "SGD, Adam 중 무엇을 써야 하는지, 그리고 왜인지 모른다" — 옵티마이저 집중 (1주)</b></summary>

<br/>

```
Day 1  Ch2-03  Gradient와 코시-슈바르츠 → 경사하강법의 출발점
       Ch3-04  조건수와 최적화 속도 → 왜 일부 문제는 수렴이 느린가
Day 2  Ch4-01  GD 수렴 증명 (볼록, L-smooth) → O(1/k) 증명
       Ch4-02  학습률 수렴 조건 → η < 2/L 유도
Day 3  Ch4-03  모멘텀과 Nesterov 가속 → O(1/k²) 수렴률
Day 4  Ch4-04  SGD 수렴 → 확률적 분석, O(1/√k)
Day 5  Ch4-05  Adam의 수학과 반례 → 왜 항상 좋지 않은가
Day 6  Ch3-02  헤시안 고유값과 안장점 → Loss Landscape 이해
       Ch4-07  Loss Landscape의 기하 → 딥러닝 극소값의 성질
Day 7  Ch4-06  뉴턴 방법과 L-BFGS → 2차 방법의 강점과 한계
```

</details>

<details>
<summary><b>🔴 "해석학 기초부터 KKT 조건까지 완전 정복한다" — 전체 정복 (7주)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 해석학 기초
        → ε-δ 증명 5개 직접 작성, Weierstrass 함수 NumPy 시각화

2주차  Chapter 2 전체 — 다변수 미적분
        → 편미분 있어도 전미분 없는 반례 NumPy로 확인
        → 야코비안·헤시안 SymPy 기호 계산

3주차  Chapter 3 전체 — 테일러 전개와 기하
        → 3D Loss Surface에서 안장점·극솟값 시각화
        → 조건수 변화에 따른 GD 궤적 비교

4주차  Chapter 4 전체 — 경사하강법 완전 분석
        → GD / Momentum / Nesterov / Adam 수렴 실험
        → Adam 발산 반례 재현 (Reddi 2018)

5주차  Chapter 5 전체 — Backpropagation
        → NumPy Autograd 엔진 직접 구현
        → PyTorch .backward() 결과와 수치 일치 검증

6주차  Chapter 6 전체 — 제약 최적화
        → SymPy로 라그랑주 문제 기호 풀이
        → KKT 조건 수치 검증, 쌍대 갭 실험

7주차  Chapter 7 전체 — AI/ML 응용
        → Softmax Jacobian 직접 계산 vs autograd 비교
        → MAML 2차 미분 JAX로 구현
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 행렬, 고유값, SVD, Spectral Theorem | Ch2-04(야코비안·헤시안은 행렬), Ch3-02(고유값과 극점), Ch5-04(스펙트럼 반경) |
| [convex-optimization-deep-dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive) | 볼록 집합·함수, 쌍대성, Interior Point, SVM | Ch4-01(볼록 수렴의 심화), Ch6-03(쌍대성 완전 전개) |
| [probability-statistics-deep-dive](https://github.com/iq-ai-lab/probability-statistics-deep-dive) | 확률론, 기댓값, MLE, 베이즈 추정 | Ch4-04(SGD: 기댓값 기반 수렴 분석) |
| [numerical-methods-deep-dive](https://github.com/iq-ai-lab/numerical-methods-deep-dive) | 수치 적분, ODE 풀이, 행렬 분해 | Ch1-03(수치 미분 유한차분 오차), Ch4-06(L-BFGS 헤시안 근사) |

> 💡 이 레포는 **미적분과 최적화의 수학적 기반**에 집중합니다. AI 경험이 없어도 Chapter 1~3은 수학 레포로 학습 가능합니다. Chapter 4~7은 딥러닝 기초(MLP, SGD 사용 경험)가 있을 때 연결이 더욱 깊어집니다.

---

## 📖 Reference

- **Principles of Mathematical Analysis** (Rudin) — ε-δ, 연속, 미분의 표준 교과서
- **Convex Optimization** (Boyd & Vandenberghe) — 제약 최적화와 쌍대성의 고전 (다음 레포에서 심화)
- **Numerical Optimization** (Nocedal & Wright) — Quasi-Newton, L-BFGS, 선탐색 이론
- **Introductory Lectures on Stochastic Optimization** (Nesterov) — 수렴률 분석의 교과서
- **Deep Learning** (Goodfellow et al.) Chapter 4, 8 — 수치 방법과 최적화
- **Automatic Differentiation in Machine Learning: a Survey** (Baydin 2018) — AD 이론 종합
- **On the Convergence of Adam and Beyond** (Reddi 2018) — Adam 수렴 반례와 AMSGrad
- **Identifying and attacking the saddle point problem in high-dimensional non-convex optimization** (Dauphin 2014) — 딥러닝 안장점 이론

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"`loss.backward()`를 호출하는 것과, 역전파가 야코비안의 연쇄 행렬곱이라는 기하학적 본질을 아는 것은 다르다"*

</div>
