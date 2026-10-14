# TorchYAND-MVSK: Mathematical Design, PyTorch API, and Numerical Stability Specification

**Version:** 0.1 design specification  
**Target:** PyTorch-native, batched, GPU-capable implementation of Yau's Affine-Normal Descent (YAND) specialized to sample-moment MVSK portfolio optimization, with reusable numerical kernels.  
**Primary implementation principle:** matrix-free + batch-first + device-preserving + numerically guarded.

---

## 0. Scope and design philosophy

This project should **not** be a line-by-line PyTorch translation of an existing MATLAB/NumPy implementation. The intended architecture is:

\[
\boxed{
\text{MVSK sample oracle}
\rightarrow
\text{matrix-free affine-normal geometry}
\rightarrow
\text{batched Krylov solves}
\rightarrow
\text{exact quartic line search}
\rightarrow
\text{batched cubic root solver}
}
\]

The most important engineering decision is that **all public numerical kernels are batch-first from day one**. A single problem is treated as a batch of size one.

The implementation should have two layers:

1. **Paper-faithful reference layer**
   - explicit reduced simplex coordinates;
   - explicit tangent frames when dimensions are small;
   - exact log-determinant correction;
   - dense linear algebra for verification.

2. **Torch-native scalable layer**
   - no explicit dense simplex basis \(U\);
   - no explicit dense Householder basis \(Q\);
   - projector-based subspace operations;
   - batched PCG;
   - stochastic trace/log-determinant correction;
   - batched analytic cubic roots;
   - fixed-shape masked active-set logic;
   - GPU-resident execution.

This split is critical: the reference layer is the correctness oracle; the scalable layer is the performance implementation.

---

# 1. Mathematical problem

Let

\[
R\in\mathbb{R}^{T\times n}
\]

be the sample return matrix, with \(T\) observations and \(n\) assets. Define

\[
\mu = \frac1T R^\top \mathbf 1,
\qquad
A = R-\mathbf 1\mu^\top.
\]

The portfolio lies on the simplex

\[
\Delta_n
=
\left\{
x\in\mathbb{R}^n:
x_i\ge 0,\;
\mathbf 1^\top x=1
\right\}.
\]

For centered projected returns

\[
z(x)=Ax,
\]

the unrestricted sample-moment MVSK objective is

\[
f(x)
=
-c_1\mu^\top x
+
\frac{c_2}{T}\sum_{t=1}^Tz_t^2
-
\frac{c_3}{T}\sum_{t=1}^Tz_t^3
+
\frac{c_4}{T}\sum_{t=1}^Tz_t^4.
\]

All \(c_i\ge 0\).

The implementation must **never construct covariance, coskewness, or cokurtosis tensors**.

---

# 2. Tensor conventions

Use batch-first tensors throughout.

## 2.1 Core shapes

Shared data:

```text
R       [T, n]
mu      [n]
A       [T, n]
```

Portfolio batch:

```text
X       [..., n]
```

where `...` denotes arbitrary leading batch dimensions, e.g.

```text
[B, n]
[B1, B2, n]
```

Projected returns:

```text
Z = X @ A.T       [..., T]
```

Directions:

```text
V, Udir, D        [..., n]
```

Objective values:

```text
f(X)              [...]
```

Preferences may be scalar/shared or batch-broadcastable:

```text
c                  [4]
or
c                  [..., 4]
```

## 2.2 Device rule

No public numerical kernel may silently move tensors between devices.

```python
assert all tensors are on the same device
```

CPU fallback is allowed only when explicitly requested by the user.

## 2.3 Dtype rule

Recommended defaults:

```text
solver default       float64
throughput mode      float32
```

The project should support:

```text
float32
float64
```

`bfloat16` and `float16` are **not supported for solver state** in v0.1.

Mixed precision may later be allowed for the large GEMM projections, but convergence-critical reductions, polynomial roots, line-search selection, and residual checks should remain at least float32 and preferably float64.

---

# 3. Package architecture

```text
torchyand/
│
├── __init__.py
├── config.py
├── types.py
│
├── data/
│   ├── mvsk_data.py
│   └── preprocessing.py
│
├── oracle/
│   ├── base.py
│   ├── mvsk.py
│   └── autograd.py
│
├── geometry/
│   ├── simplex.py
│   ├── projectors.py
│   ├── explicit_frames.py
│   └── affine_normal.py
│
├── logdet/
│   ├── exact.py
│   ├── stochastic.py
│   └── probes.py
│
├── linalg/
│   ├── pcg.py
│   ├── minres.py              # optional/future
│   ├── preconditioners.py
│   └── diagnostics.py
│
├── polynomial/
│   ├── linear.py
│   ├── quadratic.py
│   ├── cubic.py
│   ├── evaluate.py
│   └── quartic_line_search.py
│
├── constraints/
│   ├── simplex_projection.py
│   ├── feasibility.py
│   └── active_set.py
│
├── solver/
│   ├── state.py
│   ├── direction.py
│   ├── yand_mvsk.py
│   └── yand_general.py        # later
│
├── diagnostics/
│   ├── kkt.py
│   ├── stopping.py
│   └── trace.py
│
└── utils/
    ├── numerics.py
    ├── validation.py
    └── compile.py
```

---

# 4. Module A — MVSK data preprocessing

## 4.1 Mathematics

\[
\mu=\frac1T R^\top\mathbf1,
\qquad
A=R-\mathbf1\mu^\top.
\]

The centered matrix satisfies, up to floating-point error,

\[
\mathbf1^\top A \approx 0.
\]

## 4.2 API

```python
@dataclass
class MVSKData:
    A: torch.Tensor          # [T, n]
    mu: torch.Tensor         # [n]
    T: int
    n: int

def prepare_mvsk_data(
    R: torch.Tensor,
    *,
    center_dtype: torch.dtype | None = torch.float64,
    output_dtype: torch.dtype | None = None,
    check_finite: bool = True,
) -> MVSKData:
    ...
```

## 4.3 Numerical stability

1. Mean estimation should optionally accumulate in float64.
2. Avoid repeated centering inside the optimization loop.
3. Reject NaN/Inf before optimization.
4. Verify

\[
\frac{\|\mathbf1^\top A\|_\infty}
{\max(1,\|A\|_\infty)}
\le \texttt{center\_tol}.
\]

5. Do not standardize columns automatically; affine scaling is part of the geometry and preprocessing must be explicit.

---

# 5. Module B — MVSK exact sample oracle

This is the core high-throughput numerical module.

## 5.1 Cached projection

For a batch \(X\),

\[
Z=XA^\top.
\]

Cache:

\[
Z,\qquad Z^{\circ 2},\qquad Z^{\circ 3}.
\]

Use multiplication rather than generic powers:

```python
z2 = z * z
z3 = z2 * z
```

## 5.2 Objective

\[
f(x)
=
-c_1\mu^\top x
+
c_2\,\mathbb E[z^2]
-
c_3\,\mathbb E[z^3]
+
c_4\,\mathbb E[z^4].
\]

Batch implementation:

```python
z = x @ A.T
z2 = z * z
z3 = z2 * z
z4 = z2 * z2

value = (
    -c1 * (x * mu).sum(-1)
    + c2 * z2.mean(-1)
    - c3 * z3.mean(-1)
    + c4 * z4.mean(-1)
)
```

## 5.3 Gradient

\[
\nabla f(x)
=
-c_1\mu
+
\frac1T
A^\top
\left(
2c_2z
-3c_3z^{\circ2}
+4c_4z^{\circ3}
\right).
\]

Batch form:

\[
G
=
-c_1\mu
+
\frac1T
\Psi_1(Z)A.
\]

Prefer the factored expression

\[
\Psi_1(z)
=
z\left(2c_2+z(-3c_3+4c_4z)\right)
\]

to reduce temporary tensors.

## 5.4 Hessian-vector product

For direction \(v\),

\[
\nabla^2f(x)v
=
\frac1T
A^\top
\left[
\left(
2c_2
-6c_3z
+12c_4z^{\circ2}
\right)
\circ (Av)
\right].
\]

Batch:

```python
av = v @ A.T
weight2 = 2*c2 + z * (-6*c3 + 12*c4*z)
hv = (weight2 * av) @ A / T
```

No Hessian matrix is formed.

## 5.5 Directional third-order action

Define

\[
T_3(x;u,v)
=
D^3f(x)[u,v,\cdot].
\]

Then

\[
T_3(x;u,v)
=
\frac1T
A^\top
\left[
(-6c_3+24c_4z)
\circ(Au)
\circ(Av)
\right].
\]

Batch:

```python
au = u @ A.T
av = v @ A.T
weight3 = -6*c3 + 24*c4*z
third = (weight3 * au * av) @ A / T
```

## 5.6 API

```python
@dataclass
class MVSKCache:
    x: torch.Tensor
    z: torch.Tensor
    z2: torch.Tensor
    z3: torch.Tensor | None
    value: torch.Tensor | None = None
    grad: torch.Tensor | None = None

class MVSKOracle:
    def cache(self, x) -> MVSKCache: ...
    def value(self, x, cache=None) -> torch.Tensor: ...
    def value_grad(self, x, cache=None) -> tuple[torch.Tensor, torch.Tensor, MVSKCache]: ...
    def hvp(self, x, v, cache=None) -> torch.Tensor: ...
    def third_action(self, x, u, v, cache=None) -> torch.Tensor: ...
```

## 5.7 Numerical stability

- Compute moments with multiplication chains, not `pow(4)`.
- For very large \(|z|\), optionally use scale diagnostics before the fourth power.
- Return a diagnostic flag if

\[
\max_t |z_t| > z_{\text{overflow-warning}}.
\]

- Reductions may use a higher accumulator dtype.
- Never detach tensors inside the oracle unless the analytic backend explicitly does not require autograd.

---

# 6. Module C — Simplex geometry

The simplex tangent space is

\[
\mathcal T
=
\{v:\mathbf1^\top v=0\}.
\]

## 6.1 Scalable projector

For any \(v\in\mathbb R^n\),

\[
P_{\mathcal T}v
=
v-\bar v\,\mathbf1,
\qquad
\bar v=\frac1n\mathbf1^\top v.
\]

API:

```python
def project_simplex_tangent(v: torch.Tensor) -> torch.Tensor:
    return v - v.mean(dim=-1, keepdim=True)
```

## 6.2 Active-face tangent projector

Let `free_mask` indicate coordinates not fixed at the lower bound.

For \(m=\sum_i \mathbf1_{\{i\text{ free}\}}\),

\[
P_F(v)
=
M\left(
v-\frac{\mathbf1^\top Mv}{m}\mathbf1
\right),
\]

where \(M=\operatorname{diag}(\text{free mask})\).

API:

```python
def project_face_tangent(
    v: torch.Tensor,
    free_mask: torch.Tensor,
) -> torch.Tensor:
    ...
```

This fixed-shape masked representation avoids ragged lower-dimensional tensors and is particularly suitable for GPU batching.

## 6.3 Explicit reference basis

For correctness tests, expose a deterministic orthonormal basis

\[
U\in\mathbb R^{n\times(n-1)},
\qquad
U^\top U=I,
\quad
U^\top\mathbf1=0.
\]

Recommended construction: normalized Helmert basis.

API:

```python
def simplex_helmert_basis(
    n: int,
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    ...
```

**Rule:** dense \(U\) is reference/small-problem functionality, not the default scalable backend.

---

# 7. Module D — Level-set tangent geometry

Let

\[
g_T=P_F\nabla f(x).
\]

If

\[
\|g_T\| > 0,
\]

define

\[
\nu=\frac{g_T}{\|g_T\|}.
\]

The tangent space of the level set within the current simplex face is

\[
\mathcal S
=
\{v:
v\in\mathcal T_F,\;
\nu^\top v=0
\}.
\]

Because \(\nu\in\mathcal T_F\),

\[
P_{\mathcal S}v
=
P_Fv-\nu(\nu^\top P_Fv).
\]

API:

```python
@dataclass
class GeometryState:
    grad_face: torch.Tensor       # [..., n]
    grad_norm: torch.Tensor       # [...]
    nu: torch.Tensor              # [..., n]
    free_mask: torch.Tensor       # [..., n]

def project_level_tangent(
    v: torch.Tensor,
    geom: GeometryState,
) -> torch.Tensor:
    ...
```

This projector replaces the explicit product \(UQ\) in the scalable backend.

---

# 8. Module E — Explicit Householder frame (reference backend)

For reduced gradient \(\bar g\in\mathbb R^m\),

\[
\nu=\bar g/\|\bar g\|.
\]

A Householder reflector maps \(\nu\) to \(\pm e_1\). Let

\[
s=\operatorname{sign}(\nu_1)
\]

with \(s=1\) when \(\nu_1=0\), and define

\[
v=\nu+s e_1.
\]

Then

\[
H=I-2\frac{vv^\top}{v^\top v}.
\]

The remaining \(m-1\) columns form

\[
Q\in\mathbb R^{m\times(m-1)}
\]

with

\[
Q^\top Q=I,
\qquad
Q^\top\nu=0.
\]

API:

```python
def householder_complement(
    nu: torch.Tensor,
) -> torch.Tensor:
    """Reference/small-scale only."""
```

Numerical rules:

- choose the sign to avoid cancellation;
- if \(\|\nu\|\) is below tolerance, declare stationarity before calling;
- do not form batched dense \(Q\) in the large-scale backend.

---

# 9. Module F — Tangent Hessian operator

In explicit reduced notation, the YAND tangent operator is

\[
H_{T,\lambda}\eta
=
Q^\top U^\top
\nabla^2f(x)
(UQ\eta)
+\lambda\eta.
\]

In the scalable ambient projected formulation, let

\[
P_S=P_{\mathcal S}.
\]

For \(v\in\mathcal S\),

\[
\mathcal H_{S,\lambda}v
=
P_S
\left(
\nabla^2f(x)v
\right)
+\lambda v.
\]

Because \(v=P_Sv\), an implementation-safe form is

\[
\boxed{
\mathcal H_{S,\lambda}(v)
=
P_S
\nabla^2f(x)
(P_Sv)
+
\lambda P_Sv
}
\]

to correct floating-point drift from the subspace.

API:

```python
class TangentHessianOperator:
    def matvec(self, v: torch.Tensor) -> torch.Tensor: ...
```

The operator must be usable by PCG without materializing a matrix.

---

# 10. Module G — Mixed curvature vector \(h\)

The paper-level reduced formula is

\[
h
=
Q^\top U^\top
\nabla^2f(x)(U\nu).
\]

In the projector formulation,

\[
\boxed{
h_{\rm amb}
=
P_S\left(\nabla^2f(x)\nu\right).
}
\]

API:

```python
def mixed_curvature_vector(
    oracle,
    x,
    nu,
    geom,
    cache,
) -> torch.Tensor:
    ...
```

---

# 11. Module H — Log-determinant correction \(a\)

The affine-normal correction is the tangent gradient

\[
a
=
\nabla_t\log\det(H_T).
\]

The key identity is

\[
\partial_i\log\det(H_T)
=
\operatorname{tr}
\left(
H_T^{-1}\partial_iH_T
\right),
\]

which can be evaluated through linear solves and directional third-order actions rather than by forming a third-order tensor.

Two backends are required.

---

# 12. Module H1 — Exact log-determinant correction

This backend is for testing and moderate dimensions.

In a level-tangent orthonormal basis \(W\),

\[
H_T=W^\top H W.
\]

For each canonical vector \(e_j\),

\[
(H_T+\lambda I)y^{(j)}=e_j.
\]

Lift

\[
u^{(j)}=Wy^{(j)},
\qquad
v^{(j)}=We_j.
\]

Then

\[
w^{(j)}
=
D^3f(x)[u^{(j)},v^{(j)},\cdot],
\]

and

\[
a
=
\sum_j W^\top w^{(j)}.
\]

## API

```python
def exact_logdet_correction(
    oracle,
    x,
    *,
    basis_W: torch.Tensor,
    lambda_reg: torch.Tensor | float,
    cache=None,
) -> torch.Tensor:
    ...
```

## Implementation strategy

For small dimensions, vectorize the canonical directions:

```text
E       [p, p]
Y       solve(H_T + lambda I, E)
U       W @ Y
V       W @ E = W
```

Evaluate all third actions in a batch where practical.

## Numerical stability

- Prefer `torch.linalg.cholesky_ex` when the regularized tangent Hessian is SPD.
- If Cholesky fails, increase regularization.
- Dense inverse is forbidden; use solve/factorization.
- The determinant itself should not be computed.
- If a log determinant is needed for diagnostics, use `slogdet`.

---

# 13. Module H2 — Stochastic matrix-free log-determinant correction

This is the primary scalable backend.

For probes

\[
\xi^{(\ell)}
\]

in the level tangent space, solve

\[
(H_T+\lambda I)y^{(\ell)}
=
\xi^{(\ell)}.
\]

Then

\[
w^{(\ell)}
=
D^3f(x)
[
y^{(\ell)},
\xi^{(\ell)},
\cdot
]
\]

in appropriately lifted coordinates, and average.

## 13.1 Basis-free projector version

Sample ambient probes

\[
r^{(\ell)}
\]

with

\[
\mathbb E[rr^\top]=I.
\]

Project them:

\[
\xi^{(\ell)}=P_Sr^{(\ell)}.
\]

Then

\[
\mathbb E[\xi\xi^\top]=P_S,
\]

so the probe covariance is the identity on the level-tangent subspace.

Solve

\[
\mathcal H_{S,\lambda}y^{(\ell)}
=
\xi^{(\ell)}.
\]

Compute

\[
w^{(\ell)}
=
T_3(x;y^{(\ell)},\xi^{(\ell)}),
\]

and

\[
\boxed{
a_{\rm amb}
\approx
\frac1q
\sum_{\ell=1}^q
P_S w^{(\ell)}.
}
\]

This is the preferred Torch-native matrix-free design.

## API

```python
@dataclass
class LogDetResult:
    correction: torch.Tensor
    probe_count: int
    linear_iterations: torch.Tensor
    residual_norm: torch.Tensor
    success: torch.Tensor

def stochastic_logdet_correction(
    oracle,
    x,
    geom,
    *,
    q: int = 4,
    lambda_reg: float | torch.Tensor = 1e-6,
    probe: Literal["rademacher", "gaussian"] = "rademacher",
    linear_solver="pcg",
    generator: torch.Generator | None = None,
    cache=None,
) -> LogDetResult:
    ...
```

## Numerical stability

- Re-project probes after every numerically sensitive operation.
- Use deterministic seeded probes in unit tests.
- If PCG reports nonpositive curvature, increase \(\lambda\) and retry.
- Permit `q=0` only as an explicit ablation, corresponding to no logdet correction.
- Avoid normalizing projected probes; the covariance identity is more important than unit norm.

---

# 14. Module I — Batched PCG

For SPD systems

\[
A_i x_i=b_i,
\]

solve a full batch simultaneously.

## 14.1 Algorithm

For each batch item,

\[
r_0=b-Ax_0,
\qquad
z_0=M^{-1}r_0,
\qquad
p_0=z_0.
\]

Then

\[
\alpha_k
=
\frac{r_k^\top z_k}
{p_k^\top Ap_k},
\]

\[
x_{k+1}=x_k+\alpha_kp_k,
\]

\[
r_{k+1}=r_k-\alpha_kAp_k,
\]

\[
\beta_k
=
\frac{r_{k+1}^\top z_{k+1}}
{r_k^\top z_k},
\]

\[
p_{k+1}=z_{k+1}+\beta_kp_k.
\]

## 14.2 API

```python
@dataclass
class KrylovResult:
    x: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor
    residual_norm: torch.Tensor
    relative_residual: torch.Tensor
    nonpositive_curvature: torch.Tensor
    breakdown: torch.Tensor

def batched_pcg(
    matvec,
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    preconditioner=None,
    atol: float = 0.0,
    rtol: float = 1e-6,
    max_iter: int = 100,
    mode: Literal["adaptive", "fixed"] = "adaptive",
    projector=None,
) -> KrylovResult:
    ...
```

## 14.3 Active batch mask

Different systems converge at different iterations.

Maintain

```text
active [...]
```

and update with `torch.where` rather than slicing the batch.

## 14.4 Curvature guard

PCG requires positive curvature. Monitor

\[
p^\top Ap.
\]

Declare nonpositive curvature when

\[
p^\top Ap
\le
\kappa_{\rm curv}\,
\epsilon_{\rm mach}
\|p\|
\|Ap\|.
\]

On failure:

1. return a curvature flag;
2. outer direction module increases \(\lambda\);
3. restart;
4. after a bounded number of retries, fall back to a safe descent direction.

## 14.5 Residual stopping

Use

\[
\|r_k\|
\le
\texttt{atol}
+
\texttt{rtol}\,\|b\|.
\]

Do not stop solely on step size.

---

# 15. Module J — Preconditioners

## 15.1 Identity

Always available.

```python
IdentityPreconditioner()
```

## 15.2 Jacobi-like MVSK preconditioner

The ambient Hessian is

\[
H
=
\frac1T
A^\top
\operatorname{diag}(w)
A,
\]

where

\[
w
=
2c_2
-6c_3z
+12c_4z^2.
\]

Its diagonal is

\[
\operatorname{diag}(H)_i
=
\frac1T
\sum_t
w_t A_{ti}^2.
\]

Use

\[
M_i
=
\max(
|\operatorname{diag}(H)_i|+\lambda,
m_{\min}
)
\]

as a robust diagonal scale.

API:

```python
class MVSKJacobiPreconditioner:
    def build(self, oracle, cache, lambda_reg, geom): ...
    def solve(self, r): ...
```

After preconditioning, project back into the current tangent subspace.

## 15.3 Stability

Never divide by raw diagonal entries without flooring.

\[
m_{\min}
=
\kappa_M
\epsilon_{\rm mach}
\max(1,\|\operatorname{diag}(H)\|_\infty).
\]

---

# 16. Module K — Affine-normal direction

Let the face-projected gradient be \(g_T\), and

\[
\nu=\frac{g_T}{\|g_T\|}.
\]

Let the level-tangent subspace be \(\mathcal S\). Define

\[
h=P_S(H\nu)
\]

and logdet correction \(a\in\mathcal S\).

For an \(n\)-asset simplex, the reduced optimization variable has dimension \(n-1\), so the YAND denominator is

\[
(n-1)+1=n.
\]

Thus solve

\[
\boxed{
(\mathcal H_S+\lambda I)u
=
h
-
\frac{\|g_T\|}{n}
a.
}
\]

Then the ambient simplex tangent direction is

\[
\boxed{
d=u-\nu.
}
\]

Because \(u\perp\nu\),

\[
g_T^\top d
=
-\|g_T\|<0
\]

up to numerical error, so it is a descent direction before later safeguards.

## API

```python
@dataclass
class DirectionResult:
    direction: torch.Tensor
    lambda_reg: torch.Tensor
    grad_norm: torch.Tensor
    descent_dot: torch.Tensor
    linear_status: KrylovResult | None
    logdet_status: LogDetResult | None
    fallback_used: torch.Tensor

def affine_normal_direction(
    oracle,
    x,
    *,
    free_mask=None,
    lambda0=1e-6,
    lambda_growth=10.0,
    max_regularization_retries=5,
    logdet_mode="stochastic",
    q=4,
    cache=None,
) -> DirectionResult:
    ...
```

## 16.1 Descent safeguard

After construction, verify

\[
g^\top d
<
-\eta_{\rm desc}
\|g_T\|\,\|d\|.
\]

If not:

1. re-project \(d\);
2. increase regularization and recompute;
3. if still invalid, use

\[
d=-\nu
\]

or

\[
d=-g_T.
\]

Record the fallback.

---

# 17. Module L — Polynomial utilities

The polynomial package should be reusable independently of YAND.

Global rule:

> Polynomial roots are represented by `roots` plus `valid_mask`; invalid entries are filled with NaN.

---

# 18. Module L1 — Stable linear roots

Solve

\[
ax+b=0.
\]

API:

```python
def linear_roots(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    degree_tol=None,
) -> RootResult:
    ...
```

If \(|a|\) is numerically zero:

- if \(|b|\) nonzero: no roots;
- if both vanish: mark polynomial as degenerate/identically zero.

---

# 19. Module L2 — Stable quadratic roots

Solve

\[
ax^2+bx+c=0.
\]

Do not use the naive quadratic formula when roots have very different magnitudes.

For discriminant

\[
\Delta=b^2-4ac
\]

and \(\Delta\ge0\), define

\[
q
=
-\frac12
\left(
b+\operatorname{sign}(b)\sqrt{\Delta}
\right).
\]

Then

\[
x_1=\frac qa,
\qquad
x_2=\frac cq,
\]

with a guarded fallback for \(q\approx0\).

API:

```python
def quadratic_real_roots(
    a, b, c,
    *,
    degree_tol=None,
    discriminant_tol=None,
    polish: bool = True,
) -> RootResult:
    ...
```

Degenerate leading coefficient automatically falls back to the linear solver.

---

# 20. Module L3 — Batched cubic real-root solver

This is a flagship reusable kernel.

Solve

\[
ax^3+bx^2+cx+d=0
\]

for arbitrary batch shape.

## 20.1 API

```python
@dataclass
class RootResult:
    roots: torch.Tensor        # [..., max_roots]
    valid: torch.Tensor        # bool, same shape
    multiplicity_hint: torch.Tensor | None
    degree: torch.Tensor       # [...]
    residual: torch.Tensor     # [..., max_roots]

def cubic_real_roots(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    *,
    method: Literal["analytic", "companion"] = "analytic",
    promote: bool = True,
    degree_tol: float | None = None,
    discriminant_tol: float | None = None,
    polish_steps: int = 1,
) -> RootResult:
    ...
```

Inputs broadcast to a common shape.

Output:

```text
roots     [..., 3]
valid     [..., 3]
```

## 20.2 Coefficient normalization

Before classification,

\[
s=\max(|a|,|b|,|c|,|d|).
\]

For nonzero \(s\), replace

\[
(a,b,c,d)\leftarrow(a,b,c,d)/s.
\]

This avoids unnecessary overflow/underflow and makes degree thresholds scale-aware.

## 20.3 Degree detection

After normalization, classify as cubic only if

\[
|a|>\tau_{\rm degree}.
\]

Otherwise fall back to quadratic.

Recommended default:

\[
\tau_{\rm degree}
=
64\,\epsilon_{\rm mach}.
\]

Expose this as configurable.

## 20.4 Depressed cubic

For true cubic, divide by \(a\):

\[
x^3+Ax^2+Bx+C=0.
\]

Set

\[
x=y-\frac A3.
\]

Then

\[
y^3+py+q=0,
\]

with

\[
p=B-\frac{A^2}{3},
\]

\[
q=
\frac{2A^3}{27}
-\frac{AB}{3}
+C.
\]

Define

\[
\Delta
=
\left(\frac q2\right)^2
+
\left(\frac p3\right)^3.
\]

## 20.5 Discriminant tolerance

Use

\[
S_\Delta
=
\left(\frac q2\right)^2
+
\left|\frac p3\right|^3.
\]

Classify repeated-root behavior when

\[
|\Delta|
\le
\tau_\Delta,
\qquad
\tau_\Delta
=
\kappa_\Delta
\epsilon_{\rm mach}
\max(S_\Delta,\texttt{tiny}).
\]

Recommended \(\kappa_\Delta=128\).

## 20.6 Case \(\Delta>0\): one real root

Naive Cardano can lose precision when the two cube-root terms cancel.

Use the stabilized construction

\[
u
=
\sqrt[3]{
-\frac q2
-
\operatorname{sign}(q)\sqrt{\Delta}
},
\]

then, when \(u\neq0\),

\[
v=-\frac{p}{3u},
\qquad
y=u+v.
\]

If \(u\) is too small, use direct real cube roots.

Define

\[
\operatorname{cbrt}_{\mathbb R}(x)
=
\operatorname{sign}(x)|x|^{1/3}.
\]

## 20.7 Case \(\Delta\approx0\): repeated roots

Let

\[
u=\sqrt[3]{-\frac q2}.
\]

Roots are

\[
y_1=2u,
\qquad
y_2=-u.
\]

Return repeated roots with a multiplicity hint or deduplicate according to API option.

## 20.8 Case \(\Delta<0\): three real roots

Set

\[
r=2\sqrt{-p/3},
\]

\[
\theta
=
\frac13
\arccos
\left(
-\frac{q}{2\sqrt{-(p/3)^3}}
\right).
\]

Clamp the `acos` argument to \([-1,1]\).

Then

\[
y_k
=
r\cos
\left(
\theta-\frac{2\pi k}{3}
\right),
\qquad
k=0,1,2.
\]

Finally,

\[
x_k=y_k-\frac A3.
\]

## 20.9 Newton polishing

For each valid real root \(x\), evaluate

\[
P(x)=ax^3+bx^2+cx+d,
\]

\[
P'(x)=3ax^2+2bx+c.
\]

When \(|P'(x)|\) is safely nonzero,

\[
x\leftarrow x-\frac{P(x)}{P'(x)}.
\]

One guarded Newton step is the default.

## 20.10 Residual metric

Report

\[
\rho(x)
=
\frac{|P(x)|}
{|a||x|^3+|b||x|^2+|c||x|+|d|+\texttt{tiny}}.
\]

This is more meaningful than raw absolute residual.

## 20.11 Companion backend

Reference backend:

\[
C=
\begin{bmatrix}
0&0&-d/a\\
1&0&-c/a\\
0&1&-b/a
\end{bmatrix}.
\]

Use

```python
torch.linalg.eigvals(C)
```

for verification, not as the performance default.

---

# 21. Module M — Quartic line model

Along direction \(d\),

\[
w=Ad.
\]

Define mixed power sums

\[
s_{rs}(x,d)
=
\frac1T
\sum_t z_t^r w_t^s,
\qquad
r+s\le4.
\]

Then

\[
\phi(\alpha)
=
f(x+\alpha d)
=
A_0+A_1\alpha+A_2\alpha^2+A_3\alpha^3+A_4\alpha^4,
\]

where

\[
A_0=f(x),
\]

\[
A_1
=
-c_1\mu^\top d
+2c_2s_{11}
-3c_3s_{21}
+4c_4s_{31},
\]

\[
A_2
=
c_2s_{02}
-3c_3s_{12}
+6c_4s_{22},
\]

\[
A_3
=
-c_3s_{03}
+4c_4s_{13},
\]

\[
A_4
=
c_4s_{04}.
\]

## API

```python
def mvsk_quartic_coefficients(
    oracle,
    x,
    d,
    *,
    cache=None,
) -> torch.Tensor:
    """Returns [..., 5] ordered [A0,A1,A2,A3,A4]."""
```

## Numerical implementation

Compute

```python
w  = d @ A.T
z2 = z*z
w2 = w*w
```

and reuse products rather than repeatedly calling powers.

---

# 22. Module N — Feasible step cap

For the interior simplex slice

\[
\Delta_n(\tau)
=
\{x\in\Delta_n:x_i\ge\tau\},
\]

and tangent direction \(d\),

\[
\boxed{
\alpha_{\max}(x,d;\tau)
=
\min_{i:d_i<0}
\frac{x_i-\tau}{-d_i}.
}
\]

## API

```python
def simplex_step_cap(
    x: torch.Tensor,
    d: torch.Tensor,
    *,
    tau: float | torch.Tensor = 0.0,
    free_mask: torch.Tensor | None = None,
    direction_tol: float | None = None,
) -> torch.Tensor:
    ...
```

## Numerical rules

Before computing the cap:

\[
d\leftarrow P_Fd.
\]

Define a negative component only when

\[
d_i<-\tau_d,
\]

where

\[
\tau_d
=
\kappa_d\epsilon_{\rm mach}
\max(1,\|d\|_\infty).
\]

Ignore tiny negative noise.

If a nonzero tangent direction has no safely negative component, return a diagnostic failure rather than `inf`.

---

# 23. Module O — Exact quartic line search

Derivative:

\[
\phi'(\alpha)
=
A_1+2A_2\alpha+3A_3\alpha^2+4A_4\alpha^3.
\]

Thus call the cubic solver with

\[
a=4A_4,\quad
b=3A_3,\quad
c=2A_2,\quad
d=A_1.
\]

Candidate set:

\[
\mathcal C
=
\{0,\alpha_{\max}\}
\cup
\{
\alpha:
\phi'(\alpha)=0,\;
0<\alpha<\alpha_{\max}
\}.
\]

## API

```python
@dataclass
class LineSearchResult:
    alpha: torch.Tensor
    value: torch.Tensor
    candidate_alpha: torch.Tensor
    candidate_value: torch.Tensor
    candidate_valid: torch.Tensor
    root_result: RootResult
    accepted: torch.Tensor

def exact_quartic_line_search(
    coeff: torch.Tensor,          # [..., 5]
    alpha_max: torch.Tensor,      # [...]
    *,
    prefer_small_step_on_tie: bool = True,
    interval_tol: float | None = None,
    value_tol: float | None = None,
) -> LineSearchResult:
    ...
```

## 23.1 Horner evaluation

Use

\[
\phi(\alpha)
=
((((A_4\alpha+A_3)\alpha+A_2)\alpha+A_1)\alpha+A_0)
\]

instead of explicit powers.

## 23.2 Boundary inclusion

Roots satisfying

\[
-\tau_\alpha
\le \alpha
\le \alpha_{\max}+\tau_\alpha
\]

may be clamped into the interval.

## 23.3 Tie breaking

If candidate values differ by less than

\[
\tau_f
=
\texttt{value\_tol}\,
\max(1,|\phi(0)|),
\]

prefer the smaller step. This avoids gratuitous boundary jumps.

## 23.4 Descent acceptance

Require

\[
\phi(\alpha^\star)
\le
\phi(0)+\tau_f.
\]

If no candidate produces acceptable descent, return \(\alpha=0\) and a failure flag.

---

# 24. Module P — Simplex projection

For boundary repair, project onto

\[
\Delta_n(\tau)
=
\{x_i\ge\tau,\;\sum_i x_i=1\}.
\]

Let

\[
y=x-\tau\mathbf1.
\]

Then project \(y\) onto

\[
\{y_i\ge0,\;\sum_i y_i=1-n\tau\}.
\]

Use the standard sorting-based simplex projection.

## API

```python
def project_simplex(
    x: torch.Tensor,
    *,
    tau: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    ...
```

Batch sorting is supported by `torch.sort`.

## Stability

After projection, perform one final sum correction on the free/support coordinates if needed.

Check

\[
|\mathbf1^\top x-1|
\le\tau_{\rm sum}
\]

and

\[
\min_i x_i\ge\tau-\tau_{\rm feas}.
\]

---

# 25. Module Q — Active-set / face continuation

Use fixed ambient shape and a boolean mask rather than ragged reduced vectors.

```text
free_mask [..., n]
```

Coordinates not free are fixed at the lower bound.

## 25.1 Face tangent space

All geometry is replaced by `project_face_tangent`.

## 25.2 Activation

If a line search reaches the boundary and

\[
x_i\le\tau+\tau_{\rm active},
\]

the coordinate may be marked active.

## 25.3 KKT-based release

At a candidate face-stationary point, let

\[
\gamma
=
\operatorname{mean}_{i\in F}g_i.
\]

For minimization on the simplex:

- free coordinates should satisfy

\[
g_i\approx\gamma;
\]

- active coordinates should satisfy

\[
g_i\ge\gamma.
\]

An active coordinate violates KKT when

\[
g_i<\gamma-\tau_{\rm KKT}.
\]

Release the most violating coordinate or a bounded number per iteration.

## API

```python
@dataclass
class ActiveSetState:
    free_mask: torch.Tensor
    activation_count: torch.Tensor
    release_count: torch.Tensor

def update_active_set(
    x,
    grad,
    state,
    *,
    tau=0.0,
    active_tol=None,
    kkt_tol=None,
) -> ActiveSetState:
    ...
```

This module is an engineering realization of boundary-aware face continuation and should be carefully cross-validated against the reference reduced-face implementation.

---

# 26. Module R — Projected/KKT residual

A solver must not terminate using raw gradient norm on the constrained problem.

For a face with free set \(F\), estimate

\[
\gamma
=
\operatorname{mean}_{i\in F}g_i.
\]

Define free residual

\[
r_i^{\rm free}
=
g_i-\gamma.
\]

For active coordinates, KKT violation is

\[
r_i^{\rm active}
=
\min(0,g_i-\gamma).
\]

One infinity-norm residual is

\[
\boxed{
r_{\rm KKT}
=
\max
\left(
\max_{i\in F}|g_i-\gamma|,
\max_{i\notin F}\max(0,\gamma-g_i)
\right).
}
\]

API:

```python
def simplex_kkt_residual(
    x,
    grad,
    *,
    tau=0.0,
    free_mask=None,
) -> torch.Tensor:
    ...
```

---

# 27. Module S — YAND iteration state

```python
@dataclass
class YANDState:
    x: torch.Tensor
    value: torch.Tensor
    grad: torch.Tensor
    iteration: int

    free_mask: torch.Tensor
    lambda_reg: torch.Tensor

    grad_norm: torch.Tensor
    kkt_residual: torch.Tensor

    last_alpha: torch.Tensor
    last_direction_norm: torch.Tensor

    converged: torch.Tensor
    stalled: torch.Tensor
    failed: torch.Tensor

    cache: MVSKCache | None
```

No hidden global mutable state.

---

# 28. Module T — Main YAND-MVSK iteration

One outer iteration:

```text
1. Build/cache z = A x
2. Evaluate value and gradient
3. Compute simplex/face KKT residual
4. Stop converged batch members
5. Build face gradient g_T and nu
6. Build level-set projector P_S
7. Compute h = P_S H nu
8. Estimate exact/stochastic logdet correction a
9. Solve regularized tangent system for u
10. Form d = u - nu
11. Project and verify descent
12. Compute alpha_max
13. Assemble exact quartic coefficients
14. Solve derivative cubic in batch
15. Select exact feasible quartic minimizer
16. Update x
17. Apply boundary repair if needed
18. Update active set / face
19. Update lambda and stall diagnostics
```

---

# 29. Module U — Solver API

## 29.1 High-level specialized solver

```python
solver = YANDMVSK(
    R,
    coefficients=(c1, c2, c3, c4),
    dtype=torch.float64,
    device="cuda",
    logdet_mode="stochastic",
    logdet_probes=4,
    linear_solver="pcg",
    preconditioner="mvsk_jacobi",
    line_search="exact_quartic",
    tau=0.0,
)

result = solver.solve(
    x0,
    max_iter=200,
    kkt_rtol=1e-6,
)
```

## 29.2 Direct preprocessed form

```python
data = prepare_mvsk_data(R)

oracle = MVSKOracle(data, coefficients=c)

solver = YANDMVSK.from_oracle(
    oracle,
    ...
)
```

## 29.3 Batch solve

```python
X0 = torch.full(
    (B, n),
    1.0 / n,
    device="cuda",
    dtype=torch.float64,
)

result = solver.solve(X0)
```

The solver must preserve the leading batch shape.

---

# 30. Solver result

```python
@dataclass
class YANDResult:
    x: torch.Tensor
    value: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor
    kkt_residual: torch.Tensor
    status: list[str] | torch.Tensor

    num_hvp: torch.Tensor
    num_third: torch.Tensor
    num_pcg_iter: torch.Tensor
    num_root_solves: torch.Tensor

    regularization: torch.Tensor
    history: dict | None
```

---

# 31. Regularization policy

The tangent operator can be ill-conditioned or indefinite.

Start with

\[
\lambda_0
=
\lambda_{\rm rel}
\max(1,s_H)
\]

where \(s_H\) is a local curvature scale.

A simple scale is

\[
s_H
=
\max_i |\operatorname{diag}(H)_i|.
\]

On PCG curvature failure or breakdown,

\[
\lambda
\leftarrow
\gamma_\lambda\lambda,
\qquad
\gamma_\lambda\approx10.
\]

On consistently easy successful solves, optionally decrease \(\lambda\) slowly.

API policy:

```python
lambda_policy = AdaptiveRegularization(
    initial_rel=1e-6,
    growth=10.0,
    shrink=0.3,
    max_retries=5,
)
```

Never silently continue after a Krylov breakdown.

---

# 32. Stall recovery

A batch member is stalled when, for a configurable number of iterations:

- objective improvement is tiny;
- step size is tiny;
- KKT residual is not improving.

Possible recovery sequence:

```text
1. increase lambda
2. increase logdet probe count
3. rebuild/release active set
4. fall back to projected negative gradient direction
5. restart from current feasible point
6. declare stalled
```

All events must be recorded in diagnostics.

---

# 33. Numerical tolerance policy

Do not scatter arbitrary constants throughout the code.

Centralize them:

```python
@dataclass
class NumericalTolerances:
    degree_factor: float = 64.0
    discriminant_factor: float = 128.0
    curvature_factor: float = 128.0
    projection_factor: float = 128.0
    root_residual_factor: float = 1024.0
    descent_factor: float = 64.0
```

Given dtype epsilon

```python
eps = torch.finfo(dtype).eps
```

derive scale-aware thresholds.

General principle:

\[
\tau
=
\kappa\epsilon_{\rm mach}
\times
\text{local numerical scale}.
\]

Avoid fixed `1e-12` everywhere.

---

# 34. Float32 vs float64 policy

## float64

Default for:

- solver verification;
- root solver correctness tests;
- reference experiments;
- exact logdet backend;
- difficult ill-conditioned problems.

## float32

Allowed for:

- GPU throughput benchmarking;
- large batched objectives;
- large \(T,n\) when memory is limiting.

Recommended hybrid option:

```text
A, X GEMM                 float32
critical reductions       float64 where supported
cubic coefficients        promoted to float64
root solver               float64 on same device
line-search comparison    float64
```

This should be configurable because device FP64 throughput varies greatly.

---

# 35. `torch.compile` strategy

Do not require the entire optimizer loop to compile.

Prioritize compilation of pure kernels:

```text
MVSK objective/gradient
HVP
third action
projectors
PCG fixed-iteration inner loop
cubic analytic solver
quartic evaluation
```

Dynamic active-set and adaptive regularization logic can remain eager.

Expose:

```python
solver.compile_kernels()
```

rather than forcing global compilation.

---

# 36. Module V — Generic autograd oracle (phase 2)

A later general-purpose YAND backend may accept arbitrary smooth PyTorch objectives.

Protocol:

```python
class ObjectiveOracle(Protocol):
    def value_grad(self, x): ...
    def hvp(self, x, v): ...
    def third_action(self, x, u, v): ...
```

For scalar `f(x)`:

\[
Hv
=
D(\nabla f)[v].
\]

Use `torch.func.jvp` on the gradient.

For third action,

\[
D^3f(x)[u,v,\cdot]
=
D(H(x)v)[u].
\]

This can be implemented by nested JVPs.

API:

```python
oracle = AutogradOracle(objective)
```

The analytic MVSK oracle remains the performance backend.

---

# 37. Module W — Validation framework

Every optimized kernel must have a slower reference implementation.

## 37.1 Oracle tests

Compare:

\[
\nabla f_{\rm analytic}
\]

against autograd.

Compare HVP against:

\[
\nabla^2f(x)v
\]

from autograd.

Compare third action against nested autodiff.

Test both single and batched inputs.

## 37.2 Projector tests

Check

\[
P^2v\approx Pv,
\]

\[
\mathbf1^\top Pv\approx0,
\]

and for level projector,

\[
\nu^\top P_Sv\approx0.
\]

## 37.3 PCG tests

Generate SPD systems with controlled condition numbers.

Check residuals and compare against `torch.linalg.solve`.

## 37.4 Cubic tests

Cases must include:

- three distinct real roots;
- one real root;
- double root;
- triple root;
- nearly repeated roots;
- extremely large/small coefficient scale;
- cubic degenerating to quadratic;
- quadratic degenerating to linear;
- zero polynomial;
- random polynomials validated against companion eigenvalues.

Property test:

\[
\rho(r_i)
\]

must be below a dtype-scaled threshold for every valid root.

## 37.5 Quartic line-search tests

Generate random quartics and bounded intervals.

Compare against a dense high-resolution grid plus a reference polynomial-root routine.

Verify:

\[
\phi(\alpha^\star)
\le
\phi(\alpha)
\]

for every candidate stationary point and endpoint.

## 37.6 Full solver tests

Small \(n\):

- compare explicit-frame backend and projector backend;
- compare exact and high-probe stochastic logdet;
- compare dense direct tangent solve and PCG;
- verify final simplex feasibility;
- verify KKT residual.

---

# 38. Benchmark plan

The benchmark suite should answer **where Torch/GPU becomes worthwhile**, not merely report one favorable case.

## 38.1 Cubic solver

Batch sizes:

\[
B\in
\{1,16,256,4096,65536,10^6\}.
\]

Compare:

```text
NumPy polynomial roots / reference
Torch companion CPU
Torch analytic CPU
Torch analytic CUDA
```

Metrics:

```text
roots/second
latency
relative polynomial residual
failure rate
```

## 38.2 Oracle kernels

Grid:

\[
T\in\{252,1000,10000,100000\},
\]

\[
n\in\{50,200,1000,5000\},
\]

\[
B\in\{1,8,64,256\}.
\]

Measure:

```text
value+grad
HVP
third action
memory
```

## 38.3 PCG

Vary:

```text
dimension
condition number
batch size
preconditioner
lambda
fixed/adaptive iterations
```

## 38.4 End-to-end YAND

Compare:

```text
reference CPU implementation
Torch CPU float64
Torch CUDA float64
Torch CUDA float32
Torch CUDA batched
```

Report both:

```text
wall time
iterations
KKT residual
objective
oracle counts
Krylov counts
```

A speedup with worse optimality residual is not a valid win.

---

# 39. Proposed implementation milestones

## Milestone 1 — Independent numerical kernels

Implement and test:

```text
linear roots
stable quadratic roots
batched cubic roots
quartic Horner evaluation
exact quartic minimization
```

Deliverable: reusable `torchyand.polynomial`.

## Milestone 2 — Exact MVSK oracle

Implement:

```text
data preprocessing
value
gradient
HVP
third action
cache
```

Cross-check with autograd.

## Milestone 3 — Geometry + Krylov

Implement:

```text
simplex tangent projector
level tangent projector
batched PCG
Jacobi preconditioner
adaptive regularization
```

## Milestone 4 — Logdet correction

Implement:

```text
explicit exact backend
stochastic matrix-free backend
```

Cross-check on small instances.

## Milestone 5 — Full interior YAND

No active-set complexity yet:

```text
tau > 0
interior simplex slice
exact quartic line search
```

## Milestone 6 — Boundary/face continuation

Add:

```text
simplex projection
active masks
KKT release
face continuation
```

## Milestone 7 — GPU batching

Optimize:

```text
batched portfolios
batched preference vectors
torch.compile kernels
fixed-iteration PCG mode
```

## Milestone 8 — General YAND

Add `AutogradOracle` only after MVSK is stable.

---

# 40. Recommended v0.1 public API

```python
import torch
from torchyand import YANDMVSK

R = torch.randn(
    10_000,
    2_000,
    device="cuda",
    dtype=torch.float64,
)

solver = YANDMVSK(
    R,
    coefficients=(1.0, 1.0, 0.2, 0.1),
    logdet_mode="stochastic",
    logdet_probes=4,
    lambda0=1e-6,
    line_search="exact_quartic",
    linear_solver="pcg",
    preconditioner="mvsk_jacobi",
)

x0 = torch.full(
    (128, 2_000),
    1 / 2_000,
    device="cuda",
    dtype=torch.float64,
)

result = solver.solve(
    x0,
    max_iter=200,
    kkt_rtol=1e-6,
)

print(result.x.shape)
# [128, 2000]
```

---

# 41. Standalone polynomial API

The cubic solver should be useful even without YAND.

```python
from torchyand.polynomial import cubic_real_roots

roots = cubic_real_roots(
    a,
    b,
    c,
    d,
    method="analytic",
)
```

And quartic minimization:

```python
from torchyand.polynomial import minimize_quartic_interval

result = minimize_quartic_interval(
    coeff,          # [..., 5]
    lower=0.0,
    upper=alpha_max,
)
```

This makes the numerical kernel independently useful and independently benchmarkable.

---

# 42. Important distinction: reference fidelity vs Torch-specific extensions

## Directly faithful to the YAND/MVSK formulation

- sample-return oracle;
- matrix-free gradient/HVP/third-action formulas;
- simplex reduction;
- normalized reduced gradient;
- tangent Hessian system;
- log-determinant correction;
- regularization;
- exact/stochastic trace strategies;
- quartic exact line search;
- cubic derivative roots;
- feasibility cap;
- boundary-aware correction / face continuation concept.

## Proposed Torch-specific implementation extensions

- arbitrary leading batch dimensions;
- projector-only scalable geometry without explicit \(U,Q\);
- batched projected stochastic probes;
- batched PCG with per-instance masks;
- stable analytic batched cubic solver;
- automatic polynomial degree fallback;
- Newton root polishing;
- fixed-shape masked face representation;
- `torch.compile` kernel strategy;
- generic `AutogradOracle`.

These extensions should be tested against the paper-faithful reference backend rather than assumed correct.

---

# 43. Core invariants to assert during development

At every outer iteration, debug mode should check:

### Feasibility

\[
\mathbf1^\top x\approx1,
\qquad
x_i\ge\tau.
\]

### Face tangent direction

\[
\mathbf1^\top d\approx0,
\qquad
d_i=0
\quad\text{for fixed coordinates}.
\]

### Level tangent component

\[
\nu^\top u\approx0.
\]

### Descent

\[
g^\top d<0
\]

unless explicitly in a recovery state.

### Root residuals

For every accepted stationary root,

\[
\rho(\alpha)\le\tau_{\rm root}.
\]

### Line-search consistency

\[
f(x+\alpha d)
\approx
\phi(\alpha).
\]

This is a particularly important test because it validates the quartic coefficient assembly.

---

# 44. Performance-critical operations

For the scalable implementation, the dominant kernels should be:

\[
XA^\top,
\qquad
VA^\top,
\qquad
WA,
\]

plus elementwise operations and reductions.

The intended computational pattern is therefore:

```text
GEMM
→ elementwise kernel
→ GEMM
→ projector
→ Krylov recurrence
```

For batch size \(B>1\), matrix-vector products become matrix-matrix products, which is the key reason a PyTorch/GPU implementation can become attractive.

---

# 45. Memory strategy

Do not cache every possible intermediate.

At one outer iterate, the main cache should be approximately:

```text
x
z
z2
optional z3
grad
free_mask
```

For HVPs, compute `Av` transiently.

For stochastic logdet with \(q\) probes, permit two modes:

```text
probe_parallel=True
    shapes [..., q, n]
    faster, more memory

probe_parallel=False
    loop over a small q
    lower memory
```

A small Python loop over \(q=2\) or \(4\) is acceptable if the expensive HVPs remain GPU kernels. Later it can be vectorized.

---

# 46. Determinism

Expose an explicit generator:

```python
generator=torch.Generator(device=device)
```

Stochastic logdet probes should be reproducible when a seed is supplied.

Do not rely on global RNG state inside solver kernels.

---

# 47. Error/status model

Avoid raising exceptions for expected numerical events inside a batch.

Use status codes such as:

```text
CONVERGED
MAX_ITER
STALLED
PCG_BREAKDOWN
NONPOSITIVE_CURVATURE
LINESEARCH_NO_DESCENT
INFEASIBLE_NUMERIC
NAN_DETECTED
ACTIVE_SET_FAILURE
```

Fatal input errors still raise exceptions.

---

# 48. Initial recommended defaults

A conservative correctness-first configuration:

```python
dtype = torch.float64
logdet_mode = "stochastic"
logdet_probes = 4
pcg_rtol = 1e-6
pcg_max_iter = 100
lambda0_rel = 1e-6
lambda_growth = 10.0
max_regularization_retries = 5
line_search = "exact_quartic"
tau = 0.0
root_polish_steps = 1
prefer_small_step_on_tie = True
```

For performance experiments, loosen only one component at a time.

---

# 49. Research/engineering hypotheses to test

The implementation should be designed to test several concrete hypotheses.

### H1 — Cubic root batching

For sufficiently large batches, an analytic GPU cubic solver should outperform per-polynomial CPU root routines while retaining small normalized residual.

### H2 — Batch transforms GEMV into GEMM

For batch portfolio optimization,

\[
X A^\top
\]

should use the GPU substantially better than repeated single-portfolio GEMV operations.

### H3 — Implicit projectors matter

Avoiding explicit \(U\) and \(Q\) should materially reduce memory and setup cost at large \(n\).

### H4 — Stochastic logdet is the scalable regime

Exact trace should be the validation backend; small-\(q\) stochastic correction should become favorable as tangent dimension grows.

### H5 — Hybrid precision may dominate pure float64

Float32 GEMM with promoted root/line-search arithmetic may offer a better speed/accuracy trade-off on GPUs with weak FP64 throughput.

---

# 50. Minimum viable publication-quality benchmark

A convincing technical report should include:

1. **Correctness**
   - gradient/HVP/third action vs autograd;
   - projected vs explicit geometry;
   - analytic cubic roots vs companion eigenvalues;
   - exact quartic line search vs brute-force reference.

2. **Numerical robustness**
   - nearly repeated roots;
   - ill-conditioned tangent systems;
   - boundary portfolios;
   - high kurtosis / large projected returns.

3. **Performance**
   - CPU vs GPU;
   - single vs batch;
   - exact vs stochastic logdet;
   - explicit vs projector geometry;
   - float32 vs float64.

4. **Optimization quality**
   - final objective;
   - simplex feasibility;
   - KKT residual;
   - iteration/oracle counts.

---

# 51. Recommended coding order

The safest order is:

```text
1. numerics.py
2. linear/quadratic/cubic roots
3. quartic interval minimizer
4. MVSK data + oracle
5. simplex and level projectors
6. batched PCG
7. preconditioner
8. exact logdet reference
9. stochastic logdet
10. affine-normal direction
11. interior YAND solver
12. active-set/face continuation
13. benchmarking
14. generic autograd YAND
```

Do not start with the full solver. The cubic root solver and exact quartic line search should become independently trusted before they are inserted into YAND.

---

# 52. References used for the mathematical design

1. Ya-Juan Wang, Yi-Shuai Niu, Artan Sheshmani, Shing-Tung Yau (2026),  
   **Yau's Affine-Normal Descent for Large-Scale Unrestricted Higher-Moment Portfolio Optimization**, arXiv:2604.25378.

2. Yi-Shuai Niu, Artan Sheshmani, Shing-Tung Yau (2026),  
   **Affine Normal Directions via Log-Determinant Geometry: Scalable Computation under Sparse Polynomial Structure**, arXiv:2604.01163.

3. Yi-Shuai Niu, Artan Sheshmani, Shing-Tung Yau (2026),  
   **Yau's Affine Normal Descent: Algorithmic Framework and Convergence Analysis**, arXiv:2603.28448.

---

# 53. Final architecture summary

The intended scalable algorithm is

\[
\boxed{
\begin{aligned}
X
&\xrightarrow{\text{MVSK oracle}}
(f,g,Hv,T_3)
\\
&\xrightarrow{\text{simplex/face projector}}
g_T,\nu,P_S
\\
&\xrightarrow{\text{stochastic logdet + PCG}}
a,h,u
\\
&\xrightarrow{\text{affine-normal assembly}}
d=u-\nu
\\
&\xrightarrow{\text{feasibility cap}}
[0,\alpha_{\max}]
\\
&\xrightarrow{\text{quartic coefficients}}
\phi(\alpha)
\\
&\xrightarrow{\text{batched cubic roots}}
\{\alpha_i\}
\\
&\xrightarrow{\text{exact candidate comparison}}
\alpha^\star
\\
&\xrightarrow{\text{boundary / active-set update}}
X^+.
\end{aligned}
}
\]

The three most reusable numerical products of the project are:

\[
\boxed{\text{batched cubic real-root solver}}
\]

\[
\boxed{\text{batched exact quartic interval minimizer}}
\]

\[
\boxed{\text{projector-based batched matrix-free affine-normal solver}}
\]

The third item is the main Torch-specific architectural opportunity: it removes the need to materialize the dense simplex basis \(U\) and level-set basis \(Q\), keeping the scalable path in fixed-shape tensor operations suitable for GPU execution.
