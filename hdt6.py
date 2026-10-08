# %% [markdown]
# # HDT 6 - Modelación y Simulación (CC2017)
# ## Competencia entre dos plataformas: Task 1 y Task 2
#
# Sistema (Lotka-Volterra de competencia):
#
# $\dot x = r_1 x(1-x-\alpha_{12}y), \qquad \dot y = r_2 y(1-y-\alpha_{21}x)$
#
# **Grupo $g=1$** -> conjunto $((g-1)\bmod 4)+1 = 1$ -> $\alpha_{12}=0.5,\ \alpha_{21}=0.6$,
# $r_1=1$, $r_2 = 0.5+0.1((g-1)\bmod 5)=0.5$.
#
# Este archivo se ejecuta como script (`python hdt6.py`) o como notebook (`hdt6.ipynb`,
# generado con `make_notebook.py`). Escribe las figuras en `figs/` y los números en
# `results.json`. Solo se usa `solve_ivp` (DOP853) como referencia en el Task 2.1b;
# todos los demás integradores son propios (Euler y RK4).

# %%
import json
import os

import matplotlib
if "get_ipython" not in globals():          # script sin pantalla: backend de archivos
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp        # SOLO como referencia (Task 2.1b)

SEED = 2026                                   # semilla fija: todo es reproducible
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

os.makedirs("figs", exist_ok=True)
RES = {}                                      # aqui se acumulan los resultados numericos

# Paleta apta para daltonismo (Okabe-Ito) y estilo sobrio
C_BLUE, C_ORANGE, C_GREEN, C_VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
C_PURPLE, C_SKY, C_GRAY, C_INK = "#CC79A7", "#56B4E9", "#9AA0A6", "#222222"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10, "axes.grid": True,
    "grid.color": "#E5E5E5", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "legend.frameon": False,
})


def jsonable(o):
    """Convierte arreglos/valores de NumPy a tipos de Python para guardarlos en JSON."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return o


# %% [markdown]
# ## Modelo, equilibrios y jacobiana (Task 1)
#
# Equilibrios (derivados a mano en el informe): $(0,0)$, $(1,0)$, $(0,1)$ y el interior
# $x^*=\dfrac{1-\alpha_{12}}{1-\alpha_{12}\alpha_{21}},\ y^*=\dfrac{1-\alpha_{21}}{1-\alpha_{12}\alpha_{21}}$
# (solucion de $x+\alpha_{12}y=1,\ y+\alpha_{21}x=1$ por Cramer).
#
# Jacobiana (derivada de cada ecuacion respecto a cada variable):
# $J=\begin{pmatrix} r_1(1-2x-\alpha_{12}y) & -r_1\alpha_{12}x\\ -r_2\alpha_{21}y & r_2(1-2y-\alpha_{21}x)\end{pmatrix}$

# %%
A12, A21, R1, R2 = 0.5, 0.6, 1.0, 0.5         # parametros del grupo 1
G = 1


def make_field(a12, a21, r1, r2):
    """Campo vectorial f(s) del sistema; acepta estados con forma (...,2)."""
    def f(s):
        x, y = s[..., 0], s[..., 1]
        return np.stack([r1 * x * (1 - x - a12 * y),
                         r2 * y * (1 - y - a21 * x)], axis=-1)
    return f


def make_jac(a12, a21, r1, r2):
    def J(s):
        x, y = s
        return np.array([[r1 * (1 - 2 * x - a12 * y), -r1 * a12 * x],
                         [-r2 * a21 * y, r2 * (1 - 2 * y - a21 * x)]])
    return J


def equilibria(a12, a21):
    """Cuatro equilibrios; el interior por Cramer (determinante 1 - a12*a21)."""
    D = 1 - a12 * a21
    return {"E0": np.array([0.0, 0.0]), "EX": np.array([1.0, 0.0]),
            "EY": np.array([0.0, 1.0]),
            "E*": np.array([(1 - a12) / D, (1 - a21) / D])}


def classify(J, tol=1e-12):
    """Clasificacion por traza/determinante/discriminante (plano tau-Delta)."""
    tau, det = np.trace(J), np.linalg.det(J)
    disc = tau ** 2 - 4 * det
    lam = np.linalg.eigvals(J)
    if det < -tol:
        kind = "silla (inestable)"
    elif abs(det) <= tol:
        kind = "degenerado (autovalor 0)"
    elif abs(tau) <= tol:
        kind = "centro"
    else:
        stab = "estable" if tau < 0 else "inestable"
        kind = f"{'nodo' if disc >= 0 else 'foco'} {stab}"
    return dict(tau=tau, det=det, disc=disc, eig=np.sort_complex(lam), kind=kind)


f = make_field(A12, A21, R1, R2)
Jf = make_jac(A12, A21, R1, R2)
EQ = equilibria(A12, A21)

print("=== Task 1.1c / 1.2b: equilibrios del grupo 1 ===")
RES["equilibria"] = {}
for name, s in EQ.items():
    c = classify(Jf(s))
    feas = bool(np.all(s >= 0))
    print(f"{name:3s} {np.round(s, 6)} factible={feas} tau={c['tau']:.4f} "
          f"Delta={c['det']:.4f} disc={c['disc']:.4f} eig={np.round(c['eig'].real, 5)} -> {c['kind']}")
    RES["equilibria"][name] = dict(punto=s, factible=feas, **{k: v for k, v in c.items()})

lam_int = classify(Jf(EQ["E*"]))["eig"].real
RES["tiempos_recuperacion"] = {"lento": 1 / np.min(np.abs(lam_int)),
                               "rapido": 1 / np.max(np.abs(lam_int))}
print("tiempo de recuperacion 1/min|Re(lambda)| =", RES["tiempos_recuperacion"])

# %% [markdown]
# ## Task 1.4 - verificacion numerica del analisis del companero
# ($\alpha_{12}=1.4,\ \alpha_{21}=1.5,\ r_1=1,\ r_2=0.8$)

# %%
fc, Jc = make_field(1.4, 1.5, 1, 0.8), make_jac(1.4, 1.5, 1, 0.8)
Ec = equilibria(1.4, 1.5)["E*"]
Jstar = Jc(Ec)
cc = classify(Jstar)
print("\n=== Task 1.4 ===")
print("E* =", Ec, "\nJ =\n", Jstar)
print("Delta correcto = det(J) =", cc["det"], "| Delta del companero (con +) =",
      Jstar[0, 0] * Jstar[1, 1] + Jstar[0, 1] * Jstar[1, 0])
print("tau =", cc["tau"], "disc =", cc["disc"], "eig =", cc["eig"].real, "->", cc["kind"])
RES["task14"] = dict(Estar=Ec, J=Jstar, **cc,
                     delta_companero=Jstar[0, 0] * Jstar[1, 1] + Jstar[0, 1] * Jstar[1, 0])
for name, s in equilibria(1.4, 1.5).items():
    if name != "E*":
        print(name, classify(Jc(s))["eig"].real, classify(Jc(s))["kind"])
        RES["task14"][name] = classify(Jc(s))["eig"].real

# %% [markdown]
# ## Task 1.2c - plano traza-determinante

# %%
fig, ax = plt.subplots(figsize=(6.2, 4.4))
tt = np.linspace(-2.2, 2.2, 400)
ax.fill_between(tt, tt ** 2 / 4, 1.0, color="#F3E9D2", label=r"focos ($\tau^2<4\Delta$)")
ax.axhline(0, color=C_INK, lw=0.8)
ax.axvline(0, color=C_INK, lw=0.8)
ax.plot(tt, tt ** 2 / 4, color=C_INK, lw=1.2, label=r"$\tau^2=4\Delta$")
ax.text(-2.1, -0.42, "sillas ($\\Delta<0$)", color=C_INK, fontsize=9)
ax.text(-2.1, 0.62, "focos estables", fontsize=9, color="#7a6a3a")
ax.text(1.1, 0.62, "focos\ninestables", fontsize=9, color="#7a6a3a")
ax.text(-2.1, 0.02, "nodos\nestables", fontsize=9, color=C_INK)
ax.text(1.5, 0.12, "nodos inestables", fontsize=9, color=C_INK)
mk = {"E0": ("s", C_VERM, "(0,0)"), "EX": ("D", C_ORANGE, "(1,0)"),
      "EY": ("D", C_ORANGE, "(0,1)"), "E*": ("o", C_GREEN, "interior")}
for name, (m, col, lab) in mk.items():
    c = RES["equilibria"][name]
    ax.scatter(c["tau"], c["det"], marker=m, s=70, color=col, edgecolor=C_INK, zorder=5)
    ax.annotate(lab, (c["tau"], c["det"]), textcoords="offset points", xytext=(7, 6), fontsize=9)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-0.5, 0.75)
ax.set_xlabel(r"traza $\tau$")
ax.set_ylabel(r"determinante $\Delta$")
ax.set_title("Equilibrios del grupo 1 en el plano traza-determinante")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("figs/fig_traza_det.png")
plt.close(fig)

# %% [markdown]
# ## Task 1.3a/b - nulclinas, regiones y signos del campo
#
# Nulclina de $x$: $x=0$ o $L_1: x+\alpha_{12}y=1$. Nulclina de $y$: $y=0$ o $L_2: y+\alpha_{21}x=1$.

# %%
xs = np.linspace(0, 1.5, 200)
L1 = (1 - xs) / A12                  # nulclina de x:  x + a12*y = 1
L2 = 1 - A21 * xs                    # nulclina de y:  y + a21*x = 1


def unit_quiver(ax, n=18, lim=1.5, color="#B8BDC2", scale=32):
    gx, gy = np.meshgrid(np.linspace(0.04, lim - 0.04, n), np.linspace(0.04, lim - 0.04, n))
    S = np.stack([gx, gy], axis=-1)
    F = f(S)
    nrm = np.linalg.norm(F, axis=-1) + 1e-12
    ax.quiver(gx, gy, F[..., 0] / nrm, F[..., 1] / nrm, color=color, scale=scale,
              width=0.003, headwidth=4, zorder=1)


fig, ax = plt.subplots(figsize=(6.4, 5.6))
unit_quiver(ax, n=16)
ax.plot(xs, np.where(L1 <= 1.5, L1, np.nan), color=C_BLUE, lw=2, label=r"$\dot x=0$: $x+0.5\,y=1$")
ax.plot(xs, L2, color=C_VERM, lw=2, label=r"$\dot y=0$: $y+0.6\,x=1$")
ax.axvline(0, color=C_BLUE, lw=2, ls="--", label=r"$\dot x=0$: $x=0$ (eje, invariante)")
ax.axhline(0, color=C_VERM, lw=2, ls="--", label=r"$\dot y=0$: $y=0$ (eje, invariante)")
# regiones: (signo xdot, signo ydot) y flecha real del campo
reg = {"I": ((0.25, 0.30), "(+,+)"), "II": ((0.30, 1.00), "(+,$-$)"),
       "III": ((1.10, 0.20), "($-$,+)"), "IV": ((1.20, 1.00), "($-$,$-$)")}
for name, (p, sg) in reg.items():
    Fp = f(np.array(p))
    Fp = Fp / np.linalg.norm(Fp) * 0.17
    ax.annotate("", xy=(p[0] + Fp[0], p[1] + Fp[1]), xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.8))
    ax.text(p[0] - 0.02, p[1] + 0.07, f"{name} {sg}", fontsize=9, ha="center", fontweight="bold")
for name, (m, col, lab) in {"E0": ("s", C_VERM, ""), "EX": ("D", C_ORANGE, ""),
                            "EY": ("D", C_ORANGE, ""), "E*": ("o", C_GREEN, "")}.items():
    s = EQ[name]
    ax.scatter(*s, marker=m, s=70, color=col, edgecolor=C_INK, zorder=6)
ax.annotate(r"$(x^*,y^*)=(5/7,\,4/7)$", EQ["E*"], textcoords="offset points", xytext=(8, 10), fontsize=9)
ax.set_xlim(-0.02, 1.5)
ax.set_ylim(-0.02, 1.5)
ax.set_xlabel("$x$ (plataforma A)")
ax.set_ylabel("$y$ (plataforma B)")
ax.set_title("Nulclinas, regiones y direccion del campo")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("figs/fig_nulclinas.png")
plt.close(fig)

# %% [markdown]
# ## Task 2.1 - Integradores desde cero: Euler y Runge-Kutta 4
#
# Euler (orden 1): $x_{n+1}=x_n+h f(x_n)$.
#
# RK4 (orden 4): $k_1=f(x_n)$, $k_2=f(x_n+\tfrac h2k_1)$, $k_3=f(x_n+\tfrac h2k_2)$,
# $k_4=f(x_n+hk_3)$, $x_{n+1}=x_n+\tfrac h6(k_1+2k_2+2k_3+k_4)$.

# %%
def euler_step(fun, x, h):
    """Un paso de Euler explicito: x_{n+1} = x_n + h f(x_n)."""
    return x + h * fun(x)


def rk4_step(fun, x, h):
    """Un paso de RK4 clasico (k1..k4); funciona con un estado o con un lote (N,2)."""
    k1 = fun(x)
    k2 = fun(x + 0.5 * h * k1)
    k3 = fun(x + 0.5 * h * k2)
    k4 = fun(x + h * k3)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(step, fun, x0, h, T):
    """Recibe campo f, estado inicial, paso h y tiempo final T; devuelve (t, X)."""
    n = int(round(T / h))
    X = np.empty((n + 1, len(x0)))
    X[0] = x0
    for k in range(n):
        X[k + 1] = step(fun, X[k], h)
    return np.arange(n + 1) * h, X


def euler(fun, x0, h, T):
    return integrate(euler_step, fun, np.asarray(x0, float), h, T)


def rk4(fun, x0, h, T):
    return integrate(rk4_step, fun, np.asarray(x0, float), h, T)


# Referencia: solve_ivp con DOP853 y tolerancias 1e-12 (unico uso permitido de solve_ivp)
P = {"P1": (0.05, 0.05), "P2": (0.80, 0.30), "P3": (0.30, 0.80), "P4": (0.60, 0.60)}
T_REF = 10.0
sol = solve_ivp(lambda t, s: f(s), (0, T_REF), P["P2"], method="DOP853",
                rtol=1e-12, atol=1e-12)
x_ref = sol.y[:, -1]
hs = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
err_e, err_r = [], []
for h in hs:
    err_e.append(np.linalg.norm(euler(f, P["P2"], h, T_REF)[1][-1] - x_ref))
    err_r.append(np.linalg.norm(rk4(f, P["P2"], h, T_REF)[1][-1] - x_ref))
err_e, err_r = np.array(err_e), np.array(err_r)
slope_e = np.polyfit(np.log(hs), np.log(err_e), 1)
slope_r = np.polyfit(np.log(hs), np.log(err_r), 1)
# pendientes locales entre pasos consecutivos (para ver el regimen preasintotico)
loc_e = np.diff(np.log(err_e)) / np.diff(np.log(hs))
loc_r = np.diff(np.log(err_r)) / np.diff(np.log(hs))
print("\n=== Task 2.1 ===")
print("x_ref(10) =", x_ref)
for h, a, b in zip(hs, err_e, err_r):
    print(f"h={h:<7} err_Euler={a:.3e} err_RK4={b:.3e}")
print("pendiente Euler =", slope_e[0], " RK4 =", slope_r[0])
print("pendientes locales Euler:", loc_e, " RK4:", loc_r)
RES["task21"] = dict(x_ref=x_ref, h=hs, err_euler=err_e, err_rk4=err_r,
                     pendiente_euler=slope_e[0], pendiente_rk4=slope_r[0],
                     locales_euler=loc_e, locales_rk4=loc_r)

fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.loglog(hs, err_e, "o-", color=C_VERM, label=f"Euler (pendiente {slope_e[0]:.2f})")
ax.loglog(hs, err_r, "s-", color=C_BLUE, label=f"RK4 (pendiente {slope_r[0]:.2f})")
ax.loglog(hs, err_e[-1] * (hs / hs[-1]), ":", color=C_VERM, alpha=.6, label=r"referencia $\propto h$")
ax.loglog(hs, err_r[-1] * (hs / hs[-1]) ** 4, ":", color=C_BLUE, alpha=.6, label=r"referencia $\propto h^4$")
ax.set_xlabel("paso $h$")
ax.set_ylabel(r"error $\|x_h(10)-x_{\rm ref}(10)\|_2$")
ax.set_title("Convergencia de Euler y RK4 (estado P2, $T=10$)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/fig_convergencia.png")
plt.close(fig)

# %% [markdown]
# ## Task 2.2 - Retrato de fases y comparacion con las predicciones del Task 1.3c
#
# Las predicciones se escribieron en el informe (Task 1.3c) antes de ejecutar este bloque.

# %%
H_TRAJ, T_TRAJ = 0.02, 80.0
# 12+ estados iniciales: cerca de los ejes, esquinas y aleatorios (semilla fija)
near_axes = [(0.01, 1.40), (0.01, 0.60), (0.01, 0.10), (0.02, 0.02), (1.40, 0.01),
             (0.60, 0.01), (0.10, 0.01), (1.45, 1.45), (1.45, 0.60), (0.55, 1.45)]
rand_ics = [tuple(v) for v in RNG.uniform(0.05, 1.45, size=(8, 2))]
other_ics = near_axes + rand_ics
print("\n=== Task 2.2: trayectorias auxiliares:", len(other_ics), "+ 4 de referencia ===")

fig, ax = plt.subplots(figsize=(7.2, 6.4))
unit_quiver(ax, n=22, scale=40, color="#C9CDD1")
for s0 in other_ics:
    _, X = rk4(f, s0, H_TRAJ, T_TRAJ)
    ax.plot(X[:, 0], X[:, 1], color="#8FA3B5", lw=0.9, alpha=0.9, zorder=2)
    ax.plot(*s0, ".", color="#8FA3B5", ms=4, zorder=2)
ax.plot(xs, np.where(L1 <= 1.5, L1, np.nan), color=C_BLUE, lw=1.6, ls="--", label=r"nulclina $\dot x=0$")
ax.plot(xs, L2, color=C_VERM, lw=1.6, ls="--", label=r"nulclina $\dot y=0$")
pstyle = {"P1": (C_BLUE, "-"), "P2": (C_PURPLE, "-"), "P3": (C_GREEN, "-"), "P4": ("#000000", "-")}
TRAJ = {}
for nm, s0 in P.items():
    t_, X = rk4(f, s0, H_TRAJ, T_TRAJ)
    TRAJ[nm] = (t_, X)
    col, ls = pstyle[nm]
    ax.plot(X[:, 0], X[:, 1], color=col, lw=2.6, ls=ls, zorder=4, label=f"{nm} {s0}")
    ax.scatter(*s0, color=col, edgecolor="white", s=60, zorder=7)
    ax.annotate(nm, s0, textcoords="offset points", xytext=(6, -12), fontsize=9, fontweight="bold", color=col)
# equilibrios coloreados por tipo
ax.scatter(*EQ["E0"], marker="s", s=90, color=C_VERM, edgecolor=C_INK, zorder=8, label="nodo inestable (fuente)")
ax.scatter(*EQ["EX"], marker="D", s=90, color=C_ORANGE, edgecolor=C_INK, zorder=8, label="silla")
ax.scatter(*EQ["EY"], marker="D", s=90, color=C_ORANGE, edgecolor=C_INK, zorder=8)
ax.scatter(*EQ["E*"], marker="o", s=110, color=C_GREEN, edgecolor=C_INK, zorder=8, label="nodo estable (interior)")
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)
ax.set_xlabel("$x$ (plataforma A)")
ax.set_ylabel("$y$ (plataforma B)")
ax.set_title("Retrato de fases, grupo 1 (RK4, $h=0.02$)")
ax.legend(loc="upper right", fontsize=7.5, ncol=2)
fig.tight_layout()
fig.savefig("figs/fig_retrato_fases.png")
plt.close(fig)

# --- comparacion con las predicciones (mas fina: h=0.01, T=120) ---
Estar = EQ["E*"]
cmp_rows = {}
for nm, s0 in P.items():
    t_, X = rk4(f, s0, 0.01, 120.0)
    F = f(X)
    sx, sy = np.sign(F[:, 0]), np.sign(F[:, 1])
    out = np.where((sx < 0) | (sy < 0))[0]
    k = int(out[0]) if len(out) else None
    via = None if k is None else ("L1 (nulclina de x)" if sx[k] < 0 else "L2 (nulclina de y)")
    # invariancia de regiones II/III tras salir de I (cono de orden de un sistema competitivo)
    inv = None if k is None else bool(np.all(F[k:, 0] * F[k:, 1] <= 1e-14))
    dist = np.linalg.norm(X - Estar, axis=1)
    k3 = int(np.argmax(dist < 1e-3))
    cmp_rows[nm] = dict(
        inicial=s0, final=X[-1], dist_final=dist[-1], sale_de_I_por=via,
        t_salida=None if k is None else t_[k], punto_salida=None if k is None else X[k],
        max_x=X[:, 0].max(), max_y=X[:, 1].max(), regiones_II_III_invariantes=inv,
        t_dist_1e3=t_[k3], x_monotona_despues=bool(np.all(np.diff(X[k:, 0]) * np.sign(F[k, 0]) >= -1e-12)) if k else None)
    print(nm, {kk: (np.round(v, 4) if isinstance(v, np.ndarray) else v) for kk, v in cmp_rows[nm].items()})
RES["task22"] = cmp_rows

# %% [markdown]
# ## Task 2.3 - Mapa de cuencas (malla 60x60 en [0.01, 1.5]^2)
#
# **Criterio de convergencia.** Un estado se considera estabilizado cuando
# $\|f(x)\|_\infty<10^{-8}$ (campo practicamente nulo). Como todos los equilibrios son hiperbolicos,
# $\|x-x_{eq}\|\lesssim\|J^{-1}\|\,\|f\|$, asi que esa cota garantiza estar a menos de ~$10^{-7}$ del
# equilibrio; luego se asigna al equilibrio mas cercano exigiendo distancia $<10^{-4}$.
# Si no se cumple antes de $T_{max}=600$ se etiqueta como "no convergio".

# %%
def basin_map(fun, eqs, n=60, lo=0.01, hi=1.5, h=0.05, Tmax=600.0, tol=1e-8):
    gx = np.linspace(lo, hi, n)
    X0 = np.stack(np.meshgrid(gx, gx), axis=-1).reshape(-1, 2)   # (n*n, 2): x varia por columnas
    S = X0.copy()
    label = -np.ones(len(S), int)
    t_conv = np.full(len(S), np.nan)
    active = np.ones(len(S), bool)
    names = list(eqs)
    E = np.array([eqs[k] for k in names])
    for step in range(int(Tmax / h)):
        S[active] = rk4_step(fun, S[active], h)          # RK4 propio, vectorizado en el lote
        idx = np.where(active)[0]
        done = np.max(np.abs(fun(S[idx])), axis=1) < tol
        for i in idx[done]:
            d = np.linalg.norm(E - S[i], axis=1)
            if d.min() < 1e-4:
                label[i] = int(np.argmin(d))
            t_conv[i] = (step + 1) * h
            active[i] = False
        if not active.any():
            break
    return gx, label.reshape(n, n), t_conv.reshape(n, n), names


gx, LAB, TCONV, names = basin_map(f, EQ)
frac = {nm: float(np.mean(LAB == i)) for i, nm in enumerate(names)}
frac["no_convergio"] = float(np.mean(LAB == -1))
print("\n=== Task 2.3 ===  fracciones por cuenca:", frac,
      "| t_conv: max=%.1f media=%.1f" % (np.nanmax(TCONV), np.nanmean(TCONV)))
RES["task23"] = dict(fracciones=frac, t_conv_max=np.nanmax(TCONV), t_conv_media=np.nanmean(TCONV),
                     n_celdas=int(LAB.size))

fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.7))
ax = axs[0]
cmap_b = matplotlib.colors.ListedColormap([C_GRAY, C_VERM, C_ORANGE, C_ORANGE, C_GREEN])
ax.imshow(LAB, origin="lower", extent=[0.01, 1.5, 0.01, 1.5], cmap=cmap_b, vmin=-1, vmax=3, aspect="equal")
ax.plot(xs, np.where(L1 <= 1.5, L1, np.nan), color=C_BLUE, lw=1.2, ls="--")
ax.plot(xs, L2, color=C_VERM, lw=1.2, ls="--")
ax.scatter(*EQ["E*"], marker="o", s=70, color="white", edgecolor=C_INK, zorder=5)
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)
ax.set_title("Mapa de cuencas (60x60): 100 % -> $E^*$")
ax.set_xlabel("$x_0$")
ax.set_ylabel("$y_0$")
ax.grid(False)
ax.text(0.05, 1.40, "cuenca de $E^*=(5/7,4/7)$", color="white", fontsize=9, fontweight="bold")
ax = axs[1]
im = ax.imshow(TCONV, origin="lower", extent=[0.01, 1.5, 0.01, 1.5], cmap="viridis", aspect="equal")
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("tiempo de convergencia")
ax.set_title("Tiempo hasta $\\|f\\|_\\infty<10^{-8}$")
ax.set_xlabel("$x_0$")
ax.set_ylabel("$y_0$")
ax.grid(False)
fig.tight_layout()
fig.savefig("figs/fig_cuencas.png")
plt.close(fig)

# %% [markdown]
# ## Task 2.4 - Tasa de decaimiento, paso maximo de Euler
#
# a-b) Perturbaciones de norma $10^{-4}$ en 8 direcciones (angulos $k\pi/4$) alrededor de $E^*$ (esta en el
# interior, asi que todas las direcciones son admisibles); RK4 con $h=0.01$.
# Se ajusta una recta a $\ln\|x(t)-x^*\|$ en (i) la ventana pedida, $10^{-4}\to10^{-6}$, y
# (ii) la ventana asintotica $10^{-5}\to10^{-6}$.

# %%
Jstar_g = Jf(EQ["E*"])
w, V = np.linalg.eig(Jstar_g)
order = np.argsort(np.abs(w.real))               # primero el lento
lam_slow, lam_fast = w[order[0]].real, w[order[1]].real
v_slow, v_fast = V[:, order[0]].real, V[:, order[1]].real
v_slow, v_fast = v_slow / np.linalg.norm(v_slow), v_fast / np.linalg.norm(v_fast)
print("\n=== Task 2.4 ===  lambda_lento=%.6f lambda_rapido=%.6f" % (lam_slow, lam_fast))
print("autovector lento", v_slow, "angulo(deg)=", np.degrees(np.arctan2(v_slow[1], v_slow[0])) % 180)
print("autovector rapido", v_fast, "angulo(deg)=", np.degrees(np.arctan2(v_fast[1], v_fast[0])) % 180)


def decay_fit(t, d, hi, lo):
    """Ajuste lineal de ln d(t) donde lo <= d <= hi; devuelve -pendiente (tasa)."""
    m = (d <= hi) & (d >= lo)
    return -np.polyfit(t[m], np.log(d[m]), 1)[0]


DELTA, H24 = 1e-4, 0.01
dirs = {f"{k * 45:d} deg": np.array([np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)]) for k in range(8)}
dirs["autovector rapido"] = v_fast
dirs["autovector lento"] = v_slow
rows24 = {}
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.1))
for nm, u in dirs.items():
    s0 = EQ["E*"] + DELTA * u
    t_, X = rk4(f, s0, H24, 90.0)
    d = np.linalg.norm(X - EQ["E*"], axis=1)
    r_full = decay_fit(t_, d, 1.0000001e-4, 1e-6)
    r_late = decay_fit(t_, d, 1e-5, 1e-6)
    # tasa instantanea al inicio (primeras unidades de tiempo) para ver el transitorio
    m0 = t_ <= 2.0
    r_init = -np.polyfit(t_[m0], np.log(d[m0]), 1)[0]
    rows24[nm] = dict(r_ventana=r_full, r_tardia=r_late, r_inicial=r_init,
                      err_rel_ventana=abs(r_full - abs(lam_slow)) / abs(lam_slow),
                      err_rel_tardia=abs(r_late - abs(lam_slow)) / abs(lam_slow))
    print(f"{nm:18s} r[1e-4->1e-6]={r_full:.5f} r[1e-5->1e-6]={r_late:.5f} r[t<=2]={r_init:.4f} "
          f"err_rel(ventana)={rows24[nm]['err_rel_ventana']:.2%} err_rel(tardia)={rows24[nm]['err_rel_tardia']:.3%}")
    lw, al = (2.2, 1) if nm.startswith("autov") else (1.0, 0.85)
    axs[0].semilogy(t_, d, lw=lw, alpha=al, label=nm if nm.startswith("autov") or nm in ("0 deg", "90 deg") else None)
tt = np.linspace(0, 40, 50)
axs[0].semilogy(tt, 1e-4 * np.exp(lam_slow * tt), "k--", lw=1.3, label=r"$e^{\lambda_{lento}t}$")
axs[0].semilogy(tt, 1e-4 * np.exp(lam_fast * tt), "k:", lw=1.3, label=r"$e^{\lambda_{rapido}t}$")
axs[0].axhspan(1e-6, 1e-4, color="#F3E9D2", alpha=0.5, zorder=0)
axs[0].set_xlim(0, 40)
axs[0].set_ylim(1e-8, 2e-4)
axs[0].set_xlabel("$t$")
axs[0].set_ylabel(r"$\|x(t)-x^*\|$")
axs[0].set_title("Decaimiento hacia $E^*$ (8 direcciones + autovectores)")
axs[0].legend(fontsize=8)
RES["task24ab"] = dict(lam_lento=lam_slow, lam_rapido=lam_fast, v_lento=v_slow, v_rapido=v_fast, filas=rows24)

# panel 2: tasa local d/dt ln d para cada direccion (muestra el transitorio)
for nm, u in dirs.items():
    s0 = EQ["E*"] + DELTA * u
    t_, X = rk4(f, s0, H24, 60.0)
    d = np.linalg.norm(X - EQ["E*"], axis=1)
    rate = -np.gradient(np.log(d), t_)
    lw, al = (2.2, 1) if nm.startswith("autov") else (1.0, 0.85)
    axs[1].plot(t_, rate, lw=lw, alpha=al)
axs[1].axhline(abs(lam_slow), color="k", ls="--", lw=1.2)
axs[1].axhline(abs(lam_fast), color="k", ls=":", lw=1.2)
axs[1].set_xlim(0, 40)
axs[1].set_ylim(0, 1.3)
axs[1].set_xlabel("$t$")
axs[1].set_ylabel(r"tasa local $-\,d\ln\|x-x^*\|/dt$")
axs[1].set_title("Transitorio: la tasa local converge a $|\\lambda_{lento}|$")
fig.tight_layout()
fig.savefig("figs/fig_decaimiento.png")
plt.close(fig)

# %% [markdown]
# c) **Estabilidad de Euler** (derivacion a mano en el informe). Para $\lambda$ complejo,
# $|1+h\lambda|^2=1+2h\,\mathrm{Re}\lambda+h^2|\lambda|^2<1 \iff 0<h<-2\,\mathrm{Re}\lambda/|\lambda|^2$.
#
# d) **Bisecccion numerica** del mayor $h$ con el que Euler converge al equilibrio.

# %%
def hmax_euler(lam):
    return -2 * lam.real / abs(lam) ** 2


hmax = {"lento": hmax_euler(complex(lam_slow)), "rapido": hmax_euler(complex(lam_fast))}
H_THEORY = min(hmax.values())
print("\nh_max teorico por autovalor:", hmax, "-> mas restrictivo:", H_THEORY)


def euler_converges(h, s0, xeq, N=300_000, tol=1e-9, blow=1e6):
    """Itera x <- x + h f(x) (Euler, mismo esquema que `euler_step`, en floats por rapidez).
    Devuelve True si llega a distancia < tol de xeq en N pasos."""
    x, y = float(s0[0]), float(s0[1])
    ex, ey = float(xeq[0]), float(xeq[1])
    for k in range(N):
        fx = R1 * x * (1 - x - A12 * y)
        fy = R2 * y * (1 - y - A21 * x)
        x, y = x + h * fx, y + h * fy
        if not (abs(x) < blow and abs(y) < blow):
            return False
        if (x - ex) ** 2 + (y - ey) ** 2 < tol ** 2:
            return True
    return False


def bisect_hmax(s0, xeq, lo=0.1, hi=4.0, iters=40, **kw):
    assert euler_converges(lo, s0, xeq, **kw)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if euler_converges(mid, s0, xeq, **kw):
            lo = mid
        else:
            hi = mid
    return lo


res24d = {}
for nm, u in {"autovector rapido": v_fast, "autovector lento": v_slow,
              "direccion 45 grados": np.array([1, 1]) / np.sqrt(2)}.items():
    s0 = EQ["E*"] + 1e-3 * u
    res24d[nm] = bisect_hmax(s0, EQ["E*"])
    print(f"h_max numerico (cerca de E*, {nm}): {res24d[nm]:.5f}  | teorico {H_THEORY:.5f}")

# estados lejanos: barrido en h y biseccion hasta el primer fallo
FAR_STATES = {"P1": P["P1"], "P2": P["P2"], "(1.0,1.0)": (1.0, 1.0), "(1.5,1.5)": (1.5, 1.5)}
far = {}
for nm, s0 in FAR_STATES.items():
    scan = [(round(h, 2), euler_converges(h, s0, EQ["E*"], N=20000)) for h in np.arange(0.1, 3.01, 0.1)]
    first_fail = next((h for h, ok in scan if not ok), None)
    lo = max(h for h, ok in scan if ok and (first_fail is None or h < first_fail))
    hb = bisect_hmax(s0, EQ["E*"], lo=lo, hi=first_fail, N=20000) if first_fail else None
    far[nm] = dict(escaneo=scan, primer_fallo=first_fail, h_max_bisec=hb)
    print(f"Euler desde {nm}: converge hasta h ~ {hb}; escaneo:", "".join("1" if ok else "0" for _, ok in scan))
RES["task24cd"] = dict(hmax_por_autovalor=hmax, h_teorico=H_THEORY, h_numerico_cerca=res24d, lejos=far)

# %% [markdown]
# ## Task 2.5 - Intervencion sobre la plataforma con menos usuarios
#
# En el regimen de coexistencia la plataforma con menor base es la B ($y^*=4/7<x^*=5/7$).
# Intervenciones: (i) campana puntual $y_0\to y_0+\Lambda$; (ii) diferenciacion: reducir $\alpha_{21}$.

# %%
print("\n=== Task 2.5 ===  x*=%.4f y*=%.4f -> plataforma de interes: %s" %
      (*EQ["E*"], "B" if EQ["E*"][1] < EQ["E*"][0] else "A"))
_, XP2 = rk4(f, P["P2"], 0.01, 150.0)
print("P2 termina en", XP2[-1])

LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
rows25a = {}
for L in LAMBDAS:
    s0 = np.array([P["P2"][0], P["P2"][1] + L])
    t_, X = rk4(f, s0, 0.01, 200.0)
    d = np.linalg.norm(X - EQ["E*"], axis=1)
    k5 = int(np.argmax(d < 0.05))
    kb = int(np.argmax(d < 1e-3))
    rows25a[L] = dict(final=X[-1], dist_final=d[-1], t_5pct=t_[k5], t_1e3=t_[kb])
    print(f"Lambda={L:5.2f} final={np.round(X[-1], 6)} dist={d[-1]:.1e} t(d<0.05)={t_[k5]:.1f} t(d<1e-3)={t_[kb]:.1f}")
# verificacion con el mapa de cuencas: segmento x0=0.8, y0 = 0.3 + Lambda dentro de [0.01,1.5]
seg_labels = []
for L in np.arange(0, 1.2001, 0.02):
    i = int(np.argmin(np.abs(gx - (0.3 + L))))
    j = int(np.argmin(np.abs(gx - 0.8)))
    seg_labels.append(int(LAB[i, j]))
print("etiquetas del mapa de cuencas a lo largo de Lambda in [0,1.2]:", set(seg_labels), "(3 = E*)")
RES["task25a"] = dict(filas=rows25a, etiquetas_segmento=sorted(set(seg_labels)))

# (b) valor critico de alpha21: autovalor del equilibrio (1,0) es r2(1-alpha21) -> 0 en alpha21 = 1
print("\nalpha21 critico (analitico) = 1  [lambda_(1,0) = r2 (1 - alpha21) = 0]")
a21_grid = np.linspace(0.0, 0.99, 100)
slow_rates, y_star, fast_rates = [], [], []
for a in a21_grid:
    Jg = make_jac(A12, a, R1, R2)(equilibria(A12, a)["E*"])
    ev = np.sort(np.abs(np.linalg.eigvals(Jg).real))
    slow_rates.append(ev[0])
    fast_rates.append(ev[1])
    y_star.append(equilibria(A12, a)["E*"][1])
slow_rates, y_star = np.array(slow_rates), np.array(y_star)
rows25c = {}
for a in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    E_a = equilibria(A12, a)["E*"]
    ev = np.sort(np.abs(np.linalg.eigvals(make_jac(A12, a, R1, R2)(E_a)).real))
    rows25c[a] = dict(x=E_a[0], y=E_a[1], lam_lento=ev[0], tau_rec=1 / ev[0], margen=1 - a)
    print(f"alpha21={a:4.2f} E*=({E_a[0]:.3f},{E_a[1]:.3f}) |lambda_lento|={ev[0]:.4f} "
          f"t_rec={1 / ev[0]:7.2f} margen a la bifurcacion={1 - a:.2f}")
RES["task25c"] = rows25c
# sensibilidad del autovalor lento a alpha21 (derivada por diferencia central)
def slow_of(a):
    return np.sort(np.abs(np.linalg.eigvals(make_jac(A12, a, R1, R2)(equilibria(A12, a)["E*"])).real))[0]
dl = (slow_of(0.601) - slow_of(0.599)) / 0.002
print("d|lambda_lento|/d alpha21 en 0.6 =", dl)
RES["task25c_sens"] = dl

# simulaciones a cada lado del valor critico desde P2
sims = {}
for a in (0.9, 1.1):
    fa = make_field(A12, a, R1, R2)
    t_, X = rk4(fa, P["P2"], 0.02, 400.0)
    sims[a] = (t_, X)
    print(f"alpha21={a}: estado final en t=400 -> {X[-1]}")
RES["task25b"] = {a: dict(final=sims[a][1][-1], eq_esperado=equilibria(A12, a)["E*"],
                          eig_lento=np.sort(np.abs(np.linalg.eigvals(
                              make_jac(A12, a, R1, R2)(equilibria(A12, a)["E*"] if a < 1 else np.array([1.0, 0.0]))).real))[0])
                  for a in sims}

fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.9))
ax = axs[0]
for a, col in zip((0.9, 1.1), (C_BLUE, C_VERM)):
    t_, X = sims[a]
    ax.plot(t_, X[:, 0], color=col, lw=1.6, label=rf"$x$, $\alpha_{{21}}={a}$")
    ax.plot(t_, X[:, 1], color=col, lw=1.6, ls="--", label=rf"$y$, $\alpha_{{21}}={a}$")
ax.set_xlim(0, 300)
ax.set_xlabel("$t$")
ax.set_ylabel("usuarios")
ax.set_title("(b) Desde P2, a cada lado de $\\alpha_{21}=1$")
ax.legend(fontsize=7.5)
ax = axs[1]
ax.plot(a21_grid, slow_rates, color=C_BLUE, lw=2)
ax.axvline(A21, color=C_GRAY, ls=":", lw=1.2)
ax.scatter([A21], [slow_of(A21)], color=C_GREEN, zorder=5, s=50)
ax.annotate("valor actual", (A21, slow_of(A21)), xytext=(8, 14), textcoords="offset points", fontsize=8)
ax.set_xlabel(r"$\alpha_{21}$")
ax.set_ylabel(r"$|\lambda_{lento}|$ en $E^*$")
ax.set_title(r"Alejarse de $\alpha_{21}=1$ aleja $\lambda$ del eje imaginario")
ax = axs[2]
ax.plot(a21_grid, 1 / slow_rates, color=C_VERM, lw=2)
ax.axvline(A21, color=C_GRAY, ls=":", lw=1.2)
ax.set_ylim(0, 60)
ax.set_xlabel(r"$\alpha_{21}$")
ax.set_ylabel(r"tiempo de recuperacion $1/|\lambda_{lento}|$")
ax.set_title("Tiempo de recuperacion vs. $\\alpha_{21}$")
fig.tight_layout()
fig.savefig("figs/fig_intervencion.png")
plt.close(fig)

# %% [markdown]
# ## Guardado de resultados

# %%
with open("results.json", "w", encoding="utf-8") as fh:
    json.dump(jsonable(RES), fh, indent=1, ensure_ascii=False)
print("\nListo: figuras en figs/ y resultados en results.json")
