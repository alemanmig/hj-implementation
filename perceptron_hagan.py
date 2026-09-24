"""
Perceptrón simple — Clasificador Manzana / Naranja
Referencia: Hagan, Demuth, Beale — Neural Network Design, Caps. 3 y 4.

Sensores (entradas bipolares ±1):
    p[0] = forma   : +1 redonda,  -1 elíptica
    p[1] = textura : +1 lisa,     -1 rugosa
    p[2] = peso    : +1 > 1 lb,   -1 < 1 lb

Prototipos:
    naranja  p1 = [ 1, -1, -1]
    manzana  p2 = [ 1,  1, -1]

Estructura del script:
    1. Funciones de transferencia (hardlim, hardlims)
    2. Perceptrón diseñado a mano (Cap. 3): W = [0 1 0], b = 0, salida hardlims
    3. Regla de aprendizaje del perceptrón (Cap. 4): W += e p^T, b += e
    4. Evaluación exhaustiva de las 8 combinaciones posibles de sensores
    5. Inferencia sólo con enteros (vista previa de la versión embebida)
"""

import itertools
import numpy as np

# ---------------------------------------------------------------------------
# 1. Funciones de transferencia
# ---------------------------------------------------------------------------
def hardlim(n):
    """a = 1 si n >= 0, a = 0 si n < 0   (Hagan, Tabla 2.1)"""
    return np.where(n >= 0, 1, 0)


def hardlims(n):
    """a = +1 si n >= 0, a = -1 si n < 0 (versión simétrica)"""
    return np.where(n >= 0, 1, -1)


# ---------------------------------------------------------------------------
# Datos del problema
# ---------------------------------------------------------------------------
P_ORANGE = np.array([1, -1, -1])
P_APPLE = np.array([1, 1, -1])

# Cap. 4 usa hardlim, así que los targets son 0 (naranja) y 1 (manzana)
TRAINING_SET = [(P_ORANGE, 0), (P_APPLE, 1)]

NAMES_HARDLIM = {0: "naranja", 1: "manzana"}
NAMES_HARDLIMS = {-1: "naranja", 1: "manzana"}


# ---------------------------------------------------------------------------
# 2. Perceptrón diseñado gráficamente (Cap. 3, Ec. 3.6–3.14)
# ---------------------------------------------------------------------------
def perceptron_chapter3():
    W = np.array([[0, 1, 0]])   # ortogonal al plano p2 = 0
    b = np.array([0])
    print("=" * 64)
    print("CAPÍTULO 3 — Diseño gráfico: W = [0 1 0], b = 0, hardlims")
    print("=" * 64)
    tests = {
        "Naranja prototipo      ": P_ORANGE,
        "Manzana prototipo      ": P_APPLE,
        "Naranja elíptica (3.13)": np.array([-1, -1, -1]),
    }
    for label, p in tests.items():
        n = (W @ p + b).item()
        a = hardlims(n).item()
        print(f"  {label} p={p}  n={n:+d}  a={a:+d} -> {NAMES_HARDLIMS[a]}")
    return W, b


# ---------------------------------------------------------------------------
# 3. Regla de aprendizaje del perceptrón (Cap. 4, Ec. 4.38–4.39)
# ---------------------------------------------------------------------------
def train_perceptron(training_set, W0, b0, max_epochs=100, verbose=True):
    """
    Entrenamiento en línea (una muestra a la vez), como en el libro:
        e     = t - a
        W_new = W_old + e p^T
        b_new = b_old + e
    Se detiene cuando una época completa no produce errores.
    """
    W = np.array(W0, dtype=float).reshape(1, -1)
    b = np.array(b0, dtype=float).reshape(1)
    step = 0

    if verbose:
        print("\n" + "=" * 64)
        print("CAPÍTULO 4 — Regla de aprendizaje del perceptrón (hardlim)")
        print("=" * 64)
        print(f"  Inicial: W = {W.ravel()}, b = {b.item()}")

    for epoch in range(1, max_epochs + 1):
        errors = 0
        for p, t in training_set:
            step += 1
            n = (W @ p + b).item()
            a = hardlim(n).item()
            e = t - a
            if e != 0:
                errors += 1
                W = W + e * p.reshape(1, -1)
                b = b + e
            if verbose:
                flag = "ACTUALIZA" if e != 0 else "ok"
                print(f"  paso {step:2d} | p={p} t={t} n={n:+5.2f} a={a} "
                      f"e={e:+d} -> W={W.ravel()} b={b.item():+.2f}  [{flag}]")
        if errors == 0:
            if verbose:
                print(f"  Convergió en la época {epoch} ({step} presentaciones).")
            return W, b, epoch
    raise RuntimeError("No convergió (¿problema no linealmente separable?)")


# ---------------------------------------------------------------------------
# 4. Evaluación exhaustiva (el espacio de entrada sólo tiene 2^3 = 8 puntos)
# ---------------------------------------------------------------------------
def nearest_prototype(p):
    """Referencia: clase del prototipo más cercano (distancia de Hamming)."""
    d_orange = np.sum(p != P_ORANGE)
    d_apple = np.sum(p != P_APPLE)
    if d_orange == d_apple:
        return None, d_orange, d_apple
    return (0 if d_orange < d_apple else 1), d_orange, d_apple


def evaluate_all(W, b, title):
    print("\n" + "=" * 64)
    print(f"Evaluación de las 8 entradas posibles — {title}")
    print("=" * 64)
    print("  forma text  peso |    n   a  clase     | d_nar d_man  ref")
    for combo in itertools.product([-1, 1], repeat=3):
        p = np.array(combo)
        n = (W @ p + b).item()
        a = hardlim(n).item()
        ref, dn, dm = nearest_prototype(p)
        ref_txt = NAMES_HARDLIM[ref] if ref is not None else "empate"
        match = "" if ref is None else ("✓" if ref == a else "✗")
        print(f"  {p[0]:+3d}  {p[1]:+3d}  {p[2]:+3d} | {n:+5.1f}  {a}  "
              f"{NAMES_HARDLIM[a]:<8}  |   {dn}     {dm}   {ref_txt:<8}{match}")


# ---------------------------------------------------------------------------
# 5. Vista previa embebida: inferencia sólo con enteros
# ---------------------------------------------------------------------------
def to_fixed_point(W, b, scale):
    """Cuantiza W y b a enteros multiplicando por 'scale' (potencia de 2)."""
    Wq = np.round(W * scale).astype(np.int8)
    bq = np.round(b * scale).astype(np.int8)
    return Wq, bq


def infer_int(Wq, bq, p):
    """
    Lo que haría un microcontrolador o un bloque en HDL:
    sumas/restas de enteros (las entradas son ±1, no hay multiplicaciones reales)
    y la función de activación es sólo el bit de signo del acumulador.
    """
    acc = int(bq[0])
    for w, x in zip(Wq.ravel(), p):
        acc += int(w) if x > 0 else -int(w)
    return 1 if acc >= 0 else 0, acc


def check_fixed_point(W, b, scale=2):
    Wq, bq = to_fixed_point(W, b, scale)
    print("\n" + "=" * 64)
    print(f"Vista previa embebida — pesos enteros (escala x{scale})")
    print("=" * 64)
    print(f"  W flotante = {W.ravel()}, b = {b.item()}")
    print(f"  W entero   = {Wq.ravel()}, b = {bq.item()}  (int8)")
    all_ok = True
    for combo in itertools.product([-1, 1], repeat=3):
        p = np.array(combo)
        a_float = hardlim((W @ p + b).item()).item()
        a_int, acc = infer_int(Wq, bq, p)
        ok = a_float == a_int
        all_ok &= ok
        print(f"  p={p}  acc={acc:+3d}  a_int={a_int}  a_float={a_float}  "
              f"{'✓' if ok else '✗ DIFERENTE'}")
    print(f"  Resultado: {'idéntico en los 8 casos' if all_ok else 'HAY DIFERENCIAS'}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    perceptron_chapter3()

    # Condiciones iniciales del libro (Ec. 4.40) para reproducir 4.41–4.53
    W, b, _ = train_perceptron(TRAINING_SET, W0=[0.5, -1, -0.5], b0=0.5)
    evaluate_all(W, b, "perceptrón entrenado")
    check_fixed_point(W, b, scale=2)

    # Robustez: entrenar desde condiciones iniciales aleatorias
    print("\n" + "=" * 64)
    print("Convergencia desde 1000 inicializaciones aleatorias U(-1, 1)")
    print("=" * 64)
    rng = np.random.default_rng(0)
    epochs = []
    for _ in range(1000):
        _, _, ep = train_perceptron(TRAINING_SET, rng.uniform(-1, 1, 3),
                                    rng.uniform(-1, 1), verbose=False)
        epochs.append(ep)
    epochs = np.array(epochs)
    print(f"  Todas convergieron. Épocas: min={epochs.min()}, "
          f"media={epochs.mean():.2f}, max={epochs.max()}")

