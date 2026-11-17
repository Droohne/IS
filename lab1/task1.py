import csv
import heapq
import math
import random
import sys
import time
import os
from operator import mul

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "datasets", "WA_Fn-UseC_-Telco-Customer-Churn.xls")
TARGET = "Churn"
ID_COL = "customerID"
SEED = 42
TEST_SIZE = 0.25 


# Разбор данных, отбрасывание некорректных значений. 

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames), rows


def is_missing(v):
    return v is None or v.strip() == ""


def to_float(v):
    """Строка -> float, пустая строка/пробел -> None."""
    if is_missing(v):
        return None
    return float(v)


def is_numeric_column(rows, col):
    """Столбец числовой, если все непустые значения разбираются как float."""
    seen = False
    for r in rows:
        v = r[col]
        if is_missing(v):
            continue
        seen = True
        try:
            float(v)
        except ValueError:
            return False
    return seen


# Сводка по данным
def explore(columns, rows, numeric_cols, categorical_cols):
    print("=" * 70)
    print("1. РАЗБОР ДАННЫХ")
    print("=" * 70)
    print(f"Строк: {len(rows)}, столбцов: {len(columns)}")
    print(f"\nЧисловые признаки ({len(numeric_cols)}): {', '.join(numeric_cols)}")
    print(f"Категориальные признаки ({len(categorical_cols)}):")
    print("  " + ", ".join(categorical_cols))

    print("\nПропуски (пустые или состоящие из пробелов значения):")
    any_missing = False
    for c in columns:
        m = sum(1 for r in rows if is_missing(r[c]))
        if m:
            any_missing = True
            print(f"  {c}: {m}")
    if not any_missing:
        print("  нет")

    counts = {}
    for r in rows:
        counts[r[TARGET]] = counts.get(r[TARGET], 0) + 1
    print(f"\nЦелевая переменная {TARGET}:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]} ({counts[k] / len(rows):.1%})")
    print()




def train_test_split(n, test_size=TEST_SIZE, seed=SEED):
    """Перемешивает индексы с фиксированным seed и делит их на train/test."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = math.ceil(n * test_size)
    return idx[n_test:], idx[:n_test]


class Preprocessor:
    """StandardScaler для числовых + OneHotEncoder для категориальных признаков.
    Всё обучается только на train, на test применяется transform.
    Пропуски в числовых признаках заменяются медианой train."""

    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def fit(self, rows):
        self.num_stats = {}  # col -> (median, mean, std)
        for c in self.numeric_cols:
            known = sorted(v for v in (to_float(r[c]) for r in rows) if v is not None)
            mid = len(known) // 2
            median = known[mid] if len(known) % 2 else (known[mid - 1] + known[mid]) / 2
            filled = [to_float(r[c]) if not is_missing(r[c]) else median for r in rows]
            mean = sum(filled) / len(filled)
            var = sum((v - mean) ** 2 for v in filled) / len(filled)
            std = math.sqrt(var) or 1.0
            self.num_stats[c] = (median, mean, std)

        self.categories = {c: sorted({r[c] for r in rows}) for c in self.categorical_cols}
        self.feature_names = list(self.numeric_cols)
        for c in self.categorical_cols:
            self.feature_names += [f"{c}={cat}" for cat in self.categories[c]]
        return self

    def transform(self, rows):
        out = []
        for r in rows:
            vec = []
            for c in self.numeric_cols:
                median, mean, std = self.num_stats[c]
                v = to_float(r[c])
                if v is None:
                    v = median
                vec.append((v - mean) / std)
            for c in self.categorical_cols:
                val = r[c]
                # неизвестная категория -> все нули (как handle_unknown="ignore")
                vec.extend(1.0 if val == cat else 0.0 for cat in self.categories[c])
            out.append(vec)
        return out


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticRegression:
    """Логистическая регрессия: полный градиентный спуск, L2-регуляризация.
    Минимизируется средний log-loss + (1 / (2*C*n)) * ||w||^2
    (та же цель, что у sklearn при C=1)."""

    def __init__(self, lr=0.5, n_iter=500, C=1.0, verbose=True): # После n_iter=500 улучшения незначительны loss сходится
        self.lr, self.n_iter, self.C, self.verbose = lr, n_iter, C, verbose

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        cols = list(zip(*X))  # столбцы признаков, чтобы считать градиент через map/sum
        lam = 1.0 / (self.C * n)
        self.w = [0.0] * d
        self.b = 0.0
        for it in range(1, self.n_iter + 1):
            errs = [sigmoid(self.b + sum(map(mul, self.w, xi))) - yi for xi, yi in zip(X, y)]
            grad = [sum(map(mul, errs, cj)) / n + lam * wj for cj, wj in zip(cols, self.w)]
            gb = sum(errs) / n
            self.w = [wj - self.lr * g for wj, g in zip(self.w, grad)]
            self.b -= self.lr * gb
            if self.verbose and it % 100 == 0:
                loss = 0.0
                for xi, yi in zip(X, y):
                    p = min(max(sigmoid(self.b + sum(map(mul, self.w, xi))), 1e-12), 1 - 1e-12)
                    loss -= yi * math.log(p) + (1 - yi) * math.log(1 - p)
                print(f"    итерация {it:4d}: log-loss = {loss / n:.4f}")
        return self

    def predict_proba(self, X):
        return [sigmoid(self.b + sum(map(mul, self.w, x))) for x in X]

    def predict(self, X):
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "label")

    def __init__(self, label=None, feature=None, threshold=None, left=None, right=None):
        self.label, self.feature, self.threshold = label, feature, threshold
        self.left, self.right = left, right


class DecisionTreeClassifier:
    """Дерево решений CART, критерий Джини, бинарные пороги вида x[f] <= t.
    По умолчанию (как в sklearn) глубина не ограничена."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
        n, d = len(X), len(X[0])
        self.y = y
        self.cols = list(zip(*X))
        # Сортировка индексов по каждому признаку делается один раз,
        # дальше порядок только "фильтруется" при разбиении узлов.
        orders = [sorted(range(n), key=self.cols[f].__getitem__) for f in range(d)]
        self.root = self._build(list(range(n)), orders, 0)
        del self.cols, self.y
        return self

    @staticmethod
    def _gini(pos, n):
        p = pos / n
        return 1.0 - p * p - (1.0 - p) * (1.0 - p)

    def _best_split(self, orders, n, pos):
        y, msl = self.y, self.min_samples_leaf
        best_score = self._gini(pos, n) - 1e-12  # нужен строго положительный выигрыш
        best = None
        for f, order in enumerate(orders):
            col = self.cols[f]
            left_n = left_pos = 0
            for k in range(n - 1):
                i = order[k]
                left_n += 1
                left_pos += y[i]
                v, nv = col[i], col[order[k + 1]]
                if v == nv:
                    continue
                right_n = n - left_n
                if left_n < msl or right_n < msl:
                    continue
                score = (left_n * self._gini(left_pos, left_n)
                         + right_n * self._gini(pos - left_pos, right_n)) / n
                if score < best_score:
                    best_score, best = score, (f, (v + nv) / 2.0)
        return best

    def _build(self, indices, orders, depth):
        n = len(indices)
        pos = sum(self.y[i] for i in indices)
        label = 1 if pos * 2 > n else 0
        if (pos == 0 or pos == n or n < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return _Node(label=label)
        split = self._best_split(orders, n, pos)
        if split is None:
            return _Node(label=label)
        f, thr = split
        col = self.cols[f]
        left_idx = [i for i in indices if col[i] <= thr]
        right_idx = [i for i in indices if col[i] > thr]
        left_orders = [[i for i in o if col[i] <= thr] for o in orders]
        right_orders = [[i for i in o if col[i] > thr] for o in orders]
        return _Node(feature=f, threshold=thr,
                     left=self._build(left_idx, left_orders, depth + 1),
                     right=self._build(right_idx, right_orders, depth + 1))

    def predict(self, X):
        out = []
        for x in X:
            node = self.root
            while node.label is None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out.append(node.label)
        return out

    def depth(self):
        def d(node):
            return 0 if node.label is not None else 1 + max(d(node.left), d(node.right))
        return d(self.root)


class KNeighborsClassifier:
    """k ближайших соседей: евклидово расстояние, равные веса, голосование."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        self.sq = [sum(map(mul, t, t)) for t in X]
        return self

    def predict(self, X):
        n_train, k = len(self.X), self.k
        preds = []
        for x in X:
            # |x - t|^2 = |x|^2 - 2 x.t + |t|^2; |x|^2 одинаково для всех t -> опускаем
            dist = [ts - 2.0 * sum(map(mul, x, t)) for t, ts in zip(self.X, self.sq)]
            nearest = heapq.nsmallest(k, range(n_train), key=dist.__getitem__)
            votes = sum(self.y[i] for i in nearest)
            preds.append(1 if votes * 2 > k else 0)
        return preds


# Сравнение и метрики
def accuracy_score(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix(y_true, y_pred):
    """[[TN, FP], [FN, TP]] -- тот же порядок, что у sklearn (класс 1 = ушёл)."""
    tn = fp = fn = tp = 0
    for a, b in zip(y_true, y_pred):
        if a == 0 and b == 0:
            tn += 1
        elif a == 0 and b == 1:
            fp += 1
        elif a == 1 and b == 0:
            fn += 1
        else:
            tp += 1
    return [[tn, fp], [fn, tp]]



# Визуализация от нейронки

def print_table(results):
    print(f"{'Модель':<24}{'Accuracy':>10}")
    print("-" * 34)
    for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"{name:<24}{acc:>10.4f}")


def print_bars(results, width=40):
    print()
    for name, acc in results.items():
        print(f"{name:<24}|{'#' * round(acc * width):<{width}}| {acc:.2%}")


def save_svg_bar_chart(results, path, title="Accuracy моделей на тесте"):
    w, h, left, bottom, top = 520, 340, 60, 50, 50
    ph = h - bottom - top
    bar_w = (w - left - 30) / len(results)
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'font-family="sans-serif" font-size="12">',
             f'<rect width="{w}" height="{h}" fill="white"/>',
             f'<text x="{w / 2}" y="24" text-anchor="middle" font-size="15" '
             f'font-weight="bold">{title}</text>']
    for t in range(0, 11, 2):  # оси: сетка 0.0 ... 1.0
        v = t / 10
        y = top + ph * (1 - v)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w - 20}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    for i, (name, acc) in enumerate(results.items()):
        bh = ph * acc
        x = left + i * bar_w + bar_w * 0.15
        bw = bar_w * 0.7
        parts.append(f'<rect x="{x:.1f}" y="{top + ph - bh:.1f}" width="{bw:.1f}" '
                     f'height="{bh:.1f}" fill="{colors[i % len(colors)]}"/>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{top + ph - bh - 6:.1f}" '
                     f'text-anchor="middle" font-weight="bold">{acc:.3f}</text>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{h - bottom + 20}" '
                     f'text-anchor="middle">{name}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    columns, rows = load_csv(CSV_PATH)

    features = [c for c in columns if c not in (TARGET, ID_COL)]
    numeric_cols = [c for c in features if is_numeric_column(rows, c)]
    categorical_cols = [c for c in features if c not in numeric_cols]

    explore(columns, rows, numeric_cols, categorical_cols)

    # --- подготовка ---
    print("=" * 70)
    print("2. ПОДГОТОВКА ДАННЫХ")
    print("=" * 70)
    y_all = [1 if r[TARGET].strip() == "Yes" else 0 for r in rows]
    train_idx, test_idx = train_test_split(len(rows))
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    y_train = [y_all[i] for i in train_idx]
    y_test = [y_all[i] for i in test_idx]

    prep = Preprocessor(numeric_cols, categorical_cols).fit(train_rows)
    X_train = prep.transform(train_rows)
    X_test = prep.transform(test_rows)
    print(f"train: {len(X_train)} ({1 - TEST_SIZE:.0%}), test: {len(X_test)} ({TEST_SIZE:.0%})")
    print(f"Признаков после кодирования: {len(prep.feature_names)}\n")

    # --- обучение ---
    print("=" * 70)
    print("3. ОБУЧЕНИЕ И ПРОВЕРКА")
    print("=" * 70)
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "KNN (k=5)": KNeighborsClassifier(k=5),
    }
    results, predictions = {}, {}
    for name, model in models.items():
        print(f"{name}: обучение...")
        t0 = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds
        results[name] = accuracy_score(y_test, preds)
        extra = f", глубина дерева = {model.depth()}" if isinstance(model, DecisionTreeClassifier) else ""
        print(f"  accuracy на тесте = {results[name]:.4f}  ({time.time() - t0:.1f} с{extra})\n")

    # --- анализ ---
    print("=" * 70)
    print("4. АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 70)
    print_table(results)
    print_bars(results)
    save_svg_bar_chart(results, "accuracy_comparison.svg")
    print("\nГрафик сохранён: accuracy_comparison.svg")

    baseline = 1 - sum(y_test) / len(y_test)
    print(f"Для сравнения: модель «все остались» дала бы accuracy = {baseline:.4f}")

    best = max(results, key=results.get)
    print(f"\nЛучшая модель: {best} (accuracy = {results[best]:.4f})")

    (tn, fp), (fn, tp) = confusion_matrix(y_test, predictions[best])
    print(f"\nМатрица ошибок для {best}:")
    print(f"{'':>22}{'предсказано: остался':>22}{'предсказано: ушёл':>20}")
    print(f"{'на самом деле остался':>22}{tn:>22}{fp:>20}")
    print(f"{'на самом деле ушёл':>22}{fn:>22}{tp:>20}")
    print(f"\nОшибочно предсказаны как ушедшие (False Positive): {fp}")
    print(f"Ошибочно предсказаны как лояльные (False Negative): {fn}")
    if tp + fn:
        print(f"Из реально ушедших клиентов модель нашла {tp / (tp + fn):.1%} (recall)")


if __name__ == "__main__":
    main()