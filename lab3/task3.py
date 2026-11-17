import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_recall_fscore_support,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Пути и настройки
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "datasets")
FILE_STEM = "WineQT"
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")

RANDOM_STATE = 42
TARGET_COL = "quality"
MIN_CLASS_SAMPLES = 100   # классы с меньшим числом объектов отбрасываем

os.makedirs(OUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def class_label(k):
    return f"Качество {int(k)}"


def find_data_path():
    for ext in (".csv", ".xlsx", ".xls"):
        path = os.path.join(DATA_DIR, FILE_STEM + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Не найден {FILE_STEM}.csv/.xls в {DATA_DIR}")


def load_data(path):
    try:
        df = pd.read_csv(path)
    except (UnicodeDecodeError, pd.errors.ParserError):
        df = pd.read_excel(path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df


# --------------------------------------------------------------------------
# Оценка и графики
# --------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test, cm_filename=None):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, pred, zero_division=0, digits=3))

    if cm_filename:
        labels = sorted(set(y_test) | set(pred))
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, labels=labels,
            display_labels=[class_label(k) for k in labels],
            cmap="Blues", colorbar=False, xticks_rotation=45, ax=ax)
        ax.set_title(f"Матрица ошибок — {name}")
        fig.savefig(os.path.join(OUT_DIR, cm_filename), dpi=130, bbox_inches="tight")
        plt.close(fig)
    return pred, acc


def worst_class(y_test, pred, name):
    labels = sorted(set(y_test) | set(pred))
    p, r, f, s = precision_recall_fscore_support(
        y_test, pred, labels=labels, zero_division=0)
    metrics = pd.DataFrame({
        "Класс": [class_label(k) for k in labels],
        "Precision": p, "Recall": r, "F1": f,
        "Объектов в тесте": s,
    })
    worst = metrics.sort_values("Recall").iloc[0]
    print(f"\n[{name}] Худший класс по Recall:")
    print(f"  {worst['Класс']}: recall={worst['Recall']:.3f}, "
          f"F1={worst['F1']:.3f}, объектов в тесте={int(worst['Объектов в тесте'])}")
    return worst


# --------------------------------------------------------------------------
# Основной сценарий
# --------------------------------------------------------------------------
def main():
    path = find_data_path()
    df = load_data(path)

    # ======================================================================
    section("1. ЗАГРУЗКА ДАННЫХ И EDA (полный датасет)")
    # ======================================================================
    print(f"Файл: {path}")
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Признаки: {list(df.columns.drop(TARGET_COL))}")
    print(f"Пропусков: {df.isnull().sum().sum()}")

    counts_all = df[TARGET_COL].value_counts().sort_index()
    print(f"\nРаспределение классов (до фильтрации):")
    print(counts_all.to_string())
    print(f"\nДоли классов (%):")
    print((counts_all / len(df) * 100).round(2).to_string())
    print(f"Коэффициент дисбаланса (max/min): {counts_all.max() / counts_all.min():.2f}")

    # ======================================================================
    section(f"2. ФИЛЬТРАЦИЯ КЛАССОВ: оставляем только классы с >= {MIN_CLASS_SAMPLES} объектами")
    # ======================================================================
    keep_mask = counts_all[counts_all >= MIN_CLASS_SAMPLES].index
    drop_mask = counts_all[counts_all < MIN_CLASS_SAMPLES].index

    print(f"Оставляем классы: {sorted(keep_mask.tolist())}")
    print(f"Отбрасываем классы (мало объектов): "
          f"{sorted(drop_mask.tolist())} "
          f"(их количество: {[int(counts_all[k]) for k in sorted(drop_mask.tolist())]})")
    print(f"Классы НЕ сливаем — только удаляем редкие.")

    df = df[df[TARGET_COL].isin(keep_mask)].reset_index(drop=True)

    counts = df[TARGET_COL].value_counts().sort_index()
    print(f"\nРазмер после фильтрации: {df.shape[0]} строк")
    print(f"Распределение классов (после фильтрации):")
    print(counts.to_string())
    print(f"\nДоли классов (%):")
    print((counts / len(df) * 100).round(2).to_string())
    print(f"Коэффициент дисбаланса (max/min): {counts.max() / counts.min():.2f}")

    # График: до / после фильтрации
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(x=[class_label(k) for k in counts_all.index], y=counts_all.values,
                hue=[class_label(k) for k in counts_all.index],
                palette="Reds_r", legend=False, ax=axes[0])
    axes[0].set_title("ДО фильтрации (все классы)")
    axes[0].set_ylabel("Количество образцов")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(x=[class_label(k) for k in counts.index], y=counts.values,
                hue=[class_label(k) for k in counts.index],
                palette="Greens_r", legend=False, ax=axes[1])
    axes[1].set_title(f"ПОСЛЕ фильтрации (классы >= {MIN_CLASS_SAMPLES})")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "1_class_distribution.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ======================================================================
    section("3. РАЗДЕЛЕНИЕ TRAIN/TEST (stratify=y)")
    # ======================================================================
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    print(f"Классы в train: {dict(y_train.value_counts().sort_index())}")
    print(f"Классы в test:  {dict(y_test.value_counts().sort_index())}")

    # ======================================================================
    section("4. ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ")
    # ======================================================================
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    print(f"Обучен RandomForestClassifier (n_estimators={rf.n_estimators}, "
          f"class_weight=None)")
    pred_rf, acc_rf = evaluate(
        "RandomForest (базовый)", rf, X_test, y_test, "2_confusion_rf.png")

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    lr.fit(X_train, y_train)
    print("\nОбучен LogisticRegression (StandardScaler + L2, max_iter=2000)")
    pred_lr, acc_lr = evaluate(
        "LogisticRegression", lr, X_test, y_test, "3_confusion_lr.png")

    # ======================================================================
    section("5. АНАЛИЗ: КАКОЙ КЛАСС ПРЕДСКАЗЫВАЕТСЯ ХУЖЕ ВСЕГО")
    # ======================================================================
    worst_rf = worst_class(y_test, pred_rf, "RandomForest (базовый)")
    worst_lr = worst_class(y_test, pred_lr, "LogisticRegression")
    print(
        "\nПричина: после удаления сверхредких классов (3, 4, 8) оставшийся\n"
        "наименьший класс 7 (~143 объекта, ~13%) предсказывается хуже всего:\n"
        "модель, оптимизирующая accuracy, «сдвигает» предсказания в сторону\n"
        "крупных классов 5 и 6. Recall класса 7 заметно ниже recall 5 и 6\n"
        "(но не ноль, как было бы при единичных объектах).\n"
        "Меры: class_weight='balanced', SMOTE, оптимизация f1_macro."
    )

    # ======================================================================
    section("6. СРАВНЕНИЕ MACRO-F1")
    # ======================================================================
    f1_rf = f1_score(y_test, pred_rf, average="macro")
    f1_lr = f1_score(y_test, pred_lr, average="macro")
    print(f"Macro-F1 RandomForest (базовый): {f1_rf:.4f}")
    print(f"Macro-F1 LogisticRegression:     {f1_lr:.4f}")
    if f1_rf > f1_lr:
        print("→ RandomForest лучше балансирует между классами (по macro-F1).")
    else:
        print("→ LogisticRegression лучше балансирует между классами (по macro-F1).")

    # ======================================================================
    section("7. RANDOMFOREST С class_weight='balanced'")
    # ======================================================================
    rf_bal = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE)
    rf_bal.fit(X_train, y_train)
    print("Обучен RandomForestClassifier (class_weight='balanced')")
    pred_bal, acc_bal = evaluate(
        "RandomForest (balanced)", rf_bal, X_test, y_test,
        "4_confusion_rf_balanced.png")

    f1_bal = f1_score(y_test, pred_bal, average="macro")
    worst_bal = worst_class(y_test, pred_bal, "RandomForest (balanced)")

    worst_class_id = int(str(worst_rf["Класс"]).split()[-1])
    _, r_rf, _, _ = precision_recall_fscore_support(
        y_test, pred_rf, labels=[worst_class_id], zero_division=0)
    _, r_bal, _, _ = precision_recall_fscore_support(
        y_test, pred_bal, labels=[worst_class_id], zero_division=0)

    print("\nВлияние class_weight='balanced':")
    print(f"  Accuracy:  {acc_rf:.4f}  ->  {acc_bal:.4f}")
    print(f"  Macro-F1:  {f1_rf:.4f}  ->  {f1_bal:.4f}")
    print(f"  Recall класса {worst_class_id}: "
          f"{r_rf[0]:.4f}  ->  {r_bal[0]:.4f}")

    # ======================================================================
    section("8. ИТОГОВОЕ СРАВНЕНИЕ (тест)")
    # ======================================================================
    def row(name, pred, acc):
        return {
            "Модель": name,
            "Accuracy": acc,
            "Macro-F1": f1_score(y_test, pred, average="macro"),
            "Macro-Precision": precision_score(
                y_test, pred, average="macro", zero_division=0),
            "Macro-Recall": recall_score(
                y_test, pred, average="macro", zero_division=0),
        }

    table = pd.DataFrame([
        row("RandomForest (базовый)", pred_rf, acc_rf),
        row("LogisticRegression", pred_lr, acc_lr),
        row("RandomForest (balanced)", pred_bal, acc_bal),
    ]).set_index("Модель").round(3)
    print(table.to_string())
    table.to_csv(os.path.join(OUT_DIR, "comparison.csv"), encoding="utf-8-sig")

    # ======================================================================
    section("9. ИТОГОВЫЙ ВЫВОД")
    # ======================================================================
    print(
        f"После удаления классов с <{MIN_CLASS_SAMPLES} объектами (3, 4, 8)\n"
        f"остались классы 5, 6, 7 (не слиты). Дисбаланс снизился до ~3.4:1.\n"
        f"\n"
        "Для прототипа выбираем RandomForestClassifier с class_weight='balanced':\n"
        f"  • Базовая модель RF: accuracy={acc_rf:.3f}, macro-F1={f1_rf:.3f}.\n"
        f"  • balanced: accuracy={acc_bal:.3f}, macro-F1={f1_bal:.3f} — "
        f"прирост macro-F1 в пользу редкого класса 7.\n"
        f"  • Редкие классы 3, 4, 8 удалены как статистически непредставимые\n"
        f"    (1–33 объекта); обучать на них модель бессмысленно.\n"
        f"  • Итоговый прототип распознаёт три класса качества: 5, 6, 7.\n"
        f"\n"
        "Дальнейшее улучшение — SMOTE и GridSearchCV (практика №4)."
    )
    print(f"\nГрафики и comparison.csv сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    main()