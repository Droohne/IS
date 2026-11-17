import os
import sys
sys.path.insert(0, "/home/andrew/projects/univer/IS/lab1")   # добавили lab1 в список поиска
import task1                   
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Пути и настройки
# --------------------------------------------------------------------------
matplotlib.use("Agg")  # Результат в файлы
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "datasets")
FILE_STEM = "WA_Fn-UseC_-Telco-Customer-Churn"
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")

RANDOM_STATE = 42
LABELS = ["Остался", "Ушёл"]
USE_SMOTE = False        # False -> class_weight='balanced', True -> SMOTE из imblearn
SCORING = "f1"           # метрика для GridSearchCV (F1 по классу «ушёл»)
TARGET_RECALL = 0.70     # цель оптимизации

os.makedirs(OUT_DIR, exist_ok=True)

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def find_data_path():
    for ext in (".csv", ".xls"):
        path = os.path.join(DATA_DIR, FILE_STEM + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Не найден {FILE_STEM}.csv/.xls в {DATA_DIR}")


# Старая модель: всё берётся из task1.py

def run_old_model(path):
    columns, rows = task1.load_csv(path)
    features = [c for c in columns if c not in (task1.TARGET, task1.ID_COL)]
    numeric_cols = [c for c in features if task1.is_numeric_column(rows, c)]
    categorical_cols = [c for c in features if c not in numeric_cols]

    y_all = np.array([1 if r[task1.TARGET].strip() == "Yes" else 0 for r in rows])
    train_idx, test_idx = task1.train_test_split(len(rows))  # тот же split, что в лабе 1

    prep = task1.Preprocessor(numeric_cols, categorical_cols).fit([rows[i] for i in train_idx])
    X_train = prep.transform([rows[i] for i in train_idx])
    X_test = prep.transform([rows[i] for i in test_idx])

    model = task1.LogisticRegression(verbose=False).fit(X_train, y_all[train_idx].tolist())
    proba = np.array(model.predict_proba(X_test))
    pred = np.array(model.predict(X_test))
    return train_idx, test_idx, y_all, pred, proba


# Новая модель: sklearn

def load_dataframe(path):
    try:
        df = pd.read_csv(path)       # файл с расширением .xls может быть обычным CSV
    except (UnicodeDecodeError, pd.errors.ParserError):
        df = pd.read_excel(path)     # настоящий Excel (нужен xlrd/openpyxl)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # пробелы -> NaN
    return df.drop(columns=["customerID"])


def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def build_pipeline(X):
    pre = build_preprocessor(X)
    if USE_SMOTE:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        # SMOTE применяется только к train-части каждого фолда
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=RANDOM_STATE)),
                            ("clf", LogisticRegression(max_iter=2000))])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def metrics_row(name, y, pred, proba):
    return {
        "Модель": name,
        "Accuracy": accuracy_score(y, pred),
        "Precision (ушёл)": precision_score(y, pred),
        "Recall (ушёл)": recall_score(y, pred),
        "F1 (ушёл)": f1_score(y, pred),
        "Recall (остался)": recall_score(y, pred, pos_label=0),
        "AUC-ROC": roc_auc_score(y, proba),
    }


def main():
    path = find_data_path()
    section("1. ДИАГНОСТИКА МОДЕЛИ-ПОБЕДИТЕЛЯ (самописная LogisticRegression, lab1/task1.py)")
    train_idx, test_idx, y_all, old_pred, old_proba_1 = run_old_model(path)
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    print(f"train: {len(train_idx)}, test: {len(test_idx)} "
          f"(тот же split, что в практике №1)")
    print(f"Accuracy на тесте = {accuracy_score(y_test, old_pred):.4f} "
          f"(должна совпасть с результатом практики №1)\n")

    share = pd.Series(y_train).value_counts(normalize=True)
    print(f"Баланс классов в train: остался {share[0]:.1%}, ушёл {share[1]:.1%} "
          f"(соотношение ~{share[0] / share[1]:.1f}:1)\n")
    print(classification_report(y_test, old_pred, target_names=LABELS, digits=3))

    ConfusionMatrixDisplay.from_predictions(y_test, old_pred, display_labels=LABELS,
                                            cmap="Blues")
    plt.title("Модель из практики №1: матрица ошибок")
    plt.savefig(os.path.join(OUT_DIR, "1_confusion_old.png"), dpi=130, bbox_inches="tight")
    plt.close()

    old_recall = recall_score(y_test, old_pred)
    print("Анализ:")
    print(f"  * Дисбаланс есть: уходит лишь ~{share[1]:.0%} клиентов, поэтому accuracy "
          f"обманчива (модель «все остались» даёт ~{share[0]:.0%}).")
    print("  * Критичнее False Negative: уходящий клиент, которого модель не заметила, "
          "потерян вместе с выручкой, а ложная тревога (FP) стоит лишь цены "
          "удерживающего предложения.")
    print(f"  * Recall по классу «ушёл» у модели из практики №1 = {old_recall:.3f}.")
    print(f"  ЦЕЛЬ ОПТИМИЗАЦИИ: поднять Recall класса «Клиент ушёл» до {TARGET_RECALL}, "
          "не допустив сильного падения F1 и Precision.")

    # 2-3. Pipeline + GridSearchCV

    section("2-3. PIPELINE И ПОДБОР ГИПЕРПАРАМЕТРОВ (GridSearchCV)")
    df = load_dataframe(path)
    X = df.drop(columns=["Churn"])
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    method = "SMOTE" if USE_SMOTE else "class_weight='balanced'"
    print(f"Борьба с дисбалансом: {method}")
    pipe = build_pipeline(X_train)

    param_grid = {
        "clf__C": [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100],  # обратная сила L2-регуляризации
        "clf__solver": ["lbfgs", "liblinear"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, param_grid, scoring=SCORING, cv=cv, n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)

    print(f"Метрика: {SCORING}, 5-fold CV, комбинаций: "
          f"{len(grid.cv_results_['params'])}")
    print(f"best_params_ = {grid.best_params_}")
    print(f"best_score_  = {grid.best_score_:.4f}  (средний {SCORING} по 5 фолдам)")

    cv_table = (pd.DataFrame(grid.cv_results_)
                [["param_clf__C", "param_clf__solver", "mean_test_score", "std_test_score"]]
                .sort_values("mean_test_score", ascending=False).head(5))
    print("\nТоп-5 комбинаций:")
    print(cv_table.to_string(index=False))

    # 4. Финальная оценка
    section("4. ФИНАЛЬНАЯ ОЦЕНКА НА ОТЛОЖЕННОЙ ВЫБОРКЕ")
    best = grid.best_estimator_  # уже обучен на всём train (refit=True)
    new_pred = best.predict(X_test)
    new_proba = best.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, new_pred, target_names=LABELS, digits=3))

    ConfusionMatrixDisplay.from_predictions(y_test, new_pred, display_labels=LABELS,
                                            cmap="Greens")
    plt.title("Оптимизированная модель: матрица ошибок")
    plt.savefig(os.path.join(OUT_DIR, "2_confusion_new.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # ROC-кривые обеих моделей на одном графике
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, old_proba_1, name="Практика №1", ax=ax)
    RocCurveDisplay.from_predictions(y_test, new_proba, name="Оптимизированная", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Случайный классификатор")
    ax.set_title("ROC-кривая")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(OUT_DIR, "3_roc.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Важность признаков: коэффициенты логистической регрессии
    clf = best.named_steps["clf"]
    names = [n.split("__", 1)[1] for n in best.named_steps["pre"].get_feature_names_out()]
    coefs = pd.Series(clf.coef_[0], index=names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).head(15).index)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=["#d62728" if v > 0 else "#2ca02c" for v in top])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Коэффициент (красный: повышает риск ухода, зелёный: снижает)")
    ax.set_title("Топ-15 признаков по |коэффициенту|")
    fig.savefig(os.path.join(OUT_DIR, "4_feature_importance.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print("Топ-5 признаков, повышающих риск ухода:")
    print(coefs.sort_values(ascending=False).head(5).round(3).to_string())
    print("\nТоп-5 признаков, снижающих риск ухода:")
    print(coefs.sort_values().head(5).round(3).to_string())

    # ----------------------------------------------------------------------
    # Итоговое сравнение
    # ----------------------------------------------------------------------
    section("ИТОГОВОЕ СРАВНЕНИЕ: МОДЕЛЬ ИЗ ПРАКТИКИ №1 И ОПТИМИЗИРОВАННАЯ (тест)")
    table = pd.DataFrame([
        metrics_row("Практика №1 (самописная)", y_test, old_pred, old_proba_1),
        metrics_row("Оптимизированная (sklearn)", y_test, new_pred, new_proba),
    ]).set_index("Модель").round(3)
    print(table.to_string())
    table.to_csv(os.path.join(OUT_DIR, "comparison.csv"), encoding="utf-8-sig")

    new_recall = table.iloc[1]["Recall (ушёл)"]
    verdict = "достигнута" if new_recall >= TARGET_RECALL else "НЕ достигнута"
    print(f"\nЦель Recall ≥ {TARGET_RECALL}: {verdict} (Recall = {new_recall:.3f}, "
          f"было {old_recall:.3f}).")
    print(f"Графики и таблица сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    main()