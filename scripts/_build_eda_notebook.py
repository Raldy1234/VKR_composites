"""Служебный скрипт для сборки notebooks/01_EDA.ipynb через nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------- Заголовок
md("""# Разведочный анализ данных (EDA)
## ВКР: прогнозирование свойств композиционных материалов

Блок 1 выпускной квалификационной работы. В этом ноутбуке выполняется:

1. Загрузка и объединение исходных данных.
2. Описательная статистика.
3. Визуализация распределений и корреляций.
4. Поиск и удаление выбросов методом IQR.
5. Нормализация признаков (MinMaxScaler).
6. Сохранение очищенного и нормализованного датасетов.
""")

code("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

FIGURES_DIR = "../figures"
DATA_DIR = "../data"
PROCESSED_DIR = "../data/processed"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

RANDOM_STATE = 42
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
""")

# ---------------------------------------------------------------- 1. Загрузка
md("""## 1. Загрузка и объединение данных

Загружаем два файла: `X_bp.xlsx` содержит параметры композита и его
компонентов (связующего и наполнителя), `X_nup.xlsx` содержит параметры
нашивки (угол нашивки, шаг нашивки, плотность нашивки). Первый столбец в
каждом файле — индекс образца (`index_col=0`). Объединяем таблицы по индексу
через **INNER JOIN**, чтобы оставить только образцы, присутствующие в обоих
файлах.""")

code("""df_bp = pd.read_excel(os.path.join(DATA_DIR, "X_bp.xlsx"), index_col=0)
df_nup = pd.read_excel(os.path.join(DATA_DIR, "X_nup.xlsx"), index_col=0)

print("X_bp.xlsx  shape:", df_bp.shape)
print("X_nup.xlsx shape:", df_nup.shape)

df = df_bp.join(df_nup, how="inner")
print("Объединённый датасет (INNER JOIN) shape:", df.shape)
df.head()
""")

code("""df.info()
""")

md("""### Типы данных, пропуски и дубликаты""")

code("""dtypes_table = df.dtypes.rename("Тип данных").to_frame()
display(dtypes_table)

missing = df.isnull().sum().rename("Количество пропусков").to_frame()
missing["Доля пропусков, %"] = (missing["Количество пропусков"] / len(df) * 100).round(2)
display(missing)

n_duplicates = df.duplicated().sum()
print(f"Количество полных дубликатов строк: {n_duplicates}")
print(f"Итоговый размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
""")

# ---------------------------------------------------------------- 2. Статистика
md("""## 2. Описательная статистика

Для каждого из 13 признаков рассчитываем среднее, медиану, стандартное
отклонение, минимум, максимум и квартили. Таблица сохраняется в
`figures/descriptive_stats.csv`.""")

code("""desc = df.describe().T
desc["медиана"] = df.median()
desc = desc.rename(columns={
    "count": "количество",
    "mean": "среднее",
    "std": "стд. отклонение",
    "min": "минимум",
    "25%": "25-й перцентиль",
    "50%": "50-й перцентиль (медиана, дубль)",
    "75%": "75-й перцентиль",
    "max": "максимум",
})
desc = desc[["количество", "среднее", "медиана", "стд. отклонение",
             "минимум", "25-й перцентиль", "75-й перцентиль", "максимум"]]

desc.to_csv(os.path.join(FIGURES_DIR, "descriptive_stats.csv"), encoding="utf-8-sig")
desc
""")

# ---------------------------------------------------------------- 3. Визуализация
md("""## 3. Визуализация данных

### 3.1 Гистограммы с KDE

Для каждой из 13 переменных строим гистограмму распределения с наложенной
кривой плотности (KDE), чтобы оценить форму распределения и наличие
асимметрии.""")

code("""cols = df.columns.tolist()
n_cols = 3
n_rows = int(np.ceil(len(cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel("Частота", fontsize=8)

for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Распределения признаков (гистограммы с KDE)", fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "histograms_kde.png"), dpi=300, bbox_inches="tight")
plt.show()
""")

md("""### 3.2 Boxplot-ы (диаграммы размаха)

Диаграммы размаха помогают визуально оценить разброс значений и наличие
потенциальных выбросов по каждому признаку.""")

code("""fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.boxplot(y=df[col], ax=axes[i], color="lightcoral")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel(col, fontsize=8)

for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Диаграммы размаха признаков (до очистки от выбросов)", fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "boxplots_before_cleaning.png"), dpi=300, bbox_inches="tight")
plt.show()
""")

md("""### 3.3 Pairplot (диаграммы рассеяния по парам признаков)

Матрица диаграмм рассеяния показывает взаимосвязи между всеми парами
признаков одновременно.""")

code("""pairplot = sns.pairplot(df, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 12})
pairplot.fig.suptitle("Диаграммы рассеяния по парам признаков", y=1.01, fontsize=16)
pairplot.savefig(os.path.join(FIGURES_DIR, "pairplot.png"), dpi=300, bbox_inches="tight")
plt.show()
""")

md("""### 3.4 Тепловая карта корреляций

Матрица корреляций Пирсона между всеми числовыми признаками.""")

code("""corr = df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"label": "Коэффициент корреляции"}, ax=ax)
ax.set_title("Тепловая карта корреляций между признаками", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=300, bbox_inches="tight")
plt.show()
""")

code("""corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
corr_pairs = corr_pairs.sort_values(key=lambda s: s.abs(), ascending=False)
print("Топ-10 пар признаков по модулю коэффициента корреляции:")
corr_pairs.head(10)
""")

# ---------------------------------------------------------------- 4. Outliers
md("""## 4. Поиск и удаление выбросов методом IQR

Для каждого признака вычисляем межквартильный размах
IQR = Q3 - Q1 и границы выбросов:

- нижняя граница: Q1 - 1.5 * IQR
- верхняя граница: Q3 + 1.5 * IQR

Значения за пределами границ считаются выбросами. Строка удаляется из
датасета, если хотя бы один признак в ней является выбросом.""")

code("""outlier_counts = {}
outlier_mask_total = pd.Series(False, index=df.index)

for col in cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    col_mask = (df[col] < lower) | (df[col] > upper)
    outlier_counts[col] = int(col_mask.sum())
    outlier_mask_total |= col_mask

outliers_table = pd.Series(outlier_counts, name="Количество выбросов").to_frame()
outliers_table["Доля от выборки, %"] = (outliers_table["Количество выбросов"] / len(df) * 100).round(2)
display(outliers_table)

n_rows_to_remove = int(outlier_mask_total.sum())
print(f"Строк с хотя бы одним выбросом: {n_rows_to_remove}")
print(f"Размер датасета до очистки: {df.shape}")

df_clean = df.loc[~outlier_mask_total].copy()
print(f"Размер датасета после очистки: {df_clean.shape}")
""")

md("""### Диаграммы размаха после очистки от выбросов""")

code("""fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.boxplot(y=df_clean[col], ax=axes[i], color="mediumseagreen")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel(col, fontsize=8)

for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Диаграммы размаха признаков (после очистки от выбросов)", fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "boxplots_after_cleaning.png"), dpi=300, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------- 5. Нормализация
md("""## 5. Нормализация методом MinMaxScaler

Масштабируем все признаки очищенного датасета в диапазон [0, 1] с помощью
`MinMaxScaler`. Сравниваем гистограммы и таблицу минимумов/максимумов до и
после нормализации.""")

code("""scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=df_clean.columns,
    index=df_clean.index,
)

minmax_before_after = pd.DataFrame({
    "минимум до": df_clean.min(),
    "максимум до": df_clean.max(),
    "минимум после": df_normalized.min(),
    "максимум после": df_normalized.max(),
})
minmax_before_after.to_csv(os.path.join(FIGURES_DIR, "minmax_before_after.csv"), encoding="utf-8-sig")
minmax_before_after
""")

code("""fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.histplot(df_clean[col], kde=True, ax=axes[i], color="steelblue", label="До нормализации", alpha=0.5)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel("Частота", fontsize=8)

for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Распределения признаков ДО нормализации (после очистки от выбросов)", fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "histograms_before_normalization.png"), dpi=300, bbox_inches="tight")
plt.show()
""")

code("""fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.histplot(df_normalized[col], kde=True, ax=axes[i], color="darkorange", label="После нормализации", alpha=0.5)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel("Частота", fontsize=8)

for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Распределения признаков ПОСЛЕ нормализации (MinMaxScaler)", fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "histograms_after_normalization.png"), dpi=300, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------- 6. Сохранение
md("""## 6. Сохранение обработанных данных

Сохраняем очищенный (но ещё не нормализованный) датасет и нормализованный
датасет в `data/processed/`.""")

code("""clean_path = os.path.join(PROCESSED_DIR, "data_clean.csv")
normalized_path = os.path.join(PROCESSED_DIR, "data_normalized.csv")

df_clean.to_csv(clean_path, encoding="utf-8-sig")
df_normalized.to_csv(normalized_path, encoding="utf-8-sig")

print(f"Очищенный датасет сохранён: {clean_path}, размер {df_clean.shape}")
print(f"Нормализованный датасет сохранён: {normalized_path}, размер {df_normalized.shape}")
""")

md("""## Выводы

- Исходные файлы объединены по индексу через INNER JOIN, пропусков и
  дубликатов в объединённом датасете не обнаружено.
- Рассчитана описательная статистика по всем 13 признакам.
- Построены гистограммы, boxplot-ы, pairplot и тепловая карта корреляций.
- Методом IQR (1.5·IQR) обнаружены и удалены выбросы.
- Признаки нормализованы методом MinMaxScaler в диапазон [0, 1].
- Итоговые датасеты сохранены в `data/processed/data_clean.csv` и
  `data/processed/data_normalized.csv` для использования в следующих блоках
  ВКР (построение моделей машинного обучения).""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (.venv)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

with open("notebooks/01_EDA.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook создан: notebooks/01_EDA.ipynb")
