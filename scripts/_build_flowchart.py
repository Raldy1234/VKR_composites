"""Строит блок-схему всего процесса ВКР и сохраняет её в figures/flowchart.png
(300 dpi, подписи на русском языке).

Схема показывает общую цепочку (загрузка, объединение, EDA, очистка выбросов),
затем три ветви — визуализация EDA, модели машинного обучения и нейронная
сеть — и их слияние в веб-приложении Flask.

Запуск из корня проекта:
    .venv\\Scripts\\python scripts\\_build_flowchart.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Обычный дефис вместо юникодного минуса (надёжнее отображается шрифтами)
plt.rcParams["axes.unicode_minus"] = False

# Рисунок сохраняем в figures/ в корне проекта (скрипт лежит в scripts/)
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Цвета блоков: основная цепочка (сине-зелёная), ветви моделей (зелёная) и EDA (жёлтая)
STAGE_COLOR = "#dbe9ee"
STAGE_EDGE = "#2c5f6f"
SPLIT_COLOR = "#eaf5ee"
SPLIT_EDGE = "#4a8f63"
TEXT_COLOR = "#22333b"

EDA_COLOR = "#f3ecdc"
EDA_EDGE = "#a67c2e"

# Холст схемы: своя система координат (0–10 по X, 2–33 по Y), оси скрыты
fig, ax = plt.subplots(figsize=(11, 16))
ax.set_xlim(0, 10)
ax.set_ylim(2.0, 33.0)
ax.axis("off")


# Рисует скруглённый блок с текстом по центру и возвращает (cx, cy, w, h) — по ним строятся стрелки
def box(cx, cy, w, h, text, facecolor=STAGE_COLOR, edgecolor=STAGE_EDGE, fontsize=10.5):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.12,rounding_size=0.15",
        linewidth=1.6, facecolor=facecolor, edgecolor=edgecolor,
    )
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             color=TEXT_COLOR, linespacing=1.3, wrap=True)
    return (cx, cy, w, h)


# Рисует стрелку от нижней границы одного блока к верхней границе другого (точки можно задать вручную)
def arrow(b_from, b_to, dy_from=None, dy_to=None):
    x1, y1, w1, h1 = b_from
    x2, y2, w2, h2 = b_to
    p1 = (x1, y1 - h1 / 2) if dy_from is None else dy_from
    p2 = (x2, y2 + h2 / 2) if dy_to is None else dy_to
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.4, color=STAGE_EDGE)
    ax.add_patch(a)


# --- Основная цепочка (общая для всех данных) ---
b1 = box(5, 31.6, 7.6, 1.6, "Загрузка данных\nX_bp.xlsx и X_nup.xlsx")
b2 = box(5, 29.2, 7.6, 1.6, "Объединение INNER JOIN\nпо индексу (index_col=0)")
b3 = box(5, 26.8, 7.6, 1.8,
         "Разведочный анализ (EDA)\nстатистика, гистограммы, boxplot,\npairplot, корреляции")
b4 = box(5, 24.3, 7.6, 1.6, "Очистка выбросов\nметод IQR")

arrow(b1, b2)
arrow(b2, b3)
arrow(b3, b4)

# --- Разветвление на три направления ---
# X-координаты трёх колонок ветвей (EDA, машинное обучение, нейронная сеть) и их ширина
COL_EDA, COL_ML, COL_NN = 1.7, 5.0, 8.3
COL_W = 2.9


# Заголовок над колонкой ветви; белая подложка, чтобы стрелки не пересекали текст
def branch_header(cx, text, color):
    ax.text(cx, 22.7, text, ha="center", va="center", fontsize=9.5,
             fontweight="bold", color=color,
             bbox=dict(facecolor="white", edgecolor="none", pad=2))


branch_header(COL_EDA, "EDA (визуализация)", EDA_EDGE)
branch_header(COL_ML, "Модели машинного обучения", SPLIT_EDGE)
branch_header(COL_NN, "Нейронная сеть", SPLIT_EDGE)

eda1 = box(COL_EDA, 21.0, COL_W, 1.9,
           "Нормализация MinMax\n(весь датасет, только\nдля визуализации)",
           facecolor=EDA_COLOR, edgecolor=EDA_EDGE, fontsize=8.7)
ml1 = box(COL_ML, 21.0, COL_W, 1.9, "train_test_split\n70/30 (random_state=42)",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
nn1 = box(COL_NN, 21.0, COL_W, 1.9, "train_test_split\n70/30 (random_state=42)",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)

arrow(b4, eda1)
arrow(b4, ml1)
arrow(b4, nn1)

eda2 = box(COL_EDA, 18.4, COL_W, 1.9,
           "Гистограммы\nдо / после нормализации\n(только для EDA)",
           facecolor=EDA_COLOR, edgecolor=EDA_EDGE, fontsize=8.7)
ml2 = box(COL_ML, 18.4, COL_W, 1.9,
          "Pipeline: MinMaxScaler\n(fit только на train)\n+ модель",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
nn2 = box(COL_NN, 18.4, COL_W, 1.9,
          "MinMaxScaler\n(fit только на train)",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)

arrow(eda1, eda2)
arrow(ml1, ml2)
arrow(nn1, nn2)

ml3 = box(COL_ML, 15.8, COL_W, 1.9, "GridSearchCV, cv=10\n(по каждой целевой)",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
nn3 = box(COL_NN, 15.8, COL_W, 1.9, "Keras:\n3 архитектуры",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
arrow(ml2, ml3)
arrow(nn2, nn3)

ml4 = box(COL_ML, 13.2, COL_W, 1.9,
          "Отбор модели по CV_R2\n(сравнение с DummyRegressor)",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
nn4 = box(COL_NN, 13.2, COL_W, 1.9, "Отбор архитектуры\nпо val_loss",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
arrow(ml3, ml4)
arrow(nn3, nn4)

ml5 = box(COL_ML, 10.6, COL_W, 1.9,
          "Оценка на тесте\nMAE, MSE, RMSE, R2",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
nn5 = box(COL_NN, 10.6, COL_W, 1.9,
          "Оценка на тесте\nMAE, MSE, RMSE, R2",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
arrow(ml4, ml5)
arrow(nn4, nn5)

ml6 = box(COL_ML, 8.0, COL_W, 1.9,
          "Сохранение модели\napp/models/*.joblib",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
nn6 = box(COL_NN, 8.0, COL_W, 1.9,
          "Сохранение модели\nnn_ratio.keras,\nnn_scaler.joblib",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=8.7)
arrow(ml5, ml6)
arrow(nn5, nn6)

# --- Слияние двух направлений моделирования в приложении ---
b9 = box(5, 5.7, 7.6, 1.8, "Веб-приложение Flask\n(app/app.py)")
arrow(ml6, b9)
arrow(nn6, b9)

b10 = box(5, 3.4, 7.6, 1.6,
          "Пользователь: ввод характеристик\nматериала → прогноз (ориентировочный)")
arrow(b9, b10)

# Сохраняем схему в PNG (300 dpi); поля pad_inches предотвращают обрез блоков по краю
fig.tight_layout()
out_path = os.path.join(FIGURES_DIR, "flowchart.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
print("Сохранено:", out_path)
