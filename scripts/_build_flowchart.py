"""Строит блок-схему всего процесса ВКР и сохраняет её в figures/flowchart.png
(300 dpi, подписи на русском языке)."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["axes.unicode_minus"] = False

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

STAGE_COLOR = "#dbe9ee"
STAGE_EDGE = "#2c5f6f"
SPLIT_COLOR = "#eaf5ee"
SPLIT_EDGE = "#4a8f63"
TEXT_COLOR = "#22333b"

fig, ax = plt.subplots(figsize=(9, 15))
ax.set_xlim(0, 10)
ax.set_ylim(0, 30)
ax.axis("off")


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


def arrow(b_from, b_to, dy_from=None, dy_to=None):
    x1, y1, w1, h1 = b_from
    x2, y2, w2, h2 = b_to
    p1 = (x1, y1 - h1 / 2) if dy_from is None else dy_from
    p2 = (x2, y2 + h2 / 2) if dy_to is None else dy_to
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.4, color=STAGE_EDGE)
    ax.add_patch(a)


# --- Основная цепочка ---
b1 = box(5, 29, 7.6, 1.6, "Загрузка данных\nX_bp.xlsx и X_nup.xlsx")
b2 = box(5, 26.6, 7.6, 1.6, "Объединение INNER JOIN\nпо индексу (index_col=0)")
b3 = box(5, 24.2, 7.6, 1.6, "Разведочный анализ (EDA)\nописательная статистика, графики")
b4 = box(5, 21.8, 7.6, 1.6, "Очистка выбросов\nметод IQR")
b5 = box(5, 19.4, 7.6, 1.6, "Нормализация\nMinMaxScaler")
b6 = box(5, 17.0, 7.6, 1.9,
         "Разбиение выборки\ntrain_test_split(test_size=0.3, random_state=42)")

arrow(b1, b2)
arrow(b2, b3)
arrow(b3, b4)
arrow(b4, b5)
arrow(b5, b6)

# --- Разветвление на два направления моделирования ---
b7a = box(2.7, 14.0, 4.4, 2.1,
          "Модели ML (GridSearchCV, cv=10)\nМодуль упругости при растяжении, ГПа\nПрочность при растяжении, МПа",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=9.5)
b7b = box(7.3, 14.0, 4.4, 2.1,
          "Нейронная сеть (Keras)\nСоотношение\nматрица-наполнитель",
          facecolor=SPLIT_COLOR, edgecolor=SPLIT_EDGE, fontsize=9.5)

arrow(b6, b7a, p1 := (5, 17.0 - 1.9 / 2), p2 := (2.7, 14.0 + 2.1 / 2))
arrow(b6, b7b, p1 := (5, 17.0 - 1.9 / 2), p2 := (7.3, 14.0 + 2.1 / 2))

b8 = box(5, 11.0, 7.6, 2.1,
         "Сравнение с DummyRegressor / baseline\nотбор модели по CV_R2 (валидация),\nоценка MAE, MSE, RMSE, R2 на train/test")

arrow(b7a, b8, p1 := (2.7, 14.0 - 2.1 / 2), p2 := (5, 11.0 + 2.1 / 2))
arrow(b7b, b8, p1 := (7.3, 14.0 - 2.1 / 2), p2 := (5, 11.0 + 2.1 / 2))

b9 = box(5, 8.2, 7.6, 1.9,
         "Сохранение моделей\napp/models/*.joblib, nn_ratio.keras, nn_scaler.joblib")
arrow(b8, b9)

b10 = box(5, 5.6, 7.6, 1.9,
          "Веб-приложение Flask (app/app.py)\nПрогноз свойств /\nРекомендация соотношения")
arrow(b9, b10)

b11 = box(5, 3.0, 7.6, 1.6,
          "Пользователь: ввод характеристик\nматериала → прогноз (ориентировочный)")
arrow(b10, b11)

ax.text(5, 29.9, "Блок-схема процесса ВКР: от данных до веб-приложения",
        ha="center", va="bottom", fontsize=13, fontweight="bold", color=TEXT_COLOR)

fig.tight_layout()
out_path = os.path.join(FIGURES_DIR, "flowchart.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print("Сохранено:", out_path)
