"""Веб-приложение на Flask для ВКР: прогноз механических свойств
и рекомендация соотношения матрица-наполнитель по обученным моделям.

Запуск:
    .venv\\Scripts\\python app\\app.py
Приложение поднимется на http://127.0.0.1:5000/
"""
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request
from tensorflow import keras

# Пути: очищенный датасет (нужен для статистики признаков) и папка с сохранёнными моделями
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "processed" / "data_clean.csv"
MODELS_DIR = BASE_DIR / "models"

# Порядок признаков соответствует feature_names_in_ сохранённых пайплайнов/скейлера.
FEATURES_ML = [
    "Соотношение матрица-наполнитель",
    "Плотность, кг/м3",
    "модуль упругости, ГПа",
    "Количество отвердителя, м.%",
    "Содержание эпоксидных групп,%_2",
    "Температура вспышки, С_2",
    "Поверхностная плотность, г/м2",
    "Потребление смолы, г/м2",
    "Угол нашивки, град",
    "Шаг нашивки",
    "Плотность нашивки",
]

# Входы нейросети: все столбцы, кроме целевого «Соотношение матрица-наполнитель»;
# порядок совпадает с порядком при обучении скейлера
FEATURES_NN = [
    "Плотность, кг/м3",
    "модуль упругости, ГПа",
    "Количество отвердителя, м.%",
    "Содержание эпоксидных групп,%_2",
    "Температура вспышки, С_2",
    "Поверхностная плотность, г/м2",
    "Модуль упругости при растяжении, ГПа",
    "Прочность при растяжении, МПа",
    "Потребление смолы, г/м2",
    "Угол нашивки, град",
    "Шаг нашивки",
    "Плотность нашивки",
]

# Примечания об ограниченной точности: показываются пользователю рядом с результатом
ACCURACY_NOTE = (
    "Точность моделей ограничена (R² на кросс-валидации близок к нулю), "
    "поэтому прогноз носит ориентировочный характер и не заменяет "
    "лабораторные испытания."
)

ACCURACY_NOTE_NN = (
    "Точность нейронной сети ограничена (R² на тестовой выборке близок к "
    "нулю), поэтому рекомендация носит ориентировочный характер и не "
    "заменяет лабораторные испытания."
)

# Приложение Flask
app = Flask(__name__)

# Датасет нужен только для статистики признаков (минимум, медиана, максимум):
# по ней задаются значения по умолчанию в форме и диапазоны для проверки ввода
_data = pd.read_csv(DATA_PATH, index_col=0)
_stats = _data.describe().loc[["min", "50%", "max"]].rename(index={"50%": "median"})


def _stat(feature, name):
    """Возвращает статистику признака по обучающим данным: min, median или max."""
    return float(_stats.loc[name, feature])


def defaults_for(features):
    """Возвращает словарь «признак → медиана» — значения по умолчанию для формы."""
    return {f: round(_stat(f, "median"), 4) for f in features}


def parse_and_validate(features, form, prefix):
    """Читает поля формы, проверяет, что значения числовые, и предупреждает
    о выходе за диапазон обучающей выборки. Возвращает значения (только
    успешно распознанные), исходные строки (для повторного отображения
    формы), список ошибок и список предупреждений — всё на русском."""
    values = {}
    raw = {}
    errors = []
    warnings = []
    for i, feat in enumerate(features):
        # Имя поля формы: префикс вкладки («ml» или «nn») и порядковый номер признака
        text = form.get(f"{prefix}_{i}", "").strip()
        raw[feat] = text
        # Пустое поле — ошибка: прогноз в этом случае не выполняется
        if not text:
            errors.append(f"«{feat}»: поле не заполнено.")
            continue
        # Запятая допускается как десятичный разделитель; нечисловой ввод — ошибка
        try:
            val = float(text.replace(",", "."))
        except ValueError:
            errors.append(f"«{feat}»: введите числовое значение (получено «{text}»).")
            continue
        # Выход за диапазон обучающих данных — только предупреждение: значение принимается,
        # но прогноз в этом случае является экстраполяцией
        lo, hi = _stat(feat, "min"), _stat(feat, "max")
        if val < lo or val > hi:
            warnings.append(
                f"«{feat}» = {val:g} выходит за диапазон обучающих данных "
                f"[{lo:g}; {hi:g}]."
            )
        values[feat] = val
    return values, raw, errors, warnings


def load_models():
    """Загружает обученные модели и скейлер из app/models/ и возвращает кортеж:
    (модель модуля упругости, модель прочности, скейлер нейросети, нейросеть)."""
    # Пайплайны (скейлер + модель) для двух механических свойств
    model_modulus = joblib.load(MODELS_DIR / "best_model_modulus.joblib")
    model_strength = joblib.load(MODELS_DIR / "best_model_strength.joblib")
    # Скейлер и нейросеть для соотношения матрица-наполнитель
    nn_scaler = joblib.load(MODELS_DIR / "nn_scaler.joblib")
    nn_model = keras.models.load_model(MODELS_DIR / "nn_ratio.keras")
    return model_modulus, model_strength, nn_scaler, nn_model


# Модели загружаются один раз при запуске приложения, а не при каждом запросе
MODEL_MODULUS, MODEL_STRENGTH, NN_SCALER, NN_MODEL = load_models()


def render_page(active_tab, result_ml=None, result_nn=None,
                 raw_ml=None, raw_nn=None, errors_ml=None, errors_nn=None,
                 warnings_ml=None, warnings_nn=None):
    """Отрисовывает страницу index.html для обеих вкладок: передаёт в шаблон списки
    признаков, значения по умолчанию, введённые значения, результаты, ошибки и
    предупреждения; active_tab задаёт открытую вкладку («ml» или «nn»)."""
    return render_template(
        "index.html",
        features_ml=FEATURES_ML,
        features_nn=FEATURES_NN,
        defaults_ml=defaults_for(FEATURES_ML),
        defaults_nn=defaults_for(FEATURES_NN),
        raw_ml=raw_ml or {},
        raw_nn=raw_nn or {},
        result_ml=result_ml,
        result_nn=result_nn,
        errors_ml=errors_ml or [],
        errors_nn=errors_nn or [],
        warnings_ml=warnings_ml or [],
        warnings_nn=warnings_nn or [],
        active_tab=active_tab,
        accuracy_note=ACCURACY_NOTE,
        accuracy_note_nn=ACCURACY_NOTE_NN,
    )


@app.route("/", methods=["GET"])
def index():
    """Главная страница: открывается вкладка прогноза механических свойств."""
    return render_page(active_tab="ml")


@app.route("/predict_properties", methods=["POST"])
def predict_properties():
    """Вкладка «ML»: проверяет введённые признаки и по двум моделям прогнозирует
    модуль упругости и прочность при растяжении."""
    # Читаем форму и проверяем ввод
    values, raw, errors, warnings = parse_and_validate(FEATURES_ML, request.form, "ml")
    result = None
    # Прогноз выполняется только при отсутствии ошибок ввода
    if not errors:
        # Одна строка с признаками в том порядке, который ожидают обученные пайплайны
        row = pd.DataFrame([values], columns=FEATURES_ML)
        # Пайплайн сам масштабирует признаки внутри predict, отдельный скейлер не нужен
        modulus = float(MODEL_MODULUS.predict(row)[0])
        strength = float(MODEL_STRENGTH.predict(row)[0])
        # Для отображения округляем до двух знаков
        result = {
            "modulus": round(modulus, 2),
            "strength": round(strength, 2),
        }
    # При ошибках ввода result равен None: на странице показываются сообщения об ошибках
    return render_page(
        active_tab="ml",
        result_ml=result,
        raw_ml=raw,
        errors_ml=errors,
        warnings_ml=warnings,
    )


@app.route("/predict_ratio", methods=["POST"])
def predict_ratio():
    """Вкладка «Нейросеть»: проверяет введённые признаки, масштабирует их и
    прогнозирует соотношение матрица-наполнитель."""
    # Читаем форму и проверяем ввод
    values, raw, errors, warnings = parse_and_validate(FEATURES_NN, request.form, "nn")
    result = None
    # Прогноз выполняется только при отсутствии ошибок ввода
    if not errors:
        # Одна строка с признаками в порядке обучения нейросети
        row = pd.DataFrame([values], columns=FEATURES_NN)
        # Нейросеть обучалась на масштабированных данных: применяем сохранённый
        # MinMaxScaler (только transform, без повторного обучения)
        scaled = NN_SCALER.transform(row)
        # verbose=0 — без вывода прогресса Keras; берём единственное значение выхода сети
        pred = float(NN_MODEL.predict(scaled, verbose=0)[0][0])
        # Для отображения округляем до двух знаков
        result = {"ratio": round(pred, 2)}
    # При ошибках ввода result равен None: на странице показываются сообщения об ошибках
    return render_page(
        active_tab="nn",
        result_nn=result,
        raw_nn=raw,
        errors_nn=errors,
        warnings_nn=warnings,
    )


# Локальный запуск сервера разработки (режим отладки включён)
if __name__ == "__main__":
    app.run(debug=True)
