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

app = Flask(__name__)

_data = pd.read_csv(DATA_PATH, index_col=0)
_stats = _data.describe().loc[["min", "50%", "max"]].rename(index={"50%": "median"})


def _stat(feature, name):
    return float(_stats.loc[name, feature])


def defaults_for(features):
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
        text = form.get(f"{prefix}_{i}", "").strip()
        raw[feat] = text
        if not text:
            errors.append(f"«{feat}»: поле не заполнено.")
            continue
        try:
            val = float(text.replace(",", "."))
        except ValueError:
            errors.append(f"«{feat}»: введите числовое значение (получено «{text}»).")
            continue
        lo, hi = _stat(feat, "min"), _stat(feat, "max")
        if val < lo or val > hi:
            warnings.append(
                f"«{feat}» = {val:g} выходит за диапазон обучающих данных "
                f"[{lo:g}; {hi:g}]."
            )
        values[feat] = val
    return values, raw, errors, warnings


def load_models():
    model_modulus = joblib.load(MODELS_DIR / "best_model_modulus.joblib")
    model_strength = joblib.load(MODELS_DIR / "best_model_strength.joblib")
    nn_scaler = joblib.load(MODELS_DIR / "nn_scaler.joblib")
    nn_model = keras.models.load_model(MODELS_DIR / "nn_ratio.keras")
    return model_modulus, model_strength, nn_scaler, nn_model


MODEL_MODULUS, MODEL_STRENGTH, NN_SCALER, NN_MODEL = load_models()


def render_page(active_tab, result_ml=None, result_nn=None,
                 raw_ml=None, raw_nn=None, errors_ml=None, errors_nn=None,
                 warnings_ml=None, warnings_nn=None):
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
    return render_page(active_tab="ml")


@app.route("/predict_properties", methods=["POST"])
def predict_properties():
    values, raw, errors, warnings = parse_and_validate(FEATURES_ML, request.form, "ml")
    result = None
    if not errors:
        row = pd.DataFrame([values], columns=FEATURES_ML)
        modulus = float(MODEL_MODULUS.predict(row)[0])
        strength = float(MODEL_STRENGTH.predict(row)[0])
        result = {
            "modulus": round(modulus, 2),
            "strength": round(strength, 2),
        }
    return render_page(
        active_tab="ml",
        result_ml=result,
        raw_ml=raw,
        errors_ml=errors,
        warnings_ml=warnings,
    )


@app.route("/predict_ratio", methods=["POST"])
def predict_ratio():
    values, raw, errors, warnings = parse_and_validate(FEATURES_NN, request.form, "nn")
    result = None
    if not errors:
        row = pd.DataFrame([values], columns=FEATURES_NN)
        scaled = NN_SCALER.transform(row)
        pred = float(NN_MODEL.predict(scaled, verbose=0)[0][0])
        result = {"ratio": round(pred, 2)}
    return render_page(
        active_tab="nn",
        result_nn=result,
        raw_nn=raw,
        errors_nn=errors,
        warnings_nn=warnings,
    )


if __name__ == "__main__":
    app.run(debug=True)
