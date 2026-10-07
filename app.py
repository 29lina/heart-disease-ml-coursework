from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_heart_disease_model.joblib"


def get_float(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt))

            if min_value is not None and value < min_value:
                print(f"Значення не може бути меншим за {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Значення не може бути більшим за {max_value}.")
                continue

            return value

        except ValueError:
            print("Помилка. Введіть числове значення.")


def get_int(prompt, allowed_values=None):
    while True:
        try:
            value = int(input(prompt))

            if allowed_values is not None and value not in allowed_values:
                print(
                    "Допустимі значення:",
                    ", ".join(map(str, allowed_values))
                )
                continue

            return value

        except ValueError:
            print("Помилка. Введіть ціле число.")


def main():

    print("=" * 65)
    print("     СИСТЕМА ОЦІНКИ РИЗИКУ СЕРЦЕВОГО ЗАХВОРЮВАННЯ")
    print("=" * 65)

    if not MODEL_PATH.exists():
        print("\nПомилка: файл моделі не знайдено.")
        print("Спочатку виконайте Jupyter Notebook і збережіть модель.")
        return

    model = joblib.load(MODEL_PATH)

    print("\nВведіть клінічні показники пацієнта.\n")

    age = get_float(
        "1. Вік, років: ",
        1,
        120
    )

    sex = get_int(
        "2. Стать (0 — жінка, 1 — чоловік): ",
        [0, 1]
    )

    cp = get_int(
        "3. Тип болю у грудях (1, 2, 3 або 4): ",
        [1, 2, 3, 4]
    )

    trestbps = get_float(
        "4. Артеріальний тиск у стані спокою, мм рт. ст.: ",
        50,
        300
    )

    chol = get_float(
        "5. Рівень холестерину, мг/дл: ",
        50,
        700
    )

    fbs = get_int(
        "6. Цукор натще > 120 мг/дл? (0 — ні, 1 — так): ",
        [0, 1]
    )

    restecg = get_int(
        "7. Результат ЕКГ у стані спокою (0, 1 або 2): ",
        [0, 1, 2]
    )

    thalach = get_float(
        "8. Максимальна частота серцевих скорочень: ",
        40,
        250
    )

    exang = get_int(
        "9. Стенокардія при фізичному навантаженні? (0 — ні, 1 — так): ",
        [0, 1]
    )

    oldpeak = get_float(
        "10. Депресія сегмента ST (oldpeak): ",
        0,
        10
    )

    slope = get_int(
        "11. Нахил сегмента ST (1, 2 або 3): ",
        [1, 2, 3]
    )

    ca = get_int(
        "12. Кількість великих судин, забарвлених флюороскопією (0–3): ",
        [0, 1, 2, 3]
    )

    thal = get_int(
        "13. Thal (3 — норма, 6 — фіксований дефект, 7 — оборотний дефект): ",
        [3, 6, 7]
    )

    patient = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }]
    )

    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0][1]

    print("\n" + "=" * 65)
    print("                       РЕЗУЛЬТАТ")
    print("=" * 65)

    if prediction == 1:
        print("\nРизик серцевого захворювання: ВИСОКИЙ")
    else:
        print("\nРизик серцевого захворювання: НИЗЬКИЙ")

    print(
        f"Оцінена моделлю ймовірність наявності захворювання: "
        f"{probability * 100:.2f}%"
    )

    print(
        "\nУвага: результат є демонстрацією роботи моделі "
        "машинного навчання і не є медичним діагнозом."
    )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()