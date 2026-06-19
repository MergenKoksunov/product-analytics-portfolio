"""
Генератор синтетического датасета для кейса «Облака» (EdTech, онлайн-школа).

Данные клиента под NDA — структура воссоздана синтетически с правдоподобными
закономерностями из реального проекта:

1. Воронка обучения: регистрация -> занятия 1..8 -> завершение курса
2. Главный обрыв воронки: ~40% студентов отсеиваются между 2-м и 3-м занятием
3. A/B-тест дашборда прогресса + еженедельной рассылки:
   доходимость до конца курса +~15% в тест-группе
4. Каналы привлечения с разной мотивацией: студенты из реферальной программы
   завершают курс примерно в 2 раза чаще, чем из таргетированной рекламы
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)

START = pd.Timestamp("2024-10-01")
END = pd.Timestamp("2025-04-30")
N_LESSONS = 8  # занятий в курсе; завершение = пройдено 8-е

# Каналы: доля трафика + мотивационный сдвиг (аддитивно к вероятности перехода)
CHANNELS = {
    "Реферальная программа":     {"share": 0.20, "mot": 0.030},
    "Органический поиск":        {"share": 0.18, "mot": 0.018},
    "Email-рассылка":            {"share": 0.16, "mot": 0.0},
    "Контекстная реклама":       {"share": 0.16, "mot": -0.018},
    "Таргетированная реклама":   {"share": 0.30, "mot": -0.040},
}

# Базовые вероятности перехода с занятия N на N+1 (control, средний канал).
# Индекс 0 = регистрация->занятие1, индекс 2 = занятие2->занятие3 (обрыв 40%).
BASE_PASS = [0.92, 0.84, 0.59, 0.86, 0.88, 0.89, 0.91, 0.93]

# A/B: дашборд прогресса показывается тест-группе. Эффект — на переходах
# начиная с 3-го занятия (когда уже накоплен прогресс, который видно).
AB_BOOST = [0.0, 0.0, 0.036, 0.028, 0.026, 0.022, 0.018, 0.018]


def gen_students(n=7000):
    months = pd.date_range(START, END, freq="MS")
    weights = np.linspace(1.0, 1.3, len(months))  # лёгкий рост набора
    weights /= weights.sum()
    counts = RNG.multinomial(n, weights)

    rows, sid = [], 1
    for m, cnt in zip(months, counts):
        days = (m + pd.offsets.MonthEnd(0)).day
        for _ in range(cnt):
            reg = m + pd.Timedelta(days=int(RNG.integers(0, days)),
                                   hours=int(RNG.integers(8, 23)))
            rows.append((sid, reg))
            sid += 1
    df = pd.DataFrame(rows, columns=["student_id", "registered_at"])
    df["channel"] = RNG.choice(list(CHANNELS), size=len(df),
                               p=[c["share"] for c in CHANNELS.values()])
    df["ab_group"] = RNG.choice(["control", "test"], size=len(df), p=[0.5, 0.5])
    return df


def simulate_progress(df):
    n = len(df)
    mot = df["channel"].map(lambda c: CHANNELS[c]["mot"]).to_numpy()
    is_test = (df["ab_group"] == "test").to_numpy()

    max_lesson = np.zeros(n, dtype=int)  # сколько занятий пройдено
    rand = RNG.random((n, N_LESSONS))

    for i in range(n):
        reached = 0
        for k in range(N_LESSONS):
            p = BASE_PASS[k] + mot[i]
            if is_test[i]:
                p += AB_BOOST[k]
            p = min(max(p, 0.04), 0.985)
            if rand[i, k] < p:
                reached = k + 1
            else:
                break
        max_lesson[i] = reached

    df["lessons_completed"] = max_lesson
    df["completed_course"] = max_lesson >= N_LESSONS
    return df


if __name__ == "__main__":
    students = gen_students()
    students = simulate_progress(students)
    students.to_csv("students.csv", index=False)

    # ---- Калибровочная сводка
    n = len(students)
    print(f"Студентов: {n:,}")

    # Воронка по занятиям
    reached = [(students["lessons_completed"] >= k).sum() for k in range(0, N_LESSONS + 1)]
    print("\nВоронка (дошли до занятия k):")
    for k in range(1, N_LESSONS + 1):
        prev = reached[k - 1] if k >= 1 else n
        step = reached[k] / reached[k - 1] if reached[k - 1] else 0
        print(f"  Занятие {k}: {reached[k]:5,}  переход с пред.={step:.0%}")

    # Обрыв 2->3
    drop_23 = 1 - reached[3] / reached[2]
    print(f"\nОтсев между 2-м и 3-м занятием: {drop_23:.0%}")

    # A/B по завершению
    g = students.groupby("ab_group")["completed_course"].agg(["sum", "count"])
    g["cr"] = g["sum"] / g["count"]
    uplift = g.loc["test", "cr"] / g.loc["control", "cr"] - 1
    print(f"\nA/B завершение курса: control={g.loc['control','cr']:.1%}, "
          f"test={g.loc['test','cr']:.1%}, uplift={uplift:+.1%}")

    # Каналы: завершение
    ch = students.groupby("channel")["completed_course"].mean().sort_values(ascending=False)
    print("\nЗавершение курса по каналам:")
    print((ch * 100).round(1).to_string())
    ratio = ch["Реферальная программа"] / ch["Таргетированная реклама"]
    print(f"\nРеферальная программа vs таргет: x{ratio:.1f}")
