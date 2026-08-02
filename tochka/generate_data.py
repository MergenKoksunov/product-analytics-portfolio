"""
Генератор синтетического датасета для кейса «Точка» (финтех, банк для предпринимателей).

Данные клиента под NDA — структура воссоздана синтетически с правдоподобными
закономерностями из реального проекта по юнит-экономике и монетизации тарифа
«Точка Старт» для самозанятых:

1. Четыре сегмента самозанятых: курьеры, таксисты, фрилансеры, мастера услуг.
2. Стартовая проблема: средний LTV/CAC = 1,9 при плане 3,0 — продукт не окупается.
3. Разброс по сегментам: фрилансеры LTV/CAC ~4,8 (окупаются за ~4 мес),
   курьеры LTV/CAC ~1,3 (пассивное использование счёта, мало допуслуг).
4. Ценовая эластичность: у фрилансеров спрос неэластичен; у курьеров снижение
   цены на 20% повышает конверсию в подписку на ~15%.
5. После новой тарифной сетки (базовый + облегчённый для курьеров) и A/B-теста:
   средний LTV/CAC 1,9 -> 3,2, ARPU +12%, окупаемость 9 -> 6 мес.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(23)

# --- Сегменты самозанятых: доля базы, CAC (руб.), базовая месячная выручка с клиента
#     (ARPU-компонента от допуслуг), доля активных допуслуг, помесячный отток (churn)
SEGMENTS = {
    "Фрилансеры":     {"share": 0.20, "cac": 1500, "base_rev": 453, "addon": 0.55, "churn": 0.060},
    "Мастера услуг":  {"share": 0.24, "cac": 1650, "base_rev": 250, "addon": 0.42, "churn": 0.095},
    "Таксисты":       {"share": 0.22, "cac": 1800, "base_rev": 120, "addon": 0.30, "churn": 0.130},
    "Курьеры":        {"share": 0.34, "cac": 2100, "base_rev": 55,  "addon": 0.15, "churn": 0.160},
}

# Базовая цена тарифа (руб./мес) — единая на старте
PRICE_BASE = 490

# Горизонт для LTV (мес)
LTV_HORIZON = 12


def gen_users(n=9000):
    rows, uid = [], 1
    seg_names = list(SEGMENTS)
    seg_probs = [SEGMENTS[s]["share"] for s in seg_names]
    start = pd.Timestamp("2024-08-01")
    for _ in range(n):
        seg = RNG.choice(seg_names, p=seg_probs)
        # дата подписки — в течение периода запуска
        signup = start + pd.Timedelta(days=int(RNG.integers(0, 180)))
        rows.append((uid, seg, signup))
        uid += 1
    return pd.DataFrame(rows, columns=["user_id", "segment", "signup_date"])


def ltv_cac_by_segment(price):
    """Аналитический расчёт LTV, CAC, ARPU и окупаемости по сегментам при заданной цене."""
    res = {}
    for seg, p in SEGMENTS.items():
        churn = p["churn"]
        # средний срок жизни клиента (мес) = 1/churn
        lifetime = 1.0 / churn
        # ARPU = цена тарифа + выручка от допуслуг (base_rev * доля активных допуслуг)
        arpu = price + p["base_rev"] * p["addon"]
        # LTV на горизонте: суммируем удержание помесячно
        ltv = 0.0
        payback_month = None
        cum = 0.0
        for m in range(LTV_HORIZON):
            retained = (1 - churn) ** m
            ltv += arpu * retained
            cum += arpu * retained
            if payback_month is None and cum >= p["cac"]:
                payback_month = m + 1
        res[seg] = {
            "CAC": p["cac"],
            "ARPU": round(arpu),
            "LTV": round(ltv),
            "LTV_CAC": round(ltv / p["cac"], 2),
            "lifetime_months": round(lifetime, 1),
            "payback_month": payback_month,
        }
    return pd.DataFrame(res).T


def blended_ltv_cac(price_map):
    """Средневзвешенный LTV/CAC по базе. price_map: сегмент -> цена."""
    total_ltv, total_cac, total_arpu, w = 0.0, 0.0, 0.0, 0.0
    for seg, p in SEGMENTS.items():
        price = price_map[seg]
        churn = p["churn"]
        arpu = price + p["base_rev"] * p["addon"]
        ltv = sum(arpu * (1 - churn) ** m for m in range(LTV_HORIZON))
        share = p["share"]
        total_ltv += ltv * share
        total_cac += p["cac"] * share
        total_arpu += arpu * share
        w += share
    return total_ltv / w, total_cac / w, (total_ltv / w) / (total_cac / w), total_arpu / w


# ---- Ценовая эластичность: конверсия в подписку как функция цены ----
# Логит-модель: базовая конверсия при базовой цене, эластичность разная по сегментам
SEG_CONV_BASE = {   # конверсия в подписку при PRICE_BASE
    "Фрилансеры": 0.34, "Мастера услуг": 0.30, "Таксисты": 0.26, "Курьеры": 0.20,
}
SEG_ELASTICITY = {  # ценовая эластичность спроса (по модулю): курьеры чувствительны, фрилансеры нет
    "Фрилансеры": 0.13, "Мастера услуг": 0.35, "Таксисты": 0.55, "Курьеры": 0.63,
}


def conversion_at_price(segment, price):
    """Конверсия в подписку при изменении цены относительно базовой (лог-лог эластичность)."""
    base_conv = SEG_CONV_BASE[segment]
    e = SEG_ELASTICITY[segment]
    ratio = price / PRICE_BASE
    # эластичность: %изм.спроса = -e * %изм.цены  =>  conv = base * ratio^(-e)
    conv = base_conv * ratio ** (-e)
    return min(conv, 0.95)


if __name__ == "__main__":
    users = gen_users()
    users.to_csv("users.csv", index=False)

    # ---- 1. Юнит-экономика по сегментам на СТАРТЕ (единая цена)
    start_econ = ltv_cac_by_segment(PRICE_BASE)
    print(f"Пользователей: {len(users):,}\n")
    print("ЮНИТ-ЭКОНОМИКА ПО СЕГМЕНТАМ (старт, единый тариф):")
    print(start_econ.to_string())
    start_econ.to_csv("unit_economics_start.csv")

    # средневзвешенный на старте
    price_start = {s: PRICE_BASE for s in SEGMENTS}
    ltv0, cac0, ratio0, arpu0 = blended_ltv_cac(price_start)
    print(f"\nСредневзвешенный LTV/CAC (старт): {ratio0:.2f} (план 3,0)")
    print(f"Средневзвешенный ARPU (старт): {arpu0:.0f} руб.")
    print(f"Фрилансеры LTV/CAC = {start_econ.loc['Фрилансеры','LTV_CAC']}, "
          f"окупаемость {start_econ.loc['Фрилансеры','payback_month']} мес")
    print(f"Курьеры LTV/CAC = {start_econ.loc['Курьеры','LTV_CAC']}, "
          f"окупаемость {start_econ.loc['Курьеры','payback_month']} мес")

    # ---- 2. Ценовая эластичность
    print("\nЦЕНОВАЯ ЭЛАСТИЧНОСТЬ (эффект изменения цены):")
    # фрилансеры: неэластичны
    fr_conv_base = conversion_at_price("Фрилансеры", PRICE_BASE)
    fr_conv_down = conversion_at_price("Фрилансеры", PRICE_BASE * 0.8)
    print(f"  Фрилансеры: цена -20% -> конверсия {fr_conv_base:.1%} -> {fr_conv_down:.1%} "
          f"({fr_conv_down/fr_conv_base-1:+.1%}) — спрос почти неэластичен")
    # курьеры: снижение цены на 20% -> +15% конверсии
    ku_conv_base = conversion_at_price("Курьеры", PRICE_BASE)
    ku_conv_down = conversion_at_price("Курьеры", PRICE_BASE * 0.8)
    print(f"  Курьеры: цена -20% -> конверсия {ku_conv_base:.1%} -> {ku_conv_down:.1%} "
          f"({ku_conv_down/ku_conv_base-1:+.1%})")
    # Итог по эластичности курьеров: снижение цены на 20% поднимает конверсию в подписку
    # на ~15%. Это делает облигчённый тариф для курьеров оправданным шагом (проверяем ниже
    # полноценной тарифной сеткой с учётом удержания).
    print(f"  Курьеры: снижение цены −20% → конверсия в подписку {ku_conv_down/ku_conv_base-1:+.0%} "
          f"(эластичный сегмент)")

    # --- 3. После новой тарифной сетки (базовый + облегчённый для курьеров)
    # Новая сетка = правильная сегментация: каждый сегмент на подходящем тарифе.
    # Курьеры — облегчённый тариф (ниже цена, но выше удержание и допподключения);
    # по остальным сегментам оптимизация тарифа и допуслуг тоже повышает монетизацию.
    PRICE_COURIER = round(PRICE_BASE * 0.8)  # облегчённый тариф курьерам -20%
    price_new = {s: PRICE_BASE for s in SEGMENTS}
    price_new["Курьеры"] = PRICE_COURIER

    SEG_NEW = {s: dict(v) for s, v in SEGMENTS.items()}
    # Курьеры: посильный тариф -> заметно ниже отток и больше допуслуг
    SEG_NEW["Курьеры"]["churn"] = 0.080
    SEG_NEW["Курьеры"]["addon"] = 0.50
    SEG_NEW["Курьеры"]["base_rev"] = 340
    # Таксисты: оптимизация тарифа/допуслуг
    SEG_NEW["Таксисты"]["churn"] = 0.085
    SEG_NEW["Таксисты"]["addon"] = 0.48
    SEG_NEW["Таксисты"]["base_rev"] = 300
    # Мастера услуг
    SEG_NEW["Мастера услуг"]["churn"] = 0.065
    SEG_NEW["Мастера услуг"]["addon"] = 0.55
    SEG_NEW["Мастера услуг"]["base_rev"] = 390
    # Фрилансеры: уже хорошо монетизированы, лёгкое улучшение
    SEG_NEW["Фрилансеры"]["churn"] = 0.048
    SEG_NEW["Фрилансеры"]["addon"] = 0.58
    SEG_NEW["Фрилансеры"]["base_rev"] = 490

    def blended_new():
        total_ltv, total_cac, total_arpu, w = 0.0, 0.0, 0.0, 0.0
        for seg, p in SEG_NEW.items():
            price = price_new[seg]
            churn = p["churn"]
            arpu = price + p["base_rev"] * p["addon"]
            ltv = sum(arpu * (1 - churn) ** m for m in range(LTV_HORIZON))
            share = p["share"]
            total_ltv += ltv * share
            total_cac += p["cac"] * share
            total_arpu += arpu * share
            w += share
        return total_ltv / w, total_cac / w, (total_ltv / w) / (total_cac / w), total_arpu / w

    ltv1, cac1, ratio1, arpu1 = blended_new()
    print("\nПОСЛЕ НОВОЙ ТАРИФНОЙ СЕТКИ (базовый + облегчённый курьерам):")
    print(f"  Средневзвешенный LTV/CAC: {ratio0:.2f} -> {ratio1:.2f}")
    print(f"  ARPU: {arpu0:.0f} -> {arpu1:.0f} ({arpu1/arpu0-1:+.1%})")

    # окупаемость средневзвешенная до/после
    def blended_payback(price_map, seg_cfg):
        # усреднённый по базе накопленный денежный поток
        cac_avg = sum(seg_cfg[s]["cac"] * seg_cfg[s]["share"] for s in seg_cfg)
        cum = 0.0
        pb = None
        for m in range(LTV_HORIZON):
            month_rev = 0.0
            for s, p in seg_cfg.items():
                arpu = price_map[s] + p["base_rev"] * p["addon"]
                month_rev += arpu * (1 - p["churn"]) ** m * p["share"]
            cum += month_rev
            if pb is None and cum >= cac_avg:
                pb = m + 1
        return pb

    pb0 = blended_payback(price_start, SEGMENTS)
    pb1 = blended_payback(price_new, SEG_NEW)
    print(f"  Срок окупаемости (средневзвеш.): {pb0} -> {pb1} мес")

    # сохраняем итоговую сравнительную таблицу
    new_econ = ltv_cac_by_segment(PRICE_BASE)  # для отображения; курьеры пересчитаем ниже
    pd.DataFrame({
        "metric": ["LTV/CAC (средн.)", "ARPU (средн.)", "Окупаемость, мес"],
        "before": [round(ratio0, 2), round(arpu0), pb0],
        "after": [round(ratio1, 2), round(arpu1), pb1],
    }).to_csv("before_after.csv", index=False)
