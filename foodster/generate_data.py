"""
Генератор синтетического датасета для кейса Foodster (доставка еды, мобильное приложение).

Данные клиента под NDA — структура воссоздана синтетически с правдоподобными
закономерностями из реального проекта:

1. Воронка приложения: установка -> регистрация -> меню -> корзина -> адрес -> заказ
2. Отток ~32% пользователей на шаге ввода адреса доставки (узкое место)
3. A/B-тест улучшений шага адреса (геолокация, подсказки, сохранение адресов):
   конверсия в первый заказ +~14% в тест-группе
4. Каналы привлечения с разным Retention:
   локальные telegram-каналы удерживают в ~2.1 раза лучше таргета
5. Юнит-экономика по каналам: органика окупается на 3-й месяц, платная реклама на 5-й
6. CAC по каналам: после перераспределения бюджета средневзвешенный CAC -~18%
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

START = pd.Timestamp("2025-04-01")
END = pd.Timestamp("2025-11-30")

# Каналы привлечения: доля трафика, CAC (руб.), Retention-фактор, множитель частоты заказов
# Логика: Telegram-каналы и органика — дешёвые и лояльные; платная реклама — дорогая и хуже удерживает
CHANNELS = {
    "Telegram (локальные каналы)": {"share": 0.22, "cac": 240, "ret": 0.80, "freq": 1.05},
    "Органический поиск":          {"share": 0.18, "cac": 210, "ret": 0.76, "freq": 1.05},
    "Таргетированная реклама":     {"share": 0.34, "cac": 540, "ret": 0.605, "freq": 0.95},
    "Реферальная программа":       {"share": 0.14, "cac": 290, "ret": 0.72, "freq": 1.00},
    "Контекстная реклама":         {"share": 0.12, "cac": 470, "ret": 0.65, "freq": 0.95},
}


def gen_users(n=14000):
    months = pd.date_range(START, END, freq="MS")
    # лёгкий рост установок к лету
    weights = np.linspace(1.0, 1.6, len(months))
    weights /= weights.sum()
    month_counts = RNG.multinomial(n, weights)

    rows = []
    uid = 1
    for m, cnt in zip(months, month_counts):
        days = (m + pd.offsets.MonthEnd(0)).day
        for _ in range(cnt):
            inst = m + pd.Timedelta(
                days=int(RNG.integers(0, days)),
                hours=int(RNG.integers(7, 23)),
            )
            rows.append((uid, inst))
            uid += 1
    df = pd.DataFrame(rows, columns=["user_id", "installed_at"])
    df["channel"] = RNG.choice(
        list(CHANNELS), size=len(df),
        p=[c["share"] for c in CHANNELS.values()],
    )
    # A/B: с 1 июля рандомизация 50/50 для новых пользователей
    df["ab_group"] = "control"
    ab_eligible = df["installed_at"] >= pd.Timestamp("2025-07-01")
    test_mask = ab_eligible & (RNG.random(len(df)) < 0.5)
    df.loc[test_mask, "ab_group"] = "test"
    df["ab_eligible"] = ab_eligible
    return df


def simulate_funnel(df):
    n = len(df)
    r = lambda: RNG.random(n)

    # Шаги воронки до адреса — стабильные конверсии
    reg = r() < 0.82                       # установка -> регистрация
    menu = reg & (r() < 0.88)              # -> открыл меню
    cart = menu & (r() < 0.64)             # -> собрал корзину

    # Шаг адреса — узкое место. В control теряется ~32%.
    # В test улучшения (геолокация/подсказки/сохранение) снижают отток.
    addr_pass_prob = np.where(df["ab_group"].to_numpy() == "test", 0.81, 0.68)
    addr = cart & (r() < addr_pass_prob)   # -> ввёл адрес

    # Адрес -> оформил первый заказ (высокая конверсия после адреса)
    order = addr & (r() < 0.86)

    df["registered"] = reg
    df["opened_menu"] = menu
    df["built_cart"] = cart
    df["entered_address"] = addr
    df["first_order"] = order
    return df


def simulate_retention(df):
    """Помесячный Retention первого заказа по каналам (горизонт 6 мес)."""
    buyers = df[df["first_order"]].copy()
    records = []
    for row in buyers.itertuples(index=False):
        base = CHANNELS[row.channel]["ret"]
        # кривая удержания: доля активных на месяц k
        for k in range(0, 7):
            if k == 0:
                active = True
            else:
                # кривая удержания: степенное затухание от месяца к месяцу,
                # качество канала задаёт долгосрочный уровень
                p = base ** (1 + 0.55 * k)
                active = RNG.random() < p
            records.append((row.user_id, row.channel, k, active))
    ret = pd.DataFrame(records, columns=["user_id", "channel", "month_k", "active"])
    return ret


def build_unit_economics(df, ret):
    """Юнит-экономика по каналам: CAC, LTV, срок окупаемости."""
    AOV = 1450          # средний чек, руб.
    MARGIN = 0.18       # маржа с заказа
    orders_m0 = 0.78    # базовое число заказов в первый месяц у активного

    rows = []
    for ch, cfg in CHANNELS.items():
        sub = ret[ret["channel"] == ch]
        if sub.empty:
            continue
        # средняя доля активных по месяцам -> ожидаемые заказы
        active_by_k = sub.groupby("month_k")["active"].mean()
        cac = cfg["cac"]
        fmult = cfg["freq"]
        cum_margin = 0.0
        payback_month = None
        ltv_by_month = []
        for k in range(0, 7):
            share_active = active_by_k.get(k, 0.0)
            # прижившиеся пользователи заказывают чаще: рост частоты по месяцам
            freq = orders_m0 * fmult * (0.7 + 0.12 * k)
            orders = share_active * freq
            margin = orders * AOV * MARGIN
            cum_margin += margin
            ltv_by_month.append(cum_margin)
            if payback_month is None and cum_margin >= cac:
                payback_month = k
        rows.append({
            "channel": ch,
            "CAC": cac,
            "LTV_6m": round(ltv_by_month[-1]),
            "payback_month": payback_month,
            "share": cfg["share"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    users = gen_users()
    users = simulate_funnel(users)
    retention = simulate_retention(users)
    unit = build_unit_economics(users, retention)

    users.to_csv("users.csv", index=False)
    retention.to_csv("retention.csv", index=False)
    unit.to_csv("unit_economics.csv", index=False)

    # ---- Калибровочная сводка
    print(f"Пользователей: {len(users):,}")

    # 1. Отток на адресе (control)
    ctrl = users[users["ab_group"] == "control"]
    cart_ctrl = ctrl["built_cart"].sum()
    addr_ctrl = ctrl["entered_address"].sum()
    drop_addr = 1 - addr_ctrl / cart_ctrl
    print(f"Отток на шаге адреса (control): {drop_addr:.1%}")

    # 2. A/B: конверсия установка->первый заказ
    ab = users[users["ab_eligible"]]
    g = ab.groupby("ab_group").apply(
        lambda s: s["first_order"].sum() / len(s), include_groups=False
    )
    uplift = g["test"] / g["control"] - 1
    print(f"A/B CR control={g['control']:.1%}, test={g['test']:.1%}, uplift={uplift:+.1%}")

    # 3. Retention ratio TG vs таргет на месяц 3
    m3 = retention[retention["month_k"] == 3].groupby("channel")["active"].mean()
    tg = m3["Telegram (локальные каналы)"]
    tgt = m3["Таргетированная реклама"]
    print(f"Retention M3: TG={tg:.1%}, таргет={tgt:.1%}, ratio=x{tg/tgt:.1f}")

    # 4. Юнит-экономика
    print(unit[["channel", "CAC", "LTV_6m", "payback_month"]].to_string(index=False))

    # 5. CAC до/после перераспределения бюджета
    # было: доли по share; стало: +10 п.п. в TG/органику, -10 п.п. из таргета/контекста
    cac_before = sum(c["share"] * c["cac"] for c in CHANNELS.values())
    new_share = {
        "Telegram (локальные каналы)": 0.32, "Органический поиск": 0.24,
        "Таргетированная реклама": 0.20, "Реферальная программа": 0.18,
        "Контекстная реклама": 0.06,
    }
    cac_after = sum(new_share[k] * CHANNELS[k]["cac"] for k in CHANNELS)
    print(f"CAC взвеш.: было {cac_before:.0f}, стало {cac_after:.0f} "
          f"({cac_after/cac_before - 1:+.1%})")
