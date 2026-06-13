"""
Генератор синтетического датасета для кейса JoinPay (интернет-эквайринг).

Данные клиента находятся под NDA, поэтому для публичной демонстрации
воссоздана СТРУКТУРА данных с правдоподобными закономерностями:

1. Рост регистраций ~x3 после раунда инвестиций (апрель 2025)
2. Деградация ручного онбординга под нагрузкой: время интеграции растёт
3. Отток ~34% мерчантов на этапе технической интеграции
4. Скорость старта интеграции после верификации критична:
   <=2 дней -> ~75% доходят до первого платежа, >7 дней -> ~30%
5. Пилот персонального сопровождения (с сентября 2025): +~21% к конверсии
6. Отраслевая разница: магазины одежды ~x3 транзакций к электронике
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2025-11-30")

INDUSTRIES = {
    "Одежда и обувь": 0.26,
    "Электроника": 0.18,
    "Красота и здоровье": 0.14,
    "Товары для дома": 0.12,
    "Продукты питания": 0.10,
    "Услуги и подписки": 0.12,
    "Хобби и DIY": 0.08,
}

CHANNELS = {
    "Отдел продаж": 0.38,
    "Контекстная реклама": 0.27,
    "Партнёрские программы": 0.20,
    "Органика / сайт": 0.15,
}


def month_volume(month: pd.Timestamp) -> int:
    """Регистраций в месяц: ~350 до раунда, рост к ~1050 после апреля."""
    base = 350
    if month < pd.Timestamp("2025-04-01"):
        vol = base * RNG.normal(1.0, 0.06)
    else:
        # плавный разгон маркетинга после раунда
        k = min(3.0, 1.0 + 0.55 * ((month.month - 3)))
        vol = base * k * RNG.normal(1.0, 0.05)
    return int(vol)


def gen_merchants() -> pd.DataFrame:
    rows = []
    mid = 1
    months = pd.date_range(START, END, freq="MS")
    for m in months:
        n = month_volume(m)
        days_in_month = (m + pd.offsets.MonthEnd(0)).day
        for _ in range(n):
            reg = m + pd.Timedelta(
                days=int(RNG.integers(0, days_in_month)),
                hours=int(RNG.integers(8, 22)),
            )
            rows.append((mid, reg))
            mid += 1
    df = pd.DataFrame(rows, columns=["merchant_id", "registered_at"])
    df["industry"] = RNG.choice(
        list(INDUSTRIES), size=len(df), p=list(INDUSTRIES.values())
    )
    df["channel"] = RNG.choice(
        list(CHANNELS), size=len(df), p=list(CHANNELS.values())
    )
    return df


def overload_factor(reg: pd.Timestamp) -> float:
    """Перегрузка менеджеров онбординга: растёт после масштабирования."""
    if reg < pd.Timestamp("2025-04-01"):
        return 0.0
    months_after = (reg.year - 2025) * 12 + reg.month - 4
    return min(1.0, 0.25 + 0.16 * months_after)


def simulate_funnel(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # --- Верификация (KYC): проходит ~88%, занимает 1-4 дня
    verified = RNG.random(n) < 0.88
    ver_delay = RNG.integers(1, 5, size=n)
    df["verified_at"] = pd.NaT
    df.loc[verified, "verified_at"] = df.loc[verified, "registered_at"] + (
        pd.to_timedelta(ver_delay[verified], unit="D")
    )

    # --- Пилот персонального сопровождения: с 1 сентября, рандомизация 50/50
    pilot_eligible = df["verified_at"] >= pd.Timestamp("2025-09-01")
    df["onboarding_group"] = "стандарт"
    pilot_mask = pilot_eligible & (RNG.random(n) < 0.5)
    df.loc[pilot_mask, "onboarding_group"] = "персональное сопровождение"

    # --- Задержка старта интеграции после верификации
    # Без сопровождения: под нагрузкой менеджеры "теряют" мерчантов
    delays = np.full(n, np.nan)
    for i, row in enumerate(df.itertuples(index=False)):
        if pd.isna(row.verified_at):
            continue
        ovl = overload_factor(row.registered_at)
        fast_p = 0.55 - 0.30 * ovl
        if row.onboarding_group == "персональное сопровождение":
            # сопровождение сокращает задержку выхода менеджера,
            # но не устраняет её полностью
            fast_p += 0.22
        if RNG.random() < fast_p:
            d = RNG.integers(0, 3)
        else:
            d = RNG.integers(3, 21) + int(10 * ovl * RNG.random())
        delays[i] = d
    df["integration_start_delay_days"] = delays
    df["integration_started_at"] = df["verified_at"] + pd.to_timedelta(
        df["integration_start_delay_days"], unit="D"
    )

    # --- Завершение интеграции: вероятность зависит от скорости старта
    # и от сопровождения (вшитый отток ~34% на этом шаге)
    completed = np.zeros(n, dtype=bool)
    for i, row in enumerate(df.itertuples(index=False)):
        if pd.isna(row.integration_started_at):
            continue
        d = row.integration_start_delay_days
        if d <= 2:
            p = 0.86
        elif d <= 7:
            p = 0.62
        else:
            p = 0.50
        if row.onboarding_group == "персональное сопровождение":
            p = min(0.97, p + 0.04)
        completed[i] = RNG.random() < p
    df["integration_completed_at"] = pd.NaT
    dur = RNG.integers(2, 12, size=n)
    df.loc[completed, "integration_completed_at"] = df.loc[
        completed, "integration_started_at"
    ] + pd.to_timedelta(dur[completed], unit="D")

    # --- Первый платёж: зависит от той же скорости старта
    paid = np.zeros(n, dtype=bool)
    for i, row in enumerate(df.itertuples(index=False)):
        if pd.isna(row.integration_completed_at):
            continue
        d = row.integration_start_delay_days
        if d <= 2:
            p = 0.875
        elif d <= 7:
            p = 0.62
        else:
            p = 0.60
        paid[i] = RNG.random() < p
    df["first_payment_at"] = pd.NaT
    pay_delay = RNG.integers(0, 6, size=n)
    df.loc[paid, "first_payment_at"] = df.loc[
        paid, "integration_completed_at"
    ] + pd.to_timedelta(pay_delay[paid], unit="D")

    return df


def simulate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Транзакции за первые 90 дней жизни мерчанта (для отраслевых когорт)."""
    base = {
        "Одежда и обувь": 540,
        "Электроника": 180,
        "Красота и здоровье": 320,
        "Товары для дома": 250,
        "Продукты питания": 410,
        "Услуги и подписки": 290,
        "Хобби и DIY": 160,
    }
    active = df[df["first_payment_at"].notna()].copy()
    mu = active["industry"].map(base).to_numpy(dtype=float)
    counts = RNG.lognormal(mean=np.log(mu), sigma=0.55)
    avg_check = {
        "Одежда и обувь": 3400,
        "Электроника": 12800,
        "Красота и здоровье": 2600,
        "Товары для дома": 4900,
        "Продукты питания": 1900,
        "Услуги и подписки": 1500,
        "Хобби и DIY": 2700,
    }
    chk = active["industry"].map(avg_check).to_numpy(dtype=float)
    tx = pd.DataFrame(
        {
            "merchant_id": active["merchant_id"].to_numpy(),
            "industry": active["industry"].to_numpy(),
            "tx_count_90d": counts.round().astype(int),
            "avg_check_rub": (chk * RNG.normal(1.0, 0.12, size=len(active)))
            .round(0)
            .astype(int),
        }
    )
    tx["gmv_90d_rub"] = tx["tx_count_90d"] * tx["avg_check_rub"]
    return tx


if __name__ == "__main__":
    merchants = gen_merchants()
    merchants = simulate_funnel(merchants)
    transactions = simulate_transactions(merchants)

    merchants.to_csv("merchants.csv", index=False)
    transactions.to_csv("transactions.csv", index=False)

    # --- Контрольная сводка для калибровки
    ver = merchants["verified_at"].notna()
    started = merchants["integration_started_at"].notna()
    comp = merchants["integration_completed_at"].notna()
    paid = merchants["first_payment_at"].notna()

    print(f"Мерчантов всего: {len(merchants)}")
    print(f"Верифицировано: {ver.mean():.1%}")
    drop_integration = 1 - comp[ver].mean()
    print(f"Отток на интеграции (от верифицированных): {drop_integration:.1%}")

    v = merchants[ver].copy()
    fast = v["integration_start_delay_days"] <= 2
    slow = v["integration_start_delay_days"] > 7
    print(f"Конверсия в платёж, старт <=2 дней: {v.loc[fast, 'first_payment_at'].notna().mean():.1%}")
    print(f"Конверсия в платёж, старт >7 дней: {v.loc[slow, 'first_payment_at'].notna().mean():.1%}")

    pilot = v[v["verified_at"] >= "2025-09-01"]
    g = pilot.groupby("onboarding_group")["first_payment_at"].apply(
        lambda s: s.notna().mean()
    )
    print("Пилот (сентябрь+):")
    print(g.to_string())
    if "персональное сопровождение" in g and "стандарт" in g:
        uplift = g["персональное сопровождение"] / g["стандарт"] - 1
        print(f"Uplift пилота: {uplift:+.1%}")
