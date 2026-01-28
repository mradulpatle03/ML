import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path

np.random.seed(42)
random.seed(42)

def arrival_rate(hour):
    """Time-dependent arrival intensity"""
    if 9 <= hour < 11:
        return 10
    elif 11 <= hour < 14:
        return 25  # lunch peak
    elif 14 <= hour < 17:
        return 18
    else:
        return 6

def generate_queue_data(days=60):
    data = []
    start_date = datetime.now() - timedelta(days=days)
    queue_length = 0

    for day in range(days):
        base_date = start_date + timedelta(days=day)
        day_of_week = base_date.weekday()

        for hour in range(9, 18):
            lam = arrival_rate(hour)

            # Weekend adjustment
            if day_of_week >= 5:
                lam *= 0.6

            arrivals = np.random.poisson(lam)

            # Dynamic counters (staff shifts)
            if hour in [12, 13]:
                active_counters = random.randint(2, 3)  # lunch break
            else:
                active_counters = random.randint(3, 6)

            for _ in range(arrivals):
                arrival_time = base_date + timedelta(
                    hours=hour,
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )

                # Long-tail service time (realistic)
                service_time = np.random.lognormal(mean=1.4, sigma=0.4)

                # Queue evolution
                service_capacity = active_counters * (60 / service_time)
                queue_length = max(
                    0,
                    queue_length + 1 - service_capacity / arrivals
                )

                # Occasional anomaly (system slowdown)
                anomaly_factor = 1
                if random.random() < 0.03:
                    anomaly_factor = random.uniform(1.5, 2.5)

                wait_time = max(
                    0,
                    (queue_length / max(active_counters, 1)) * service_time * anomaly_factor
                )

                data.append([
                    arrival_time,
                    day_of_week,
                    hour,
                    int(queue_length),
                    active_counters,
                    round(service_time, 2),
                    round(wait_time, 2)
                ])

    columns = [
        "timestamp",
        "day_of_week",
        "hour",
        "customers_in_queue",
        "active_counters",
        "service_time",
        "wait_time"
    ]

    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    df = generate_queue_data(60)

    # Resolve project root (ML01)
    BASE_DIR = Path(__file__).resolve().parent.parent

    data_dir = BASE_DIR / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(data_dir / "queue_data.csv", index=False)
    print("Queue data generated successfully!")