import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path

np.random.seed(42)

def generate_queue_data(days=30):
    data = []
    start_date = datetime.now() - timedelta(days=days)

    for day in range(days):
        current_day = start_date + timedelta(days=day)
        day_of_week = current_day.weekday()

        # Peak multiplier
        peak_factor = 1.5 if day_of_week < 5 else 1.1

        for hour in range(9, 18):  # Service hours
            arrivals = np.random.poisson(lam=20 * peak_factor)

            active_counters = random.randint(2, 6)
            avg_service_time = np.random.uniform(3, 8)

            queue_length = max(0, arrivals - active_counters * 5)

            for _ in range(arrivals):
                wait_time = max(
                    0,
                    (queue_length / active_counters) * avg_service_time
                    + np.random.normal(0, 2)
                )

                data.append([
                    current_day + timedelta(hours=hour),
                    day_of_week,
                    hour,
                    queue_length,
                    active_counters,
                    avg_service_time,
                    round(wait_time, 2)
                ])

    columns = [
        "timestamp",
        "day_of_week",
        "hour",
        "customers_in_queue",
        "active_counters",
        "avg_service_time",
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