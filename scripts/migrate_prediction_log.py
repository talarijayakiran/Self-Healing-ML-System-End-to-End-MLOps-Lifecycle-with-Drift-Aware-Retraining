from __future__ import annotations

import csv
import uuid
from pathlib import Path


LOG_PATH = Path(
    "data/monitoring/predictions.csv"
)

BACKUP_PATH = Path(
    "data/monitoring/predictions.backup.csv"
)

TEMP_PATH = Path(
    "data/monitoring/predictions.migrated.csv"
)


CANONICAL_COLUMNS = [
    "timestamp",
    "request_id",
    "model_version",
    "date",
    "category",
    "region",
    "price",
    "promo",
    "prediction",
]


def migrate() -> None:

    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {LOG_PATH}"
        )

    if not BACKUP_PATH.exists():
        raise FileNotFoundError(
            "Backup file does not exist. "
            "Create predictions.backup.csv first."
        )

    migrated_rows = []

    with LOG_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.reader(file)

        header = next(reader)

        print(
            f"Original header: {header}"
        )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):

            if not row:
                continue

            # Legacy 7-column format
            if len(row) == 7:

                (
                    date,
                    category,
                    region,
                    price,
                    promo,
                    prediction,
                    timestamp,
                ) = row

                migrated_rows.append(
                    [
                        timestamp,
                        f"legacy-{uuid.uuid4()}",
                        "unknown",
                        date,
                        category,
                        region,
                        price,
                        promo,
                        prediction,
                    ]
                )

            # New 9-column format
            elif len(row) == 9:

                migrated_rows.append(row)

            else:

                raise ValueError(
                    "Unexpected row width at "
                    f"line {line_number}: "
                    f"{len(row)} columns"
                )

    with TEMP_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            CANONICAL_COLUMNS
        )

        writer.writerows(
            migrated_rows
        )

    TEMP_PATH.replace(
        LOG_PATH
    )

    print(
        f"Migration complete: "
        f"{len(migrated_rows)} rows"
    )

    print(
        "Canonical schema:"
    )

    print(
        ",".join(CANONICAL_COLUMNS)
    )


if __name__ == "__main__":
    migrate()