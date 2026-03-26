"""City correlation matrix computed from model forecast deviations."""
from __future__ import annotations

import logging
from typing import Any

from weather_edge.config import CITIES
from weather_edge.models.enums import City

logger = logging.getLogger(__name__)


def compute_correlation_matrix(city_data: dict[str, Any]) -> dict:
    """Compute pairwise correlation between city forecast deviations.

    For each city, compute how much each model deviates from the city's
    model consensus (mean). If city A and city B both have models that
    predict warmer-than-consensus, they are positively correlated.

    Args:
        city_data: The cities dict from dashboard state, containing
                   model forecasts per city.

    Returns:
        Dict with 'cities' (list of city IDs), 'matrix' (2D correlation grid),
        and 'pairs' (notable correlations).
    """
    # Extract deviation vectors: for each city, how each model deviates from mean
    city_deviations: dict[str, dict[str, float]] = {}

    for city_id, info in city_data.items():
        models = info.get("models", {})
        if not models:
            continue

        # Get temp_max values from each model
        values = {}
        for model_name, model_data in models.items():
            t = model_data.get("temp_max_c")
            if t is not None:
                values[model_name] = t

        if len(values) < 2:
            continue

        mean_val = sum(values.values()) / len(values)
        # Store deviation from mean for each model
        deviations = {m: v - mean_val for m, v in values.items()}
        city_deviations[city_id] = deviations

    city_ids = sorted(city_deviations.keys())
    n = len(city_ids)

    if n < 2:
        return {"cities": city_ids, "matrix": [], "pairs": []}

    # Compute pairwise correlation using shared models
    matrix: list[list[float | None]] = []
    notable_pairs: list[dict] = []

    for i, c1 in enumerate(city_ids):
        row: list[float | None] = []
        devs1 = city_deviations[c1]

        for j, c2 in enumerate(city_ids):
            if i == j:
                row.append(1.0)
                continue

            devs2 = city_deviations[c2]

            # Find shared models
            shared = set(devs1.keys()) & set(devs2.keys())
            if len(shared) < 3:
                row.append(None)
                continue

            # Pearson correlation
            vals1 = [devs1[m] for m in shared]
            vals2 = [devs2[m] for m in shared]

            mean1 = sum(vals1) / len(vals1)
            mean2 = sum(vals2) / len(vals2)

            cov = sum((a - mean1) * (b - mean2) for a, b in zip(vals1, vals2))
            std1 = (sum((a - mean1) ** 2 for a in vals1)) ** 0.5
            std2 = (sum((b - mean2) ** 2 for b in vals2)) ** 0.5

            if std1 < 1e-10 or std2 < 1e-10:
                # No variance in one city's deviations
                row.append(0.0)
            else:
                corr = cov / (std1 * std2)
                # Clamp to [-1, 1]
                corr = max(-1.0, min(1.0, corr))
                row.append(round(corr, 3))

                # Track notable pairs (only upper triangle)
                if i < j and abs(corr) > 0.5:
                    c1_name = CITIES.get(City(c1), None)
                    c2_name = CITIES.get(City(c2), None)
                    notable_pairs.append({
                        "city_a": c1,
                        "city_b": c2,
                        "city_a_name": c1_name.name if c1_name else c1,
                        "city_b_name": c2_name.name if c2_name else c2,
                        "correlation": row[-1],
                        "type": "correlated" if corr > 0 else "anti-correlated",
                    })

        matrix.append(row)

    # Sort notable pairs by absolute correlation descending
    notable_pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    # Add city names for display
    city_labels = []
    for cid in city_ids:
        try:
            cfg = CITIES[City(cid)]
            city_labels.append({"id": cid, "name": cfg.name})
        except (ValueError, KeyError):
            city_labels.append({"id": cid, "name": cid.upper()})

    return {
        "cities": city_labels,
        "matrix": matrix,
        "pairs": notable_pairs[:20],  # Top 20 notable pairs
    }
