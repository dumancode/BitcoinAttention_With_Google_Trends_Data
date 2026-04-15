from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.api import VAR


RANDOM_SEED = 42
FIGURE_DIR = Path("figures")
OUTPUT_DIR = Path("outputs")


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def generate_demo_attention_data() -> pd.DataFrame:
    """Create a reproducible Bitcoin/Google-Trends-like dataset for the public demo."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2012-07-01", "2017-06-30", freq="D")
    n = len(dates)

    returns = np.zeros(n)
    log_volume = np.zeros(n)
    attention = np.zeros(n)
    volatility = np.zeros(n)

    returns[0] = rng.normal(0.001, 0.035)
    log_volume[0] = 0.0
    attention[0] = 0.0
    volatility[0] = 0.035

    for t in range(1, n):
        shock = rng.normal(0, 1)
        volatility[t] = (
            0.50 * volatility[t - 1]
            + 0.18 * abs(returns[t - 1])
            + 0.004 * abs(attention[t - 1])
            + rng.normal(0, 0.004)
        )
        volatility[t] = np.clip(volatility[t], 0.005, 0.075)
        returns[t] = 0.0006 + 0.10 * returns[t - 1] + volatility[t] * shock
        returns[t] = np.clip(returns[t], -0.18, 0.18)
        attention[t] = (
            0.58 * attention[t - 1]
            + 2.6 * abs(returns[t - 1])
            + 1.2 * volatility[t - 1]
            + rng.normal(0, 0.08)
        )
        attention[t] = np.clip(attention[t], -2.5, 2.5)
        log_volume[t] = (
            0.68 * log_volume[t - 1]
            + 0.10 * attention[t - 1]
            + 1.1 * abs(returns[t])
            + rng.normal(0, 0.08)
        )
        log_volume[t] = np.clip(log_volume[t], -2.0, 2.0)

    close = 10 * np.exp(np.cumsum(returns))
    realized_volatility = pd.Series(returns).rolling(7).std().bfill() * np.sqrt(365)
    search_count = 50 + 45 * (attention - attention.min()) / (attention.max() - attention.min())
    search_count = np.clip(search_count + rng.normal(0, 5, n), 1, 100)
    volume = np.exp(13 + log_volume)

    data = pd.DataFrame(
        {
            "Date": dates,
            "close": close,
            "volume": volume,
            "SearchCount": search_count,
            "log-Returns": returns,
            "RV": realized_volatility,
        }
    )

    return add_transformed_variables(data)


def add_transformed_variables(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    data["SearchCount"] = pd.to_numeric(data["SearchCount"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    data["RV"] = pd.to_numeric(data["RV"], errors="coerce")

    data.loc[data["SearchCount"] <= 0, "SearchCount"] = np.nan
    data.loc[data["volume"] <= 0, "volume"] = np.nan
    data.loc[data["RV"] <= 0, "RV"] = np.nan

    data["stan-SQ"] = data["SearchCount"] / data["SearchCount"].mean(skipna=True)
    data["log-SQ"] = np.log(data["stan-SQ"])
    data["log-RV"] = np.log(data["RV"])
    data["log-VO"] = np.log(data["volume"])

    return data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    variables = ["log-SQ", "log-RV", "log-VO", "log-Returns"]
    stats = data[variables].agg(["mean", "std", "min", "max", "skew", "kurt"]).T
    stats.columns = ["mean", "std", "min", "max", "skewness", "kurtosis"]
    return stats


def fit_var_models(data: pd.DataFrame, lags: int = 7) -> dict:
    model_specs = {
        "attention_volatility": ["log-SQ", "log-RV"],
        "attention_volume": ["log-SQ", "log-VO"],
        "full_system": ["log-SQ", "log-RV", "log-VO", "log-Returns"],
    }

    results = {}
    for name, columns in model_specs.items():
        model_data = data.set_index("Date")[columns].dropna()
        model_data = model_data.asfreq("D")
        results[name] = VAR(model_data).fit(lags)
    return results


def granger_summary(var_results: dict) -> pd.DataFrame:
    tests = [
        ("attention_volatility", "log-RV", ["log-SQ"]),
        ("attention_volatility", "log-SQ", ["log-RV"]),
        ("attention_volume", "log-VO", ["log-SQ"]),
        ("attention_volume", "log-SQ", ["log-VO"]),
        ("full_system", "log-Returns", ["log-SQ"]),
        ("full_system", "log-SQ", ["log-Returns"]),
        ("full_system", "log-RV", ["log-SQ", "log-VO", "log-Returns"]),
    ]

    rows = []
    for model_name, caused, causing in tests:
        result = var_results[model_name].test_causality(caused=caused, causing=causing, kind="f")
        rows.append(
            {
                "model": model_name,
                "causing": " + ".join(causing),
                "caused": caused,
                "F statistic": result.test_statistic,
                "p-value": result.pvalue,
                "significance": significance_stars(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def plot_attention_price(data: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(data["Date"], data["close"], color="#f2b66d", label="Bitcoin close")
    ax1.set_ylabel("Synthetic BTC close")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(data["Date"], data["SearchCount"], color="#89a7ff", alpha=0.75, label="Google Trends attention")
    ax2.set_ylabel("Search attention index")

    ax1.set_title("Bitcoin Price and Search Attention")
    ax1.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bitcoin_attention_price.png", dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    corr = data[["log-SQ", "log-RV", "log-VO", "log-Returns"]].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Between Attention and Bitcoin Variables")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "bitcoin_attention_correlation.png", dpi=180)
    plt.close()


def plot_irf(var_results: dict, steps: int = 40) -> None:
    irf = var_results["full_system"].irf(steps)
    fig = irf.plot(orth=False)
    fig.set_size_inches(12, 9)
    fig.suptitle("Impulse Response Functions for Bitcoin Attention VAR", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bitcoin_attention_irf.png", dpi=180)
    plt.close(fig)


def run_subsample_analysis(data: pd.DataFrame) -> pd.DataFrame:
    split_date = pd.Timestamp("2013-10-28")
    ranges = {
        "early_period": data[data["Date"] < split_date],
        "later_period": data[data["Date"] >= split_date],
    }

    rows = []
    for period, frame in ranges.items():
        results = fit_var_models(frame, lags=5)
        summary = granger_summary(results)
        summary.insert(0, "period", period)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = generate_demo_attention_data()
    stats = descriptive_statistics(data)
    var_results = fit_var_models(data)
    granger = granger_summary(var_results)
    subsamples = run_subsample_analysis(data)

    data.to_csv(OUTPUT_DIR / "demo_bitcoin_attention_data.csv", index=False)
    stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")
    granger.to_csv(OUTPUT_DIR / "granger_causality_summary.csv", index=False)
    subsamples.to_csv(OUTPUT_DIR / "subsample_granger_summary.csv", index=False)

    plot_attention_price(data)
    plot_correlation_heatmap(data)
    plot_irf(var_results)

    print("Descriptive statistics")
    print(stats.round(4).to_string())
    print("\nGranger causality summary")
    print(granger.round(4).to_string(index=False))
    print("\nSaved outputs under outputs/ and figures/.")


if __name__ == "__main__":
    main()
