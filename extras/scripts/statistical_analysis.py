"""
Statistical analysis module for federated learning experimental results.

This module performs comprehensive statistical analysis on federated learning
experimental results, including Wilcoxon signed-rank tests and Friedman tests
to evaluate the significance of performance differences between models.

Classes:
    None

Functions:
    format_data: Formats experimental data for statistical analysis.

Constants:
    JSON_PATH: Path to the directory containing experimental results.
    INDOMAIN_PATH: Path to the in-domain experimental results JSON file.
    IMNET_PATH: Path to the ImageNet experimental results JSON file.
    MAIN_MODEL: Name of the main baseline model for comparison.

Statistical Tests:
    - Wilcoxon Signed-Rank Test: Non-parametric test for comparing paired samples
      to determine if there are statistically significant differences between
      the main model and other models across datasets.
    - Friedman Test: Non-parametric test for comparing multiple related samples
      to detect differences in treatments across multiple test attempts.

Usage:
    Run this module directly to perform statistical analysis on the experimental
    results stored in the extras/results directory.

Author: Matteo Caligiuri
        Francesco Barbato
"""

from pathlib import Path
import json
from typing import Dict, List, Tuple, Union

from scipy.stats import wilcoxon, friedmanchisquare


JSON_PATH = Path(__file__).parent.parent / "results"
INDOMAIN_PATH = JSON_PATH / "indomain.json"
IMNET_PATH = JSON_PATH / "imnet.json"
MAIN_MODEL = "FedPromo"


def format_data(
    data: Dict[str, Dict[str, List[float]]], out_type: str = "wilcoxon"
) -> Dict[str, Dict[str, Union[Tuple[List[float], List[float]], List[float]]]]:
    """
    Format the data from the JSON file into a structure suitable for statistical analysis.

    Args:
        data (Dict[str, Dict[str, List[float]]]): The raw data loaded from
            the JSON file, where each key is a dataset name and each value is a
            dictionary containing model names as keys and their corresponding metrics
            as lists of floats.
        out_type (str): The type of statistical test to prepare the data for.
            Supported values are "wilcoxon" and "friedman".

    Returns:
        Dict[str, Dict[str, Union[Tuple[List[float], List[float]], List[float]]]]:
            A dictionary where each key is a model name, and each value is another
            dictionary with keys "acc-1" and "acc-5". Each of these keys maps to a tuple
            containing two lists of floats: the first list contains the model's metrics
            across all datasets, and the second list contains the main model's metrics
            across the same datasets.
    """

    # Define the dataset_lists
    dts_lists = list(data)

    if out_type == "wilcoxon":
        return {
            model_name: {
                "acc-1": (
                    [data[dts_name][model_name][0] for dts_name in dts_lists]
                    + [data[dts_name][model_name][2] for dts_name in dts_lists],
                    [data[dts_name][MAIN_MODEL][0] for dts_name in dts_lists]
                    + [data[dts_name][MAIN_MODEL][2] for dts_name in dts_lists],
                ),
                "acc-5": (
                    [data[dts_name][model_name][1] for dts_name in dts_lists]
                    + [data[dts_name][model_name][3] for dts_name in dts_lists],
                    [data[dts_name][MAIN_MODEL][1] for dts_name in dts_lists]
                    + [data[dts_name][MAIN_MODEL][3] for dts_name in dts_lists],
                ),
            }
            for model_name in data[dts_lists[0]]
            if model_name != MAIN_MODEL
        }
    elif out_type == "friedman":
        return {
            model_name: {
                "acc-1": [data[dts_name][model_name][0] for dts_name in dts_lists]
                + [data[dts_name][model_name][2] for dts_name in dts_lists],
                "acc-5": [data[dts_name][model_name][1] for dts_name in dts_lists]
                + [data[dts_name][model_name][3] for dts_name in dts_lists],
            }
            for model_name in data[dts_lists[0]]
        }
    else:
        raise ValueError(
            f"Unknown type: {type}. Supported types are 'wilcoxon' and 'friedman'."
        )


if __name__ == "__main__":
    print("🚀 Starting Statistical Analysis...")
    print("=" * 60)

    # Load the jsons
    print("📂 Loading experimental data files...")
    with open(INDOMAIN_PATH, "r", encoding="utf-8") as f:
        indomain_data = json.load(f)[0]
    with open(IMNET_PATH, "r", encoding="utf-8") as f:
        imnet_data = json.load(f)[0]
    print("✅ Data files loaded successfully")

    # Format the data for the wilcoxon test
    print("\n🔧 Formatting data for Wilcoxon signed-rank test...")
    formatted_indomain = format_data(indomain_data, out_type="wilcoxon")
    formatted_imnet = format_data(imnet_data, out_type="wilcoxon")

    # Compute wilcoxon test
    print("\n📊 Performing Wilcoxon Signed-Rank Tests...")
    print("=" * 60)
    wilcoxon_results = []

    for (model_name, indomain_metrics), imnet_metrics in zip(
        formatted_indomain.items(), formatted_imnet.values()
    ):
        for (metric_name, (indomain_model_data, indomain_main_data)), (
            imnet_model_data,
            imnet_main_data,
        ) in zip(indomain_metrics.items(), imnet_metrics.values()):
            _, p_value = wilcoxon(
                indomain_main_data + imnet_main_data,
                indomain_model_data + imnet_model_data,
                alternative="greater",
            )

            # Determine significance level
            if p_value < 0.001:
                SIGNIFICANCE = "*** (p < 0.001)"
                EMOJI = "🔥"
            elif p_value < 0.01:
                SIGNIFICANCE = "** (p < 0.01)"
                EMOJI = "⭐"
            elif p_value < 0.05:
                SIGNIFICANCE = "* (p < 0.05)"
                EMOJI = "✨"
            else:
                SIGNIFICANCE = "n.s. (p ≥ 0.05)"
                EMOJI = "💤"

            result_line = (
                f"{EMOJI} {model_name:<20} - {metric_name}: "
                + f"p-value={p_value:1.0e} {SIGNIFICANCE}"
            )
            print(result_line)
            wilcoxon_results.append((model_name, metric_name, p_value))

    print("=" * 60)

    # Format the data for the Friedman test
    print("\n🔧 Formatting data for Friedman test...")
    formatted_indomain = format_data(indomain_data, out_type="friedman")
    formatted_imnet = format_data(imnet_data, out_type="friedman")

    # Compute Friedman test
    print("\n🎯 Performing Friedman Tests...")
    print("=" * 60)
    friedman_results = []

    for metric_name in ["acc-1", "acc-5"]:
        concat_data = {
            model_name: formatted_indomain[model_name][metric_name]
            + formatted_imnet[model_name][metric_name]
            for model_name in formatted_indomain
        }

        _, p_value = friedmanchisquare(*concat_data.values())

        # Determine significance level
        if p_value < 0.001:
            SIGNIFICANCE = "*** (p < 0.001)"
            EMOJI = "🔥"
        elif p_value < 0.01:
            SIGNIFICANCE = "** (p < 0.01)"
            EMOJI = "⭐"
        elif p_value < 0.05:
            SIGNIFICANCE = "* (p < 0.05)"
            EMOJI = "✨"
        else:
            SIGNIFICANCE = "n.s. (p ≥ 0.05)"
            EMOJI = "💤"

        result_line = (
            f"{EMOJI} Friedman test for {metric_name:<8}: "
            + f"p-value={p_value:1.0e} {SIGNIFICANCE}"
        )
        print(result_line)
        friedman_results.append((metric_name, p_value))

    print("=" * 60)

    # Summary section
    print("\n📈 STATISTICAL ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"📋 Main Baseline Model: {MAIN_MODEL}")
    print("🔍 Tests Performed:")
    print(f"   • Wilcoxon Signed-Rank Tests: {len(wilcoxon_results)} comparisons")
    print(f"   • Friedman Tests: {len(friedman_results)} metrics")

    # Count significant results
    significant_wilcoxon = sum(1 for _, _, p in wilcoxon_results if p < 0.05)
    significant_friedman = sum(1 for _, p in friedman_results if p < 0.05)

    print("\n💫 Significant Results (p < 0.05):")
    print(
        f"   • Wilcoxon Tests: {significant_wilcoxon}/{len(wilcoxon_results)} "
        + f"({significant_wilcoxon/len(wilcoxon_results)*100:.1f}%)"
    )
    print(
        f"   • Friedman Tests: {significant_friedman}/{len(friedman_results)} "
        + f"({significant_friedman/len(friedman_results)*100:.1f}%)"
    )

    print("\n🎉 Statistical analysis completed successfully!")
    print("=" * 60)
