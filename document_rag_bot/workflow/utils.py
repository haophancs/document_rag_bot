from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

from document_rag_bot.workflow.structs import (
    GraphConfig,
    MetadataConfig,
    NodeConfig,
    PromptTemplateConfig,
    WorkflowConfig,
)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    :param file_path: Path to the YAML file
    :returns: Dictionary containing the YAML file contents
    :raises yaml.YAMLError: If there's an error parsing the YAML file
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_workflow_config(
    config_dir: str = "document_rag_bot/workflow/config",
) -> WorkflowConfig:
    """
    Load the entire workflow configuration from a directory of YAML files.

    :param config_dir: Directory containing the configuration files
    :returns: WorkflowConfig object containing the entire workflow configuration
    :raises FileNotFoundError: If any required configuration file is missing
    :raises yaml.YAMLError: If there's an error parsing any YAML file
    """
    config_dir: Path = Path(config_dir)  # type: ignore
    prompt_template_config: PromptTemplateConfig = PromptTemplateConfig(
        **load_yaml(str(config_dir / "templates.yaml")),  # type: ignore
    )
    node_configs: List[NodeConfig] = [
        NodeConfig(**load_yaml(str(node_file)))
        for node_file in (config_dir / "nodes").glob("*.yaml")  # type: ignore
    ]
    metadata_config: MetadataConfig = MetadataConfig(
        **load_yaml(str(config_dir / "metadata.yaml")),  # type: ignore
    )
    graph_config: GraphConfig = GraphConfig(
        **load_yaml(str(config_dir / "graph.yaml")),  # type: ignore
    )
    return WorkflowConfig(
        prompt_templates=prompt_template_config,
        nodes=node_configs,
        graph=graph_config,
        metadata=metadata_config,
    )


class RetrievalMetricCalculator:
    """Stateless metric calculator for retrieval tasks."""

    @staticmethod
    def _get_confusion_matrix(
        needed: Set[str],
        retrieved: List[str],
        k: int,
        return_members: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate TP, TN, FP, FN for the given k.

        :param needed: Set of needed items.
        :param retrieved: List of retrieved items.
        :param k: The number of top items to consider.
        :param return_members: Whether to return the members of each category.
        :returns: Dictionary containing TP, TN, FP, FN and optionally their members.
        """
        retrieved_set = set(retrieved[:k])
        tp_items = needed & retrieved_set
        fp_items = retrieved_set - needed
        fn_items = needed - retrieved_set
        total_items = len(needed | retrieved_set)
        tn = total_items - len(tp_items) - len(fp_items) - len(fn_items)

        result = {
            "TP": len(tp_items),
            "TN": tn,
            "FP": len(fp_items),
            "FN": len(fn_items),
        }

        if return_members:
            result.update(
                {
                    "TP_items": list(tp_items),  # type: ignore
                    "FP_items": list(fp_items),  # type: ignore
                    "FN_items": list(fn_items),  # type: ignore
                },
            )

        return result

    @staticmethod
    def recall_at_k(needed: List[str], retrieved: List[str], k: int) -> float:
        """
        Calculate Recall@K.

        :param needed: List of needed items.
        :param retrieved: List of retrieved items.
        :param k: The number of top items to consider.
        :returns: Recall@K value.
        """
        if not needed:
            return 0.0
        cm = RetrievalMetricCalculator._get_confusion_matrix(set(needed), retrieved, k)
        return cm["TP"] / (cm["TP"] + cm["FN"])

    @staticmethod
    def precision_at_k(needed: List[str], retrieved: List[str], k: int) -> float:
        """
        Calculate Precision@K.

        :param needed: List of needed items.
        :param retrieved: List of retrieved items.
        :param k: The number of top items to consider.
        :returns: Precision@K value.
        """
        if k == 0:
            return 0.0
        cm = RetrievalMetricCalculator._get_confusion_matrix(set(needed), retrieved, k)
        return cm["TP"] / (cm["TP"] + cm["FP"]) if cm["TP"] + cm["FP"] > 0 else 0.0

    @staticmethod
    def accuracy_at_k(needed: List[str], retrieved: List[str], k: int) -> float:
        """
        Calculate Accuracy@K.

        :param needed: List of needed items.
        :param retrieved: List of retrieved items.
        :param k: The number of top items to consider.
        :returns: Accuracy@K value.
        """
        cm = RetrievalMetricCalculator._get_confusion_matrix(set(needed), retrieved, k)
        total_items = len(set(needed) | set(retrieved))
        return (cm["TP"] + cm["TN"]) / total_items

    @staticmethod
    def f1_at_k(needed: List[str], retrieved: List[str], k: int) -> float:
        """
        Calculate F1@K.

        :param needed: List of needed items.
        :param retrieved: List of retrieved items.
        :param k: The number of top items to consider.
        :returns: F1@K value.
        """
        precision = RetrievalMetricCalculator.precision_at_k(needed, retrieved, k)
        recall = RetrievalMetricCalculator.recall_at_k(needed, retrieved, k)
        return (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

    @staticmethod
    def generate_report(
        needed: List[str],
        retrieved: List[str],
        k_values: List[int],
        return_members: bool = False,
        include_recall: bool = True,
        include_precision: bool = True,
        include_accuracy: bool = True,
        include_f1: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report of all evaluation metrics for multiple K values.

        :param needed: List of needed items.
        :param retrieved: List of retrieved items.
        :param k_values: List of K values to evaluate.
        :param return_members: Whether to return the members of each category.
        :param include_recall: Whether to return the recall scores.
        :param include_precision: Whether to return the precision scores.
        :param include_accuracy: Whether to return the accuracy scores.
        :param include_f1: Whether to return the f1 scores.
        :returns: A dictionary containing all evaluation metrics for each K value.
        """
        report = {}

        for k in k_values:
            cm = RetrievalMetricCalculator._get_confusion_matrix(
                set(needed),
                retrieved,
                k,
                return_members,
            )
            if include_recall:
                report.update(
                    {
                        f"Recall@{k}": RetrievalMetricCalculator.recall_at_k(
                            needed,
                            retrieved,
                            k,
                        ),
                    },
                )
            if include_precision:
                report.update(
                    {
                        f"Precision@{k}": RetrievalMetricCalculator.precision_at_k(
                            needed,
                            retrieved,
                            k,
                        ),
                    },
                )
            if include_accuracy:
                report.update(
                    {
                        f"Accuracy@{k}": RetrievalMetricCalculator.accuracy_at_k(
                            needed,
                            retrieved,
                            k,
                        ),
                    },
                )
            if include_f1:
                report.update(
                    {
                        f"F1@{k}": RetrievalMetricCalculator.f1_at_k(
                            needed,
                            retrieved,
                            k,
                        ),
                        **{
                            f"{key}@{k}": cm[key]
                            for key in cm
                            if not key.endswith("_items")
                        },
                    },
                )

            if return_members:
                report.update(
                    {
                        f"TP_items@{k}": cm.get("TP_items", []),
                        f"FP_items@{k}": cm.get("FP_items", []),
                        f"FN_items@{k}": cm.get("FN_items", []),
                    },
                )

        return report
