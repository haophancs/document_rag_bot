from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator


class EdgeConfig(BaseModel):
    """Configuration for standard edge."""

    from_node: str
    to_node: str


class ConditionalEdgesConfig(BaseModel):
    """Configuration for conditional edges."""

    from_node: str
    route_function: str
    intermediates: List[str]


class PromptTemplateConfig(BaseModel):
    """Configuration for prompt templates."""

    system: str
    human: str


class GraphConfig(BaseModel):
    """Configuration for workflow graph."""

    edge_list: List[EdgeConfig]
    conditional_edges_list: List[ConditionalEdgesConfig]
    entrypoint: str


class FieldConfig(BaseModel):
    """Configuration for a field."""

    type: str
    description: Optional[str] = None
    annotated_op: Optional[str] = None

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Overloaded method of original one in pydantic BaseModel."""
        exclude_fields = {"annotated_op", *kwargs.get("", set())}
        kwargs["exclude"] = exclude_fields
        return super().dict(*args, **kwargs)


class NodeConfig(BaseModel):
    """Configuration for a node in the workflow."""

    node_id: str
    role: str

    backstory: Optional[str] = None
    goal: Optional[str] = None
    task: Optional[str] = None

    input_structure: Optional[Dict[str, FieldConfig]] = None
    output_structure: Optional[Dict[str, FieldConfig]] = None
    example_output: Optional[str] = None

    chat_model: Optional[str] = None
    tool: Optional[str] = None

    max_retries: int = 3

    @model_validator(mode="before")
    @classmethod
    def _validate_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        chat_model = values.get("chat_model")
        tool = values.get("tool")
        if chat_model is None and tool is None:
            raise ValueError("Either chat_model or tool must be provided.")
        return values


class ScoringConditions(BaseModel):
    """Configuration for scoring conditions."""

    Low: str
    Medium: str
    High: str


class CriterionDefinition(BaseModel):
    """Configuration for criterion definition."""

    description: str
    scoring_conditions: ScoringConditions


class MetadataConfig(BaseModel):
    """Configuration for workflow metadata."""

    selected_embeddings: List[str]
    messages_limit: int
    number_k_retriever: int
    number_k_reranker: int
    rank_fusion_algorithm: str
    disable_reranker: bool
    domain_info: str


class WorkflowConfig(BaseModel):
    """Overall configuration for the workflow."""

    prompt_templates: PromptTemplateConfig
    nodes: List[NodeConfig]
    graph: GraphConfig
    metadata: MetadataConfig
