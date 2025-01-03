import abc
import uuid
from typing import ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CollectionItemSchema(abc.ABC, BaseModel):
    """Abstract class for data model in collection."""

    collection_name: ClassVar[str] = "documents"


class DocumentSchema(CollectionItemSchema):
    """Data model of the document."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the document",
    )
    text: str = Field(description="Text content of the document")
    url: Optional[str] = Field(description="Source URL of the document")
    embeddings: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Dictionary of embedding vectors by deployment ID",
    )

    model_config = ConfigDict(coerce_numbers_to_str=True)

    collection_name: ClassVar[str] = "covid_documents"

    def __str__(self) -> str:
        return self.text
