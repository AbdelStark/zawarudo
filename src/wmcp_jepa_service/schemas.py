from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class TensorRef(BaseModel):
    kind: Literal["tensor"] = "tensor"
    encoding: Literal["inline", "base64", "uri"]
    dtype: Literal["uint8", "float16", "float32", "float64", "int32", "int64"]
    shape: list[int]
    layout: str
    data: Any | None = None
    data_b64: str | None = None
    uri: str | None = None
    sha256: str | None = None

    @model_validator(mode="after")
    def validate_encoding_payload(self) -> "TensorRef":
        if self.encoding == "inline" and self.data is None:
            raise ValueError("inline tensors require data")
        if self.encoding == "base64" and not self.data_b64:
            raise ValueError("base64 tensors require data_b64")
        if self.encoding == "uri" and not self.uri:
            raise ValueError("uri tensors require uri")
        return self


class Observation(BaseModel):
    modality: Literal["rgb", "depth", "state", "proprioception", "latent"]
    tensor: TensorRef
    preprocessing: dict[str, Any] = Field(default_factory=dict)


class ActionTensor(BaseModel):
    space: Literal["continuous", "discrete", "hybrid"] = "continuous"
    tensor: TensorRef
    normalization: str | None = None
    bounds: dict[str, list[float]] | None = None


class RequestEnvelope(BaseModel):
    wmcp_version: str = "0.1"
    request_id: str
    operation: Literal["metadata", "encode", "predict", "rollout", "score", "plan"]
    model: str
    model_revision: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    return_options: dict[str, Any] = Field(default_factory=dict)


class ResponseEnvelope(BaseModel):
    wmcp_version: str = "0.1"
    request_id: str
    operation: str
    model: str
    model_revision: str | None = None
    outputs: dict[str, Any]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    wmcp_version: str = "0.1"
    request_id: str | None = None
    error: dict[str, Any]


class ModelMetadata(BaseModel):
    model_id: str
    model_revision: str = "mock"
    model_family: str = "jepa"
    model_type: str = "action_conditioned_world_model"
    task: str = "pusht"
    supported_operations: list[str] = Field(default_factory=lambda: ["metadata", "encode", "predict", "rollout", "score", "plan"])
    input_shapes: dict[str, Any] = Field(default_factory=dict)
    latent_space: dict[str, Any] = Field(default_factory=lambda: {"dimension": 192, "dtype": "float32"})
    limits: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
