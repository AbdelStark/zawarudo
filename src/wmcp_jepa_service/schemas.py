from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class TensorRef(BaseModel):
    kind: Literal["tensor"] = "tensor"
    encoding: Literal["inline", "base64", "uri"]
    dtype: Literal["uint8", "float16", "float32", "float64", "int32", "int64"]
    shape: List[int]
    layout: str
    data: Optional[Any] = None
    data_b64: Optional[str] = None
    uri: Optional[str] = None
    sha256: Optional[str] = None

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
    preprocessing: Dict[str, Any] = Field(default_factory=dict)


class ActionTensor(BaseModel):
    space: Literal["continuous", "discrete", "hybrid"] = "continuous"
    tensor: TensorRef
    normalization: Optional[str] = None
    bounds: Optional[Dict[str, List[float]]] = None


class RequestEnvelope(BaseModel):
    wmcp_version: str = "0.1"
    request_id: str
    operation: Literal["metadata", "encode", "predict", "rollout", "score", "plan"]
    model: str
    model_revision: Optional[str] = None
    trace: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    return_options: Dict[str, Any] = Field(default_factory=dict)


class ResponseEnvelope(BaseModel):
    wmcp_version: str = "0.1"
    request_id: str
    operation: str
    model: str
    model_revision: Optional[str] = None
    outputs: Dict[str, Any]
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    wmcp_version: str = "0.1"
    request_id: Optional[str] = None
    error: Dict[str, Any]


class ModelMetadata(BaseModel):
    model_id: str
    model_revision: str = "mock"
    model_family: str = "jepa"
    model_type: str = "action_conditioned_world_model"
    task: str = "pusht"
    supported_operations: List[str] = Field(default_factory=lambda: ["metadata", "encode", "predict", "rollout", "score", "plan"])
    input_shapes: Dict[str, Any] = Field(default_factory=dict)
    latent_space: Dict[str, Any] = Field(default_factory=lambda: {"dimension": 192, "dtype": "float32"})
    limits: Dict[str, Any] = Field(default_factory=dict)
    runtime: Dict[str, Any] = Field(default_factory=dict)
