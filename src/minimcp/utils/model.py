from typing import Any

from pydantic import BaseModel


def to_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(by_alias=True, mode="json", exclude_none=True)


def to_json(model: BaseModel) -> str:
    return model.model_dump_json(by_alias=True, exclude_none=True)
