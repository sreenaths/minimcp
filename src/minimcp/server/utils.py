from typing import Any

from pydantic import BaseModel


def to_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(by_alias=True, mode="json", exclude_none=True)
