from typing import Optional
class Pipeline:
    async def __call__(self, task_description: str, data: Optional[dict] = None) -> str:
        return ""
