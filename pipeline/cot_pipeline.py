from typing import Optional
from templates.pipeline_template import Pipeline


workflow = """
import base64
from utils.cost_manager import CostManager
from configs.models_config import ModelsConfig
from templates.cot_prompt import COT_PROMPT, parse_prediction

from provider.llm_provider_registry import create_llm_instance

llm = create_llm_instance(ModelsConfig.default().get("gpt-4o-mini"))
llm.cost_manager = CostManager()

class Workflow:
    async def __call__(self, data: dict) -> str:
        image_paths = data["image_paths"]
        video_paths = data["video_paths"]
        audio_paths = data["audio_paths"]
        text_data = data["text_data"]
        task_description = '''{task_description}'''
        if len(video_paths) > 0 or len(audio_paths) > 0:
            return ""
        images = []
        for image_path in image_paths:
            with open(image_path, "rb") as f:
                images.append(base64.b64encode(f.read()).decode("utf-8"))
        response = await llm.aask(
            COT_PROMPT.format(description=task_description, input_text=text_data), images=images
        )
        response = parse_prediction(response)
        return response
"""

class FewShotPipeline(Pipeline):
    async def __call__(self, task_description: str, metadata: Optional[dict] = None) -> str:
        return workflow.format(task_description=task_description)



