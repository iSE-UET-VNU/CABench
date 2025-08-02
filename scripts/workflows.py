import os
import sys
import importlib
import importlib.util
import inspect
from typing import Optional

from utils.logs import logger
from templates.workflow_template import Workflow
from templates.pipeline_template import Pipeline


def load_pipeline(path: str) -> Pipeline:
    """Load pipeline from file path"""
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Cannot create module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if spec.loader is None:
        raise ImportError(f"Module loader is None for {path}")
    spec.loader.exec_module(module)
    pipeline_classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (issubclass(obj, Pipeline) and 
            obj is not Pipeline and 
            obj.__module__ == module_name):
            pipeline_classes.append(obj)
    if not pipeline_classes:
        raise ValueError(f"Can not find Pipeline class in {path}")
    pipeline_class = pipeline_classes[0]
    pipeline = pipeline_class()
    return pipeline


def load_workflow(path: str, round_num: Optional[int] = None) -> Workflow:
    """Load workflow from path with optional round number"""
    path = path.replace("\\", ".").replace("/", ".")
    while path.endswith("."):
        path = path[:-1]
    if round_num is not None:
        workflow_module_name = f"{path}.workflow_{round_num}"
    else:
        workflow_module_name = f"{path}.workflow"
    print(workflow_module_name)
    try:
        workflow_module = __import__(workflow_module_name, fromlist=[""])
        workflow_class = getattr(workflow_module, "Workflow")
        workflow = workflow_class()
        return workflow
    except ImportError as e:
        logger.info(f"Error loading workflow: {e}")
        raise ValueError(f"Error loading workflow: {e}")
    except Exception as e:
        logger.info(f"Error loading workflow: {e}")
        raise ValueError(f"Error loading workflow: {e}")


def write_workflow(path: str, code: str, round_num: Optional[int] = None):
    """Write workflow code to file"""
    if round_num is not None:
        filename = f"workflow_{round_num}.py"
    else:
        filename = f"workflow.py"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), "w", encoding="utf-8") as f:
        f.write(code) 