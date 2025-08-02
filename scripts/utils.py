import os
import uuid
import click
import pandas as pd


def validate_save_dir(ctx, param, value):
    """Validate that save directory is a direct subdirectory of results/"""
    results_root = os.path.abspath("results")
    abs_value = os.path.abspath(value)
    if not os.path.isdir(abs_value):
        raise click.BadParameter(f"{value} is not a directory.")
    if os.path.dirname(abs_value) != results_root:
        raise click.BadParameter(f"{value} must be a direct subdirectory of {results_root}")
    return value


def get_save_task_dir(task_dir: str, save_dir: str) -> str:
    """Get the save directory for a specific task"""
    if not os.path.normpath(task_dir).startswith("tasks"):
        raise ValueError(f"task_dir have to start with 'tasks/', but got: {task_dir}")
    
    rel_path = os.path.relpath(task_dir, start="tasks")
    return os.path.join(save_dir, rel_path)


def save_csv(path: str, file_name: str, df: pd.DataFrame):
    """Save DataFrame to CSV with collision handling"""
    try:
        df.to_csv(os.path.join(path, file_name), index=False)
    except PermissionError:
        file_name = f"{file_name}_{uuid.uuid4()}.csv"
        save_csv(path, file_name, df) 