import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import asyncio

from utils.logs import logger
from scripts.calculator import Calculator, MetricType
from scripts.data import (
    load_data, load_metadata, load_task_description, 
    load_label, load_prediction, load_task_dirs
)
from scripts.utils import get_save_task_dir, save_csv
from scripts.workflows import load_pipeline, load_workflow, write_workflow


async def generate_workflow_for_task(task_dir, save_dir, pipeline_path, rounds, p_name):
    """Generate workflow for a single task"""
    for round in tqdm(range(rounds), leave=False):
        tqdm.write(f"[{p_name}] Generate workflow: {Path(task_dir).name} for round: {round}")
        validation_data = load_data(os.path.join(task_dir, "validation"))
        data = {
            "validation_data": validation_data 
        }
        task_description = load_task_description(task_dir)
        pipeline = load_pipeline(pipeline_path)
        try:
            workflow = await pipeline(task_description, data)
        except Exception as e:
            logger.error(f"Error generating workflow: {e}")
            workflow = ""
        round_num = round if rounds > 1 else None
        save_task_dir = get_save_task_dir(task_dir, save_dir)
        write_workflow(save_task_dir, workflow, round_num=round_num)


async def run_workflow_for_task(task_dir, save_dir, rounds, p_name):
    """Run workflow for a single task"""
    inputs = load_data(os.path.join(task_dir, "test"))
    for round in tqdm(range(rounds), leave=False):
        tqdm.write(f"[{p_name}] Run workflow: {Path(task_dir).name} for round: {round}")
        round_num = round if rounds > 1 else None
        predictions = []
        save_task_dir = get_save_task_dir(task_dir, save_dir)
        workflow = load_workflow(save_task_dir, round_num=round_num)
        for input_id in tqdm(inputs, leave=False):
            tqdm.write(f"[{p_name}] Run workflow: {Path(task_dir).name} for round: {round} for input: {input_id}")
            input_data = inputs[input_id]
            try:
                prediction = await workflow(input_data)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Error running workflow: {e}")
                prediction = ""
            predictions.append({"id": input_id, "prediction": prediction})
        if round_num is not None:
            filename = f"predictions_{round_num}.csv"
        else:
            filename = f"predictions.csv"
        save_csv(save_task_dir, filename, pd.DataFrame(predictions))


def calculate_scores_for_task(task_dir, save_dir, rounds, p_name):
    """Calculate scores for a single task"""
    tqdm.write(f"[{p_name}] Calculate score: {Path(task_dir).name}")
    metadata = load_metadata(task_dir)
    task_name = Path(task_dir).name
    metric = metadata["metric"]
    metric_type = MetricType(metric)
    test_dir = os.path.join(task_dir, "test")
    print(test_dir)
    label = load_label(test_dir)
    df_scores = []
    calculator = Calculator()
    
    for round in range(rounds):
        round_num = round if rounds > 1 else None
        save_task_dir = get_save_task_dir(task_dir, save_dir)
        prediction = load_prediction(save_task_dir, round_num)
        _, df_score = calculator(metric_type, prediction, label)
        df_scores.append(df_score)
    
    df_avg_score = df_scores[0].copy()
    for col in df_avg_score.columns:
        if col == "score" and pd.api.types.is_numeric_dtype(df_avg_score[col]):
            col_sum = df_avg_score[col].copy()
            for df in df_scores[1:]:
                if col in df.columns:
                    col_sum += df[col]
            df_avg_score[col] = col_sum / len(df_scores)
    
    round_num = None if rounds == 1 else None  # For file naming
    save_task_dir = get_save_task_dir(task_dir, save_dir)
    if round_num is not None:
        save_csv(save_task_dir, f"scores_{round_num}.csv", pd.DataFrame(df_avg_score))
    else:
        save_csv(save_task_dir, f"scores.csv", pd.DataFrame(df_avg_score))
    
    return {
        "id": task_name,
        "score": df_avg_score["score"].mean()
    }


async def generate_workflows(paths, save_dir, pipeline_path, rounds):
    """Generate workflows for given paths"""
    for p in paths:
        task_dirs = load_task_dirs(p)
        for task_dir in tqdm(task_dirs, leave=False):
            tqdm.write(f"[{Path(p).name}] Generate workflow: {Path(task_dir).name}")
            await generate_workflow_for_task(task_dir, save_dir, pipeline_path, rounds, Path(p).name)


async def run_workflows(paths, save_dir, rounds):
    """Run workflows for given paths"""
    for p in paths:
        task_dirs = load_task_dirs(p)
        for task_dir in tqdm(task_dirs, leave=False):
            tqdm.write(f"[{Path(p).name}] Run workflow: {Path(task_dir).name}")
            await run_workflow_for_task(task_dir, save_dir, rounds, Path(p).name)


def calculate_scores(paths, save_dir, rounds):
    """Calculate scores for given paths"""
    for p in paths:
        task_dirs = load_task_dirs(p)
        final_score = []
        for task_dir in tqdm(task_dirs, leave=False):
            score_result = calculate_scores_for_task(task_dir, save_dir, rounds, Path(p).name)
            final_score.append(score_result)
        print("Average score: ", pd.DataFrame(final_score)["score"].mean())
        save_p = get_save_task_dir(p, save_dir)
        save_csv(save_p, f"final_scores.csv", pd.DataFrame(final_score))


async def generate_and_run(paths, save_dir, pipeline_path, rounds, run_after=False, calculate_after=False):
    """Generate workflows and optionally run and calculate for each task sequentially"""
    for p in paths:
        task_dirs = load_task_dirs(p)
        final_scores = []
        
        for task_dir in tqdm(task_dirs, leave=False):
            tqdm.write(f"[{Path(p).name}] Processing task: {Path(task_dir).name}")
            
            # Step 1: Generate workflow for this task
            await generate_workflow_for_task(task_dir, save_dir, pipeline_path, rounds, Path(p).name)
            
            # Step 2: Run workflow if requested
            if run_after:
                print(f"Running workflow for {Path(task_dir).name}...")
                await run_workflow_for_task(task_dir, save_dir, rounds, Path(p).name)
                
                # Step 3: Calculate scores if requested
                if calculate_after:
                    print(f"Calculating scores for {Path(task_dir).name}...")
                    score_result = calculate_scores_for_task(task_dir, save_dir, rounds, Path(p).name)
                    final_scores.append(score_result)
        
        # Save final scores if we calculated them
        if calculate_after and run_after and final_scores:
            print("Average score: ", pd.DataFrame(final_scores)["score"].mean())
            save_p = get_save_task_dir(p, save_dir)
            save_csv(save_p, f"final_scores.csv", pd.DataFrame(final_scores))


async def run_and_calculate(paths, save_dir, rounds, calculate_after=False):
    """Run workflows and optionally calculate scores for each task sequentially"""
    for p in paths:
        task_dirs = load_task_dirs(p)
        final_scores = []
        
        for task_dir in tqdm(task_dirs, leave=False):
            tqdm.write(f"[{Path(p).name}] Processing task: {Path(task_dir).name}")
            
            # Step 1: Run workflow for this task
            await run_workflow_for_task(task_dir, save_dir, rounds, Path(p).name)
            
            # Step 2: Calculate scores if requested
            if calculate_after:
                print(f"Calculating scores for {Path(task_dir).name}...")
                score_result = calculate_scores_for_task(task_dir, save_dir, rounds, Path(p).name)
                final_scores.append(score_result)
        
        # Save final scores if we calculated them
        if calculate_after and final_scores:
            print("Average score: ", pd.DataFrame(final_scores)["score"].mean())
            save_p = get_save_task_dir(p, save_dir)
            save_csv(save_p, f"final_scores.csv", pd.DataFrame(final_scores)) 