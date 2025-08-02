import os
import click
import pandas as pd
from pathlib import Path
import asyncio

from scripts.operations import (
    generate_and_run, run_and_calculate, calculate_scores
)
from scripts.data import download_datasets, get_available_datasets
from scripts.utils import validate_save_dir


@click.group()
def cli():
    """Runs generations and evaluations"""


@cli.command("generate")
@click.option("-p", "--path", type=click.Path(exists=True, file_okay=False), required=True, multiple=True,
              help="Root path to the task directory")
@click.option("-s", "--save-dir", type=click.Path(exists=True, file_okay=False), callback=validate_save_dir, required=True, 
              help="Directory to save final results. Must be a direct subfolder of 'results/'.")
@click.option("-pl", "--pipeline_path", type=click.Path(exists=True, dir_okay=False), required=True, 
              help="Path to the pipeline to generate solutions")
@click.option("-n", "--rounds", type=int, required=False, default=1, 
              help="The number of rounds to run")
@click.option("--run-after", is_flag=True, 
              help="Run workflows immediately after generation")
@click.option("--calculate-after", is_flag=True, 
              help="Calculate scores after running (requires --run-after)")
def generate(path, save_dir, pipeline_path, rounds, run_after, calculate_after):
    """Generate workflows and optionally run and calculate scores"""
    if calculate_after and not run_after:
        raise click.BadParameter("--calculate-after requires --run-after to be set")
    
    async def process():
        await generate_and_run(
            paths=path, 
            save_dir=save_dir, 
            pipeline_path=pipeline_path, 
            rounds=rounds, 
            run_after=run_after, 
            calculate_after=calculate_after
        )
    
    asyncio.run(process())


@cli.command("run")
@click.option("-p", "--path", type=click.Path(exists=True, file_okay=False), required=True, multiple=True, 
              help="Root path to the task directory")
@click.option("-s", "--save-dir", type=click.Path(exists=True, file_okay=False), callback=validate_save_dir, required=True, 
              help="Directory to save final results. Must be a direct subfolder of 'results/'.")
@click.option("-n", "--rounds", type=int, required=False, default=1, 
              help="The number of rounds to run")
@click.option("--calculate-after", is_flag=True, 
              help="Calculate scores immediately after running")
def run(path, save_dir, rounds, calculate_after):
    """Run workflows and optionally calculate scores"""
    async def process():
        await run_and_calculate(
            paths=path, 
            save_dir=save_dir, 
            rounds=rounds, 
            calculate_after=calculate_after
        )
    
    asyncio.run(process())


@cli.command("calculate")
@click.option("-p", "--path", type=click.Path(exists=True, file_okay=False), required=True,
              multiple=True, help="Root path to the task directory")
@click.option("-s", "--save-dir", type=click.Path(exists=True, file_okay=False), callback=validate_save_dir, required=True, 
              help="Directory to save final results. Must be a direct subfolder of 'results/'.")
@click.option("-n", "--rounds", type=int, required=False, default=1, 
              help="The number of rounds to run")
def calculate(path, save_dir, rounds):
    """Calculate scores for predictions"""
    calculate_scores(paths=path, save_dir=save_dir, rounds=rounds)


@cli.command("download")
@click.option("-d", "--datasets", multiple=True, default=["datasets"],
              help="Datasets to download. Available: datasets, results")
@click.option("--skip-existing", is_flag=True,
              help="Skip downloading if datasets already exist")
@click.option("--list", "list_datasets", is_flag=True,
              help="List available datasets")
def download(datasets, skip_existing, list_datasets):
    """Download CABench datasets"""
    if list_datasets:
        available = get_available_datasets()
        click.echo("Available datasets:")
        for name, info in available.items():
            click.echo(f"  - {name}: extracts to {info['extract_path']}")
        return
    
    download_datasets(required_datasets=datasets, skip_existing=skip_existing)


def main():
    cli()
    

if __name__ == "__main__":
    main()