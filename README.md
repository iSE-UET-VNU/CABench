# CABench

## Setup

You can install `mlebench` with pip:

```console
pip install -e .
```


## Dataset

The CA-bench dataset is a collection of 70 CA problems which we use to evaluate the ML engineering capabilities of AI systems.

To install CA problems datasets, run:
```console
cabench download -d datasets
```

To install baseline and humand design results, run:

```console
cabench download -d results
```

## Usage

### Generate workflows from pipeline

To generate workflows from a specific pipeline:

```console
cabench generate -p <task_directory> -s <save_directory> -pl <pipeline_path> -n <rounds>
```

Example:
```console
cabench generate -p tasks/node-level -s results/my_experiment -pl pipeline/zeroshot_pipeline.py -n 3
```

### Run generated workflows

To run the generated workflows:

```console
cabench run -p <task_directory> -s <save_directory> -n <rounds>
```

Example:
```console
cabench run -p tasks/node-level -s results/my_experiment -n 3
```

### Calculate solution scores

To calculate scores for executed solutions:

```console
cabench calculate -p <task_directory> -s <save_directory> -n <rounds>
```

Example:
```console
cabench calculate -p tasks/node-level -s results/my_experiment -n 3
```

### Run complete pipeline

To generate, run and calculate scores in a single command:

```console
cabench generate -p <task_directory> -s <save_directory> -pl <pipeline_path> -n <rounds> --run-after --calculate-after
```

Example:
```console
cabench generate -p tasks/node-level -s results/my_experiment -pl pipeline/zeroshot_pipeline.py -n 3 --run-after --calculate-after
```

### Main parameters

- `-p, --path`: Path to task directory (multiple tasks supported)
- `-s, --save-dir`: Directory to save results (must be a subfolder of 'results/')
- `-pl, --pipeline_path`: Path to pipeline for generating solutions
- `-n, --rounds`: Number of rounds to run (default: 1)
- `--run-after`: Run workflows immediately after generation
- `--calculate-after`: Calculate scores after running (requires --run-after)

### List available datasets

```console
cabench download --list
```

