import os
import yaml
import time
import torch
import argparse
import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from ctf4science.eval_module import evaluate_custom, save_results
from ctf4science.visualization_module import Visualization
from ctf4science.data_module import load_validation_dataset, parse_pair_ids, get_applicable_plots, get_config, get_prediction_timesteps

# Delete results directory - used for storing batch_results
file_dir = Path(__file__).parent

def main(config_path: str) -> None:
    """
    Main function to run the spacetime model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    The evaluation function evaluates on validation data obtained from training data.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configurations
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load prepare command to execute
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['name']}"

    # batch_id is from optimize_parameters.py
    if 'batch_id' in config['model']:
        batch_id = config['model']['batch_id']
    else:
        batch_id = 0

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': f"hyper_opt_{batch_id}",
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Process each sub-dataset
        # Prepare commands
        cmd_1 = \
        """\
        python\
        {ode_lorenz_main_path}\
        --batch_id {batch_id}\
        --batch_size {batch_size}\
        --dataset {dataset}\
        --seed {seed}\
        --pair_id {pair_id}\
        --model {model}\
        --solver {solver}\
        --seq_length {seq_length}\
        --hidden_state_size {hidden_state_size}\
        --gradient_clip_val {gradient_clip_val}\
        --epochs {epochs}\
        --lr {lr}\
        --validation\
        """

        cmd_formatted_1 = cmd_1.format(
            ode_lorenz_main_path = file_dir / "pt_trainer.py",
            batch_id = batch_id,
            batch_size = config['model']['batch_size'],
            dataset = config['dataset']['name'],
            seed = config['model']['seed'],
            pair_id = pair_id,
            model = config['model']['model'],
            solver = config['model']['solver'],
            hidden_state_size = config['model']['hidden_state_size'],
            seq_length = config['model']['seq_length'],
            gradient_clip_val = config['model']['gradient_clip_val'],
            epochs = config['model']['epochs'],
            lr = config['model']['lr'],
        )

        # Execute command 1
        print("---------------")
        print("Python running:")
        print(cmd_formatted_1)
        print("---------------")

        out = os.system(cmd_formatted_1)
        time.sleep(1) # to allow for ctrl+c

        print("---------------")
        print(f"Returned: {out}")
        print("---------------")

        if out != 0:
            raise Exception(f"Output code {out}")

        # Load predictions
        pred_data = torch.load(file_dir / 'tmp_pred' / f'output_mat_{batch_id}.torch', weights_only=False)
        pred_data = pred_data.T

        # Evaluate predictions using default metrics
        _, val_data, _ = load_validation_dataset(dataset_name, pair_id, 0.8)
        results = evaluate_custom(dataset_name, pair_id, val_data, pred_data)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

    # Save aggregated batch results
    results_file = file_dir / f"results_{batch_id}.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)