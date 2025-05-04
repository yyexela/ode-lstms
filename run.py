import os
import yaml
import time
import torch
import argparse
import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots, get_config, get_prediction_timesteps

# file dir
file_dir = Path(__file__).parent

def main(config_path: str) -> None:
    """
    Main function to run the spacetime model with specified config file.

    Loads configuration and prepares to call the model.

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

    # Generate a unique batch_id for this run
    # Define the name of the output folder for your batch
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize Visualization object
    viz = Visualization()

    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

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
        --hidden_state_size {hidden_state_size}\
        --seq_length {seq_length}\
        --gradient_clip_val {gradient_clip_val}\
        --epochs {epochs}\
        --lr {lr}\
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
        print("pred shape:", pred_data.shape)

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)

        # Save aggregated batch results
        with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
            yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)