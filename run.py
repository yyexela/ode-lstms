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
from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots, get_config

# file dir
file_dir = Path(__file__).parent

# Configuration dictionary per dataset
id_dict_Lorenz = {
    1: {"train_ids":[1], "test_id": 1, "forecast_id": 1, "forecast_length": 2001, "burn_in": False},
    2: {"train_ids":[2], "test_id": 2, "reconstruct_id": 2},
    3: {"train_ids":[2], "test_id": 3, "forecast_id": 2, "forecast_length": 2001, "burn_in": False},
    4: {"train_ids":[3], "test_id": 4, "reconstruct_id": 3},
    5: {"train_ids":[3], "test_id": 4, "forecast_id": 3, "forecast_length": 2001, "burn_in": False},
    6: {"train_ids":[4], "test_id": 6, "forecast_id": 4, "forecast_length": 2001, "burn_in": False},
    7: {"train_ids":[5], "test_id": 7, "forecast_id": 5, "forecast_length": 2001, "burn_in": False},
    8: {"train_ids":[6, 7, 8], "test_id": 8, "forecast_id": 9, "forecast_length": 2001, "burn_in": True},
    9: {"train_ids":[6, 7, 8], "test_id": 9, "forecast_id": 10, "forecast_length": 2001, "burn_in": True},
}

id_dict_KS = {
    1: {"train_ids":[1], "test_id": 1, "forecast_id": 1, "forecast_length": 1000, "burn_in": False},
    2: {"train_ids":[2], "test_id": 2, "reconstruct_id": 2},
    3: {"train_ids":[2], "test_id": 3, "forecast_id": 2, "forecast_length": 1000, "burn_in": False},
    4: {"train_ids":[3], "test_id": 4, "reconstruct_id": 3},
    5: {"train_ids":[3], "test_id": 4, "forecast_id": 3, "forecast_length": 1000, "burn_in": False},
    6: {"train_ids":[4], "test_id": 6, "forecast_id": 4, "forecast_length": 1000, "burn_in": False},
    7: {"train_ids":[5], "test_id": 7, "forecast_id": 5, "forecast_length": 1000, "burn_in": False},
    8: {"train_ids":[6, 7, 8], "test_id": 8, "forecast_id": 9, "forecast_length": 1000, "burn_in": True},
    9: {"train_ids":[6, 7, 8], "test_id": 9, "forecast_id": 10, "forecast_length": 1000, "burn_in": True},
}

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
    ds_config = get_config(config['dataset']['name'])

    # Load prepare command to execute
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    if dataset_name == "ODE_Lorenz":
        id_dict = id_dict_Lorenz
    elif dataset_name == "PDE_KS":
        id_dict = id_dict_KS

    model_name = f"{config['model']['name']}"

    # Generate a unique batch_id for this run
    # Define the name of the output folder for your batch
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'sub_datasets': []
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
        --dataset {dataset}\
        --seed {seed}\
        --pair_id {pair_id}\
        --model {model}\
        --solver {solver}\
        --hidden_state_size {hidden_state_size}\
        --train_ids {train_ids}\
        --seq_length {seq_length}\
        --gradient_clip_val {gradient_clip_val}\
        --accelerator {accelerator}\
        --log_every_n_steps {log_every_n_steps}\
        --epochs {epochs}\
        --gpu {gpu}\
        --lr {lr}\
        """

        cmd_formatted_1 = cmd_1.format(
            ode_lorenz_main_path = file_dir / "pt_trainer.py",
            dataset = config['dataset']['name'],
            seed = config['model']['seed'],
            pair_id = pair_id,
            model = config['model']['model'],
            solver = config['model']['solver'],
            hidden_state_size = config['model']['hidden_state_size'],
            train_ids = ' '.join([str(i) for i in id_dict[pair_id]["train_ids"]]),
            seq_length = config['model']['seq_length'],
            gradient_clip_val = config['model']['gradient_clip_val'],
            accelerator = config['model']['accelerator'],
            log_every_n_steps = config['model']['log_every_n_steps'],
            epochs = config['model']['epochs'],
            gpu = config['model']['gpu'],
            lr = config['model']['lr'],
        )

        if id_dict[pair_id].get('forecast_id', None) is not None:
            cmd_2 = \
            """\
            python\
            {ode_lorenz_main_path}\
            --dataset {dataset}\
            --seed {seed}\
            --pair_id {pair_id}\
            --forecast_id {forecast_id}\
            --forecast_length {forecast_length}\
            --train_ids {train_ids}\
            --seq_length {seq_length}\
            --gpu {gpu}\
            """

            cmd_formatted_2 = cmd_2.format(
                ode_lorenz_main_path = file_dir / "ts_evaluation.py",
                dataset = config['dataset']['name'],
                seed = config['model']['seed'],
                pair_id = pair_id,
                forecast_id = int(id_dict[pair_id]["forecast_id"]),
                forecast_length = int(id_dict[pair_id]["forecast_length"]),
                train_ids = ' '.join([str(i) for i in id_dict[pair_id]["train_ids"]]),
                seq_length = config['model']['seq_length'],
                gpu = config['model']['gpu'],
            )

            if id_dict[pair_id]['burn_in']:
                cmd_formatted_2 = cmd_formatted_2 + " --burn_in"

        elif id_dict[pair_id].get('reconstruct_id', None) is not None:
            cmd_2 = \
            """\
            python\
            {ode_lorenz_main_path}\
            --dataset {dataset}\
            --seed {seed}\
            --pair_id {pair_id}\
            --reconstruct_id {reconstruct_id}\
            --train_ids {train_ids}\
            --seq_length {seq_length}\
            --gpu {gpu}\
            """

            cmd_formatted_2 = cmd_2.format(
                ode_lorenz_main_path = file_dir / "ts_evaluation.py",
                dataset = config['dataset']['name'],
                seed = config['model']['seed'],
                pair_id = pair_id,
                reconstruct_id = int(id_dict[pair_id]["reconstruct_id"]),
                train_ids = ' '.join([str(i) for i in id_dict[pair_id]["train_ids"]]),
                seq_length = config['model']['seq_length'],
                gpu = config['model']['gpu'],
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

        # Execute command 2
        print("---------------")
        print("Python running:")
        print(cmd_formatted_2)
        print("---------------")

        out = os.system(cmd_formatted_2)
        time.sleep(1) # to allow for ctrl+c

        print("---------------")
        print(f"Returned: {out}")
        print("---------------")

        if out != 0:
            raise Exception(f"Output code {out}")

        # Load predictions
        pred_data = torch.load(file_dir / 'tmp_pred' / 'output_mat.torch', weights_only=False)

        # Load test data
        _, test_data, _ = load_dataset(dataset_name, pair_id)

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, test_data, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        results_for_yaml = {key: float(value) for key, value in results.items()}
        batch_results['sub_datasets'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type)

        # Save aggregated batch results
        with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
            yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)