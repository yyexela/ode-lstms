from pathlib import Path
import yaml

config_path = Path(__file__).parent.parent / 'tuning_config' / 'config_lstm_ODE_Lorenz_1.yaml'
with open(config_path, 'r') as f:
    hp_config = yaml.safe_load(f)
print(hp_config)

for model in ['ode-lstm', 'lstm']:
    for dataset in ['ODE_Lorenz', "PDE_KS"]:
        for pair_id in range(1,9+1):

            # Fill data
            hp_config['hyperparameters']['hidden_state_size']['lower_bound'] = 8
            hp_config['hyperparameters']['hidden_state_size']['upper_bound'] = 256
            hp_config['hyperparameters']['lr']['lower_bound'] = 0.00001
            hp_config['hyperparameters']['lr']['upper_bound'] = 0.01
            hp_config['dataset']['pair_id'] = [pair_id]
            hp_config['dataset']['name'] = dataset

            if pair_id in list(range(7,10+1)):
                # limited data
                hp_config['hyperparameters']['seq_length']['lower_bound'] = 5
                hp_config['hyperparameters']['seq_length']['upper_bound'] = 45
            else:
                # normal data
                hp_config['hyperparameters']['seq_length']['lower_bound'] = 8
                hp_config['hyperparameters']['seq_length']['upper_bound'] = 256
            
            # save
            output_path = Path(__file__).parent.parent / 'tuning_config' / f'config_{model}_{dataset}_{pair_id}.yaml'
            with open(output_path, 'w') as f:
                yaml.dump(hp_config, f)