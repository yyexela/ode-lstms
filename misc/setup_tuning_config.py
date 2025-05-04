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
            hp_config['dataset']['name'] = dataset
            hp_config['dataset']['pair_id'] = [pair_id]
            hp_config['hyperparameters']['lr']['lower_bound'] = 0.00001
            hp_config['hyperparameters']['lr']['upper_bound'] = 0.01

            if pair_id in list(range(6,7+1)):
                # limited data
                hp_config['hyperparameters']['seq_length']['lower_bound'] = 5
                hp_config['hyperparameters']['seq_length']['upper_bound'] = 39
                hp_config['model']['batch_size'] = 5
            elif pair_id in list(range(8,10+1)):
                # limited burn-in data but not limited training data 
                hp_config['hyperparameters']['seq_length']['lower_bound'] = 5
                hp_config['hyperparameters']['seq_length']['upper_bound'] = 39
                hp_config['model']['batch_size'] = 128
            else:
                # normal data
                hp_config['hyperparameters']['seq_length']['lower_bound'] = 8
                hp_config['hyperparameters']['seq_length']['upper_bound'] = 256
                hp_config['model']['batch_size'] = 128

            if dataset in ['ODE_Lorenz']:
                hp_config['hyperparameters']['hidden_state_size']['lower_bound'] = 3
                hp_config['hyperparameters']['hidden_state_size']['upper_bound'] = 32
            else:
                hp_config['hyperparameters']['hidden_state_size']['lower_bound'] = 8
                hp_config['hyperparameters']['hidden_state_size']['upper_bound'] = 256
            
            # save
            output_path = Path(__file__).parent.parent / 'tuning_config' / f'config_{model}_{dataset}_{pair_id}.yaml'
            with open(output_path, 'w') as f:
                yaml.dump(hp_config, f)