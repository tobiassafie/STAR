import wandb

# --------------------------
# TOGGLEABLE CONFIGURATION
# --------------------------
config_flags = {
    'sweep_hidden_layers': True,
    'sweep_neurons': True,
    'sweep_activation': True,
    'sweep_init': True,
    'sweep_learning_rate': False,
    'sweep_loss_weights': False,
    'sweep_method': 'random'  # Choose from: 'random', 'bayes', 'grid'
}

# --------------------------
# DEFAULT VALUES (used if not swept)
# --------------------------
fixed_defaults = {
    'hidden_layers': 4,
    'neurons_per_layer': 64,
    'activation': 'tanh',
    'init_method': 'xavier',
    'initial_lr': 0.001,
    'pde_weight_scale': 1.0,
    'bc_weight_scale': 1.0,
    'ic_weight_scale': 1.0
}

# --------------------------
# DYNAMIC SWEEP CONFIG BUILDER
# --------------------------
def generate_sweep_config(flags, defaults):
    sweep_config = {
        'method': flags['sweep_method'],
        'metric': {
            'name': 'final_total_loss',
            'goal': 'minimize'
        },
        'parameters': {}
    }

    if flags['sweep_hidden_layers']:
        sweep_config['parameters']['hidden_layers'] = {
            'values': [2, 4, 6, 8]
        }
    else:
        sweep_config['parameters']['hidden_layers'] = {
            'value': defaults['hidden_layers']
        }

    if flags['sweep_neurons']:
        sweep_config['parameters']['neurons_per_layer'] = {
            'values': [32, 64, 128, 256]
        }
    else:
        sweep_config['parameters']['neurons_per_layer'] = {
            'value': defaults['neurons_per_layer']
        }

    if flags['sweep_activation']:
        sweep_config['parameters']['activation'] = {
            'values': ['tanh', 'relu', 'silu']
        }
    else:
        sweep_config['parameters']['activation'] = {
            'value': defaults['activation']
        }

    if flags['sweep_init']:
        sweep_config['parameters']['init_method'] = {
            'values': ['xavier', 'kaiming', 'normal']
        }
    else:
        sweep_config['parameters']['init_method'] = {
            'value': defaults['init_method']
        }

    if flags['sweep_learning_rate']:
        sweep_config['parameters']['initial_lr'] = {
            'min': 0.0001,
            'max': 0.01,
            'distribution': 'log_uniform_values'
        }
    else:
        sweep_config['parameters']['initial_lr'] = {
            'value': defaults['initial_lr']
        }

    if flags['sweep_loss_weights']:
        sweep_config['parameters']['pde_weight_scale'] = {
            'min': 0.1, 'max': 10.0, 'distribution': 'log_uniform_values'
        }
        sweep_config['parameters']['bc_weight_scale'] = {
            'min': 0.1, 'max': 10.0, 'distribution': 'log_uniform_values'
        }
        sweep_config['parameters']['ic_weight_scale'] = {
            'min': 0.1, 'max': 10.0, 'distribution': 'log_uniform_values'
        }
    else:
        sweep_config['parameters']['pde_weight_scale'] = {'value': defaults['pde_weight_scale']}
        sweep_config['parameters']['bc_weight_scale'] = {'value': defaults['bc_weight_scale']}
        sweep_config['parameters']['ic_weight_scale'] = {'value': defaults['ic_weight_scale']}

    return sweep_config

# --------------------------
# INSTANTIATE SWEEP
# --------------------------
sweep_config = generate_sweep_config(config_flags, fixed_defaults)
sweep_id = wandb.sweep(sweep_config, project="PINN_Sweep_Modular")

print(f"Sweep ID: {sweep_id}")
print("wandb agent " + sweep_id)