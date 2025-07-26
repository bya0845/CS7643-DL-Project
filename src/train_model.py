import yaml
import os
import sys  
import argparse
from solver import Solver

script_dir = os.path.dirname(__file__)
project_base_dir = os.path.abspath(script_dir)
sys.path.append(project_base_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, nargs='?', const=128)
    parser.add_argument("-lr", type=float, nargs='?', const=0.001)
    parser.add_argument("-reg", type=float, nargs='?', const=0.6)
    parser.add_argument("-epochs", type=int, nargs='?', const=100)
    parser.add_argument("-warmup", type=int, nargs='?', const=10)
    parser.add_argument("-momentum", type=float, nargs='?', const=0.9)
    parser.add_argument("-scheduler", type=str, nargs='?', const="cosine")
    parser.add_argument("-min_lr", type=float, nargs='?', const=0.00001)
    parser.add_argument("-optimizer", type=str, nargs='?', const="adamw")
    parser.add_argument("-nesterov", type=bool, nargs='?', const=True)

    parser.add_argument("-save_best", type=bool, nargs='?', const=True)
    parser.add_argument("-dataset", type=str, nargs='?', const="Food101")
    parser.add_argument("-img_size", type=int, nargs='?', const=64)
    parser.add_argument("-output_filename", type=str, nargs='?', const=None)

    parser.add_argument("-mlflow", type=bool, nargs='?', const=True)
    parser.add_argument("-experiment_name", type=str, nargs='?', const="CCT Hyperparameter Tuning")
    parser.add_argument("-training_info", type=str, nargs='?', const="CCT Initial Hyperparameter Tuning")
    parser.add_argument("-training_title", type=str, nargs='?', const="CCT Tuning")

    parser.add_argument("-prune", type=bool, nargs='?', const=True)
    parser.add_argument("-prune_unstructured", type=float, nargs='?', const=0.1)
    parser.add_argument("-prune_structured", type=float, nargs='?', const=0.1)

    parser.add_argument("-num_layers", type=int, nargs='?', const=6)
    parser.add_argument("-embedding_dim", type=int, nargs='?', const=256)
    parser.add_argument("-mlp_ratio", type=float, nargs='?', const=2.0)
    parser.add_argument("-num_heads", type=int, nargs='?', const=4)

    args = parser.parse_args()

    try:
        model_name = os.environ['MODEL_NAME']
        config_file_name = f"config_{model_name}.yaml"
    except KeyError as e:
        missing_var = str(e).strip("'")
        print(f'Missing required environment variable: {missing_var}')
        sys.exit(1)

    CONFIGS_DIR = os.path.join(project_base_dir, "configs")
    config_file_path = os.path.join(CONFIGS_DIR, config_file_name)

    try:
        with open(config_file_path, "r") as read_file:
            config = yaml.safe_load(read_file)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_file_path}")
        sys.exit(1)

    kwargs = {}
    for key in config:
        if isinstance(config[key], dict):
            for k, v in config[key].items():
                if k != 'description':
                    kwargs[k] = v

    kwargs['path_prefix'] = project_base_dir
    kwargs['model_name'] = model_name

    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            config_name = 'learning_rate' if arg_name == 'lr' else arg_name
            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                print(f'Set {config_name} to {arg_value} via command line')
            kwargs[config_name] = arg_value
 
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print("\n=== SOLVER CONFIGURATION ===")
        for key, value in kwargs.items():
            if key != 'databricks_token':
                print(f"{key:20}: {value}")
        print("============================")

    solver = Solver(**kwargs)
    solver.run()

    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        for key, value in kwargs.items():
            if key != 'databricks_token':
                print(f"{key:20}: {value}")
        print("="*60)

if __name__ == "__main__":
    main()
    