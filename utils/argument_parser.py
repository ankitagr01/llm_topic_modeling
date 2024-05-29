import argparse
import yaml

class ArgumentParser:
    """
    Class for parsing command-line arguments and loading configurations.

    Attributes:
        config_file (str): Path to the configuration file.
    """

    def __init__(self, config_file: str = "config/config.yaml"):
        self.config_file = config_file

    def parse_args(self):
        """
        Parses command-line arguments.

        Returns:
            Namespace: Parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(description="LLaMA Topic Modeling")
        parser.add_argument("--config", type=str, default=self.config_file, help="Path to the config file.")
        parser.add_argument("--model_name", type=str, default=None, help="Name of the student model to use.")
        args = parser.parse_args()
        return args

    def load_config(self, args):
        """
        Loads the configuration from a YAML file and updates it with command-line arguments.

        Args:
            args (Namespace): Parsed command-line arguments.

        Returns:
            dict: Loaded configuration.
        """
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

        if args.model_name:
            config['model']['student_model_name'] = args.model_name

        return config
