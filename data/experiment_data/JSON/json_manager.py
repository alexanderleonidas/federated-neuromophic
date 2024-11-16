import json

class JSONManager:
    @staticmethod
    def save_to_json(data, file_path):
        """
        Saves a list of dictionaries or a dictionary to a JSON file.

        Args:
            data (list of dict or dict): The data to save.
            file_path (str): The path to the file where data will be saved.
        """
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    @staticmethod
    def load_from_json(file_path):
        """
        Loads data from a JSON file.

        Args:
            file_path (str): The path to the file to read data from.

        Returns:
            list of dict or dict: The loaded data.
        """
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
