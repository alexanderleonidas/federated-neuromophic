import json
import os

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

    @staticmethod
    def load_json_files_from_directory(directory_path):
        result_arrays = []
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.json'):
                with open(os.path.join(directory_path, file_name)) as file:
                    results = json.load(file)
                    result_arrays.append(results)
        return result_arrays
