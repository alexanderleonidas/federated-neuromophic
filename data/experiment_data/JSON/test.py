
from data.experiment_data.JSON.json_manager import JSONManager
import os

def test_json_parsing_and_modeling():
    # Step 1: Create sample ResultModel instances
    normal_result = {
        "t_statistic": 1.9, "p_value": 0.045, "precision": 0.88,
        "recall": 0.86, "f1_score": 0.87, "total": 200, "correct": 172
    }
    neuromorphic_result = {
        "t_statistic": 2.1, "p_value": 0.03, "precision": 0.85,
        "recall": 0.83, "f1_score": 0.84, "total": 180, "correct": 150
    }


    # Step 3: Save the data to a JSON file
    file_path = 'normal_backprop_single_results_10epochs_standardtrainingparameters.json'
    JSONManager.save_to_json(normal_result, file_path)

    # Step 4: Load the data back from the JSON file
    loaded_dataset_model = JSONManager.load_from_json(file_path)

    # Step 5: Detailed assertions on loaded data
    assert len(loaded_dataset_model.normal) == 1, "Normal results count mismatch after loading"
    assert len(loaded_dataset_model.neuromorphic) == 1, "Neuromorphic results count mismatch after loading"

    # Cleanup the test file
    if os.path.exists(file_path):
        os.remove(file_path)

    print("Test passed: JSON parsing, modeling, and saving/loading functionality verified successfully.")

test_json_parsing_and_modeling()