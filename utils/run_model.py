from time import strftime
import subprocess



# Function to run the speciesnet model and output json with the timestamp
def run_speciesnet(input_folder, model_path):
    timestamp = strftime("%Y%m%d-%H%M%S")
    output_json = f"predictions_{timestamp}.json"
    command = [
        "python", "-m", "speciesnet.scripts.run_model",
        "--folders", input_folder,
        "--predictions_json", output_json,
        "--model", model_path
    ]
    subprocess.run(command)
    print(f"SpeciesNet model run completed. Predictions saved to {output_json}")