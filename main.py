import argparse
import glob
import os
import re
from nndriver import Processor, FeatureTransformer, NNModel, NNTrainer, NNDriver, Visualizer
from utils import read_csv_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="train, infer, realtime, process, visualize-realtime, visualize-realtime-absolute or visualize-inferred")
    parser.add_argument("--csv", type=str,
                        help="Path pattern to CSV files (can include wildcards in quotes)")
    parser.add_argument("--csv2", type=str,
                        help="Path to second CSV file")
    parser.add_argument("--json", type=str, default="default.json",
                        help="Path pattern to track JSON files (can include wildcards in quotes)")
    parser.add_argument("--transformer", type=str, default="transformer.joblib",
                        help="Path to save/load the fitted scaler/PCA")
    parser.add_argument("--model", type=str, default="nn_model.pt",
                        help="Path to save/load the trained model")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output path pattern for processed CSV files (can include wildcards in quotes)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="TCP server host")
    parser.add_argument("--port", type=int, default=65432,
                        help="TCP server port")
    parser.add_argument("--option", type=str, default="onthefly")
    args = parser.parse_args()

    # Create common components for the pipeline
    processor = Processor()
    transformer = FeatureTransformer()
    nn_model = NNModel()

    mode = args.mode.lower()
    use_postprocessed = (args.option.lower() == 'postprocessed')

    if mode == "process":
        if not args.csv:
            print("Must provide --csv for processing.")
            return
        if not args.output_csv:
            print("Must provide --output_csv for saving processed data.")
            return

        # Expand wildcards in the input CSV pattern
        input_files = glob.glob(args.csv)
        if not input_files:
            print(f"No files match the pattern: {args.csv}")
            return

        # Save the provided patterns (they should include a single '*' placeholder)
        json_pattern = args.json
        output_pattern = args.output_csv

        # Create output directory if needed
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Process each CSV file
        for input_file in input_files:
            filename = os.path.basename(input_file)
            # Use regex to capture digits after 'track'
            match = re.search(r'track(\d+)', filename)
            if match:
                track_number = match.group(1)
            else:
                print(f"Could not extract track number from {filename}, skipping.")
                continue

            # Build JSON and output file names by replacing '*' with the track number
            json_file = json_pattern.replace('*', track_number)
            output_file = output_pattern.replace('*', track_number)

            print(f"[Processor] Processing track {track_number}:")
            print(f"  Input: {input_file}")
            print(f"  JSON: {json_file}")
            print(f"  Output: {output_file}")

            if not os.path.exists(json_file):
                print(f"  Error: JSON file {json_file} not found. Skipping.")
                continue

            data_dict = read_csv_data(input_file)
            if data_dict is None:
                print(f"  Error: Could not load CSV data from {input_file}. Skipping.")
                continue

            track_data = processor.build_track_data(json_file)
            df_features = processor.process_csv(data_dict, track_data)
            df_features.to_csv(output_file, index=False)
            print(f"  Success: Processed CSV saved to {output_file}")

    elif mode == "train":
        if not args.csv:
            print("Must provide --csv for training.")
            return
        trainer = NNTrainer(processor, transformer, nn_model)
        trainer.train(csv_path=args.csv,
                      json_path=args.json,
                      output_csv_path=args.output_csv,
                      use_postprocessed=use_postprocessed)
    elif mode == "infer":
        if not args.csv:
            print("Must provide --csv for inference.")
            return
        driver = NNDriver(processor, transformer, nn_model, output_csv=args.output_csv)
        driver.inference_mode(csv_path=args.csv, json_path=args.json)
    elif mode == "realtime":
        driver = NNDriver(processor, transformer, nn_model)
        driver.realtime_mode(json_path=args.json, host=args.host, port=args.port)
    elif mode == "visualize-relative":
        if not args.csv:
            print("Must provide --csv for visualize-realtime.")
            return
        viz = Visualizer(processor, transformer, nn_model)
        viz.visualize_relative(csv_path=args.csv, json_path=args.json, use_postprocessed=use_postprocessed)
    elif mode == "visualize-absolute":
        if not args.csv:
            print("Must provide --csv for visualize-realtime-absolute.")
            return
        viz = Visualizer(processor, transformer, nn_model)
        viz.visualize_absolute(csv_path=args.csv, json_path=args.json, use_postprocessed=use_postprocessed)
    elif mode == "visualize-inferred":
        if not args.csv:
            print("Must provide --csv for visualize-inferred")
            return
        viz = Visualizer(processor, transformer, nn_model)
        viz.visualizer_inferred(actual_csv_path=args.csv, inferred_csv_path=args.csv2)
    else:
        print("Unknown mode. Use --mode train, infer, realtime, visualize-realtime, or visualize-realtime-absolute.")

if __name__ == "__main__":
    main()
