import argparse
from nndriver import Processor, FeatureTransformer, NNModel, NNTrainer, NNDriver, Visualizer
from utils import read_csv_data  # Make sure this import is available

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="train, infer, realtime, process, visualize-realtime, visualize-realtime-absolute or visualize-inferred")
    parser.add_argument("--csv", type=str,
                        help="Path to CSV file (for train, infer, process, visualize-realtime, or visualize-realtime-absolute)")
    parser.add_argument("--csv2", type=str,
                        help="Path to second CSV file")
    parser.add_argument("--json", type=str, default="default.json",
                        help="Path to track JSON")
    parser.add_argument("--transformer", type=str, default="transformer.joblib",
                        help="Path to save/load the fitted scaler/PCA")
    parser.add_argument("--model", type=str, default="nn_model.pt",
                        help="Path to save/load the trained model")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional path to save processed/output CSV")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="TCP server host")
    parser.add_argument("--port", type=int, default=65432,
                        help="TCP server port")
    args = parser.parse_args()

    # Create common components for the pipeline
    processor = Processor()
    transformer = FeatureTransformer()
    nn_model = NNModel()

    mode = args.mode.lower()

    if mode == "process":
        if not args.csv:
            print("Must provide --csv for processing.")
            return
        if not args.output_csv:
            print("Must provide --output_csv for saving processed data.")
            return
            
        print("[Processor] Processing CSV mode...")
        data_dict = read_csv_data(args.csv)
        if data_dict is None:
            print("Could not load CSV data.")
            return
            
        track_data = processor.build_track_data(args.json)
        df_features = processor.process_csv(data_dict, track_data)
        df_features.to_csv(args.output_csv, index=False)
        print(f"[Processor] Processed CSV saved to {args.output_csv}")
    elif mode == "train":
        if not args.csv:
            print("Must provide --csv for training.")
            return
        trainer = NNTrainer(processor, transformer, nn_model)
        trainer.train(csv_path=args.csv,
                      json_path=args.json,
                      output_csv_path=args.output_csv)
    elif mode == "infer":
        if not args.csv:
            print("Must provide --csv for inference.")
            return
        driver = NNDriver(processor, transformer, nn_model, output_csv = args.output_csv)
        driver.inference_mode(csv_path=args.csv, json_path=args.json)
    elif mode == "realtime":
        driver = NNDriver(processor, transformer, nn_model)
        driver.realtime_mode(json_path=args.json, host=args.host, port=args.port)
    elif mode == "visualize-realtime":
        if not args.csv:
            print("Must provide --csv for visualize-realtime.")
            return
        viz = Visualizer(processor, transformer, nn_model)
        viz.visualize_relative(csv_path=args.csv, json_path=args.json)
    elif mode == "visualize-realtime-absolute":
        if not args.csv:
            print("Must provide --csv for visualize-realtime-absolute.")
            return
        viz = Visualizer(processor, transformer, nn_model)
        viz.visualize_absolute(csv_path=args.csv, json_path=args.json)
    elif mode == "visualize-inferred":
        if not args.csv:
            print("Must provide --csv for visualize-inferred")
        viz = Visualizer(processor, transformer, nn_model)
        viz.visualizer_inferred(actual_csv_path=args.csv, inferred_csv_path=args.csv2)
    else:
        print("Unknown mode. Use --mode train, infer, realtime, visualize-realtime, or visualize-realtime-absolute.")


if __name__ == "__main__":
    main()
