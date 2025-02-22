import argparse
# Import or define your refactored classes:
from nndriver import Processor, FeatureTransformer, NNModel, NNTrainer, NNDriver, Visualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="train, infer, realtime, visualize-realtime, or visualize-realtime-absolute")
    parser.add_argument("--csv", type=str,
                        help="Path to CSV file (for train, infer, visualize-realtime, or visualize-realtime-absolute)")
    parser.add_argument("--json", type=str, default="default.json",
                        help="Path to track JSON")
    parser.add_argument("--transformer", type=str, default="transformer.joblib",
                        help="Path to save/load the fitted scaler/PCA")
    parser.add_argument("--model", type=str, default="nn_model.joblib",
                        help="Path to save/load the trained model")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional path to save postprocessed CSV in train mode")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="TCP server host")
    parser.add_argument("--port", type=int, default=65432,
                        help="TCP server port")
    args = parser.parse_args()

    # Create common components for the pipeline
    processor = Processor()
    transformer = FeatureTransformer()  # must implement fit_transform, transform, save, load, etc.
    nn_model = NNModel()                # your neural network model class

    mode = args.mode.lower()

    if mode == "train":
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
        driver = NNDriver(processor, transformer, nn_model)
        driver.inference_mode(csv_path=args.csv, json_path=args.json)
    elif mode == "realtime":
        driver = NNDriver(processor, transformer, nn_model)
        driver.realtime_mode(track_json=args.json, host=args.host, port=args.port)
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
    else:
        print("Unknown mode. Use --mode train, infer, realtime, visualize-realtime, or visualize-realtime-absolute.")

if __name__ == "__main__":
    main()
