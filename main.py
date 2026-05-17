import argparse
from src.utils.io import save_df_to_csv
from src.utils.config import load_config


def main(mode: str):
    config_path="configs/config.yaml"
    config = load_config(config_path)

    mode_splitted = mode.split(" ")
    
    if "scrape" in mode_splitted:
        from src.scraping.scraper import run_scraper

        run_scraper(config)
    
    if "clean" in mode_splitted:
        from src.data.cleaning import run_cleaning

        cleaned_data = run_cleaning(config)
        save_df_to_csv(cleaned_data, config["data"]["interim_path"])

    if "featuring" in mode_splitted:
        from src.data.featuring import run_featuring

        data = run_featuring(config)
        save_df_to_csv(data, config["data"]["processed_path"])

    if "train" in mode_splitted:
        from src.models.train import run_training

        eval_errors = run_training(config)
    
        for name, score in eval_errors.items():
            print(f"{name}: MAE = {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    main(args.mode)
