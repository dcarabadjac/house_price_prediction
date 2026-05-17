import csv  # BEGIN: ed8c6549bwf9
import logging  # END: ed8c6549bwf9
import os  # BEGIN: ed8c6549bwf9

def load_csv():
    pass

def save_df_to_csv(df, filename):
    """
    Save a pandas DataFrame to a CSV file.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be saved.
    filename (str): The name of the file where the DataFrame will be saved.

    The function saves the DataFrame to a CSV file with UTF-8 encoding and logs
    the number of records saved.
    """
    df.to_csv(filename, index=False, encoding="utf-8")
    logging.info(f"Сохранено в {filename} ({len(df)} записей)")


def save_items_to_csv(items, filename):
    """
    Save a list of dictionaries to a CSV file.

    Parameters:
    items (list of dict): A list of dictionaries containing data to be saved.
    filename (str): The name of the file where the data will be saved.

    The function writes the keys "id", "title", "price", and "link" as the header
    of the CSV file and populates the rows with the corresponding values from the
    dictionaries in the items list. It logs the number of records saved.
    """
    if not items:
        logging.warning("Нет данных для сохранения")
        return

    # Собираем все уникальные колонки
    keys = set()
    for it in items:
        keys.update(it.keys())
    keys = list(keys)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for it in items:
            writer.writerow({k: it.get(k, "") for k in keys})
    logging.info(f"Сохранено в {filename} ({len(items)} записей)")