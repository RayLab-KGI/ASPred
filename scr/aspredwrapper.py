

"""
Reads the DB for get pending sequences.
Run pending sequences
Retrieve results from aspre output
Sends the results to the DB
"""

import mysql.connector
import csv
import subprocess
import os
import random
from datetime import datetime

from dotenv import load_dotenv


#pp python path
PyP = "/home/sbassi/miniconda3/envs/prost_t5_cuda12.4.1/bin/python"
#mp model path
PrP = '/home/sbassi/src/scr/run_new_set.py'
MP = "/home/sbassi/PT5_finetuned.pth"

load_dotenv('.env.prod')
db_config = {
        'user': 'sbassimain',
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'database': 'aspred'
    }


def generate_aspred_input(config):

    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        # Query to get pending sequences
        query = """
        SELECT id, sequence 
        FROM sequence_analyzer_sequencesubmission 
        WHERE status = 'pending'
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        id_lst = [row[0] for row in results]
        
        with open('forASPRED.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["sequence", "label"])
            for _, sequence in results:
                writer.writerow([sequence, 0])
        
        print(f"Created forASPRED.csv with {len(results)} sequences")
        print(f"ID list contains {len(id_lst)} IDs")
        
        return id_lst
        
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        if 'conn' in locals():
            conn.close()


def run_prediction(python_path, prg_path, model_path):
    print("Running prediction script...")
    subprocess.run(f"{python_path} {prg_path} {model_path} forASPRED.csv", shell=True, check=True)


def run_prediction_test():
    """Create mock predictions file"""
    try:
        with open('forASPRED.csv', 'r', newline='') as infile:
            reader = csv.reader(infile)
            sequences = list(reader)
        with open('forASPRED_predictions.csv', 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['sequence', 'label', 'predicted_probs', 'predicted_labels'])
            for sequence, label in sequences:
                prob = random.random()
                pred_label = 1 if prob >= 0.5 else 0
                writer.writerow([sequence, label, prob, pred_label])
        print("Created mock predictions file: forASPRED_predictions.csv")
    except FileNotFoundError:
        print("Error: forASPRED.csv not found")
    except Exception as e:
        print(f"Error creating predictions file: {e}")


def read_output():
    """Read the third column from the predictions CSV file"""
    predictions = []
    try:
        with open('forASPRED_predictions.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if len(row) >= 3:
                    predictions.append(row[2]) 
        return predictions
    except FileNotFoundError:
        print("Error: forASPRED_predictions.csv not found")
        return []
    except Exception as e:
        print(f"Error reading predictions file: {e}")
        return []


def update_database(id_lst, predictions, config):
    """Update the database with prediction results"""

    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        now = datetime.now()
        for id_val, prediction in zip(id_lst, predictions):
            result = float(prediction)
            query = """
            UPDATE sequence_analyzer_sequencesubmission
            SET result = %s, result_date = %s, status = 'done'
            WHERE id = %s
            """
            cursor.execute(query, (result, now, id_val))
        conn.commit()
        print(f"Updated {len(id_lst)} rows in the database")

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    id_lst = generate_aspred_input(db_config)
    print("ID list:", id_lst)
    run_prediction(PyP, PrP, MP)
    ##run_prediction_test()
    preds_lst = read_output()
    print(preds_lst)
    update_database(id_lst, preds_lst, db_config)
