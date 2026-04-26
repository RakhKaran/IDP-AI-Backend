import json
import os


LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"


def get_transaction_id(process_instance_id, local_download_dir=LOCAL_DOWNLOAD_DIR):
    tid_path = os.path.join(
        local_download_dir,
        f"process-instance-{process_instance_id}",
        "tid.json",
    )

    if not os.path.exists(tid_path):
        return None

    try:
        with open(tid_path, "r", encoding="utf-8") as file:
            return json.load(file).get("transactionId")
    except Exception as exc:
        print(f"Warning: failed to read transaction id from {tid_path}: {exc}")
        return None


def sync_stage_status(cursor, process_instance_id, current_stage, is_instance_running=1):
    transaction_id = get_transaction_id(process_instance_id)

    if transaction_id:
        cursor.execute(
            """
            UPDATE ProcessInstanceTransactions
            SET currentStage = %s, updatedAt = NOW()
            WHERE id = %s
            """,
            (current_stage, transaction_id),
        )
    else:
        print(
            f"Warning: no transaction id found for process_instance_id={process_instance_id}; "
            "skipping ProcessInstanceTransactions update."
        )

    cursor.execute(
        """
        UPDATE ProcessInstances
        SET currentStage = %s, isInstanceRunning = %s, updatedAt = NOW()
        WHERE id = %s
        """,
        (current_stage, is_instance_running, process_instance_id),
    )

    return transaction_id
