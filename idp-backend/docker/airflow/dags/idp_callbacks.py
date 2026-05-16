import os
import traceback
from datetime import datetime

from transaction_status import pause_process_instance


LOG_TYPE = {
    "info": 0,
    "error": 1,
    "success": 2,
    "warning": 3,
}


STAGE_BY_DAG_ID = {
    "idp_service_orchestrator": "Service-Orchestrator",
    "ingest_documents_dag": "Ingestion",
    "classify_documents_dag": "Classify",
    "extract_documents_dag": "Extract",
    "highlight_extracted_fields_dag": "Validate",
    "deliver_dag": "Deliver",
    "external_data_sources_dag": "External Data Sources",
    "document_index_dag": "Document Index",
    "document_query_dag": "Document Query",
    "integration_dag": "Integration",
    "image_processing_dag": "Image Processing",
    "code_node_dag": "Code",
    "ai_analyser_dag": "AI Analyser",
    "validate_documents_dag": "Validate",
    "ml_classify_documents_dag": "Classify",
    "field_extractor_retrain_dag": "Extract",
    "classifier_retrain_dag": "Classify",
}


def get_process_instance_id_from_context(context) -> str | None:
    dag_run = context.get("dag_run")
    if dag_run and getattr(dag_run, "conf", None):
        return dag_run.conf.get("id")
    return None


def get_stage_from_context(context, fallback: str = "Unknown") -> str:
    dag = context.get("dag")
    dag_id = getattr(dag, "dag_id", None) or context.get("dag_id")
    return STAGE_BY_DAG_ID.get(dag_id, dag_id or fallback)


def log_event(
    context,
    message: str,
    *,
    level: str = "info",
    node_name: str | None = None,
    remark: str = "",
    extra: dict | None = None,
):
    process_instance_id = get_process_instance_id_from_context(context)
    stage = node_name or get_stage_from_context(context)
    log_to_mongo(
        process_instance_id=process_instance_id,
        node_name=stage,
        message=message,
        log_type=LOG_TYPE.get(level, LOG_TYPE["info"]),
        remark=remark,
        extra=extra,
    )


def _get_transaction_id_from_tid_file(process_instance_id: str, local_download_dir: str) -> str | None:
    tid_path = os.path.join(
        local_download_dir,
        f"process-instance-{process_instance_id}",
        "tid.json",
    )
    if not os.path.exists(tid_path):
        return None
    try:
        import json

        with open(tid_path, "r", encoding="utf-8") as f:
            return json.load(f).get("transactionId")
    except Exception:
        return None


def log_to_mongo(
    *,
    process_instance_id: str | None,
    node_name: str,
    message: str,
    log_type: int = LOG_TYPE["info"],
    remark: str = "",
    extra: dict | None = None,
):
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        return

    # Lazy import so DAG parsing doesn't fail in environments missing pymongo.
    from pymongo import MongoClient

    local_download_dir = os.getenv("LOCAL_DOWNLOAD_DIR", "/opt/airflow/downloaded_docs")
    transaction_id = (
        _get_transaction_id_from_tid_file(process_instance_id, local_download_dir)
        if process_instance_id
        else None
    )

    client = MongoClient(mongo_uri)
    try:
        collection = client["idp"]["LogEntry"]
        doc = {
            "processInstanceId": process_instance_id,
            "processInstanceTransactionId": transaction_id,
            "nodeName": node_name,
            "logsDescription": message,
            "logType": log_type,  # 0=info, 1=error, 2=success, 3=warning
            "isDeleted": False,
            "isActive": True,
            "remark": remark,
            "createdAt": datetime.utcnow(),
        }
        if extra:
            doc.update(extra)
        collection.insert_one(doc)
    finally:
        client.close()


def task_failure_callback(context):
    """
    Airflow callback to:
    - log the failure to MongoDB (frontend-visible)
    - pause the process instance in MySQL by setting isInstanceRunning=0
    - update currentStage for both ProcessInstances and ProcessInstanceTransactions
    """
    dag_run = context.get("dag_run")
    dag = context.get("dag")
    task_instance = context.get("task_instance")
    exception = context.get("exception")

    dag_id = getattr(dag, "dag_id", None) or context.get("dag_id")
    task_id = getattr(task_instance, "task_id", None) if task_instance else None

    process_instance_id = None
    if dag_run and getattr(dag_run, "conf", None):
        process_instance_id = dag_run.conf.get("id")

    stage = STAGE_BY_DAG_ID.get(dag_id, dag_id or "Unknown")
    error_message = f"Task failed: dag_id={dag_id} task_id={task_id}"
    if exception:
        error_message = f"{error_message} | error={exception!r}"

    log_to_mongo(
        process_instance_id=process_instance_id,
        node_name=stage,
        message=error_message,
        log_type=LOG_TYPE["error"],
        extra={
            "dagId": dag_id,
            "taskId": task_id,
            "tryNumber": getattr(task_instance, "try_number", None) if task_instance else None,
            "traceback": traceback.format_exc(),
        },
    )

    print("processs instance id:", process_instance_id)
    print("stage:", stage)
    print("error message:", error_message)

    if process_instance_id:
        # Pause instance so UI doesn't keep progressing it.
        pause_process_instance(process_instance_id, 'failed')

        # Clean up tid.json
        local_download_dir = os.getenv("LOCAL_DOWNLOAD_DIR", "/opt/airflow/downloaded_docs")
        tid_path = os.path.join(local_download_dir, f"process-instance-{process_instance_id}", "tid.json")
        if os.path.exists(tid_path):
            try:
                os.remove(tid_path)
                print(f"✅ Cleaned up tid.json for failed process instance {process_instance_id}")
            except Exception as e:
                print(f"⚠️ Warning: Could not remove tid.json: {e}")

