from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import json
import os
import subprocess
from transaction_status import sync_stage_status

load_dotenv()

LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
CODE_NODE_TIMEOUT_SECONDS = int(os.getenv("CODE_NODE_TIMEOUT_SECONDS", "30"))

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client["idp"]["LogEntry"]


def _get_transaction_id(process_instance_id):
    tid_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"process-instance-{process_instance_id}", "tid.json")
    if not os.path.exists(tid_path):
        return None
    try:
        with open(tid_path, "r", encoding="utf-8") as f:
            return json.load(f).get("transactionId")
    except Exception as exc:
        print(f"Warning: failed to read transaction id from {tid_path}: {exc}")
        return None


def log_to_mongo(process_instance_id, node_name, message, log_type=1, remark=""):
    try:
        mongo_collection.insert_one({
            "processInstanceId": process_instance_id,
            "processInstanceTransactionId": _get_transaction_id(process_instance_id),
            "nodeName": node_name,
            "logsDescription": message,
            "logType": log_type,
            "isDeleted": False,
            "isActive": True,
            "remark": remark,
            "createdAt": datetime.utcnow()
        })
    except Exception as mongo_err:
        print(f"Failed to log to MongoDB: {mongo_err}")


def _read_json(path, fallback):
    if not os.path.exists(path):
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _fetch_blueprint(process_instance_id, process_instance_dir, cursor):
    blueprint_path = os.path.join(process_instance_dir, "blueprint.json")
    if os.path.exists(blueprint_path):
        return _read_json(blueprint_path, [])

    cursor.execute(
        """
        SELECT b.bluePrint
        FROM ProcessInstances pi
        JOIN Processes p ON p.id = pi.processesId
        JOIN BluePrint b ON b.id = p.bluePrintId
        WHERE pi.id = %s
        """,
        (process_instance_id,)
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        raise ValueError("Blueprint not found for process instance")

    blueprint = json.loads(row[0])
    _write_json(blueprint_path, blueprint)
    return blueprint


def _find_node_component(blueprint):
    for node in blueprint:
        node_name = str(node.get("nodeName", "")).strip().lower()
        if node_name == "code":
            return node.get("component", {}) or {}
    return {}


def _build_runner_script():
    return """const fs = require("fs");
const vm = require("vm");

async function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  const payload = JSON.parse(fs.readFileSync(inputPath, "utf8"));
  const capturedLogs = [];

  const sandbox = {
    console: {
      log: (...args) => capturedLogs.push(args.map((arg) => {
        if (typeof arg === "string") return arg;
        try {
          return JSON.stringify(arg);
        } catch (error) {
          return String(arg);
        }
      }).join(" ")),
      error: (...args) => capturedLogs.push(args.map((arg) => String(arg)).join(" ")),
      warn: (...args) => capturedLogs.push(args.map((arg) => String(arg)).join(" "))
    },
    JSON,
    Math,
    Date,
    Array,
    Object,
    String,
    Number,
    Boolean,
    RegExp,
    Promise,
    result: null,
    input: payload.input || {},
    context: payload.context || {}
  };

  vm.createContext(sandbox);

  const wrappedCode = `(async function(input, context) {
${payload.code}
})`;

  const script = new vm.Script(wrappedCode);
  const userFunction = script.runInContext(sandbox, { timeout: payload.timeoutMs || 5000 });
  sandbox.result = await userFunction(sandbox.input, sandbox.context);

  fs.writeFileSync(outputPath, JSON.stringify({
    success: true,
    result: sandbox.result,
    logs: capturedLogs
  }, null, 2));
}

main().catch((error) => {
  const outputPath = process.argv[3];
  fs.writeFileSync(outputPath, JSON.stringify({
    success: false,
    error: {
      message: error && error.message ? error.message : String(error),
      stack: error && error.stack ? error.stack : null
    }
  }, null, 2));
  process.exit(1);
});
"""


def _execute_code(process_instance_dir, code, payload):
    input_path = os.path.join(process_instance_dir, "code_node_input.json")
    output_path = os.path.join(process_instance_dir, "code_node_execution.json")
    runner_path = os.path.join(process_instance_dir, "code_node_runner.js")

    _write_json(input_path, payload)
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(_build_runner_script())

    completed = subprocess.run(
        ["node", runner_path, input_path, output_path],
        capture_output=True,
        text=True,
        timeout=CODE_NODE_TIMEOUT_SECONDS,
        cwd=process_instance_dir,
        check=False,
    )

    result = _read_json(output_path, {})
    return completed, result, output_path


def run_code_node(**context):
    process_instance_id = context["dag_run"].conf.get("id")
    if not process_instance_id:
        raise ValueError("Missing process_instance_id in dag_run.conf")

    process_instance_dir = os.path.join(
        LOCAL_DOWNLOAD_DIR,
        f"process-instance-{process_instance_id}"
    )
    os.makedirs(process_instance_dir, exist_ok=True)

    hook = MySqlHook(mysql_conn_id="idp_mysql")
    conn = hook.get_conn()
    cursor = conn.cursor()

    try:
        transaction_id = sync_stage_status(cursor, process_instance_id, "Code", 1)
        conn.commit()

        blueprint = _fetch_blueprint(process_instance_id, process_instance_dir, cursor)
        component = _find_node_component(blueprint)
        if not component:
            raise ValueError("Code node not found in blueprint")

        code = str(component.get("code", "")).strip()
        if not code:
            raise ValueError("Code node is missing JavaScript code")

        mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
        mcp_context = _read_json(mcp_context_path, {})
        execution_payload = {
            "code": code,
            "timeoutMs": CODE_NODE_TIMEOUT_SECONDS * 1000,
            "input": mcp_context,
            "context": {
                "processInstanceId": process_instance_id,
                "nodeName": "Code",
            },
        }

        log_to_mongo(
            process_instance_id,
            "Code",
            "Executing JavaScript code node",
            log_type=0,
        )

        completed, execution_result, execution_output_path = _execute_code(
            process_instance_dir,
            code,
            execution_payload,
        )

        if completed.stdout.strip():
            log_to_mongo(
                process_instance_id,
                "Code",
                f"Node stdout: {completed.stdout.strip()}",
                log_type=0,
            )

        if completed.stderr.strip():
            log_to_mongo(
                process_instance_id,
                "Code",
                f"Node stderr: {completed.stderr.strip()}",
                log_type=3,
            )

        if completed.returncode != 0 or not execution_result.get("success"):
            error_message = (
                execution_result.get("error", {}).get("message")
                or completed.stderr.strip()
                or "Unknown code node execution error"
            )
            raise RuntimeError(error_message)

        response_payload = {
            "sourceType": "code",
            "language": "javascript",
            "result": execution_result.get("result"),
            "logs": execution_result.get("logs", []),
        }
        response_path = os.path.join(process_instance_dir, "code_node_response.json")
        _write_json(response_path, response_payload)

        mcp_context["code_node_response_path"] = response_path
        mcp_context["code_node_output"] = execution_result.get("result")
        mcp_context["code_node_logs"] = execution_result.get("logs", [])
        mcp_context["code_node_execution_details_path"] = execution_output_path
        _write_json(mcp_context_path, mcp_context)

        log_to_mongo(
            process_instance_id,
            "Code",
            "Code node executed successfully",
            log_type=2,
        )

    except subprocess.TimeoutExpired as exc:
        conn.rollback()
        log_to_mongo(
            process_instance_id,
            "Code",
            f"Code node timed out after {CODE_NODE_TIMEOUT_SECONDS} seconds",
            log_type=1,
            remark="code_node_dag timeout",
        )
        raise RuntimeError(f"Code node timed out after {CODE_NODE_TIMEOUT_SECONDS} seconds") from exc
    except Exception as exc:
        conn.rollback()
        log_to_mongo(
            process_instance_id,
            "Code",
            f"Code node failed: {type(exc).__name__}: {exc}",
            log_type=1,
            remark="code_node_dag failure",
        )
        raise
    finally:
        cursor.close()
        conn.close()


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}


with DAG(
    dag_id="code_node_dag",
    default_args=default_args,
    description="Execute custom JavaScript code node for process workflows",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["process", "code"],
) as dag:

    run_task = PythonOperator(
        task_id="run_code_node",
        python_callable=run_code_node,
    )

    run_task
