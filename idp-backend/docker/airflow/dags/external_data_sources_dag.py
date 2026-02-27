from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
from collections import deque
from urllib.parse import urlencode
from urllib.parse import urljoin, urlparse
import json
import os
import re
import requests

load_dotenv()

LOCAL_DOWNLOAD_DIR = "/opt/airflow/downloaded_docs"
EXTERNAL_DATA_TIMEOUT_SECONDS = int(os.getenv("EXTERNAL_DATA_TIMEOUT_SECONDS", "60"))

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client["idp"]["LogEntry"]


def log_to_mongo(process_instance_id, node_name, message, log_type=1, remark=""):
    try:
        mongo_collection.insert_one({
            "processInstanceId": process_instance_id,
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
    candidate_names = {"external data sources", "external data source"}
    for node in blueprint:
        node_name = str(node.get("nodeName", "")).strip().lower()
        if node_name in candidate_names:
            return node.get("component", {}) or {}
    return {}


def _to_dict(items):
    out = {}
    if not isinstance(items, list):
        return out
    for item in items:
        key = str(item.get("key", "")).strip()
        if not key:
            continue
        out[key] = str(item.get("value", ""))
    return out


def _apply_path_params(url, params_map):
    result = url
    for key, value in params_map.items():
        result = result.replace(f"{{{{{key}}}}}", value)
        result = result.replace(f":{key}", value)
    return result


def _split_lines(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).splitlines() if item.strip()]


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _extract_links(html, base_url):
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html or "", flags=re.IGNORECASE)
    links = []
    for href in hrefs:
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme in {"http", "https"}:
            links.append(absolute)
    return links


def _extract_text(html):
    if not html:
        return ""
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _get_title(html):
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html or "")
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _matches_patterns(url, include_patterns, exclude_patterns):
    if include_patterns and not any(pattern in url for pattern in include_patterns):
        return False
    if exclude_patterns and any(pattern in url for pattern in exclude_patterns):
        return False
    return True


def _is_same_scope(root_url, candidate_url, follow_subdomains):
    root_host = urlparse(root_url).hostname or ""
    candidate_host = urlparse(candidate_url).hostname or ""
    if not root_host or not candidate_host:
        return False
    if follow_subdomains:
        return candidate_host == root_host or candidate_host.endswith(f".{root_host}")
    return candidate_host == root_host


def _run_api_connector(component, process_instance_id, process_instance_dir):
    method = str(component.get("apiMethod", "GET")).upper().strip()
    url = str(component.get("apiUrl", "")).strip()
    if not url:
        raise ValueError("External Data Sources API URL is missing")

    headers = _to_dict(component.get("headers", []))
    query_params = _to_dict(component.get("queryParams", []))
    body = component.get("body", "")

    final_url = _apply_path_params(url, {})
    if query_params:
        separator = "&" if "?" in final_url else "?"
        final_url = f"{final_url}{separator}{urlencode(query_params)}"

    payload = {}
    if method in {"POST", "PUT", "PATCH", "DELETE"}:
        if isinstance(body, (dict, list)):
            payload["json"] = body
        else:
            raw_body = str(body or "").strip()
            if raw_body:
                try:
                    payload["json"] = json.loads(raw_body)
                except Exception:
                    payload["data"] = raw_body

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"Calling API connector {method} {final_url}",
        log_type=0,
    )

    response = requests.request(
        method=method,
        url=final_url,
        headers=headers,
        timeout=EXTERNAL_DATA_TIMEOUT_SECONDS,
        **payload,
    )

    response_payload = {
        "sourceType": "api",
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": None,
    }

    try:
        response_payload["body"] = response.json()
    except Exception:
        response_payload["body"] = response.text

    response_path = os.path.join(process_instance_dir, "external_data_sources_api_response.json")
    _write_json(response_path, response_payload)

    mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
    mcp_context = _read_json(mcp_context_path, {})
    mcp_context["external_data_sources_response_path"] = response_path
    mcp_context["external_data_sources_type"] = "api"
    mcp_context["external_data_sources_status_code"] = response.status_code
    _write_json(mcp_context_path, mcp_context)

    if not response.ok:
        raise RuntimeError(f"External API call returned status {response.status_code}")

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"API connector succeeded with status {response.status_code}",
        log_type=2,
    )


def _run_website_connector(component, process_instance_id, process_instance_dir):
    start_urls = _split_lines(component.get("websiteStartUrls", ""))
    if not start_urls:
        raise ValueError("Website connector requires at least one start URL")

    max_depth = int(component.get("websiteMaxDepth", 1))
    max_pages = int(component.get("websiteMaxPages", 20))
    include_patterns = _split_lines(component.get("websiteIncludePatterns", ""))
    exclude_patterns = _split_lines(component.get("websiteExcludePatterns", ""))
    follow_subdomains = _to_bool(component.get("websiteFollowSubdomains", False))
    respect_robots_txt = _to_bool(component.get("websiteRespectRobotsTxt", True))
    render_js = _to_bool(component.get("websiteRenderJs", False))
    requirement = str(component.get("websiteSpecificRequirement", "")).strip()
    output_schema = str(component.get("websiteOutputSchema", "")).strip()

    max_depth = max(0, min(max_depth, 5))
    max_pages = max(1, min(max_pages, 200))

    queue = deque((url, 0, url) for url in start_urls)
    visited = set()
    crawled_pages = []

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"Website crawl started for {len(start_urls)} URL(s), depth={max_depth}, maxPages={max_pages}",
        log_type=0,
    )
    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        "websiteRespectRobotsTxt and websiteRenderJs are captured for future provider integration; current crawler ignores them.",
        log_type=3,
    )

    while queue and len(crawled_pages) < max_pages:
        current_url, depth, root_url = queue.popleft()
        if current_url in visited:
            continue
        visited.add(current_url)

        if not _matches_patterns(current_url, include_patterns, exclude_patterns):
            continue

        try:
            response = requests.get(
                current_url,
                timeout=EXTERNAL_DATA_TIMEOUT_SECONDS,
                headers={"User-Agent": "IDP-ExternalDataCrawler/1.0"},
            )
            response.raise_for_status()
            content_type = str(response.headers.get("content-type", "")).lower()
            if "text/html" not in content_type:
                continue

            html = response.text
            crawled_pages.append({
                "url": current_url,
                "depth": depth,
                "title": _get_title(html),
                "text": _extract_text(html),
            })

            if depth < max_depth:
                for next_url in _extract_links(html, current_url):
                    if _is_same_scope(root_url, next_url, follow_subdomains):
                        queue.append((next_url, depth + 1, root_url))
        except Exception as exc:
            log_to_mongo(
                process_instance_id,
                "External Data Sources",
                f"Website crawl skipped {current_url}: {type(exc).__name__}: {exc}",
                log_type=3,
            )

    result_payload = {
        "sourceType": "website",
        "summary": {
            "startUrls": start_urls,
            "maxDepth": max_depth,
            "maxPages": max_pages,
            "followSubdomains": follow_subdomains,
            "respectRobotsTxt": respect_robots_txt,
            "renderJs": render_js,
            "visitedCount": len(visited),
            "crawledCount": len(crawled_pages),
        },
        "specificRequirement": requirement,
        "outputSchema": output_schema,
        "pages": crawled_pages,
    }

    response_path = os.path.join(process_instance_dir, "external_data_sources_website_response.json")
    _write_json(response_path, result_payload)

    mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
    mcp_context = _read_json(mcp_context_path, {})
    mcp_context["external_data_sources_response_path"] = response_path
    mcp_context["external_data_sources_type"] = "website"
    _write_json(mcp_context_path, mcp_context)

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"Website crawl completed with {len(crawled_pages)} pages",
        log_type=2,
    )


def _run_db_connector_placeholder(component, process_instance_id):
    db_type = component.get("dbType") or "unknown"
    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"DB connector selected ({db_type}). MCP mapping placeholder reached.",
        log_type=3,
        remark="TODO: implement MCP DB connector mapping and invocation",
    )
    raise NotImplementedError("DB connector execution is pending MCP integration.")


def _run_bigdata_connector_placeholder(component, process_instance_id):
    big_data_type = component.get("bigDataType") or "unknown"
    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"Big Data connector selected ({big_data_type}). MCP mapping placeholder reached.",
        log_type=3,
        remark="TODO: implement MCP big data connector mapping and invocation",
    )
    raise NotImplementedError("Big Data connector execution is pending MCP integration.")


def run_external_data_sources(**context):
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
        cursor.execute(
            """
            UPDATE ProcessInstances
            SET currentStage = %s, isInstanceRunning = 1, updatedAt = NOW()
            WHERE id = %s
            """,
            ("External Data Sources", process_instance_id),
        )
        conn.commit()

        blueprint = _fetch_blueprint(process_instance_id, process_instance_dir, cursor)
        component = _find_node_component(blueprint)
        if not component:
            raise ValueError("External Data Sources node not found in blueprint")

        source_type = str(component.get("sourceType", "")).strip().lower()
        if source_type == "api":
            _run_api_connector(component, process_instance_id, process_instance_dir)
        elif source_type == "website":
            _run_website_connector(component, process_instance_id, process_instance_dir)
        elif source_type == "db":
            _run_db_connector_placeholder(component, process_instance_id)
        elif source_type == "bigdata":
            _run_bigdata_connector_placeholder(component, process_instance_id)
        else:
            raise ValueError(f"Unsupported sourceType '{source_type}' in External Data Sources node")

    except Exception as exc:
        conn.rollback()
        log_to_mongo(
            process_instance_id,
            "External Data Sources",
            f"External Data Sources failed: {type(exc).__name__}: {exc}",
            log_type=1,
            remark="external_data_sources_dag failure",
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
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="external_data_sources_dag",
    default_args=default_args,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["idp", "external-data-sources"],
) as dag:
    external_data_sources_task = PythonOperator(
        task_id="run_external_data_sources",
        python_callable=run_external_data_sources,
    )
