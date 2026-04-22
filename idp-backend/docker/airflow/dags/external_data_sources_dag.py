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
EXTERNAL_MCP_BASE_URL = os.getenv("EXTERNAL_MCP_BASE_URL", "https://mcp.jnanic.com").rstrip("/")
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN")

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


def _write_external_response(process_instance_dir, source_type, payload):
    response_path = os.path.join(process_instance_dir, f"external_data_sources_{source_type}_response.json")
    _write_json(response_path, payload)

    mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
    mcp_context = _read_json(mcp_context_path, {})
    mcp_context["external_data_sources_response_path"] = response_path
    mcp_context["external_data_sources_type"] = source_type
    _write_json(mcp_context_path, mcp_context)

    return response_path


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

    _write_external_response(process_instance_dir, "api", response_payload)

    mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
    mcp_context = _read_json(mcp_context_path, {})
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

    _write_external_response(process_instance_dir, "website", result_payload)

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"Website crawl completed with {len(crawled_pages)} pages",
        log_type=2,
    )


def _run_postgres_connector(component, process_instance_id, process_instance_dir):
    sql = str(component.get("dbQuery", "")).strip()
    connection_ref = str(component.get("dbConnectionRef", "")).strip()
    connector_name = str(component.get("dbConnectorName", "")).strip()

    if not sql:
        raise ValueError("PostgreSQL query is required")

    first_keyword = sql.split(None, 1)[0].lower() if sql.split() else ""
    if first_keyword not in {"select", "with", "show", "explain"}:
        raise ValueError("Only read-only PostgreSQL queries are allowed")

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"Calling PostgreSQL MCP tool for connector '{connector_name or 'postgres'}' and ref '{connection_ref or 'default'}'",
        log_type=0,
    )

    response_payload = _post_mcp(
        "/call_tool",
        {
            "tool_name": "Postgres.execute_query",
            "parameters": {
                "sql": sql,
            },
        },
    )

    result_payload = {
        "sourceType": "db",
        "dbType": "postgresql",
        "connectorName": connector_name,
        "connectionRef": connection_ref,
        "toolName": "Postgres.execute_query",
        "query": sql,
        "response": response_payload,
    }

    _write_external_response(process_instance_dir, "postgresql", result_payload)

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        "PostgreSQL MCP query executed successfully",
        log_type=2,
    )


def _run_db_connector(component, process_instance_id, process_instance_dir):
    db_type = str(component.get("dbType") or "unknown").strip().lower()
    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"DB connector selected ({db_type}).",
        log_type=3,
        remark="TODO: implement MCP DB connector mapping and invocation",
    )

    if db_type in {"mysql", "mongodb", "sqlserver"}:
        log_to_mongo(
            process_instance_id,
            "External Data Sources",
            f"(Database selected: {db_type}).",
            log_type=3,
            remark="TODO: implement MCP DB connector mapping and invocation",
        )
        raise RuntimeError(f"Integration pending for selected database: {db_type}")

    if db_type == "postgresql":
        log_to_mongo(
            process_instance_id,
            "External Data Sources",
            "PostgreSQL MCP connector selected.",
            log_type=0,
        )
        _run_postgres_connector(component, process_instance_id, process_instance_dir)
        return

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        f"(Invalid database selected: {db_type}).",
        log_type=1,
        remark="TODO: implement MCP DB connector mapping and invocation",
    )
    raise RuntimeError(f"Invalid database selected: {db_type}")
    

def _mcp_headers():
    headers = {"Content-Type": "application/json"}
    if MCP_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_AUTH_TOKEN}"
    return headers


def _post_mcp(path, payload):
    response = requests.post(
        f"{EXTERNAL_MCP_BASE_URL}{path}",
        headers=_mcp_headers(),
        json=payload,
        timeout=EXTERNAL_DATA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return {"raw_text": response.text}


def _connect_databricks_mcp(component):
    workspace_url = str(component.get("databricksWorkspaceUrl", "")).strip()
    databricks_token = str(component.get("databricksToken", "")).strip()

    if not workspace_url:
        raise ValueError("Databricks workspace URL is required")
    if not databricks_token:
        raise ValueError("Databricks token is required")

    connect_payload = {
        "mcpServers": {
            "databricks-server": {
                "command": "/usr/local/bin/python",
                "args": ["/app/mcp_server.py"],
                "env": {
                    "DATABRICKS_HOST": workspace_url,
                    "DATABRICKS_TOKEN": databricks_token,
                },
                "transport": "stdio",
            }
        }
    }
    return _post_mcp("/connect_mcp", connect_payload)


def _call_databricks_tool(tool_name, parameters):
    payload = {
        "tool_name": tool_name,
        "parameters": parameters or {},
    }
    return _post_mcp("/call_tool", payload)


def _connect_snowflake_mcp(component):
    account = str(component.get("snowflakeAccount", "")).strip()
    user = str(component.get("snowflakeUser", "")).strip()
    password = str(component.get("snowflakePassword", "")).strip()
    warehouse = str(component.get("snowflakeWarehouse", "")).strip()

    if not account:
        raise ValueError("Snowflake account is required")
    if not user:
        raise ValueError("Snowflake user is required")
    if not password:
        raise ValueError("Snowflake password is required")
    if not warehouse:
        raise ValueError("Snowflake warehouse is required")

    snowflake_env = {
        "SNOWFLAKE_ACCOUNT": account,
        "SNOWFLAKE_USER": user,
        "SNOWFLAKE_PASSWORD": password,
        "SNOWFLAKE_WAREHOUSE": warehouse,
    }
    database = str(component.get("snowflakeDatabase", "")).strip()
    schema = str(component.get("snowflakeSchema", "")).strip()
    role = str(component.get("snowflakeRole", "")).strip()
    if database:
        snowflake_env["SNOWFLAKE_DATABASE"] = database
    if schema:
        snowflake_env["SNOWFLAKE_SCHEMA"] = schema
    if role:
        snowflake_env["SNOWFLAKE_ROLE"] = role

    connect_payload = {
        "mcpServers": {
            "snowflake-server": {
                "command": "/usr/local/bin/python",
                "args": ["/app/mcp_server.py"],
                "env": snowflake_env,
                "transport": "stdio",
            }
        }
    }
    return _post_mcp("/connect_mcp", connect_payload)


def _call_snowflake_tool(tool_name, parameters):
    payload = {
        "tool_name": tool_name,
        "parameters": parameters or {},
    }
    return _post_mcp("/call_tool", payload)


def _find_first_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("warehouses", "items", "data", "result", "value"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
            if isinstance(nested, dict):
                child = _find_first_list(nested)
                if child:
                    return child
    return []


def _pick_warehouse_id(warehouses_response):
    # MCP sometimes returns JSON inside content[0]["text"]
    try:
        content = warehouses_response.get("result", {}).get("content", [])
        if content and isinstance(content, list):
            text_payload = content[0].get("text")
            if isinstance(text_payload, str):
                parsed = json.loads(text_payload)
                warehouses = parsed.get("warehouses", [])
                if warehouses:
                    return warehouses[0].get("id", "")
    except Exception as e:
        print("Failed to parse MCP warehouses response:", e)

    warehouses = _find_first_list(warehouses_response)
    for item in warehouses:
        if not isinstance(item, dict):
            continue
        warehouse_id = item.get("id") or item.get("warehouse_id")
        if isinstance(warehouse_id, str) and warehouse_id.strip():
            return warehouse_id.strip()

    return ""


def _build_databricks_statement(component):
    dataset = str(component.get("bigDataDataset", "")).strip()
    query_or_filter = str(component.get("bigDataQueryFilter", "")).strip()
    limit = int(component.get("bigDataLimit", 100) or 100)
    limit = max(1, min(limit, 10000))

    if query_or_filter:
        lowered = query_or_filter.lower()
        if lowered.startswith("select") or lowered.startswith("with"):
            return query_or_filter

    if not dataset:
        raise ValueError("Databricks requires Dataset/Table when query is not full SQL")

    if query_or_filter:
        return f"SELECT * FROM {dataset} WHERE {query_or_filter} LIMIT {limit}"
    return f"SELECT * FROM {dataset} LIMIT {limit}"


def _build_snowflake_sql(component):
    dataset = str(component.get("bigDataDataset", "")).strip()
    query_or_filter = str(component.get("bigDataQueryFilter", "")).strip()
    database = str(component.get("snowflakeDatabase", "")).strip()
    schema = str(component.get("snowflakeSchema", "")).strip()
    limit = int(component.get("bigDataLimit", 100) or 100)
    limit = max(1, min(limit, 10000))

    if query_or_filter:
        lowered = query_or_filter.lower()
        if lowered.startswith(("select", "with", "show", "describe", "desc")):
            return query_or_filter

    if not dataset:
        raise ValueError("Snowflake requires Dataset/Table when query is not full SQL")

    table_ref = dataset
    if "." not in dataset and database and schema:
        table_ref = f"{database}.{schema}.{dataset}"

    if query_or_filter:
        return f"SELECT * FROM {table_ref} WHERE {query_or_filter} LIMIT {limit}"
    return f"SELECT * FROM {table_ref} LIMIT {limit}"


def _run_bigdata_databricks_connector(component, process_instance_id, process_instance_dir):
    catalog = str(component.get("databricksCatalog", "")).strip()
    schema = str(component.get("databricksSchema", "")).strip()
    if not catalog:
        raise ValueError("Databricks catalog is required")
    if not schema:
        raise ValueError("Databricks schema is required")

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        "Connecting Databricks MCP server via /connect_mcp",
        log_type=0,
    )
    connect_response = _connect_databricks_mcp(component)

    warehouses_response = _call_databricks_tool("Databricks.list_sql_warehouses", {})
    print("warehouses_response : ", warehouses_response)
    warehouse_id = _pick_warehouse_id(warehouses_response)
    print("warehouse_id : ", warehouse_id)
    if not warehouse_id:
        raise RuntimeError("No Databricks SQL warehouse found from MCP response")

    statement = _build_databricks_statement(component)
    execute_response = _call_databricks_tool(
        "Databricks.execute_sql_statement",
        {
            "warehouse_id": warehouse_id,
            "statement": statement,
            "catalog": catalog,
            "schema": schema,
            "wait_timeout": "30s",
        },
    )

    result_payload = {
        "sourceType": "bigdata",
        "bigDataType": "databricks",
        "mcp_base_url": EXTERNAL_MCP_BASE_URL,
        "connect_response": connect_response,
        "warehouses_response": warehouses_response,
        "execute_response": execute_response,
        "statement": statement,
        "warehouse_id": warehouse_id,
        "catalog": catalog,
        "schema": schema,
    }
    response_path = os.path.join(process_instance_dir, "external_data_sources_bigdata_databricks_response.json")
    _write_json(response_path, result_payload)

    mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
    mcp_context = _read_json(mcp_context_path, {})
    mcp_context["external_data_sources_response_path"] = response_path
    mcp_context["external_data_sources_type"] = "bigdata"
    mcp_context["external_data_sources_bigdata_type"] = "databricks"
    _write_json(mcp_context_path, mcp_context)

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        "Databricks big data connector executed successfully via MCP",
        log_type=2,
    )


def _run_bigdata_snowflake_connector(component, process_instance_id, process_instance_dir):
    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        "Connecting Snowflake MCP server via /connect_mcp",
        log_type=0,
    )
    connect_response = _connect_snowflake_mcp(component)

    sql = _build_snowflake_sql(component)
    execute_response = _call_snowflake_tool(
        "Snowflake.execute_query",
        {"sql": sql},
    )

    result_payload = {
        "sourceType": "bigdata",
        "bigDataType": "snowflake",
        "mcp_base_url": EXTERNAL_MCP_BASE_URL,
        "connect_response": connect_response,
        "execute_response": execute_response,
        "sql": sql,
        "database": str(component.get("snowflakeDatabase", "")).strip(),
        "schema": str(component.get("snowflakeSchema", "")).strip(),
        "warehouse": str(component.get("snowflakeWarehouse", "")).strip(),
    }
    response_path = os.path.join(process_instance_dir, "external_data_sources_bigdata_snowflake_response.json")
    _write_json(response_path, result_payload)

    mcp_context_path = os.path.join(process_instance_dir, "mcp_context.json")
    mcp_context = _read_json(mcp_context_path, {})
    mcp_context["external_data_sources_response_path"] = response_path
    mcp_context["external_data_sources_type"] = "bigdata"
    mcp_context["external_data_sources_bigdata_type"] = "snowflake"
    _write_json(mcp_context_path, mcp_context)

    log_to_mongo(
        process_instance_id,
        "External Data Sources",
        "Snowflake big data connector executed successfully via MCP",
        log_type=2,
    )


def _run_bigdata_connector(component, process_instance_id, process_instance_dir):
    big_data_type = str(component.get("bigDataType", "")).strip().lower()
    if big_data_type == "databricks":
        _run_bigdata_databricks_connector(component, process_instance_id, process_instance_dir)
        return

    if big_data_type == "snowflake":
        _run_bigdata_snowflake_connector(component, process_instance_id, process_instance_dir)
        return

    raise ValueError(f"Unsupported bigDataType '{big_data_type}'")


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
            _run_db_connector(component, process_instance_id, process_instance_dir)
        elif source_type == "bigdata":
            _run_bigdata_connector(component, process_instance_id, process_instance_dir)
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
