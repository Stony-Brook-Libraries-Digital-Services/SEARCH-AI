<?php

/* ===== ADD Your Access Point URL(s) ===== */
$allowed_origins = [
  ' your access point URL here '
];

$origin = $_SERVER['HTTP_ORIGIN'] ?? '';

if (in_array($origin, $allowed_origins)) {
  header("Access-Control-Allow-Origin: $origin");
}

header("Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type");

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
  http_response_code(204);
  exit;
}

/* ===== ADD: JSONL logger to _logs/search.jsonl (same directory as this file) ===== */
$LOG_DIR   = __DIR__ . '/_logs';
$LOG_FILE  = $LOG_DIR . '/search.jsonl';
$log_ok    = true;

// try to create _logs if it doesn't exist
if (!@is_dir($LOG_DIR)) { @mkdir($LOG_DIR, 0775, true); }

// if not writable, fall back to system temp
if (!@is_dir($LOG_DIR) || !@is_writable($LOG_DIR)) {
  $LOG_FILE = rtrim(sys_get_temp_dir(), '/\\') . '/search.jsonl';
  if (!@is_writable(dirname($LOG_FILE))) {
    $log_ok = false;
  }
}

function log_jsonl($data) {
  global $LOG_FILE, $log_ok;
  if (!$log_ok) return;
  @file_put_contents(
    $LOG_FILE,
    json_encode($data, JSON_UNESCAPED_SLASHES) . "\n",
    FILE_APPEND | LOCK_EX
  );
}

$query = $_REQUEST['query'] ?? '';

$log = [
  'ts'   => date('c'),
  'query'=> is_string($query) ? $query : json_encode($query),
];

if (empty($query)) {
    $log['status'] = 400;
    $log['error']  = 'missing query';
    log_jsonl($log);

    http_response_code(400);
    echo 'Error: No query provided.';
    exit;
}

$escaped_query = escapeshellarg($query);
$command = "python open.py url $escaped_query";
$log['command'] = $command;

$start  = microtime(true);
$output = shell_exec($command);
$elapsed_ms = (int)((microtime(true) - $start) * 1000);

$preview = $output === null ? '(null)'
         : (strlen($output) > 500 ? substr($output, 0, 500) . '...[truncated]' : $output);

$log['elapsed_ms']  = $elapsed_ms;
$log['output_len']  = is_string($output) ? strlen($output) : -1;
$log['output_preview'] = str_replace(["\r","\n"], ['\\r','\\n'], (string)$preview);

$url = trim((string)$output);
$log['final_url'] = $url;

if (!$url || !preg_match('/^https?:\/\//i', $url)) {
    $log['status'] = 500;
    $log['error']  = 'invalid url from script';
    log_jsonl($log);

    http_response_code(500);
    echo "Error: Script did not return a valid URL.\n" . (string)$output;
    exit;
}

$log['status'] = 200;
log_jsonl($log);

header("Content-Type: text/plain; charset=utf-8");
echo $url;
exit;
