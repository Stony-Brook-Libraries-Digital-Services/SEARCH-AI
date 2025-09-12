<?php
/**
 * SEARCH AI Proxy (Stony Brook University Libraries)
 * --------------------------------------------------
 * Purpose:
 *   Lightweight PHP proxy to accept a search `query`, call the backend
 *   pipeline, optionally log activity, and return a redirectable URL (or text).
 *
 * How it works (high level):
 *   1) Handles CORS and preflight for the configured front-end origin.
 *   2) Reads the `query` from GET/POST (form or JSON), trims & validates it.
 *   3) Constructs and executes the backend command safely.
 *   4) Logs request/response metadata when enabled.
 *   5) Returns the generated URL for the frontend to use.
 *
 * Inputs:
 *   - query: string (from request parameters).
 *
 * Responses:
 *   - 200: Plain-text URL suitable for redirect.
 *   - 400: Missing/invalid input.
 *   - 5xx: Backend or execution failure.
 *
 * Notes:
 *   - CORS is restricted to the library discovery origin.
 *   - Shell execution uses escaping; keep it that way.
 *   - Avoid logging sensitive information if forcing log in.
 *
 * Maintenance:
 *   - Keep code logic unchanged in this file; only adjust comments and
 *     configuration paths as needed.
 */

// CORS 
header("Access-Control-Allow-Origin: https://search.library.stonybrook.edu");
header("Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type");
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') { http_response_code(204); exit; }

// Read input
$query = $_REQUEST['query'] ?? '';
$query = is_string($query) ? trim($query) : '';
if ($query === '') {
  http_response_code(400);
  header("Content-Type: text/plain; charset=utf-8");
  echo "Error: No query provided.";
  exit;
}

// Paths (all relative to this file)
$baseDir = __DIR__;
$script  = $baseDir . '/openCopy.py';   // must sit next to this PHP file
$logDir  = $baseDir . '/_logs';
$logFile = $logDir . '/search.jsonl';
$debug   = $logDir . '/debug.log';

// Ensure log dir exists
if (!is_dir($logDir)) { @mkdir($logDir, 0770, true); }

// Helper: run shell command, capture out/err/code
function run_cmd($cmd, $cwd) {
  $desc = [1=>['pipe','w'], 2=>['pipe','w']];
  $proc = proc_open($cmd, $desc, $pipes, $cwd);
  if (!is_resource($proc)) return [null, "proc_open failed", 1];
  $out = stream_get_contents($pipes[1]); fclose($pipes[1]);
  $err = stream_get_contents($pipes[2]); fclose($pipes[2]);
  $code = proc_close($proc);
  return [$out, $err, $code];
}

// Build and run: try python3, then python
$escapedQuery = escapeshellarg($query);
$cmd1 = '/usr/bin/env bash -lc ' . escapeshellarg('python3 ' . escapeshellarg($script) . ' url ' . $escapedQuery);
list($out, $err, $code) = run_cmd($cmd1, $baseDir);
$cmdUsed = $cmd1;

if ($code !== 0 || $out === null || trim($out) === '') {
  $cmd2 = '/usr/bin/env bash -lc ' . escapeshellarg('python ' . escapeshellarg($script) . ' url ' . $escapedQuery);
  list($out, $err, $code) = run_cmd($cmd2, $baseDir);
  $cmdUsed = $cmd2;
}

$url = trim((string)$out);

// Log 
@file_put_contents(
  $logFile,
  json_encode(['ts'=>gmdate('c'), 'query'=>$query, 'url'=>$url], JSON_UNESCAPED_SLASHES) . PHP_EOL,
  FILE_APPEND | LOCK_EX
);

// Optional debug
@file_put_contents(
  $debug,
  '['.gmdate('c')."] code=$code\nCMD:\n$cmdUsed\nSTDERR:\n$err\nSTDOUT:\n$out\n\n",
  FILE_APPEND | LOCK_EX
);

// Respond with URL for the frontend to redirect
header("Content-Type: text/plain; charset=utf-8");
echo $url;
exit;
