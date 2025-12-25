from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latency of HTTP requests",
    ["endpoint"]
)