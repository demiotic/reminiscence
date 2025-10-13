from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class QueryMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    QUERY_MODE_UNSPECIFIED: _ClassVar[QueryMode]
    QUERY_MODE_AUTO: _ClassVar[QueryMode]
    QUERY_MODE_SEMANTIC: _ClassVar[QueryMode]
    QUERY_MODE_EXACT: _ClassVar[QueryMode]

class EvictionPolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EVICTION_POLICY_UNSPECIFIED: _ClassVar[EvictionPolicy]
    EVICTION_POLICY_FIFO: _ClassVar[EvictionPolicy]
    EVICTION_POLICY_LRU: _ClassVar[EvictionPolicy]
    EVICTION_POLICY_LFU: _ClassVar[EvictionPolicy]

class SortField(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SORT_FIELD_UNSPECIFIED: _ClassVar[SortField]
    SORT_FIELD_CREATED_AT: _ClassVar[SortField]
    SORT_FIELD_LAST_ACCESSED: _ClassVar[SortField]
    SORT_FIELD_ACCESS_COUNT: _ClassVar[SortField]
    SORT_FIELD_SIMILARITY: _ClassVar[SortField]
QUERY_MODE_UNSPECIFIED: QueryMode
QUERY_MODE_AUTO: QueryMode
QUERY_MODE_SEMANTIC: QueryMode
QUERY_MODE_EXACT: QueryMode
EVICTION_POLICY_UNSPECIFIED: EvictionPolicy
EVICTION_POLICY_FIFO: EvictionPolicy
EVICTION_POLICY_LRU: EvictionPolicy
EVICTION_POLICY_LFU: EvictionPolicy
SORT_FIELD_UNSPECIFIED: SortField
SORT_FIELD_CREATED_AT: SortField
SORT_FIELD_LAST_ACCESSED: SortField
SORT_FIELD_ACCESS_COUNT: SortField
SORT_FIELD_SIMILARITY: SortField

class MultiModalInput(_message.Message):
    __slots__ = ("text", "image", "video", "audio", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    VIDEO_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    text: str
    image: bytes
    video: bytes
    audio: bytes
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, text: _Optional[str] = ..., image: _Optional[bytes] = ..., video: _Optional[bytes] = ..., audio: _Optional[bytes] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Context(_message.Message):
    __slots__ = ("values",)
    class ValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.ScalarMap[str, str]
    def __init__(self, values: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SerializedResult(_message.Message):
    __slots__ = ("json_data",)
    JSON_DATA_FIELD_NUMBER: _ClassVar[int]
    json_data: bytes
    def __init__(self, json_data: _Optional[bytes] = ...) -> None: ...

class LookupRequest(_message.Message):
    __slots__ = ("query", "context", "similarity_threshold", "mode", "track_metrics")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    TRACK_METRICS_FIELD_NUMBER: _ClassVar[int]
    query: MultiModalInput
    context: Context
    similarity_threshold: float
    mode: QueryMode
    track_metrics: bool
    def __init__(self, query: _Optional[_Union[MultiModalInput, _Mapping]] = ..., context: _Optional[_Union[Context, _Mapping]] = ..., similarity_threshold: _Optional[float] = ..., mode: _Optional[_Union[QueryMode, str]] = ..., track_metrics: bool = ...) -> None: ...

class LookupResponse(_message.Message):
    __slots__ = ("hit", "result", "similarity", "matched_query", "age_seconds", "entry_id", "context", "ttl_remaining")
    HIT_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    MATCHED_QUERY_FIELD_NUMBER: _ClassVar[int]
    AGE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ENTRY_ID_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TTL_REMAINING_FIELD_NUMBER: _ClassVar[int]
    hit: bool
    result: SerializedResult
    similarity: float
    matched_query: str
    age_seconds: float
    entry_id: str
    context: Context
    ttl_remaining: float
    def __init__(self, hit: bool = ..., result: _Optional[_Union[SerializedResult, _Mapping]] = ..., similarity: _Optional[float] = ..., matched_query: _Optional[str] = ..., age_seconds: _Optional[float] = ..., entry_id: _Optional[str] = ..., context: _Optional[_Union[Context, _Mapping]] = ..., ttl_remaining: _Optional[float] = ...) -> None: ...

class LookupBatchRequest(_message.Message):
    __slots__ = ("requests", "track_metrics")
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    TRACK_METRICS_FIELD_NUMBER: _ClassVar[int]
    requests: _containers.RepeatedCompositeFieldContainer[LookupRequest]
    track_metrics: bool
    def __init__(self, requests: _Optional[_Iterable[_Union[LookupRequest, _Mapping]]] = ..., track_metrics: bool = ...) -> None: ...

class LookupBatchResponse(_message.Message):
    __slots__ = ("responses",)
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    responses: _containers.RepeatedCompositeFieldContainer[LookupResponse]
    def __init__(self, responses: _Optional[_Iterable[_Union[LookupResponse, _Mapping]]] = ...) -> None: ...

class StoreRequest(_message.Message):
    __slots__ = ("query", "context", "result", "metadata", "ttl_seconds", "context_threshold", "allow_errors", "mode")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    QUERY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    ALLOW_ERRORS_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    query: MultiModalInput
    context: Context
    result: SerializedResult
    metadata: _containers.ScalarMap[str, str]
    ttl_seconds: int
    context_threshold: float
    allow_errors: bool
    mode: QueryMode
    def __init__(self, query: _Optional[_Union[MultiModalInput, _Mapping]] = ..., context: _Optional[_Union[Context, _Mapping]] = ..., result: _Optional[_Union[SerializedResult, _Mapping]] = ..., metadata: _Optional[_Mapping[str, str]] = ..., ttl_seconds: _Optional[int] = ..., context_threshold: _Optional[float] = ..., allow_errors: bool = ..., mode: _Optional[_Union[QueryMode, str]] = ...) -> None: ...

class StoreResponse(_message.Message):
    __slots__ = ("success", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    def __init__(self, success: bool = ..., error: _Optional[str] = ...) -> None: ...

class StoreBatchRequest(_message.Message):
    __slots__ = ("requests", "allow_errors", "mode")
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    ALLOW_ERRORS_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    requests: _containers.RepeatedCompositeFieldContainer[StoreRequest]
    allow_errors: bool
    mode: QueryMode
    def __init__(self, requests: _Optional[_Iterable[_Union[StoreRequest, _Mapping]]] = ..., allow_errors: bool = ..., mode: _Optional[_Union[QueryMode, str]] = ...) -> None: ...

class StoreBatchResponse(_message.Message):
    __slots__ = ("responses",)
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    responses: _containers.RepeatedCompositeFieldContainer[StoreResponse]
    def __init__(self, responses: _Optional[_Iterable[_Union[StoreResponse, _Mapping]]] = ...) -> None: ...

class CheckAvailabilityRequest(_message.Message):
    __slots__ = ("query", "context", "similarity_threshold", "mode")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    query: MultiModalInput
    context: Context
    similarity_threshold: float
    mode: QueryMode
    def __init__(self, query: _Optional[_Union[MultiModalInput, _Mapping]] = ..., context: _Optional[_Union[Context, _Mapping]] = ..., similarity_threshold: _Optional[float] = ..., mode: _Optional[_Union[QueryMode, str]] = ...) -> None: ...

class CheckAvailabilityResponse(_message.Message):
    __slots__ = ("available", "age_seconds", "ttl_remaining_seconds", "similarity", "is_fresh")
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    AGE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TTL_REMAINING_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    IS_FRESH_FIELD_NUMBER: _ClassVar[int]
    available: bool
    age_seconds: float
    ttl_remaining_seconds: float
    similarity: float
    is_fresh: bool
    def __init__(self, available: bool = ..., age_seconds: _Optional[float] = ..., ttl_remaining_seconds: _Optional[float] = ..., similarity: _Optional[float] = ..., is_fresh: bool = ...) -> None: ...

class InvalidateRequest(_message.Message):
    __slots__ = ("query", "context", "older_than_seconds")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    OLDER_THAN_SECONDS_FIELD_NUMBER: _ClassVar[int]
    query: MultiModalInput
    context: Context
    older_than_seconds: float
    def __init__(self, query: _Optional[_Union[MultiModalInput, _Mapping]] = ..., context: _Optional[_Union[Context, _Mapping]] = ..., older_than_seconds: _Optional[float] = ...) -> None: ...

class InvalidateResponse(_message.Message):
    __slots__ = ("invalidated_count",)
    INVALIDATED_COUNT_FIELD_NUMBER: _ClassVar[int]
    invalidated_count: int
    def __init__(self, invalidated_count: _Optional[int] = ...) -> None: ...

class CleanupExpiredRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CleanupExpiredResponse(_message.Message):
    __slots__ = ("deleted_count",)
    DELETED_COUNT_FIELD_NUMBER: _ClassVar[int]
    deleted_count: int
    def __init__(self, deleted_count: _Optional[int] = ...) -> None: ...

class ClearRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ClearResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class CreateIndexRequest(_message.Message):
    __slots__ = ("num_partitions", "num_subvectors")
    NUM_PARTITIONS_FIELD_NUMBER: _ClassVar[int]
    NUM_SUBVECTORS_FIELD_NUMBER: _ClassVar[int]
    num_partitions: int
    num_subvectors: int
    def __init__(self, num_partitions: _Optional[int] = ..., num_subvectors: _Optional[int] = ...) -> None: ...

class CreateIndexResponse(_message.Message):
    __slots__ = ("success", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    def __init__(self, success: bool = ..., error: _Optional[str] = ...) -> None: ...

class GetIndexStatsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetIndexStatsResponse(_message.Message):
    __slots__ = ("has_index", "total_entries", "note")
    HAS_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    has_index: bool
    total_entries: int
    note: str
    def __init__(self, has_index: bool = ..., total_entries: _Optional[int] = ..., note: _Optional[str] = ...) -> None: ...

class ListEntriesRequest(_message.Message):
    __slots__ = ("limit", "offset", "context_filter", "query_filter", "sort_by", "sort_descending", "include_embeddings", "include_results")
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FILTER_FIELD_NUMBER: _ClassVar[int]
    QUERY_FILTER_FIELD_NUMBER: _ClassVar[int]
    SORT_BY_FIELD_NUMBER: _ClassVar[int]
    SORT_DESCENDING_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_RESULTS_FIELD_NUMBER: _ClassVar[int]
    limit: int
    offset: int
    context_filter: Context
    query_filter: str
    sort_by: SortField
    sort_descending: bool
    include_embeddings: bool
    include_results: bool
    def __init__(self, limit: _Optional[int] = ..., offset: _Optional[int] = ..., context_filter: _Optional[_Union[Context, _Mapping]] = ..., query_filter: _Optional[str] = ..., sort_by: _Optional[_Union[SortField, str]] = ..., sort_descending: bool = ..., include_embeddings: bool = ..., include_results: bool = ...) -> None: ...

class CacheEntry(_message.Message):
    __slots__ = ("entry_id", "query", "context", "result", "metadata", "created_at", "last_accessed_at", "access_count", "ttl_seconds", "expires_at", "embedding", "embedding_dim", "matched_context_key")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENTRY_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_ACCESSED_AT_FIELD_NUMBER: _ClassVar[int]
    ACCESS_COUNT_FIELD_NUMBER: _ClassVar[int]
    TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    MATCHED_CONTEXT_KEY_FIELD_NUMBER: _ClassVar[int]
    entry_id: str
    query: MultiModalInput
    context: Context
    result: SerializedResult
    metadata: _containers.ScalarMap[str, str]
    created_at: float
    last_accessed_at: float
    access_count: int
    ttl_seconds: int
    expires_at: float
    embedding: bytes
    embedding_dim: int
    matched_context_key: str
    def __init__(self, entry_id: _Optional[str] = ..., query: _Optional[_Union[MultiModalInput, _Mapping]] = ..., context: _Optional[_Union[Context, _Mapping]] = ..., result: _Optional[_Union[SerializedResult, _Mapping]] = ..., metadata: _Optional[_Mapping[str, str]] = ..., created_at: _Optional[float] = ..., last_accessed_at: _Optional[float] = ..., access_count: _Optional[int] = ..., ttl_seconds: _Optional[int] = ..., expires_at: _Optional[float] = ..., embedding: _Optional[bytes] = ..., embedding_dim: _Optional[int] = ..., matched_context_key: _Optional[str] = ...) -> None: ...

class ListEntriesResponse(_message.Message):
    __slots__ = ("entries", "total_count", "has_more", "returned_count")
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    HAS_MORE_FIELD_NUMBER: _ClassVar[int]
    RETURNED_COUNT_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[CacheEntry]
    total_count: int
    has_more: bool
    returned_count: int
    def __init__(self, entries: _Optional[_Iterable[_Union[CacheEntry, _Mapping]]] = ..., total_count: _Optional[int] = ..., has_more: bool = ..., returned_count: _Optional[int] = ...) -> None: ...

class GetStatsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetStatsResponse(_message.Message):
    __slots__ = ("cache_entries", "total_entries", "max_entries", "eviction_policy", "threshold", "embedding_dim", "model", "ttl_seconds", "storage", "index_created", "hits", "misses", "hit_rate", "avg_lookup_latency_ms", "avg_store_latency_ms", "lookup_errors", "store_errors", "evictions", "schedulers")
    class SchedulersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: SchedulerStats
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[SchedulerStats, _Mapping]] = ...) -> None: ...
    CACHE_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    MAX_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    EVICTION_POLICY_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    STORAGE_FIELD_NUMBER: _ClassVar[int]
    INDEX_CREATED_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    MISSES_FIELD_NUMBER: _ClassVar[int]
    HIT_RATE_FIELD_NUMBER: _ClassVar[int]
    AVG_LOOKUP_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    AVG_STORE_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    LOOKUP_ERRORS_FIELD_NUMBER: _ClassVar[int]
    STORE_ERRORS_FIELD_NUMBER: _ClassVar[int]
    EVICTIONS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULERS_FIELD_NUMBER: _ClassVar[int]
    cache_entries: int
    total_entries: int
    max_entries: int
    eviction_policy: str
    threshold: float
    embedding_dim: int
    model: str
    ttl_seconds: int
    storage: str
    index_created: bool
    hits: int
    misses: int
    hit_rate: float
    avg_lookup_latency_ms: float
    avg_store_latency_ms: float
    lookup_errors: int
    store_errors: int
    evictions: int
    schedulers: _containers.MessageMap[str, SchedulerStats]
    def __init__(self, cache_entries: _Optional[int] = ..., total_entries: _Optional[int] = ..., max_entries: _Optional[int] = ..., eviction_policy: _Optional[str] = ..., threshold: _Optional[float] = ..., embedding_dim: _Optional[int] = ..., model: _Optional[str] = ..., ttl_seconds: _Optional[int] = ..., storage: _Optional[str] = ..., index_created: bool = ..., hits: _Optional[int] = ..., misses: _Optional[int] = ..., hit_rate: _Optional[float] = ..., avg_lookup_latency_ms: _Optional[float] = ..., avg_store_latency_ms: _Optional[float] = ..., lookup_errors: _Optional[int] = ..., store_errors: _Optional[int] = ..., evictions: _Optional[int] = ..., schedulers: _Optional[_Mapping[str, SchedulerStats]] = ...) -> None: ...

class SchedulerStats(_message.Message):
    __slots__ = ("running", "total_runs", "errors", "last_run_timestamp", "next_run_timestamp")
    RUNNING_FIELD_NUMBER: _ClassVar[int]
    TOTAL_RUNS_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    LAST_RUN_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    NEXT_RUN_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    running: bool
    total_runs: int
    errors: int
    last_run_timestamp: float
    next_run_timestamp: float
    def __init__(self, running: bool = ..., total_runs: _Optional[int] = ..., errors: _Optional[int] = ..., last_run_timestamp: _Optional[float] = ..., next_run_timestamp: _Optional[float] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "checks", "metrics", "timestamp")
    class ChecksEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ComponentHealth
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ComponentHealth, _Mapping]] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CHECKS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    status: str
    checks: _containers.MessageMap[str, ComponentHealth]
    metrics: HealthMetrics
    timestamp: int
    def __init__(self, status: _Optional[str] = ..., checks: _Optional[_Mapping[str, ComponentHealth]] = ..., metrics: _Optional[_Union[HealthMetrics, _Mapping]] = ..., timestamp: _Optional[int] = ...) -> None: ...

class ComponentHealth(_message.Message):
    __slots__ = ("ok", "error", "details")
    OK_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    error: str
    details: str
    def __init__(self, ok: bool = ..., error: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...

class HealthMetrics(_message.Message):
    __slots__ = ("total_entries", "recent_errors")
    TOTAL_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    RECENT_ERRORS_FIELD_NUMBER: _ClassVar[int]
    total_entries: int
    recent_errors: ErrorCounts
    def __init__(self, total_entries: _Optional[int] = ..., recent_errors: _Optional[_Union[ErrorCounts, _Mapping]] = ...) -> None: ...

class ErrorCounts(_message.Message):
    __slots__ = ("lookup", "store")
    LOOKUP_FIELD_NUMBER: _ClassVar[int]
    STORE_FIELD_NUMBER: _ClassVar[int]
    lookup: int
    store: int
    def __init__(self, lookup: _Optional[int] = ..., store: _Optional[int] = ...) -> None: ...

class FlightTicketRequest(_message.Message):
    __slots__ = ("operation", "parameters")
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    operation: str
    parameters: str
    def __init__(self, operation: _Optional[str] = ..., parameters: _Optional[str] = ...) -> None: ...

class FlightTicketResponse(_message.Message):
    __slots__ = ("ticket", "flight_endpoint", "estimated_rows", "estimated_bytes")
    TICKET_FIELD_NUMBER: _ClassVar[int]
    FLIGHT_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_ROWS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_BYTES_FIELD_NUMBER: _ClassVar[int]
    ticket: bytes
    flight_endpoint: str
    estimated_rows: int
    estimated_bytes: int
    def __init__(self, ticket: _Optional[bytes] = ..., flight_endpoint: _Optional[str] = ..., estimated_rows: _Optional[int] = ..., estimated_bytes: _Optional[int] = ...) -> None: ...

class GetCapabilitiesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetCapabilitiesResponse(_message.Message):
    __slots__ = ("version", "flight_enabled", "flight_endpoint", "supported_features")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    FLIGHT_ENABLED_FIELD_NUMBER: _ClassVar[int]
    FLIGHT_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_FEATURES_FIELD_NUMBER: _ClassVar[int]
    version: str
    flight_enabled: bool
    flight_endpoint: str
    supported_features: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, version: _Optional[str] = ..., flight_enabled: bool = ..., flight_endpoint: _Optional[str] = ..., supported_features: _Optional[_Iterable[str]] = ...) -> None: ...
