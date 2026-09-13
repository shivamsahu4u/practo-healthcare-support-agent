"""Centralised configuration, vocabularies and path constants.


Every other module reads its settings from here, so there is exactly one place
where an environment variable is interpreted and exactly one place where the
scenario vocabularies (categories, statuses, knowledge-base topics) are defined.


Importing this module has three deliberate side effects, all of which must
happen *before* CrewAI / SentenceTransformers are imported anywhere:


1. ``.env`` is loaded (without overriding variables already in the environment).
2. CrewAI telemetry is disabled by exporting ``CREWAI_DISABLE_TELEMETRY`` and
   ``OTEL_SDK_DISABLED``. CrewAI reads these at import time, and its telemetry
   otherwise attempts an outbound network call during ``crew.kickoff()``.
3. When ``OFFLINE_MODE`` is on and model downloads are not explicitly allowed,
   ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are exported so the embedding
   library physically cannot reach the network.
"""


from __future__ import annotations


import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


from dotenv import load_dotenv


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR: Final[Path] = REPO_ROOT / "data" / "knowledge_base"
GENERATED_DIR: Final[Path] = REPO_ROOT / "data" / "generated"
REPORTS_DIR: Final[Path] = REPO_ROOT / "reports"
TRANSCRIPTS_DIR: Final[Path] = REPO_ROOT / "transcripts"
CALIBRATION_FILE: Final[Path] = GENERATED_DIR / "calibration.json"
INDEX_STATE_FILE: Final[Path] = GENERATED_DIR / "index_state.json"


# Loaded early and non-destructively: a variable already exported in the shell
# (for example by CI) wins over the file.
load_dotenv(REPO_ROOT / ".env", override=False)


# --------------------------------------------------------------------------- #
# Scenario vocabularies - the single source of truth
# --------------------------------------------------------------------------- #


#: Appointment categories given by the brief. Every value must appear in the
#: generated dataset at least ``MIN_RECORDS_PER_CATEGORY`` times.
CATEGORIES: Final[tuple[str, ...]] = (
    "General Medicine",
    "Cardiology",
    "Dermatology",
    "Pediatrics",
    "Orthopedics",
)


#: Appointment statuses given by the brief. Every value must appear at least once.
STATUSES: Final[tuple[str, ...]] = (
    "Scheduled",
    "Completed",
    "Cancelled",
    "No-Show",
    "Rescheduled",
)


@dataclass(frozen=True, slots=True)
class KbTopic:
    """One required knowledge-base topic.


    ``slug`` doubles as the parent document id used by retrieval, precision /
    recall scoring and the evaluation harness, so it must stay stable.
    """


    slug: str
    title: str


    @property
    def filename(self) -> str:
        return f"{self.slug}.md"


#: The twelve knowledge-base topics required by the brief, in brief order.
KB_TOPICS: Final[tuple[KbTopic, ...]] = (
    KbTopic("appointment_booking", "Appointment Booking Policy"),
    KbTopic("cancellation_rescheduling", "Cancellation and Rescheduling Window"),
    KbTopic("consultation_fees", "Consultation Fee Structure by Specialty"),
    KbTopic("insurance_claims", "Insurance Claim Process"),
    KbTopic("prescription_refills", "Prescription Refill Policy"),
    KbTopic("lab_turnaround", "Lab Test Turnaround Times"),
    KbTopic("telemedicine", "Telemedicine Eligibility"),
    KbTopic("emergency_visits", "Emergency Visit Protocol"),
    KbTopic("patient_privacy", "Patient Data Privacy Policy"),
    KbTopic("follow_up_discount", "Follow-up Visit Discount Policy"),
    KbTopic("second_opinion", "Second Opinion Process"),
    KbTopic("home_visits", "Home Visit Eligibility"),
)


KB_TOPIC_BY_SLUG: Final[dict[str, KbTopic]] = {topic.slug: topic for topic in KB_TOPICS}


#: Chunking strategy identifiers. Each maps to its own ChromaDB collection.
STRATEGY_FIXED: Final[str] = "fixed_size_overlap"
STRATEGY_SENTENCE: Final[str] = "sentence_based"
STRATEGIES: Final[tuple[str, ...]] = (STRATEGY_FIXED, STRATEGY_SENTENCE)


#: Embedding backend identifiers (see ``rag.embeddings.build_embedder``).
BACKEND_SENTENCE_TRANSFORMERS: Final[str] = "sentence_transformers"
BACKEND_CHROMA_ONNX: Final[str] = "chroma_onnx"
BACKEND_DETERMINISTIC_HASH: Final[str] = "deterministic_hash"


#: Crew orchestration modes (see ``agents.crew.run_crew``).
CREW_MODE_CREWAI: Final[str] = "crewai"
CREW_MODE_DIRECT: Final[str] = "direct"


#: Agent keys used by the least-autonomy tool registry in ``agents.governance``.
AGENT_RETRIEVAL: Final[str] = "retrieval_agent"
AGENT_LOOKUP: Final[str] = "lookup_agent"
AGENT_COMPOSER: Final[str] = "response_composer"


#: Tool names. These are the names CrewAI sees, and the names the registry gates.
TOOL_POLICY_LOOKUP: Final[str] = "policy_knowledge_lookup"
TOOL_APPOINTMENT_LOOKUP: Final[str] = "appointment_status_lookup"


#: The fallback text emitted whenever retrieval does not clear the calibrated
#: similarity threshold. Asserted verbatim by tests and transcripts.
FALLBACK_ANSWER: Final[str] = (
    "I don't know based on the available Practo policy knowledge base."
)


#: Exemplar appointment id used in patient-facing guidance ("ids look like ...").
#:
#: **This MUST stay digit-free.** Any appointment id that appears in an assistant
#: message is recoverable on the next turn by
#: ``agents/memory.py:resolve_record_id_from_history``, which scans the stored
#: conversation newest-first. A *real* id here - this once read ``APT-1007``,
#: which is a live record - made the assistant's own boilerplate resolve as "the
#: record currently under discussion", so a bare follow-up turn would look up and
#: disclose an appointment the patient never named.
#:
#: ``agents/routing.py:RECORD_ID_PATTERN`` requires four digits, so a digit-free
#: exemplar cannot be resolved: the guard is structural rather than a convention.
#: ``agents/composition.py`` asserts it at import time so it cannot regress.
EXEMPLAR_RECORD_ID: Final[str] = "APT-XXXX"


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #


class ConfigurationError(RuntimeError):
    """Raised when an environment variable is present but unusable."""


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _raw(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _bool_env(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigurationError(
        f"{name}={value!r} is not a boolean. Use one of: "
        f"{sorted(_TRUE)} / {sorted(_FALSE)}."
    )


def _int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    value = _raw(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name}={value!r} is not an integer.") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"{name}={parsed} must be >= {minimum}.")
    return parsed


def _float_env(name: str, default: float) -> float:
    value = _raw(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name}={value!r} is not a number.") from exc


def _optional_float_env(name: str) -> float | None:
    value = _raw(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name}={value!r} is not a number.") from exc


def _choice_env(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = _raw(name) or default
    if value not in allowed:
        raise ConfigurationError(f"{name}={value!r} must be one of {list(allowed)}.")
    return value


def _path_env(name: str, default: Path) -> Path:
    value = _raw(name)
    if value is None:
        return default
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, fully resolved runtime configuration."""


    app_env: str
    log_level: str


    mock_llm: bool
    offline_mode: bool
    allow_model_download: bool


    embedding_backend: str
    embedding_model_path: str | None
    embedding_model_name: str


    chroma_persist_directory: Path
    fixed_collection_name: str
    sentence_collection_name: str
    recommended_collection_name: str


    fixed_chunk_size: int
    fixed_chunk_overlap: int
    sentences_per_chunk: int


    top_k: int
    similarity_threshold_override: float | None
    groundedness_overlap_min: float


    crew_mode: str
    crew_max_iter: int
    review_enabled: bool


    max_request_characters: int
    max_estimated_tokens: int


    cache_enabled: bool
    cache_max_entries: int


    log_file: Path


    # ------------------------------------------------------------------ #


    @classmethod
    def from_env(cls) -> "Settings":
        fixed_size = _int_env("FIXED_CHUNK_SIZE", 480, minimum=50)
        fixed_overlap = _int_env("FIXED_CHUNK_OVERLAP", 96, minimum=0)
        if fixed_overlap >= fixed_size:
            raise ConfigurationError(
                f"FIXED_CHUNK_OVERLAP ({fixed_overlap}) must be smaller than "
                f"FIXED_CHUNK_SIZE ({fixed_size}), otherwise chunking cannot advance."
            )


        fixed_name = _raw("FIXED_COLLECTION_NAME") or "practo_kb_fixed"
        sentence_name = _raw("SENTENCE_COLLECTION_NAME") or "practo_kb_sentence"
        if fixed_name == sentence_name:
            raise ConfigurationError(
                "FIXED_COLLECTION_NAME and SENTENCE_COLLECTION_NAME must differ: the "
                "brief requires each chunking strategy to live in its own collection."
            )


        recommended = _raw("RECOMMENDED_COLLECTION_NAME") or sentence_name
        if recommended not in (fixed_name, sentence_name):
            raise ConfigurationError(
                f"RECOMMENDED_COLLECTION_NAME={recommended!r} must be either "
                f"{fixed_name!r} or {sentence_name!r}."
            )


        overlap_min = _float_env("GROUNDEDNESS_OVERLAP_MIN", 0.6)
        if not 0.0 < overlap_min <= 1.0:
            raise ConfigurationError(
                f"GROUNDEDNESS_OVERLAP_MIN={overlap_min} must be in (0, 1]."
            )


        return cls(
            app_env=_raw("APP_ENV") or "development",
            log_level=(_raw("LOG_LEVEL") or "INFO").upper(),
            mock_llm=_bool_env("MOCK_LLM", True),
            offline_mode=_bool_env("OFFLINE_MODE", True),
            allow_model_download=_bool_env("ALLOW_MODEL_DOWNLOAD", False),
            embedding_backend=_choice_env(
                "EMBEDDING_BACKEND",
                BACKEND_SENTENCE_TRANSFORMERS,
                (
                    BACKEND_SENTENCE_TRANSFORMERS,
                    BACKEND_CHROMA_ONNX,
                    BACKEND_DETERMINISTIC_HASH,
                ),
            ),
            embedding_model_path=_raw("EMBEDDING_MODEL_PATH"),
            embedding_model_name=(
                _raw("EMBEDDING_MODEL_NAME") or "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chroma_persist_directory=_path_env(
                "CHROMA_PERSIST_DIRECTORY", GENERATED_DIR / "chroma"
            ),
            fixed_collection_name=fixed_name,
            sentence_collection_name=sentence_name,
            recommended_collection_name=recommended,
            fixed_chunk_size=fixed_size,
            fixed_chunk_overlap=fixed_overlap,
            sentences_per_chunk=_int_env("SENTENCES_PER_CHUNK", 2, minimum=1),
            top_k=_int_env("TOP_K", 3, minimum=1),
            similarity_threshold_override=_optional_float_env("SIMILARITY_THRESHOLD"),
            groundedness_overlap_min=overlap_min,
            crew_mode=_choice_env(
                "CREW_MODE", CREW_MODE_CREWAI, (CREW_MODE_CREWAI, CREW_MODE_DIRECT)
            ),
            crew_max_iter=_int_env("CREW_MAX_ITER", 4, minimum=2),
            review_enabled=_bool_env("REVIEW_ENABLED", True),
            max_request_characters=_int_env("MAX_REQUEST_CHARACTERS", 2000, minimum=1),
            max_estimated_tokens=_int_env("MAX_ESTIMATED_TOKENS", 600, minimum=1),
            cache_enabled=_bool_env("CACHE_ENABLED", True),
            cache_max_entries=_int_env("CACHE_MAX_ENTRIES", 512, minimum=1),
            log_file=_path_env("LOG_FILE", GENERATED_DIR / "requests.jsonl"),
        )


    # ------------------------------------------------------------------ #


    @property
    def collection_for_strategy(self) -> dict[str, str]:
        """Map chunking strategy -> ChromaDB collection name."""
        return {
            STRATEGY_FIXED: self.fixed_collection_name,
            STRATEGY_SENTENCE: self.sentence_collection_name,
        }


    @property
    def strategy_for_collection(self) -> dict[str, str]:
        """Map ChromaDB collection name -> chunking strategy."""
        return {name: strategy for strategy, name in self.collection_for_strategy.items()}


    @property
    def all_collection_names(self) -> tuple[str, ...]:
        return (self.fixed_collection_name, self.sentence_collection_name)


    def describe(self) -> dict[str, object]:
        """JSON-safe summary for ``GET /health`` and report headers.


        Deliberately excludes anything secret-shaped; the optional real-LLM
        variables are never read into ``Settings`` at all.
        """
        return {
            "app_env": self.app_env,
            "mock_llm": self.mock_llm,
            "offline_mode": self.offline_mode,
            "allow_model_download": self.allow_model_download,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model_path or self.embedding_model_name,
            "crew_mode": self.crew_mode,
            "review_enabled": self.review_enabled,
            "cache_enabled": self.cache_enabled,
            "top_k": self.top_k,
            "collections": {
                STRATEGY_FIXED: self.fixed_collection_name,
                STRATEGY_SENTENCE: self.sentence_collection_name,
            },
            "recommended_collection": self.recommended_collection_name,
            "budget": {
                "max_request_characters": self.max_request_characters,
                "max_estimated_tokens": self.max_estimated_tokens,
            },
        }


def _apply_process_environment(settings: Settings) -> None:
    """Export the environment variables that third-party libraries read on import.


    Called once at module import, before anything imports CrewAI or
    SentenceTransformers. ``setdefault`` is used so an explicit shell export is
    never silently overwritten.
    """
    # CrewAI telemetry would otherwise make an outbound call on kickoff().
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CREWAI_DISABLE_VERSION_CHECK", "true")
    # ChromaDB's own product telemetry.
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


    if settings.offline_mode and not settings.allow_model_download:
        # Hard offline switch for the huggingface stack. With these set, a
        # missing model raises instead of quietly downloading.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


#: Process-wide settings singleton. Import this, do not re-read the environment.
SETTINGS: Final[Settings] = Settings.from_env()
_apply_process_environment(SETTINGS)


def ensure_runtime_directories(settings: Settings = SETTINGS) -> None:
    """Create the writable directories the app needs. Idempotent."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)


