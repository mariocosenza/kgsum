import json
import os
from enum import Enum, auto


class ClassifierType(Enum):
    SVM = "SVM"
    NAIVE_BAYES = "NAIVE_BAYES"
    KNN = "KNN"
    J48 = "J48"
    MISTRAL = "MISTRAL"
    MLP = "MLP"
    DEEP = "DEEP"
    BATCHNORM = "BATCHNORM"


class Phase(Enum):
    LABELING = "LABELING"
    EXTRACTION = "EXTRACTION"
    PROCESSING = "PROCESSING"
    TRAINING = "TRAINING"
    STORE = "STORE"


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "default_secret")
    USE_GEMINI: bool = True
    SEARCH_GITHUB: bool = True
    SEARCH_ZENODO: bool = True
    SEARCH_LOD_CLOUD: bool = True
    STOP_BEFORE_MERGING: bool = False
    START_OFFSET: int = 0
    LIMIT: int = 4000
    SAVE_RANGE: int = 16
    EXTRACT_SPARQL: bool = True
    QUERY_LOV: bool = True
    USE_NER: bool = True
    USE_FILTER: bool = True
    CLASSIFIER: ClassifierType = ClassifierType.NAIVE_BAYES
    FEATURES = ["CURI"]
    OVERSAMPLE: bool = False
    MAX_TOKEN: int = 800000
    USE_TF_IDF_AUTOENCODER: bool = True
    STORE_PROFILE_AFTER_TRAINING: bool = False
    STORE_PROFILE_AT_RUN: bool = False
    BASE_DOMAIN = "https://exemple.org"
    START_PHASE: Phase = Phase.TRAINING
    STOP_PHASE: Phase = Phase.TRAINING
    ALLOWED_PHASE: list[Phase] = [Phase.PROCESSING, Phase.TRAINING]
    ALLOW_UPLOAD: bool = False

    @staticmethod
    def init_configuration():
        # Try multiple possible locations for the config file
        possible_paths = [
            # Current working directory (most common for Docker)
            os.path.join(os.getcwd(), "config.json"),
            # Same directory as this Python file
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
            # Root directory (common in Docker containers)
            "/app/config.json",
            # Parent directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"),
            # Default fallback
            "config.json"
        ]

        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

        if not config_path:
            raise FileNotFoundError(
                f"Config file not found. Searched in: {possible_paths}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"File directory: {os.path.dirname(os.path.abspath(__file__))}"
            )

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            raise Exception(f"Error reading config file at {config_path}: {e}")

        Config.USE_GEMINI = bool(config["labeling"].get("use_gemini", False))
        Config.SEARCH_GITHUB = bool(config["labeling"].get("search_github", False))
        Config.SEARCH_ZENODO = bool(config["labeling"].get("search_zenodo", False))
        Config.SEARCH_LOD_CLOUD = bool(config["labeling"].get("search_lod_cloud", False))
        Config.STOP_BEFORE_MERGING = bool(config["labeling"].get("stop_before_merging", False))

        Config.START_OFFSET = max(int(config["extraction"].get("start_offset", 0)), 0)
        Config.LIMIT = max(int(config["extraction"].get("step_numbers", 0)), 0)
        Config.SAVE_RANGE = max(int(config["extraction"].get("step_range", 16)), 0)
        Config.EXTRACT_SPARQL = bool(config["extraction"].get("extract_sparql", False))
        Config.QUERY_LOV = bool(config["extraction"].get("query_lov", False))

        Config.USE_NER = bool(config["processing"].get("use_ner", False))
        Config.USE_FILTER = bool(config["processing"].get("use_filter", False))

        Config.CLASSIFIER = _assign_enum_classifier(config["training"].get("classifier", "NO"))
        Config.FEATURES = config["training"].get("feature", [])
        Config.OVERSAMPLE = bool(config["training"].get("oversample", False))
        Config.MAX_TOKEN = max(int(config["training"].get("max_token", 256)), 0)
        Config.USE_TF_IDF_AUTOENCODER = bool(config["training"].get("use_tfidf_autoencoder", False))

        Config.STORE_PROFILE_AFTER_TRAINING = bool(config["profile"].get("store_profile_after_training", False))
        Config.STORE_PROFILE_AT_RUN = bool(config["profile"].get("store_profile_at_run", False))
        Config.BASE_DOMAIN = config["profile"].get("base_domain", "https://exemple.org")

        Config.ALLOW_UPLOAD = bool(config["general_settings"].get("allow_upload", False))
        Config.START_PHASE = _assign_enum_phase(config["general_settings"].get("start_phase", "NO"))
        Config.STOP_PHASE = _assign_enum_phase(config["general_settings"].get("stop_phase", "NO"))
        _check_phase_order(Config.START_PHASE, Config.STOP_PHASE)
        Config.ALLOWED_PHASE = _phase_list(phase=Config.START_PHASE, phase_2=Config.STOP_PHASE)


class ProductionConfig(Config):
    DEBUG = False


class DevelopmentConfig(Config):
    DEBUG = True


def _assign_enum_classifier(enum_string: str):
    return ClassifierType(enum_string.upper())


def _assign_enum_phase(enum_string: str):
    return Phase(enum_string.upper())


def _check_phase_order(phase: Phase, phase_2: Phase):
    phase_order = [Phase.LABELING, Phase.EXTRACTION, Phase.PROCESSING, Phase.TRAINING, Phase.STORE]
    if phase_order.index(phase) > phase_order.index(phase_2):
        raise ValueError("Phase order is inconsistent")


def _phase_list(phase: Phase, phase_2: Phase) -> list[Phase]:
    phase_order = [Phase.LABELING, Phase.EXTRACTION, Phase.PROCESSING, Phase.TRAINING, Phase.STORE]
    list_phase = [phase]
    for phase_o in phase_order:
        if phase_order.index(phase_o) >= phase_order.index(phase) and phase_order.index(phase_o) <= phase_order.index(
                phase_2):
            list_phase.append(phase_o)
    return list_phase


Config.init_configuration()