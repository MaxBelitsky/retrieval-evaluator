import logging
import os

_NOISY_THIRD_PARTY = [
    "weaviate",
    "weaviate-client",
    "weaviate_client",
    "httpx",
    "httpcore",
    "grpc",
    "urllib3",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
]


def setup_logger(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO

    if debug:
        fmt = "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)s:%(funcName)s(%(lineno)d)  %(message)s"
    else:
        # fmt = "%(asctime)s  %(message)s"
        fmt = "%(message)s"
        # Suppress BertModel LOAD REPORT and other transformers print()-based noise.
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")

    if not debug:
        for name in _NOISY_THIRD_PARTY:
            logging.getLogger(name).setLevel(logging.ERROR)
