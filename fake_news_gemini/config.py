import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    gemini_api_key: Optional[str]
    gemini_model_name: str
    random_seed: int
    test_size: float

    @staticmethod
    def load() -> "AppConfig":
        return AppConfig(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
            random_seed=int(os.getenv("SEED", "42")),
            test_size=float(os.getenv("TEST_SIZE", "0.2")),
        )


CONFIG = AppConfig.load()