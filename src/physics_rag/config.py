"""配置管理模块 —— 所有可调参数集中管理，从 .env 文件加载。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：config.py 向上三级（src/physics_rag/config.py → 项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(p: str) -> str:
    """将相对路径转为基于项目根目录的绝对路径。"""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


class Settings(BaseSettings):
    """应用全局配置，优先级：环境变量 > .env 文件 > 默认值。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── DeepSeek API ──────────────────────────────────────────
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ── Embedding 模型 ────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # ── FAISS 向量库 ─────────────────────────────────────────
    faiss_index_path: str = "./faiss_index"
    top_k: int = 5

    # ── 文本切片 ──────────────────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # ── 数据 ──────────────────────────────────────────────────
    data_dir: str = "./data"

    @property
    def is_configured(self) -> bool:
        """检查是否已完成基本配置（至少设置了 API Key）。"""
        return bool(self.deepseek_api_key)

    def ensure_dirs(self) -> None:
        """确保所有需要的目录存在。"""
        Path(self.resolved_faiss_path).mkdir(parents=True, exist_ok=True)
        Path(self.resolved_data_dir).mkdir(parents=True, exist_ok=True)

    @property
    def resolved_data_dir(self) -> str:
        """数据目录的绝对路径。"""
        return _resolve_path(self.data_dir)

    @property
    def resolved_faiss_path(self) -> str:
        """FAISS 索引目录的绝对路径。"""
        return _resolve_path(self.faiss_index_path)


# 全局单例
settings = Settings()
