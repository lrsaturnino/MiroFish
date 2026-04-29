"""
配置管理
统一从项目根目录的 .env 文件加载配置
"""

import os
from typing import Literal

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


# Default values for the global LLM_* env vars. Centralised here so the
# class attributes (``Config.LLM_BASE_URL`` / ``Config.LLM_MODEL_NAME``) and
# the per-role resolver (``Config.llm_for``) cannot drift apart when a
# default is updated.
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL_NAME = "gpt-4o-mini"


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False
    
    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', DEFAULT_LLM_BASE_URL)
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', DEFAULT_LLM_MODEL_NAME)
    # Reasoning effort for GPT-5 / o-series / gpt-5.4-* models.
    # Valid values: none, minimal, low, medium, high, xhigh.
    # Empty string means the parameter is omitted from requests (model default).
    LLM_REASONING_EFFORT = os.environ.get('LLM_REASONING_EFFORT', '').strip()
    
    # Zep配置
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls):
        """验证必要配置"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")
        if not cls.ZEP_API_KEY:
            errors.append("ZEP_API_KEY 未配置")
        return errors

    @classmethod
    def llm_for(
        cls, role: Literal["builder", "swarm", "judge"]
    ) -> tuple[str | None, str, str]:
        """Resolve the (api_key, base_url, model) triple for a given role.

        Each role looks up its prefixed env-var group
        (``BUILDER_LLM_*`` / ``SWARM_LLM_*`` / ``JUDGE_LLM_*``) and falls
        back **per field** to the global ``LLM_*`` env vars. Missing or
        empty-string role values are treated as unset, so a blank field
        cleanly inherits the global value.

        ``os.environ`` is read at call time (not the class attributes) so
        runtime env mutations and ``monkeypatch.setenv`` take effect
        without an importlib reload.

        Args:
            role: One of ``"builder"``, ``"swarm"``, ``"judge"``.

        Returns:
            ``(api_key, base_url, model)``. ``api_key`` may be ``None``
            when neither the role env nor ``LLM_API_KEY`` is set.

        Raises:
            ValueError: If ``role`` is not one of the three known roles.

        Example:
            >>> Config.llm_for("builder")
            ('sk-...', 'https://api.openai.com/v1', 'gpt-4o-mini')
        """
        prefixes = {"builder": "BUILDER", "swarm": "SWARM", "judge": "JUDGE"}
        if role not in prefixes:
            raise ValueError(f"unknown role: {role!r}")

        prefix = prefixes[role]
        api_key = os.environ.get(f"{prefix}_LLM_API_KEY", "") or os.environ.get(
            "LLM_API_KEY", None
        )
        base_url = os.environ.get(f"{prefix}_LLM_BASE_URL", "") or os.environ.get(
            "LLM_BASE_URL", DEFAULT_LLM_BASE_URL
        )
        model = os.environ.get(f"{prefix}_LLM_MODEL_NAME", "") or os.environ.get(
            "LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME
        )
        return (api_key, base_url, model)

