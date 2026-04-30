"""
OASIS模拟管理器
管理Twitter和Reddit双平台并行模拟
使用预设脚本 + LLM智能生成配置参数
"""

import logging
import os
import json
import shutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from .zep_entity_reader import ZepEntityReader, FilteredEntities
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from .simulation_config_generator import SimulationConfigGenerator, SimulationParameters
from .agent_research_service import AgentResearchService, ResearchJsonlLogger
from .search.tavily import TavilyProvider
from .search.cache import QueryCache
from ..utils.locale import t

logger = get_logger('mirofish.simulation')

# Standard-library logger used only for the research-step defensive
# wrap below. The project-wide ``mirofish.simulation`` logger has
# ``propagate=False`` (by design — see ``app/utils/logger.py``), so its
# records do not bubble to the root logger and are therefore invisible
# to test fixtures (e.g. pytest ``caplog``) that attach at the root.
# This sub-logger uses module name and the standard logging hierarchy
# so callers / observers can configure / capture it via the standard
# ``logging`` API without touching the project logger.
_research_logger = logging.getLogger(__name__)


class SimulationStatus(str, Enum):
    """模拟状态"""
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"      # 模拟被手动停止
    COMPLETED = "completed"  # 模拟自然完成
    FAILED = "failed"


class PlatformType(str, Enum):
    """平台类型"""
    TWITTER = "twitter"
    REDDIT = "reddit"


@dataclass
class SimulationState:
    """模拟状态 (Simulation state — operator-visible persistence record).

    Persistence schema for one simulation run. Serialized via the hand-written
    ``to_dict`` below and read back via ``SimulationManager._load_simulation_state``.

    The trailing ``builder_model_name``/``swarm_model_name``/``judge_model_name``
    fields are persisted **for UI round-trip only (v1 scope)**. They do NOT,
    on their own, change the runtime LLM resolution path. The actual per-role
    LLM split takes effect via the ``BUILDER_LLM_*`` / ``SWARM_LLM_*`` /
    ``JUDGE_LLM_*`` environment variables resolved by ``Config.llm_for(role)``
    at process start (see ``app/config.py``). These three fields exist so the
    Step1 UI can read and write the operator's preferred per-role model
    identifiers; bridging UI value into the runtime resolver is out of scope
    in this TD and may be picked up by a future task. Empty string ``""`` is
    the canonical "inherit from ``LLM_*``" sentinel — never ``None``.
    """
    simulation_id: str
    project_id: str
    graph_id: str
    
    # 平台启用状态
    enable_twitter: bool = True
    enable_reddit: bool = True
    
    # 状态
    status: SimulationStatus = SimulationStatus.CREATED
    
    # 准备阶段数据
    entities_count: int = 0
    profiles_count: int = 0
    entity_types: List[str] = field(default_factory=list)
    
    # 配置生成信息
    config_generated: bool = False
    config_reasoning: str = ""
    
    # 运行时数据
    current_round: int = 0
    twitter_status: str = "not_started"
    reddit_status: str = "not_started"
    
    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 错误信息
    error: Optional[str] = None

    # UI-persisted operator hint for the BUILDER role; the actual override is
    # delivered to the runtime via the BUILDER_LLM_* env vars resolved by
    # Config.llm_for("builder"). Empty string ⇒ inherit LLM_*.
    builder_model_name: str = ""
    # UI-persisted operator hint for the SWARM role; runtime override flows
    # through SWARM_LLM_* env vars resolved by Config.llm_for("swarm").
    # Empty string ⇒ inherit LLM_*.
    swarm_model_name: str = ""
    # UI-persisted operator hint for the JUDGE role; runtime override flows
    # through JUDGE_LLM_* env vars resolved by Config.llm_for("judge").
    # Empty string ⇒ inherit LLM_*.
    judge_model_name: str = ""

    # UI-persisted operator-visible toggle for web research. Persistence-only:
    # the runtime gate continues to be the ``RESEARCH_ENABLED`` env var read
    # by ``AgentResearchService.is_enabled()`` at process start. A future
    # task may bridge this UI flag into the runtime resolver; for now it is
    # round-tripped between the operator's UI and ``state.json`` only.
    research_enabled: bool = False
    # UI-persisted operator-visible base K for per-agent research queries.
    # Slider range 1–10 in the UI; the absolute runtime ceiling stays
    # ``MAX_RESEARCH_QUERIES_PER_AGENT`` (env-driven, default 20) enforced
    # by ``AgentResearchService.budget()``. Persistence-only; the runtime
    # budget calculator does not consume this field directly.
    research_base_k: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """完整状态字典（内部使用）"""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "enable_twitter": self.enable_twitter,
            "enable_reddit": self.enable_reddit,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "config_reasoning": self.config_reasoning,
            "current_round": self.current_round,
            "twitter_status": self.twitter_status,
            "reddit_status": self.reddit_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "builder_model_name": self.builder_model_name,
            "swarm_model_name": self.swarm_model_name,
            "judge_model_name": self.judge_model_name,
            "research_enabled": self.research_enabled,
            "research_base_k": self.research_base_k,
        }
    
    def to_simple_dict(self) -> Dict[str, Any]:
        """简化状态字典（API返回使用）"""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "error": self.error,
        }


class SimulationManager:
    """
    模拟管理器
    
    核心功能：
    1. 从Zep图谱读取实体并过滤
    2. 生成OASIS Agent Profile
    3. 使用LLM智能生成模拟配置参数
    4. 准备预设脚本所需的所有文件
    """
    
    # 模拟数据存储目录
    SIMULATION_DATA_DIR = os.path.join(
        os.path.dirname(__file__), 
        '../../uploads/simulations'
    )
    
    def __init__(self):
        # 确保目录存在
        os.makedirs(self.SIMULATION_DATA_DIR, exist_ok=True)
        
        # 内存中的模拟状态缓存
        self._simulations: Dict[str, SimulationState] = {}
    
    def _get_simulation_dir(self, simulation_id: str) -> str:
        """获取模拟数据目录"""
        sim_dir = os.path.join(self.SIMULATION_DATA_DIR, simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        return sim_dir
    
    def _save_simulation_state(self, state: SimulationState):
        """保存模拟状态到文件"""
        sim_dir = self._get_simulation_dir(state.simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        state.updated_at = datetime.now().isoformat()
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        
        self._simulations[state.simulation_id] = state
    
    def _load_simulation_state(self, simulation_id: str) -> Optional[SimulationState]:
        """从文件加载模拟状态"""
        if simulation_id in self._simulations:
            return self._simulations[simulation_id]
        
        sim_dir = self._get_simulation_dir(simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        if not os.path.exists(state_file):
            return None
        
        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Safe null-coercion for the two typed-non-string research fields:
        # explicit JSON ``null`` collapses to the declared default, but a
        # legitimate persisted ``False`` / ``0`` is preserved. The unsafe
        # ``or default`` shortcut would silently rewrite ``False`` → ``False``
        # (harmless coincidence) and ``0`` → ``3`` (data corruption against
        # a hand-edited state.json). Don't take the test-passing shortcut;
        # the contract is corruption-resistance.
        raw_research_enabled = data.get("research_enabled")
        raw_research_base_k = data.get("research_base_k")

        state = SimulationState(
            simulation_id=simulation_id,
            project_id=data.get("project_id", ""),
            graph_id=data.get("graph_id", ""),
            enable_twitter=data.get("enable_twitter", True),
            enable_reddit=data.get("enable_reddit", True),
            status=SimulationStatus(data.get("status", "created")),
            entities_count=data.get("entities_count", 0),
            profiles_count=data.get("profiles_count", 0),
            entity_types=data.get("entity_types", []),
            config_generated=data.get("config_generated", False),
            config_reasoning=data.get("config_reasoning", ""),
            current_round=data.get("current_round", 0),
            twitter_status=data.get("twitter_status", "not_started"),
            reddit_status=data.get("reddit_status", "not_started"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            error=data.get("error"),
            # ``or ""`` coerces both missing keys (returns ``None``) AND
            # explicit JSON ``null`` to the canonical empty-string sentinel.
            # Limited to these three string-only fields; do NOT propagate
            # this pattern to fields where ``0`` / ``False`` / ``""`` carry
            # distinct meaning.
            builder_model_name=data.get("builder_model_name") or "",
            swarm_model_name=data.get("swarm_model_name") or "",
            judge_model_name=data.get("judge_model_name") or "",
            research_enabled=raw_research_enabled if raw_research_enabled is not None else False,
            research_base_k=raw_research_base_k if raw_research_base_k is not None else 3,
        )
        
        self._simulations[simulation_id] = state
        return state
    
    def create_simulation(
        self,
        project_id: str,
        graph_id: str,
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        builder_model_name: str = "",
        swarm_model_name: str = "",
        judge_model_name: str = "",
        research_enabled: bool = False,
        research_base_k: int = 3,
    ) -> SimulationState:
        """
        创建新的模拟

        Args:
            project_id: 项目ID
            graph_id: Zep图谱ID
            enable_twitter: 是否启用Twitter模拟
            enable_reddit: 是否启用Reddit模拟
            builder_model_name: UI-persisted operator hint for the BUILDER role.
                Empty string ⇒ inherit from the global ``LLM_*`` env vars via
                ``Config.llm_for("builder")``.
            swarm_model_name: UI-persisted operator hint for the SWARM role.
                Empty string ⇒ inherit from ``Config.llm_for("swarm")``.
            judge_model_name: UI-persisted operator hint for the JUDGE role.
                Empty string ⇒ inherit from ``Config.llm_for("judge")``.
            research_enabled: UI-persisted operator-visible web research toggle.
                Persistence-only — does not gate the runtime pipeline. The
                runtime gate stays the ``RESEARCH_ENABLED`` env var consumed
                by ``AgentResearchService.is_enabled()``.
            research_base_k: UI-persisted base K for per-agent research
                queries. Persistence-only; the runtime budget calculator is
                governed by ``MAX_RESEARCH_QUERIES_PER_AGENT`` env var.

        Returns:
            SimulationState
        """
        import uuid
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"

        state = SimulationState(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit,
            status=SimulationStatus.CREATED,
            builder_model_name=builder_model_name,
            swarm_model_name=swarm_model_name,
            judge_model_name=judge_model_name,
            research_enabled=research_enabled,
            research_base_k=research_base_k,
        )

        self._save_simulation_state(state)
        logger.info(f"创建模拟: {simulation_id}, project={project_id}, graph={graph_id}")

        return state

    def update_simulation(
        self,
        simulation_id: str,
        *,
        builder_model_name: Optional[str] = None,
        swarm_model_name: Optional[str] = None,
        judge_model_name: Optional[str] = None,
        research_enabled: Optional[bool] = None,
        research_base_k: Optional[int] = None,
    ) -> SimulationState:
        """Partial update of the operator-visible UI-persisted fields.

        Sentinel discipline:
            * ``None``  → "do not touch this field"
            * ``""``    → (string fields only) "set this field to inherit
                          from ``LLM_*``"

        This is the only place where ``None`` and ``""`` carry different
        meanings. Callers that want to clear a string field must pass ``""``
        explicitly; omitting the kwarg leaves the persisted value untouched.
        For the research bool / int fields the only "do-not-touch" sentinel
        is ``None`` — there is no analogue of the empty-string clear path.

        Example:
            ``manager.update_simulation("sim_abc", swarm_model_name="haiku")``
            mutates only ``swarm_model_name``; ``builder_model_name`` and
            ``judge_model_name`` remain at their persisted values.

        Args:
            simulation_id: Existing simulation to mutate.
            builder_model_name: New BUILDER role hint, or ``None`` to leave
                the persisted value unchanged.
            swarm_model_name: New SWARM role hint, or ``None`` to leave
                the persisted value unchanged.
            judge_model_name: New JUDGE role hint, or ``None`` to leave
                the persisted value unchanged.
            research_enabled: New web-research toggle value, or ``None`` to
                leave the persisted value unchanged.
            research_base_k: New per-agent research base K, or ``None`` to
                leave the persisted value unchanged.

        Returns:
            The updated ``SimulationState``.

        Raises:
            ValueError: If ``simulation_id`` does not exist.
        """
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"模拟不存在: {simulation_id}")

        if builder_model_name is not None:
            state.builder_model_name = builder_model_name
        if swarm_model_name is not None:
            state.swarm_model_name = swarm_model_name
        if judge_model_name is not None:
            state.judge_model_name = judge_model_name
        if research_enabled is not None:
            state.research_enabled = research_enabled
        if research_base_k is not None:
            state.research_base_k = research_base_k

        self._save_simulation_state(state)
        return state
    
    def _run_research_step(
        self,
        state: SimulationState,
        profiles: List[OasisAgentProfile],
        sim_params: SimulationParameters,
        simulation_requirement: str,
    ) -> List[OasisAgentProfile]:
        """Pipeline Step 2.5 — one-shot agent web research (planning §5.8 / §3.3).

        Constructs ``AgentResearchService`` once per ``prepare_simulation``
        invocation with all four collaborators wired explicitly
        (``TavilyProvider``, ``QueryCache``, ``LLMClient(role="builder")``,
        ``ResearchJsonlLogger``). Builds ``activity_by_user_id`` from
        ``sim_params.agent_configs`` (``user_id`` and ``agent_id`` are both
        sequential ints walking the same filtered-entities list — verified
        in ``oasis_profile_generator.py`` and ``simulation_config_generator.py``)
        and forwards ``simulation_requirement`` as the ``topic_seed`` arg
        (planning §9.3 — the operator-supplied "news subject" the agents
        are searching about).

        The service self-gates internally on ``RESEARCH_ENABLED`` +
        ``TAVILY_API_KEY``: when disabled it emits one ``WARNING`` line
        and returns the profile list unchanged at zero cost. This is why
        no external ``if service.is_enabled():`` guard is needed — the
        host always calls ``run()`` and trusts the contract.

        The full service-construct-and-run block is wrapped in a
        defensive ``try/except Exception`` so a catastrophic failure
        (``LLMClient`` constructor without env vars, provider/cache
        construction error, I/O error on the JSONL logger's first
        ``write`` — the project dir is created lazily inside ``write``,
        not in the logger's constructor) NEVER aborts the pipeline
        (planning §3.4 NFR-1). Per-agent failures are already isolated
        inside the service loop. On catch we surface the traceback via
        ``logger.exception`` and return the original profile list
        reference so ``save_profiles`` below sees the un-mutated data.

        Args:
            state: The current ``SimulationState`` (read for
                ``state.project_id``).
            profiles: The freshly-generated profile list from Phase 2.
                Mutated in place by the service's success path; returned
                unchanged on disabled or catastrophic-failure paths.
            sim_params: The Phase-3 ``SimulationParameters`` carrying
                ``agent_configs: List[AgentActivityConfig]``.
            simulation_requirement: Forwarded as ``topic_seed`` to
                ``AgentResearchService.run``.

        Returns:
            The (possibly mutated) profile list. Same list reference as
            the input on disabled and catastrophic-failure paths.
        """
        original_profiles = profiles
        try:
            project_dir = os.path.join(
                Config.UPLOAD_FOLDER, "projects", str(state.project_id)
            )
            service = AgentResearchService(
                search_provider=TavilyProvider(),
                cache=QueryCache(),
                llm_client=LLMClient(role="builder"),
                jsonl_logger=ResearchJsonlLogger(project_dir),
            )
            activity_by_user_id = {
                ac.agent_id: ac for ac in sim_params.agent_configs
            }
            if len(activity_by_user_id) != len(profiles):
                # Defensive sanity check: ``user_id`` and ``agent_id``
                # are both sequential ``int`` walking the same
                # filtered entities list. A length mismatch signals an
                # upstream filter divergence — log loudly but do not
                # raise; the service skips per-agent when ``activity is
                # None``.
                _research_logger.warning(
                    "activity_by_user_id length (%d) does not match "
                    "profiles length (%d); some agents will be skipped",
                    len(activity_by_user_id),
                    len(profiles),
                )
            return service.run(
                state.project_id,
                profiles,
                activity_by_user_id,
                simulation_requirement,
            )
        except Exception:
            # Use the standard-library sub-logger (propagates to root)
            # so this catastrophic-failure record is visible to standard
            # log handlers AND test capture fixtures (e.g. pytest
            # ``caplog``). The project ``mirofish.simulation`` logger
            # has ``propagate=False`` by design — see
            # ``app/utils/logger.py``.
            _research_logger.exception(
                "research step failed; continuing with un-researched profiles"
            )
            return original_profiles

    def prepare_simulation(
        self,
        simulation_id: str,
        simulation_requirement: str,
        document_text: str,
        defined_entity_types: Optional[List[str]] = None,
        use_llm_for_profiles: bool = True,
        progress_callback: Optional[callable] = None,
        parallel_profile_count: int = 3
    ) -> SimulationState:
        """
        准备模拟环境（全程自动化）
        
        步骤：
        1. 从Zep图谱读取并过滤实体
        2. 为每个实体生成OASIS Agent Profile（可选LLM增强，支持并行）
        3. 使用LLM智能生成模拟配置参数（时间、活跃度、发言频率等）
        4. 保存配置文件和Profile文件
        5. 复制预设脚本到模拟目录
        
        Args:
            simulation_id: 模拟ID
            simulation_requirement: 模拟需求描述（用于LLM生成配置）
            document_text: 原始文档内容（用于LLM理解背景）
            defined_entity_types: 预定义的实体类型（可选）
            use_llm_for_profiles: 是否使用LLM生成详细人设
            progress_callback: 进度回调函数 (stage, progress, message)
            parallel_profile_count: 并行生成人设的数量，默认3
            
        Returns:
            SimulationState
        """
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"模拟不存在: {simulation_id}")
        
        try:
            state.status = SimulationStatus.PREPARING
            self._save_simulation_state(state)
            
            sim_dir = self._get_simulation_dir(simulation_id)
            
            # ========== 阶段1: 读取并过滤实体 ==========
            if progress_callback:
                progress_callback("reading", 0, t('progress.connectingZepGraph'))
            
            reader = ZepEntityReader()
            
            if progress_callback:
                progress_callback("reading", 30, t('progress.readingNodeData'))
            
            filtered = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=defined_entity_types,
                enrich_with_edges=True
            )
            
            state.entities_count = filtered.filtered_count
            state.entity_types = list(filtered.entity_types)
            
            if progress_callback:
                progress_callback(
                    "reading", 100,
                    t('progress.readingComplete', count=filtered.filtered_count),
                    current=filtered.filtered_count,
                    total=filtered.filtered_count
                )
            
            if filtered.filtered_count == 0:
                state.status = SimulationStatus.FAILED
                state.error = "没有找到符合条件的实体，请检查图谱是否正确构建"
                self._save_simulation_state(state)
                return state
            
            # ========== 阶段2: 生成Agent Profile ==========
            total_entities = len(filtered.entities)
            
            if progress_callback:
                progress_callback(
                    "generating_profiles", 0,
                    t('progress.startGenerating'),
                    current=0,
                    total=total_entities
                )
            
            # 传入graph_id以启用Zep检索功能，获取更丰富的上下文
            generator = OasisProfileGenerator(graph_id=state.graph_id)
            
            def profile_progress(current, total, msg):
                if progress_callback:
                    progress_callback(
                        "generating_profiles", 
                        int(current / total * 100), 
                        msg,
                        current=current,
                        total=total,
                        item_name=msg
                    )
            
            # 设置实时保存的文件路径（优先使用 Reddit JSON 格式）
            realtime_output_path = None
            realtime_platform = "reddit"
            if state.enable_reddit:
                realtime_output_path = os.path.join(sim_dir, "reddit_profiles.json")
                realtime_platform = "reddit"
            elif state.enable_twitter:
                realtime_output_path = os.path.join(sim_dir, "twitter_profiles.csv")
                realtime_platform = "twitter"
            
            profiles = generator.generate_profiles_from_entities(
                entities=filtered.entities,
                use_llm=use_llm_for_profiles,
                progress_callback=profile_progress,
                graph_id=state.graph_id,  # 传入graph_id用于Zep检索
                parallel_count=parallel_profile_count,  # 并行生成数量
                realtime_output_path=realtime_output_path,  # 实时保存路径
                output_platform=realtime_platform  # 输出格式
            )
            
            state.profiles_count = len(profiles)

            # ========== 阶段3: LLM智能生成模拟配置 ==========
            # Moved above save_profiles so that ``sim_params.agent_configs``
            # (the per-agent activity records) is available for Step 2.5
            # below — research needs ``activity_by_user_id`` to compute
            # per-agent search budgets and read each agent's ``stance``.
            if progress_callback:
                progress_callback(
                    "generating_config", 0,
                    t('progress.analyzingRequirements'),
                    current=0,
                    total=3
                )

            config_generator = SimulationConfigGenerator()

            if progress_callback:
                progress_callback(
                    "generating_config", 30,
                    t('progress.callingLLMConfig'),
                    current=1,
                    total=3
                )

            sim_params = config_generator.generate_config(
                simulation_id=simulation_id,
                project_id=state.project_id,
                graph_id=state.graph_id,
                simulation_requirement=simulation_requirement,
                document_text=document_text,
                entities=filtered.entities,
                enable_twitter=state.enable_twitter,
                enable_reddit=state.enable_reddit
            )

            if progress_callback:
                progress_callback(
                    "generating_config", 70,
                    t('progress.savingConfigFiles'),
                    current=2,
                    total=3
                )

            # 保存配置文件
            config_path = os.path.join(sim_dir, "simulation_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(sim_params.to_json())

            state.config_generated = True
            state.config_reasoning = sim_params.generation_reasoning

            if progress_callback:
                progress_callback(
                    "generating_config", 100,
                    t('progress.configComplete'),
                    current=3,
                    total=3
                )

            # Pipeline Step 2.5 (per planning §5.8 / §3.3): one-shot
            # agent web research, after personas bake AND after activity
            # configs exist, before profiles are serialized to CSV/JSON.
            # topic_seed = simulation_requirement (planning §9.3 — the
            # operator-supplied "news subject" the agents are searching
            # about; same string forwarded into the config generator
            # above and the empirically-correct field per AC-4).
            # See ``_run_research_step`` for the contract details and
            # the catastrophic-failure isolation rationale.
            profiles = self._run_research_step(
                state, profiles, sim_params, simulation_requirement
            )

            # 保存Profile文件（注意：Twitter使用CSV格式，Reddit使用JSON格式）
            # Reddit 已经在生成过程中实时保存了，这里再保存一次确保完整性
            if progress_callback:
                progress_callback(
                    "generating_profiles", 95,
                    t('progress.savingProfiles'),
                    current=total_entities,
                    total=total_entities
                )

            if state.enable_reddit:
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "reddit_profiles.json"),
                    platform="reddit"
                )

            if state.enable_twitter:
                # Twitter使用CSV格式！这是OASIS的要求
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "twitter_profiles.csv"),
                    platform="twitter"
                )

            if progress_callback:
                progress_callback(
                    "generating_profiles", 100,
                    t('progress.profilesComplete', count=len(profiles)),
                    current=len(profiles),
                    total=len(profiles)
                )

            # 注意：运行脚本保留在 backend/scripts/ 目录，不再复制到模拟目录
            # 启动模拟时，simulation_runner 会从 scripts/ 目录运行脚本

            # 更新状态
            state.status = SimulationStatus.READY
            self._save_simulation_state(state)
            
            logger.info(f"模拟准备完成: {simulation_id}, "
                       f"entities={state.entities_count}, profiles={state.profiles_count}")
            
            return state
            
        except Exception as e:
            logger.error(f"模拟准备失败: {simulation_id}, error={str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            state.status = SimulationStatus.FAILED
            state.error = str(e)
            self._save_simulation_state(state)
            raise
    
    def get_simulation(self, simulation_id: str) -> Optional[SimulationState]:
        """获取模拟状态"""
        return self._load_simulation_state(simulation_id)
    
    def list_simulations(self, project_id: Optional[str] = None) -> List[SimulationState]:
        """列出所有模拟"""
        simulations = []
        
        if os.path.exists(self.SIMULATION_DATA_DIR):
            for sim_id in os.listdir(self.SIMULATION_DATA_DIR):
                # 跳过隐藏文件（如 .DS_Store）和非目录文件
                sim_path = os.path.join(self.SIMULATION_DATA_DIR, sim_id)
                if sim_id.startswith('.') or not os.path.isdir(sim_path):
                    continue
                
                state = self._load_simulation_state(sim_id)
                if state:
                    if project_id is None or state.project_id == project_id:
                        simulations.append(state)
        
        return simulations
    
    def get_profiles(self, simulation_id: str, platform: str = "reddit") -> List[Dict[str, Any]]:
        """获取模拟的Agent Profile"""
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"模拟不存在: {simulation_id}")
        
        sim_dir = self._get_simulation_dir(simulation_id)
        profile_path = os.path.join(sim_dir, f"{platform}_profiles.json")
        
        if not os.path.exists(profile_path):
            return []
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_simulation_config(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """获取模拟配置"""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_run_instructions(self, simulation_id: str) -> Dict[str, str]:
        """获取运行说明"""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
        
        return {
            "simulation_dir": sim_dir,
            "scripts_dir": scripts_dir,
            "config_file": config_path,
            "commands": {
                "twitter": f"python {scripts_dir}/run_twitter_simulation.py --config {config_path}",
                "reddit": f"python {scripts_dir}/run_reddit_simulation.py --config {config_path}",
                "parallel": f"python {scripts_dir}/run_parallel_simulation.py --config {config_path}",
            },
            "instructions": (
                f"1. 激活conda环境: conda activate MiroFish\n"
                f"2. 运行模拟 (脚本位于 {scripts_dir}):\n"
                f"   - 单独运行Twitter: python {scripts_dir}/run_twitter_simulation.py --config {config_path}\n"
                f"   - 单独运行Reddit: python {scripts_dir}/run_reddit_simulation.py --config {config_path}\n"
                f"   - 并行运行双平台: python {scripts_dir}/run_parallel_simulation.py --config {config_path}"
            )
        }
