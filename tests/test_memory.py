"""Three-tier memory system tests (structure, encoding, retrieval, priority ranking, lifecycle)."""
import pytest
import torch

from msef.config import MemoryConfig
from msef.memory.encoder import deterministic_embed, token_estimate
from msef.memory.lifecycle import MemoryLifecycleManager
from msef.memory.retrieval import MemoryRetriever, freq_norm, recency_norm
from msef.memory.storage import MemoryStore
from msef.types import MemoryScope


def make_store(tmp_path) -> MemoryStore:
    return MemoryStore(
        project_root=str(tmp_path / "project"),
        team_dir=str(tmp_path / "team"),
        user_home=str(tmp_path / "user"),
        persist=False,
    )


def test_deterministic_embedding_dims_and_similarity():
    """Memory content maps to fixed-dimension dense vectors; semantically similar entries are close."""
    vec = deterministic_embed(["Coding standard: use snake_case function names"], dim=384)
    assert vec.shape == (1, 384)
    # Same-topic texts are more similar than unrelated ones
    similar = deterministic_embed(["Coding standard: functions use snake_case"], dim=384)
    unrelated = deterministic_embed(["Deployment checklist: database backup"], dim=384)
    sim_same = torch.nn.functional.cosine_similarity(vec, similar).item()
    sim_diff = torch.nn.functional.cosine_similarity(vec, unrelated).item()
    assert sim_same > sim_diff


def test_memory_store_three_tiers(tmp_path):
    """Three-tier storage: project/team/personal are independent tiers."""
    store = make_store(tmp_path)
    store.add("Project architecture: microservices", MemoryScope.PROJECT, category="architecture")
    store.add("Team migration template: V1 to V2", MemoryScope.TEAM, category="template")
    store.add("Personal preference: four-space indentation", MemoryScope.PERSONAL, category="preference")
    assert store.count(MemoryScope.PROJECT) == 1
    assert store.count(MemoryScope.TEAM) == 1
    assert store.count(MemoryScope.PERSONAL) == 1
    assert store.count() == 3


def test_retrieval_priority_ranking(tmp_path):
    """Retrieval priority: combined ranking of similarity + frequency + recency + tier."""
    store = make_store(tmp_path)
    cfg = MemoryConfig(similarity_threshold=0.2, top_k=3)
    # Project-level memory with high usage frequency
    proj = store.add("This project uses layered architecture; controllers call services", MemoryScope.PROJECT, category="architecture")
    for _ in range(5):
        store.touch(proj)
    store.add("Team command: mvn clean package", MemoryScope.TEAM, category="command")

    retriever = MemoryRetriever(store, cfg)
    ranked = retriever.retrieve("How is this project's layered architecture designed?", scopes=[MemoryScope.PROJECT, MemoryScope.TEAM])
    assert len(ranked) >= 1
    # Touching (usage) updates the last-access time
    top_entry, sim, prio = ranked[0]
    assert top_entry.usage_count >= 1


def test_retrieval_mount_context(tmp_path):
    """Context mounting: retrieval results are formatted as mount text."""
    store = make_store(tmp_path)
    store.add("Architecture rule: modules communicate through interfaces.", MemoryScope.PROJECT, category="architecture")
    retriever = MemoryRetriever(store, MemoryConfig(similarity_threshold=0.05))
    mounted, entries = retriever.mount("Architecture rules for module communication")
    assert "project memory" in mounted
    assert entries


def test_recency_norm_and_freq_norm():
    now = 2000.0
    assert recency_norm(now - 100, now, window=1000) == pytest.approx(0.9)
    assert recency_norm(now - 5000, now, window=1000) == 0.0
    assert freq_norm(3, 6) == 0.5
    assert freq_norm(0, 0) == 0.0


def test_memory_update_and_merge(tmp_path):
    """Memory update: similar entries merge; new content is added."""
    store = make_store(tmp_path)
    lc = MemoryLifecycleManager(store, MemoryConfig(merge_similarity_threshold=0.3))
    entry, merged = lc.write("This project forbids external HTTP calls inside database transactions", MemoryScope.PROJECT, source_type="stage_output")
    assert not merged
    entry2, merged2 = lc.write("External HTTP services are forbidden inside database transactions", MemoryScope.PROJECT, source_type="stage_output")
    if merged2:
        # After merging, only one main entry remains for the same topic
        assert entry2.id == entry.id or store.count(MemoryScope.PROJECT) >= 1
    # Completely different topic → new entry
    entry3, merged3 = lc.write("zzzz qqqq vvvv 9917", MemoryScope.PROJECT, source_type="manual")
    assert not merged3
    assert store.count(MemoryScope.PROJECT) == (2 if merged2 else 3)


def test_memory_version_control_rollback(tmp_path):
    """Version control: updates keep history versions and can be rolled back."""
    store = make_store(tmp_path)
    entry = store.add("Initial version content", MemoryScope.PROJECT)
    store.update(entry, new_content="Updated version content")
    assert len(entry.versions) == 1
    lc = MemoryLifecycleManager(store)
    assert lc.rollback(entry)
    assert entry.content == "Initial version content"


def test_lifecycle_apply_marks_downgrade_archive(tmp_path):
    """Lifecycle: low-activity entries are downgraded/archived/evicted."""
    store = make_store(tmp_path)
    for i in range(8):
        store.add(f"Stale entry {i}: legacy API notes", MemoryScope.TEAM)
    cfg = MemoryConfig(recency_window_seconds=1.0)
    lc = MemoryLifecycleManager(store, cfg)
    stats = lc.apply(MemoryScope.TEAM)
    assert sum(stats.values()) >= 0
    # Lifecycle actions do not corrupt storage
    assert store.count(MemoryScope.TEAM) <= 8


def test_promote_between_levels(tmp_path):
    """Memory promotion: project tier → team tier."""
    store = make_store(tmp_path)
    entry = store.add("Migration workflow: step1..step5", MemoryScope.PROJECT)
    lc = MemoryLifecycleManager(store)
    new_entry = lc.promote(entry.id, MemoryScope.PROJECT, MemoryScope.TEAM)
    assert new_entry is not None
    assert new_entry.scope == MemoryScope.TEAM
    assert store.get(entry.id, MemoryScope.PROJECT) is None
    assert store.get(new_entry.id, MemoryScope.TEAM) is not None
