from darknode_ai.registry import Registry, ModelVersion


def test_add_get_and_deploy(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.add(ModelVersion(version="v0.1.0", metrics={"heldout_perplexity": 12.3}))
    reg.add(ModelVersion(version="v0.2.0", metrics={"heldout_perplexity": 9.1}))
    assert reg.get("v0.2.0").metrics["heldout_perplexity"] == 9.1

    reg.set_state("v0.1.0", "deployed")
    assert reg.deployed().version == "v0.1.0"
    # deploying another demotes the previous deployed one
    reg.set_state("v0.2.0", "deployed")
    assert reg.deployed().version == "v0.2.0"

    # persistence across instances
    reg2 = Registry(tmp_path / "registry.json")
    assert reg2.deployed().version == "v0.2.0"
    assert len(reg2.versions) == 2


def test_add_replaces_same_version(tmp_path):
    reg = Registry(tmp_path / "r.json")
    reg.add(ModelVersion(version="v1", metrics={"a": 1}))
    reg.add(ModelVersion(version="v1", metrics={"a": 2}))
    assert len(reg.versions) == 1
    assert reg.get("v1").metrics["a"] == 2
