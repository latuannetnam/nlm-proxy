from nlm_proxy.core.config import SmartRoutingSettings
import os

def test_cross_notebook_settings_defaults():
    settings = SmartRoutingSettings()
    assert settings.cross_notebook_enabled is True
    assert settings.cross_notebook_max_secondary == 2
    assert settings.cross_notebook_concurrency == 5
    assert settings.cross_notebook_synthesis_enabled is True
    assert "Cross-referenced" in settings.cross_notebook_section_marker

def test_cross_notebook_env_vars(monkeypatch):
    monkeypatch.setenv("NLM_PROXY_ROUTING_CROSS_NOTEBOOK_ENABLED", "false")
    monkeypatch.setenv("NLM_PROXY_ROUTING_CROSS_NOTEBOOK_CONCURRENCY", "10")
    monkeypatch.setenv("NLM_PROXY_ROUTING_CROSS_NOTEBOOK_MAX_SECONDARY", "3")
    
    # We need to re-instantiate or force reload logic depending on implementation, 
    # but usually Pydantic BaseSettings reads env vars on init.
    settings = SmartRoutingSettings()
    assert settings.cross_notebook_enabled is False
    assert settings.cross_notebook_concurrency == 10
    assert settings.cross_notebook_max_secondary == 3
