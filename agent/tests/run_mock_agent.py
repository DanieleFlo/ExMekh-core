from __future__ import annotations

import os

from exmekh_agent.config import config
from exmekh_agent.main import main
from exmekh_agent.state.session_manager import session_manager
from tests.mock_client import (
    MockProvider,
    create_default_hierarchical_pipeline,
    install_mock_provider,
)

# Install MockProvider before launching UI
stream_delay = float(os.getenv("EXMEKH_AGENT_MOCK_STREAM_DELAY", "0"))
if stream_delay > 0:
    pipeline = create_default_hierarchical_pipeline()
    for step in pipeline.steps:
        step.delay = stream_delay
    install_mock_provider(MockProvider(pipeline=pipeline))
else:
    install_mock_provider()

if __name__ == "__main__":
    project = session_manager.create_project(config.work_root)
    session_manager.create_session(project_id=project.id)
    main()
