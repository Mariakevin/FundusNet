"""Django settings package.
Defaults to development settings unless DJANGO_SETTINGS_MODULE is set.
"""

import os

# Default to dev settings unless specified otherwise
settings_module = os.getenv("DJANGO_SETTINGS_MODULE", "retina_project.settings.dev")
if settings_module.endswith(".prod") or "prod" in settings_module.lower():
    from .prod import *  # noqa: F401, F403
else:
    from .dev import *  # noqa: F401, F403
