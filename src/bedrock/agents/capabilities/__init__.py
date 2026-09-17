"""One file per capability; each one exports ``CAPABILITIES``.

Adding an adapter means adding a file here, never editing a list. This
``__init__`` is deliberately empty of imports so that discovering what exists
costs nothing and pulls in no HTTP client, no mail library and no settings.
"""
