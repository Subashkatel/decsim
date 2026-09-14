"""The vocabulary every component speaks, one module per record family.

The frozen values that travel between the QPU, the controller, the
buffers, the window manager, the decoders and the Pauli frame. Nothing
here has behavior beyond a value's own derived views, and nothing is
re-exported: a reader imports the module whose family it needs.
"""
