"""The build script of the machine: one module per pipeline stage.

The root calls these in the order a readout travels. gem5 keeps the
same split: the object model lives beside each component and the
build lives in configs/ (configs/common/MemConfig.py,
configs/common/CacheConfig.py).
"""
