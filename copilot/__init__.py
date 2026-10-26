"""Incident copilot: a Muse Spark-powered investigator built on PostmortemEnv.

The environment in ``engine/`` is the evidence store and the deterministic
grader. This package is the product layer on top of it: an agent that
investigates an incident blind (no oracle feedback), routes each step to the
model configured for it (Muse Spark on Meta Model API by default), and writes the postmortem.
"""
