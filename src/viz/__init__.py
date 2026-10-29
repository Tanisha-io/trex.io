"""Thesis figure regeneration (List of Figures 1.1, 6.1–6.23)."""

__all__ = ["generate_all"]


def generate_all(*args, **kwargs):
    from src.viz.generate_all import generate_all as _generate_all

    return _generate_all(*args, **kwargs)
