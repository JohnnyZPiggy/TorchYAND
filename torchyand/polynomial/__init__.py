"""Fixed-shape, batched real polynomial roots."""

from .common import RootResult
from .linear import linear_roots
from .quadratic import quadratic_real_roots
from .cubic import cubic_real_roots
from .quartic_line_search import QuarticMinResult, minimize_quartic_interval

__all__ = ["RootResult", "linear_roots", "quadratic_real_roots", "cubic_real_roots",
           "QuarticMinResult", "minimize_quartic_interval"]
