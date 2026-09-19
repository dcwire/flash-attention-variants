"""Marker for work that is scaffolded but not yet written.

Every stub in this repo calls ``todo("...")``. ``tests/conftest.py`` turns the resulting
``NotYetImplemented`` into an *xfail* (not a failure), so ``pytest -ra`` doubles as a progress
board: an ``x`` is a task you haven't done, a ``.`` is one you have, an ``F`` is a real bug.
Once you implement something, just delete the ``todo`` call; the same test then has to pass.
"""


class NotYetImplemented(NotImplementedError):
    pass


def todo(what: str) -> None:
    raise NotYetImplemented(what)
