"""Shared test setup."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "fitted_ruler: let finalize choose the shipped table with the fitted ruler "
        "and stop the blend's boosted members early",
    )


@pytest.fixture(autouse=True)
def _probe_chooses_what_ships(request, monkeypatch):
    """Finalize ships the probe's choice unless a test is marked ``fitted_ruler``.

    Most tests fix which table ships by writing probe scores and intervals into
    the journal, to test something downstream of the choice. The fitted ruler
    refits every table and would overrule those numbers; it has its own tests.
    The blend's boosted members likewise keep their library defaults (0099's
    stopping fits would triple the suite's time), except in those tests.
    """
    if request.node.get_closest_marker("fitted_ruler"):
        return
    from tabular_agent.harness import stop, tools

    def probe_choice(session, run_journal, candidates, metric):
        winner = stop.best_table(candidates, metric)
        return winner["path"] if winner else session.rel(session.data_path)

    monkeypatch.setattr(tools, "_choose_by_fitted_ruler", probe_choice)
    monkeypatch.setattr(tools, "STOPPED_MODEL_TYPES", ())
