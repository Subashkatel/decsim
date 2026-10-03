"""The logical observables each operation delivered.

A listener on OperationResults' operation_result_delivered source. The
folder delivers an operation's result once and withdraws it by
delivering None, so the ledger holds exactly what the run produced and
the RunResult's rows are read from here rather than from the window
side.
"""

import decsim.observe.result_ledger as result_ledger


def test_a_withdrawal_is_a_delivery_of_none_and_removes_the_row():
    ledger = result_ledger.ResultLedger()

    ledger.operation_result_delivered(1, (0, 1))
    ledger.operation_result_delivered(1, None)

    assert ledger.observables_for(1) is None
    assert ledger.result_by_operation == {}


def test_a_second_delivery_replaces_the_first():
    ledger = result_ledger.ResultLedger()

    ledger.operation_result_delivered(1, (0,))
    ledger.operation_result_delivered(1, (1,))

    assert ledger.observables_for(1) == (1,)
