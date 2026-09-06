"""The logical observables each operation delivered.

A listener on OperationResults' operation_result_delivered(operation_id,
logical_observables); the folder delivers an operation's result once and
withdraws it by delivering None, so the ledger holds exactly what the
run produced. The RunResult's operation_results rows are read from here,
never from the window side.
"""


class ResultLedger:
    """Every delivered result by operation id."""

    def __init__(self) -> None:
        self.result_by_operation: dict = {}

    def operation_result_delivered(
        self, operation_id, logical_observables
    ) -> None:
        """One operation's result was delivered, or withdrawn with None."""
        if logical_observables is None:
            self.result_by_operation.pop(operation_id, None)
            return
        self.result_by_operation[operation_id] = logical_observables

    def observables_for(self, operation_id):
        """The operation's delivered observables; None when it has none."""
        return self.result_by_operation.get(operation_id)
