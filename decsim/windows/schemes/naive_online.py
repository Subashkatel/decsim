"""The naive_online row: one window over the whole operation.

No windowing at all: the operation's rounds are decoded as one batch
once they have arrived, which is the baseline every windowed scheme is
measured against.
"""

import decsim.records.windows as window_records
import decsim.windows.schemes.window_data as window_data


class NaiveOnlineScheme:
    """Decode each operation as one full batch after all rounds arrive."""

    scheme_label = "naive online batch decode (no windowing)"
    # One window covers the operation, so there is nothing past its
    # commit and nothing after it to chain to.
    has_trailing_tail_context = False
    commits_in_one_serial_chain = False
    supports_dynamic_streams = False

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> window_records.OperationWindowPlan:
        """One window over the whole operation."""
        del commit_round_count, buffer_round_count
        whole = window_records.WindowGeometry(1, 1, round_count, round_count)
        return window_records.OperationWindowPlan(
            operation_id=operation_id,
            windows=(whole,),
            internal_dependencies=(),
            entry_window_indices=(0,),
            exit_window_indices=(0,),
            windowed=False,
            batch_preceding_idle_rounds=True,
        )

    def data_complete(
        self,
        window: window_records.Window,
        *,
        readiness: window_records.WindowReadiness,
    ) -> bool:
        """Whether the window has every round it reads."""
        return window_data.sliding_data_complete(window, readiness)

    def validate_buffer(self, geometry) -> None:
        """A batch decode has no buffer floor."""
