"""The narrator's two listeners: the engine builds the text, they keep it.

The text is gem5's DPRINTF shape, the tick then the component's name
then the message (src/base/trace.hh).
"""

import decsim.engine as engine_module
import decsim.observe.log_writers as log_writers


def test_the_log_writer_keeps_each_stamped_line_in_order():
    engine = engine_module.Engine()
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    engine.log("worker", "ready")
    engine.now = 1_000_000
    engine.log("worker", "ready")
    assert log.lines == [
        "[  0.000 us] worker: ready",
        "[  1.000 us] worker: ready",
    ]


def test_the_console_printer_prints_each_line_as_it_is_fired(capsys):
    engine = engine_module.Engine()
    printer = log_writers.ConsolePrinter()
    engine.line.connect(printer.write)
    engine.now = 1_000_000
    engine.log("worker", "ready")
    captured = capsys.readouterr()
    assert captured.out == "[  1.000 us] worker: ready\n"


def test_an_io_line_is_described_only_when_someone_listens():
    silent = engine_module.Engine()
    described = []
    silent.log_io("Buffer 0", lambda: described.append("walked"))
    heard = engine_module.Engine()
    log = log_writers.LogWriter()
    heard.io_line.connect(log.write)
    heard.log_io("Buffer 0", lambda: "holds 1 round")
    assert described == []
    assert log.lines == ["[  0.000 us] Buffer 0: holds 1 round"]
