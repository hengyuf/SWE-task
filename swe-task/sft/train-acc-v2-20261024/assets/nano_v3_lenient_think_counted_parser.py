# SPDX-License-Identifier: Apache-2.0
"""train-acc-v2 (2026-10-25): vLLM reasoning parser ``nano_v3_lenient_think_counted`` = ``nano_v3_lenient_think``
(eval/deepswe-v1.1/assets/nano_v3_lenient_think_parser.py, imported by path and never changed) plus log lines that
count the "missing </think>" cases. The parse result is exactly the lenient parser's; only logging is added.

Lines written to the vLLM server log (logs/vllm-<job>.log), per API-server process:
* ``TA_THINK_RECOVERED request_id=<id> reasoning_chars=<n>``: the think block was not closed and contains a
  ``<tool_call>``; the lenient rule ended the reasoning at that call (strict nano_v3 would have returned the whole
  output as reasoning and OpenCode would have ended the session on a turn without an action);
* ``TA_THINK_UNCLOSED_NO_TOOL request_id=<id> chars=<n>``: the block was not closed and has no ``<tool_call>``: the
  whole output stays reasoning (chars=0 is an end-of-sequence right after the ``<think>`` generation prefix);
* ``TA_THINK_TOOL_IN_CLOSED_THINK request_id=<id>``: the block was closed, but a ``<tool_call>`` sits INSIDE it (before
  ``</think>``): no parser recovers this; the call is reasoning text (counted within closed);
* ``TA_THINK_COUNTS pid=<pid> calls=<n> closed=<n> closed_tool_in_think=<n> recovered=<n> unclosed_no_tool=<n> empty=<n>
  thinking_off=<n>``
  on the first call and every 200 calls.
Served with ``--reasoning-parser-plugin <this file> --reasoning-parser nano_v3_lenient_think_counted``.
"""

import importlib.util
import os

from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

LENIENT_PATH = os.environ.get(
    "TA_LENIENT_PARSER_PATH",
    "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/assets/"
    "nano_v3_lenient_think_parser.py",
)
_spec = importlib.util.spec_from_file_location("ta_nano_v3_lenient_think_parser", LENIENT_PATH)
_lenient = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lenient)

logger = init_logger("vllm.reasoning.ta_think_counter")
TOOL_CALL_START = _lenient.TOOL_CALL_START
COUNTS = {"calls": 0, "closed": 0, "closed_tool_in_think": 0, "recovered": 0, "unclosed_no_tool": 0, "empty": 0,
          "thinking_off": 0}


@ReasoningParserManager.register_module("nano_v3_lenient_think_counted")
class NanoV3LenientThinkCountedReasoningParser(_lenient.NanoV3LenientThinkReasoningParser):
    """``nano_v3_lenient_think`` with counters (the parse itself is inherited unchanged)."""

    def extract_reasoning(self, model_output, request):
        result = super().extract_reasoning(model_output, request)
        try:
            self._count(model_output, request)
        except Exception as error:  # counting must never change or break a response
            logger.warning("TA_THINK_COUNTER_ERROR %s", error)
        return result

    def _count(self, model_output, request):
        COUNTS["calls"] += 1
        request_id = getattr(request, "request_id", None)
        kwargs = getattr(request, "chat_template_kwargs", None) or {}
        text = model_output.partition(self.start_token)[2] if self.start_token in model_output else model_output
        if kwargs.get("enable_thinking") is False:
            COUNTS["thinking_off"] += 1
        elif self.end_token in model_output:
            COUNTS["closed"] += 1
            if TOOL_CALL_START in text.partition(self.end_token)[0]:
                COUNTS["closed_tool_in_think"] += 1
                logger.info("TA_THINK_TOOL_IN_CLOSED_THINK request_id=%s", request_id)
        elif TOOL_CALL_START in text:
            COUNTS["recovered"] += 1
            logger.info("TA_THINK_RECOVERED request_id=%s reasoning_chars=%d", request_id,
                        text.find(TOOL_CALL_START))
        else:
            COUNTS["unclosed_no_tool"] += 1
            if not text.strip():
                COUNTS["empty"] += 1
            logger.info("TA_THINK_UNCLOSED_NO_TOOL request_id=%s chars=%d", request_id, len(text))
        if COUNTS["calls"] == 1 or COUNTS["calls"] % 200 == 0:
            logger.info("TA_THINK_COUNTS pid=%d %s", os.getpid(), " ".join("%s=%d" % kv for kv in COUNTS.items()))
