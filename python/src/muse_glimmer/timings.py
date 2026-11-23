"""Expose the serving adapter's per-turn HUD measurements on timed requests."""

import json
from contextvars import ContextVar

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .constants import TIMINGS_HEADER

_request: ContextVar[Scope | None] = ContextVar("glimmer_timed_request", default=None)


class RequestMetrics:
    """Keep the HUD collector and associate its turn with the requesting client."""

    def __init__(self, metrics, hud_enabled: bool):
        self.metrics = metrics
        self.hud_enabled = hud_enabled

    def begin(self, session_id):
        scope = _request.get()
        if scope is None and not self.hud_enabled:
            return None
        turn = self.metrics.begin(session_id)
        if scope is not None:
            scope["glimmer.turn"] = turn
        return turn

    def __getattr__(self, name):
        return getattr(self.metrics, name)


class TimingHeaders:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (
            scope["type"] != "http"
            or scope["path"] != "/v1/chat/completions"
            or Headers(scope=scope).get(TIMINGS_HEADER) != "1"
        ):
            await self.app(scope, receive, send)
            return

        async def send_with_timings(message: Message):
            if message["type"] == "http.response.start" and message["status"] == 200:
                turn = scope.get("glimmer.turn")
                # Non-streaming completions finish before their response headers are sent.
                if turn is not None and turn.ended is not None:
                    ttft = (
                        None
                        if turn.first_output is None
                        else (turn.first_output - turn.started) * 1000
                    )
                    values = {
                        "ttft_ms": ttft,
                        "prefill_tok_s": turn.prefill_rate,
                        "decode_tok_s": turn.decode_rate,
                    }
                    MutableHeaders(scope=message)[TIMINGS_HEADER] = json.dumps(
                        {key: value for key, value in values.items() if value is not None}
                    )
            await send(message)

        token = _request.set(scope)
        try:
            await self.app(scope, receive, send_with_timings)
        finally:
            _request.reset(token)
