import asyncio
import json
from types import SimpleNamespace

from muse_glimmer.timings import RequestMetrics, TimingHeaders
from starlette.datastructures import Headers


def test_concurrent_completions_return_their_own_worker_stats():
    class Metrics:
        def begin(self, session_id):
            return SimpleNamespace(
                started=2, first_output=2.25, ended=None, prefill_rate=None, decode_rate=None
            )

    metrics = RequestMetrics(Metrics(), hud_enabled=False)

    async def app(scope, receive, send):
        turn = metrics.begin(None)
        if turn is not None:
            await asyncio.sleep(0)
            turn.decode_rate = scope["rate"]
            turn.ended = 3
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def check(rate):
        async def inspect(message):
            header = Headers(raw=message["headers"]).get("x-glimmer-timings")
            if rate is None:
                assert header is None
            else:
                assert json.loads(header) == {"ttft_ms": 250, "decode_tok_s": rate}

        headers = [] if rate is None else [(b"x-glimmer-timings", b"1")]
        scope = {"type": "http", "path": "/v1/chat/completions", "headers": headers, "rate": rate}
        await TimingHeaders(app)(scope, None, inspect)

    async def run():
        await asyncio.gather(check(21), check(None), check(42))

    asyncio.run(run())
