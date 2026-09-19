"""HTTP middleware for durable Transcription Job responses."""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class TranscriptionJobHeadersMiddleware:
    """Prevent caching of every durable Transcription Job response."""

    def __init__(self, app: ASGIApp) -> None:
        """Store the next application in the ASGI chain.

        Args:
            app: Downstream ASGI application.
        """
        self._app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Set ``Cache-Control: no-store`` on Transcription Job responses.

        Args:
            scope: Current ASGI connection scope.
            receive: ASGI message receiver.
            send: Outgoing ASGI message sender.
        """
        if scope["type"] != "http" or not _is_transcription_job_path(scope["path"]):
            await self._app(scope, receive, send)
            return

        async def send_with_job_headers(message: Message) -> None:
            """Add the no-store header when the response starts."""
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = "no-store"
            await send(message)

        await self._app(scope, receive, send_with_job_headers)


def _is_transcription_job_path(path: str) -> bool:
    """Return whether a path addresses the durable job API."""
    return path == "/api/transcription-jobs" or path.startswith(
        "/api/transcription-jobs/"
    )
