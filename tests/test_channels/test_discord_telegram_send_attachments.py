"""Issue #2147: Discord and Telegram channel adapters silently drop
message attachments.

Neither ``DiscordChannel.send`` nor ``TelegramChannel.send`` inspected
``OutgoingMessage.attachments`` at all -- only ``message.content`` reached
the platform, and every generated deliverable attached via the
``attachments`` field vanished with no error, no warning, and no upload.
"""

from __future__ import annotations

import json as _json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentos.channels.discord import DiscordChannel, DiscordChannelConfig
from agentos.channels.telegram import TelegramChannel, TelegramChannelConfig
from agentos.channels.types import Attachment, OutgoingMessage

# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------


class _DiscordResponse:
    status_code = 200

    def __init__(self, message_id: str) -> None:
        self._message_id = message_id

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"id": self._message_id}


def _discord_channel() -> tuple[DiscordChannel, list[dict[str, Any]]]:
    channel = DiscordChannel(DiscordChannelConfig(token="token", application_id="app"))
    client = AsyncMock()
    calls: list[dict[str, Any]] = []

    async def _post(url: str, **kwargs: Any) -> _DiscordResponse:
        calls.append(kwargs)
        return _DiscordResponse(str(len(calls)))

    async def _patch(url: str, **kwargs: Any) -> _DiscordResponse:
        calls.append(kwargs)
        return _DiscordResponse(str(len(calls)))

    client.post = _post
    client.patch = _patch
    channel._client = client
    return channel, calls


@pytest.mark.asyncio
async def test_discord_send_uploads_an_attachment_as_multipart() -> None:
    channel, calls = _discord_channel()
    attachment = Attachment(name="report.pdf", mime_type="application/pdf", data=b"%PDF-1.4...")

    await channel.send(
        OutgoingMessage(content="Here is your file", attachments=[attachment], reply_to="channel-1")
    )

    assert len(calls) == 1
    call = calls[0]
    assert "json" not in call
    assert call["files"]["files[0]"] == ("report.pdf", b"%PDF-1.4...", "application/pdf")
    payload = _json.loads(call["data"]["payload_json"])
    assert payload["content"] == "Here is your file"


@pytest.mark.asyncio
async def test_discord_send_attachment_only_rides_the_last_chunk() -> None:
    """Same rule as embeds/components: describe the complete answer once,
    not on every continuation chunk."""
    channel, calls = _discord_channel()
    long_content = "x" * 5000
    attachment = Attachment(name="report.pdf", mime_type="application/pdf", data=b"data")

    await channel.send(
        OutgoingMessage(content=long_content, attachments=[attachment], reply_to="channel-1")
    )

    assert len(calls) > 1
    assert all("files" not in call for call in calls[:-1])
    assert "files" in calls[-1]


@pytest.mark.asyncio
async def test_discord_send_skips_a_url_only_attachment_without_data() -> None:
    """Fetching a caller-supplied URL would need the same SSRF-safe path
    inbound downloads use -- out of scope, so it's skipped, not fetched."""
    channel, calls = _discord_channel()
    attachment = Attachment(name="remote.png", url="https://example.com/remote.png")

    await channel.send(
        OutgoingMessage(content="see attached", attachments=[attachment], reply_to="channel-1")
    )

    assert len(calls) == 1
    assert "files" not in calls[0]
    assert calls[0]["json"]["content"] == "see attached"


@pytest.mark.asyncio
async def test_discord_send_drops_an_oversized_attachment_but_still_sends_text() -> None:
    """Guard: one attachment over the upload ceiling must not cost the
    caller the text reply that came with it."""
    channel, calls = _discord_channel()
    channel.MAX_FILE_BYTES = 4
    attachment = Attachment(name="big.bin", data=b"too many bytes")

    await channel.send(
        OutgoingMessage(
            content="text still arrives", attachments=[attachment], reply_to="channel-1"
        )
    )

    assert len(calls) == 1
    assert "files" not in calls[0]
    assert calls[0]["json"]["content"] == "text still arrives"


@pytest.mark.asyncio
async def test_discord_interaction_response_delivers_an_attachment_via_multipart() -> None:
    channel, calls = _discord_channel()
    attachment = Attachment(name="report.pdf", mime_type="application/pdf", data=b"data")

    await channel.send(
        OutgoingMessage(
            content="done",
            attachments=[attachment],
            reply_to="channel-1",
            metadata={
                "interaction_token": "tok",
                "interaction_application_id": "app",
                "interaction_id": "int-1",
            },
        )
    )

    assert len(calls) == 1
    assert "files" in calls[0]


@pytest.mark.asyncio
async def test_discord_send_reply_reference_embeds_and_attachment_coexist_on_one_chunk() -> None:
    """A short reply is a single segment, so index 0 is also the last chunk:
    the reply reference (first-chunk rule), embeds/components (last-chunk
    rule), and the attachment (last-chunk rule) all land in the *same*
    multipart payload rather than one silently overwriting another."""
    channel, calls = _discord_channel()
    attachment = Attachment(name="report.pdf", mime_type="application/pdf", data=b"data")

    await channel.send(
        OutgoingMessage(
            content="short reply",
            attachments=[attachment],
            reply_to="channel-1",
            metadata={
                "reply_to_message_id": "99",
                "embeds": [{"title": "t"}],
                "components": [{"type": 1}],
            },
        )
    )

    assert len(calls) == 1
    call = calls[0]
    assert "json" not in call
    assert call["files"]["files[0]"] == ("report.pdf", b"data", "application/pdf")
    payload = _json.loads(call["data"]["payload_json"])
    assert payload["content"] == "short reply"
    assert payload["message_reference"] == {"message_id": "99"}
    assert payload["embeds"] == [{"title": "t"}]
    assert payload["components"] == [{"type": 1}]


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


class _TelegramResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _telegram_channel() -> tuple[TelegramChannel, list[dict[str, Any]]]:
    channel = TelegramChannel(TelegramChannelConfig(token="token"))
    client = AsyncMock()
    calls: list[dict[str, Any]] = []

    async def _post(url: str, **kwargs: Any) -> _TelegramResponse:
        calls.append({"url": url, **kwargs})
        if url.endswith("/sendDocument"):
            return _TelegramResponse(
                {"ok": True, "result": {"message_id": len(calls), "document": {"file_id": "f1"}}}
            )
        return _TelegramResponse({"ok": True, "result": {"message_id": len(calls)}})

    client.post = _post
    channel._client = client
    channel._owns_client = False
    return channel, calls


@pytest.mark.asyncio
async def test_telegram_send_uploads_an_attachment_via_send_document() -> None:
    channel, calls = _telegram_channel()
    attachment = Attachment(name="report.pdf", mime_type="application/pdf", data=b"%PDF-1.4...")

    await channel.send(
        OutgoingMessage(content="Here is your file", attachments=[attachment], reply_to="123")
    )

    assert len(calls) == 2
    text_call, doc_call = calls
    assert text_call["url"].endswith("/sendMessage")
    assert doc_call["url"].endswith("/sendDocument")
    assert doc_call["files"]["document"] == ("report.pdf", b"%PDF-1.4...", "application/pdf")
    assert doc_call["data"]["chat_id"] == "123"


@pytest.mark.asyncio
async def test_telegram_send_uploads_multiple_attachments_as_separate_documents() -> None:
    channel, calls = _telegram_channel()
    attachments = [Attachment(name="a.txt", data=b"a"), Attachment(name="b.txt", data=b"b")]

    await channel.send(
        OutgoingMessage(content="two files", attachments=attachments, reply_to="123")
    )

    doc_calls = [c for c in calls if c["url"].endswith("/sendDocument")]
    assert len(doc_calls) == 2
    assert doc_calls[0]["files"]["document"][0] == "a.txt"
    assert doc_calls[1]["files"]["document"][0] == "b.txt"


@pytest.mark.asyncio
async def test_telegram_send_skips_a_url_only_attachment_without_data() -> None:
    channel, calls = _telegram_channel()
    attachment = Attachment(name="remote.png", url="https://example.com/remote.png")

    await channel.send(
        OutgoingMessage(content="see attached", attachments=[attachment], reply_to="123")
    )

    assert len(calls) == 1
    assert calls[0]["url"].endswith("/sendMessage")


@pytest.mark.asyncio
async def test_telegram_send_drops_an_oversized_attachment_but_still_sends_text() -> None:
    channel, calls = _telegram_channel()
    channel.MAX_FILE_BYTES = 4
    attachment = Attachment(name="big.bin", data=b"too many bytes")

    await channel.send(
        OutgoingMessage(content="text still arrives", attachments=[attachment], reply_to="123")
    )

    assert len(calls) == 1
    assert calls[0]["url"].endswith("/sendMessage")


@pytest.mark.asyncio
async def test_telegram_send_forwards_thread_id_to_send_document() -> None:
    channel, calls = _telegram_channel()
    attachment = Attachment(name="report.pdf", data=b"data")

    await channel.send(
        OutgoingMessage(
            content="here",
            attachments=[attachment],
            reply_to="123",
            metadata={"thread_id": 42},
        )
    )

    doc_call = calls[-1]
    assert doc_call["data"]["message_thread_id"] == 42
