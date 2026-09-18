"""Issue #2790: a reaction carries no text, so the group-mention gate
(``_should_skip_unmentioned`` via ``is_group_mentioned``) must not fall
through to ``is_mentioned(msg.content)`` -- ``content`` is always ``""`` for
a ``MESSAGE_REACTION_ADD`` event, so that check is unconditionally ``False``
and drops *every* reaction in *every* guild channel or thread, including one
added to a message the bot itself just sent.
"""

from __future__ import annotations

from agentos.channels.discord import DiscordChannel, DiscordChannelConfig
from agentos.gateway.channel_dispatch import _should_skip_unmentioned


def _reaction_event(*, message_id: str, channel_id: str = "guild-channel-1") -> dict:
    return {
        "message_id": message_id,
        "channel_id": channel_id,
        "guild_id": "guild-1",
        "user_id": "user-1",
        "emoji": {"name": "thumbsup"},
    }


async def test_reaction_on_the_bots_own_message_is_not_gated() -> None:
    channel = DiscordChannel(DiscordChannelConfig(token="token"))
    channel._sent_messages["bot-msg-1"] = "guild-channel-1"

    await channel._handle_dispatch("MESSAGE_REACTION_ADD", _reaction_event(message_id="bot-msg-1"))
    msg = await channel.receive()

    assert msg.metadata["is_group"] is True
    assert channel.is_group_mentioned(msg) is True
    session_key = "agent:main:discord:group:guild-channel-1"
    assert _should_skip_unmentioned(channel, msg, session_key) is False


async def test_reaction_on_someone_elses_message_is_still_gated() -> None:
    channel = DiscordChannel(DiscordChannelConfig(token="token"))

    await channel._handle_dispatch(
        "MESSAGE_REACTION_ADD", _reaction_event(message_id="someone-elses-msg")
    )
    msg = await channel.receive()

    assert msg.metadata["is_group"] is True
    assert channel.is_group_mentioned(msg) is False
    session_key = "agent:main:discord:group:guild-channel-1"
    assert _should_skip_unmentioned(channel, msg, session_key) is True


async def test_reaction_on_the_bots_own_message_in_a_thread_is_not_gated() -> None:
    """_sent_messages is keyed only by message_id, not by channel/thread --
    confirms the lookup isn't accidentally scoped to top-level channels."""
    channel = DiscordChannel(DiscordChannelConfig(token="token"))
    channel._channel_types["thread-1"] = channel._channel_type(11)  # public thread
    channel._sent_messages["bot-msg-in-thread"] = "thread-1"

    await channel._handle_dispatch(
        "MESSAGE_REACTION_ADD",
        _reaction_event(message_id="bot-msg-in-thread", channel_id="thread-1"),
    )
    msg = await channel.receive()

    assert channel.is_group_mentioned(msg) is True


async def test_a_reaction_missing_its_message_id_is_not_mentioned() -> None:
    """Defensive: a malformed/absent message_id must fail closed, not raise."""
    channel = DiscordChannel(DiscordChannelConfig(token="token"))

    event = _reaction_event(message_id="whatever")
    del event["message_id"]
    await channel._handle_dispatch("MESSAGE_REACTION_ADD", event)
    msg = await channel.receive()

    assert channel.is_group_mentioned(msg) is False
