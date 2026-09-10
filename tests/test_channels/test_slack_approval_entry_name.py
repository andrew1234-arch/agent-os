"""Regression test: a Slack channel entry's configured ``name`` never
reaches ``SlackChannel``, breaking approval sessionKey binding for any
entry not literally named "slack".

Root cause is in the registry, not slack.py's approval-check logic itself:
``_build_generic_channel``'s flat/kwargs path (used for adapters like
``SlackChannel`` that take plain constructor kwargs rather than a nested
``*ChannelConfig`` object) has no equivalent of the ``name`` re-injection
the config-wrapped path already does. ``entry.name`` is unconditionally
excluded from the dumped data (``_COMMON_ENTRY_FIELDS``) and never added
back for the flat path, so ``SlackChannel.name`` is permanently stuck at
its default regardless of how the operator configures the entry.

Any multi-account Slack setup (the exact scenario #1022 already
established as a real, supported use case) necessarily has at least one
entry not literally named "slack" (entry names are the dict key in
``ChannelManager._channels`` and must be unique). For any such entry,
every Approve/Deny click is wrongly rejected as a "mismatch".
"""

from __future__ import annotations

import pytest

from agentos.channels.registry import build_managed_channel
from agentos.channels.slack import SlackChannel
from agentos.gateway.approval_queue import get_approval_queue, reset_approval_queue
from agentos.gateway.config import SlackChannelEntry
from agentos.session.keys import build_group_key


@pytest.fixture(autouse=True)
def _clean_approval_queue():
    reset_approval_queue()
    yield
    reset_approval_queue()


def test_registry_wires_entry_name_into_slack_channel() -> None:
    """The entry's configured name must reach the built adapter -- this is
    exactly what already happens for Discord/Telegram/MSTeams via their
    nested ``*ChannelConfig`` classes; Slack's flat constructor path must
    get the same treatment.
    """
    entry = SlackChannelEntry(name="slack-support", token="xoxb-fake", slack_channel_id="C12345")

    channel = build_managed_channel(entry)

    assert isinstance(channel, SlackChannel)
    assert channel.name == "slack-support"


async def test_slack_approval_resolves_for_a_non_default_named_entry() -> None:
    """A Slack entry configured as e.g. "slack-support" (not literally
    "slack") must still be able to resolve its own pending approvals.
    """
    queue = get_approval_queue()
    session_key = build_group_key(agent_id="main", channel="slack-support", peer_id="C12345")
    approval_id = queue.request(
        "exec",
        {"argv": ["rm", "-rf"], "action_kind": "exec", "sessionKey": session_key},
    )

    entry = SlackChannelEntry(name="slack-support", token="xoxb-fake", slack_channel_id="C12345")
    channel = build_managed_channel(entry)
    assert isinstance(channel, SlackChannel)

    payload = {
        "actions": [{"value": f"approve:{approval_id}"}],
        "user": {"id": "usr123"},
        "channel": {"id": "C12345"},
        "team": {"id": "T123"},
        "message": {"text": "Approval requested"},
        # No response_url -- keeps this test focused on the binding check,
        # not Slack's outbound "update the original message" call.
    }

    await channel._handle_slack_interactive(payload)

    entry_after = queue.get(approval_id)
    assert entry_after.resolved is True, (
        "approval was never resolved -- the sessionKey check wrongly treated "
        "a correctly-matching entry name as a mismatch"
    )
    assert entry_after.approved is True
