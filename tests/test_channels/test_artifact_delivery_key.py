"""``artifact_delivery_key`` identity — delivery is keyed on content *and*
name, not either alone (issue #2133).

``sha256`` alone was consulted first, so two artifacts with different names
and matching bytes -- two empty CSVs, a template rendered per region, two
placeholder images -- produced the same key and the second was silently
dropped by ``dedupe_artifacts_for_channel_delivery`` before delivery, with
no log line and no text-fallback mention.

The naive fix of preferring ``id`` over content is also wrong:
``ArtifactStore.publish_bytes()`` mints a fresh random id on every call
regardless of content, so every real artifact carries a unique id, and
preferring it would make two identical republishes of the same file under
the same name look like two different artifacts, silently duplicating
delivery instead of dropping it -- the opposite failure, but still wrong.
"""

from __future__ import annotations

from agentos.channels.artifact_delivery import (
    artifact_delivery_key,
    dedupe_artifacts_for_channel_delivery,
)

SHA = "a" * 64
OTHER_SHA = "b" * 64


def test_same_content_different_names_are_distinct() -> None:
    """The issue's own reported bug: two files under different names must
    not collapse into one just because their bytes match."""
    first = {"sha256": SHA, "name": "q1.csv"}
    second = {"sha256": SHA, "name": "q2.csv"}

    assert artifact_delivery_key(first) != artifact_delivery_key(second)


def test_same_content_same_name_is_a_duplicate() -> None:
    first = {"sha256": SHA, "name": "report.csv"}
    second = {"sha256": SHA, "name": "report.csv"}

    assert artifact_delivery_key(first) == artifact_delivery_key(second)


def test_id_does_not_override_content_and_name_identity() -> None:
    """Regression guard: a naive fix that checks ``id`` before content+name
    would give every real artifact (id is always unique) a distinct key,
    reintroducing duplicate delivery for a genuine republish of the same
    file under the same name. ``id`` must not be allowed to win here."""
    first = {"id": "art-aaaaaaaaaaaaaaaaaa", "sha256": SHA, "name": "image.png"}
    second = {"id": "art-bbbbbbbbbbbbbbbbbb", "sha256": SHA, "name": "image.png"}

    assert artifact_delivery_key(first) == artifact_delivery_key(second)


def test_dedupe_keeps_distinct_names_with_identical_content() -> None:
    """Two empty CSVs the user asked for by name are two deliveries."""
    artifacts = [
        {"sha256": SHA, "name": "q1.csv"},
        {"sha256": SHA, "name": "q2.csv"},
        {"sha256": SHA, "name": "q1.csv"},
    ]

    kept = dedupe_artifacts_for_channel_delivery(artifacts)

    assert [item["name"] for item in kept] == ["q1.csv", "q2.csv"]


def test_dedupe_still_collapses_true_duplicates() -> None:
    artifacts = [
        {"id": "art-1", "sha256": SHA, "name": "same.png"},
        {"id": "art-2", "sha256": SHA, "name": "same.png"},
    ]

    assert len(dedupe_artifacts_for_channel_delivery(artifacts)) == 1


def test_different_content_same_name_are_distinct() -> None:
    """Boundary this fix does not touch: same name, different bytes, is
    still two artifacts -- content matters, not just the name."""
    first = {"sha256": SHA, "name": "report.csv"}
    second = {"sha256": OTHER_SHA, "name": "report.csv"}

    assert artifact_delivery_key(first) != artifact_delivery_key(second)


def test_falls_back_through_the_field_order() -> None:
    assert artifact_delivery_key({"path": "/tmp/a.bin"}).startswith("path:")
    assert artifact_delivery_key({"id": "art-1"}).startswith("id:")
    assert artifact_delivery_key({"name": "only.txt"}) == "name:only.txt"


def test_missing_name_still_yields_a_content_key() -> None:
    assert artifact_delivery_key({"sha256": SHA}) == f"sha256:{SHA}"


def test_non_string_name_is_ignored() -> None:
    assert artifact_delivery_key({"sha256": SHA, "name": 42}) == f"sha256:{SHA}"


def test_empty_artifact_has_no_key() -> None:
    assert artifact_delivery_key({}) == ""


def test_keyless_artifacts_are_never_deduped_together() -> None:
    """An artifact with no identity field must not collapse into another."""
    artifacts = [{}, {}]

    assert len(dedupe_artifacts_for_channel_delivery(artifacts)) == 2
