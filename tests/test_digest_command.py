"""`/digest` driven through the real Textual app, same as the other app tests."""

from textual.widgets import Input

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.ui.widgets import SystemNotice

from tests.conftest import FakeBackend

CSV = "\n".join(
    ["sample,ph,note"] + [f"S{i:04d},{6.0 + i % 20 * 0.1:.1f},routine" for i in range(200)]
)


def make_app(tmp_path, backend: FakeBackend) -> BedrockApp:
    return BedrockApp(
        settings=Settings(model="test-model", context_dir=str(tmp_path / "context")),
        backend=backend,
    )


def notices(app: BedrockApp) -> str:
    return "\n".join(str(widget.source) for widget in app.query(SystemNotice))


async def submit(app: BedrockApp, pilot, text: str) -> None:
    """Type a command and run it to completion.

    The text is set on the input rather than pressed key by key: these
    commands carry long temporary file paths, and eighty individual key events
    per test is most of the suite's runtime for no extra coverage. Submitting
    still goes through the real ``Input.Submitted`` handler.
    """
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_digest_a_file_reports_both_savings(tmp_path, fake_backend: FakeBackend):
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    app = make_app(tmp_path, fake_backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")

        text = notices(app)
        assert "Compressed `assay.csv`" in text
        assert "smaller" in text and "cheaper to read" in text
        assert len(app.context_store.list()) == 1


async def test_digest_with_no_arguments_lists_what_is_stored(tmp_path, fake_backend: FakeBackend):
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    app = make_app(tmp_path, fake_backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")
        await submit(app, pilot, f"/digest")

        assert "Stored context" in notices(app)
        assert "assay.csv" in notices(app)


async def test_digest_search_returns_the_matching_content(tmp_path, fake_backend: FakeBackend):
    path = tmp_path / "assay.csv"
    path.write_text(CSV.replace("S0100,6.0,routine", "S0100,9.9,CONTAMINATION found"))
    app = make_app(tmp_path, fake_backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")
        await submit(app, pilot, f"/digest search contamination")

        assert "CONTAMINATION found" in notices(app)


async def test_digest_show_and_forget(tmp_path, fake_backend: FakeBackend):
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    app = make_app(tmp_path, fake_backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")
        artifact_id = app.context_store.list()[0].id

        await submit(app, pilot, f"/digest show {artifact_id}")
        assert "rows x 3 columns" in notices(app)

        await submit(app, pilot, f"/digest forget {artifact_id}")
        assert "Forgot `assay.csv`" in notices(app)
        assert app.context_store.list() == []


async def test_a_missing_file_is_a_plain_message_not_a_crash(tmp_path, fake_backend: FakeBackend):
    app = make_app(tmp_path, fake_backend)
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest /no/such/file.csv")

        assert "No file at" in notices(app)
        assert app.context_store.list() == []


async def test_prose_gets_the_model_summary_on_top_of_the_computed_one(tmp_path):
    """The one case the model earns its keep: prose, where rules have little
    to say. The backend is a fake, so no server is involved."""
    path = tmp_path / "protocol.md"
    path.write_text("# Protocol\n\n" + "Samples rest at ambient temperature before loading. " * 20)
    backend = FakeBackend(reply="GIST: How to handle samples before an assay.\n- Rest at ambient first")
    app = make_app(tmp_path, backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")

        assert "How to handle samples before an assay." in notices(app)
        stored = app.context_store.list()[0]
        assert stored.digest.generated_by == "test-model"


async def test_structured_data_never_calls_the_model(tmp_path):
    """A table's computed schema is better than a summary of it, and asking a
    model anyway would spend time to make the digest worse."""
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    backend = FakeBackend(reply="GIST: nonsense\n- nonsense")
    app = make_app(tmp_path, backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")

        assert app.context_store.list()[0].digest.generated_by == "rules"
        assert backend.requests == []
