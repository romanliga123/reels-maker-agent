import subprocess

import pytest

from reels_agent import config


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    """Изолирует тест от настоящих storage/uploads|work|output — каждый тест получает свои папки."""
    uploads = tmp_path / "uploads"
    work = tmp_path / "work"
    output = tmp_path / "output"
    for d in (uploads, work, output):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(config, "WORK_DIR", work)
    monkeypatch.setattr(config, "OUTPUT_DIR", output)
    return {"uploads": uploads, "work": work, "output": output}


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory):
    """5-секундное синтетическое видео 1280x720 с тоновым аудио — офлайн, без сети."""
    out_dir = tmp_path_factory.mktemp("fixtures")
    path = out_dir / "synth.mp4"
    cmd = [
        config.FFMPEG_BIN, "-y",
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=5",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=5",
        "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=60)
    return path


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory):
    """Видео без аудиодорожки — для проверки отказа при отсутствии звука."""
    out_dir = tmp_path_factory.mktemp("fixtures_silent")
    path = out_dir / "silent.mp4"
    cmd = [
        config.FFMPEG_BIN, "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24:duration=2",
        "-an", "-c:v", "libx264", str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=60)
    return path


@pytest.fixture(scope="session")
def corrupt_video(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("fixtures_corrupt")
    path = out_dir / "corrupt.mp4"
    path.write_text("этот файл не настоящее видео")
    return path


@pytest.fixture
def fake_transcript():
    from reels_agent.models import TranscriptSegment, Word
    return [
        TranscriptSegment(text="Привет, это первый сегмент.", start=0.0, end=4.0, words=[
            Word("Привет,", 0.0, 0.6), Word("это", 0.6, 0.9),
            Word("первый", 0.9, 1.5), Word("сегмент.", 1.5, 2.2),
        ]),
        TranscriptSegment(text="А это второй сегмент подряд.", start=4.0, end=8.0, words=[
            Word("А", 4.0, 4.1), Word("это", 4.1, 4.4), Word("второй", 4.4, 5.0),
            Word("сегмент", 5.0, 5.6), Word("подряд.", 5.6, 6.2),
        ]),
        TranscriptSegment(text="И третий, последний.", start=8.0, end=12.0, words=[
            Word("И", 8.0, 8.1), Word("третий,", 8.1, 8.7), Word("последний.", 8.7, 9.5),
        ]),
    ]


@pytest.fixture
def long_transcript():
    """~40с транскрипт из 7 сегментов — достаточно длинный, чтобы тесты на
    CLIP_MIN_SEC(15с)/CLIP_MAX_SEC(90с) клампинг не "съедали" весь транскрипт целиком."""
    from reels_agent.models import TranscriptSegment, Word
    texts = [
        "Привет, это первый сегмент.",
        "А это второй сегмент подряд.",
        "И третий, продолжаем дальше.",
        "Четвёртый сегмент тоже здесь.",
        "Пятый сегмент почти у цели.",
        "Шестой сегмент перед финалом.",
        "И седьмой, последний сегмент.",
    ]
    segments = []
    t = 0.0
    for text in texts:
        words = []
        word_start = t
        for w in text.split():
            w_end = word_start + 0.5
            words.append(Word(w, word_start, w_end))
            word_start = w_end + 0.1
        segments.append(TranscriptSegment(text=text, start=t, end=t + 4.0, words=words))
        t += 6.0
    return segments


def has_groq_key() -> bool:
    return bool(config.GROQ_API_KEY)
