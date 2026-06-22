from pathlib import Path

from reels_agent.pipeline.render import _escape_filter_path


def test_drive_colon_gets_double_backslash_escaped():
    p = Path("C:/Users/someone/clip.ass")
    escaped = _escape_filter_path(p)
    assert escaped == "C\\\\:/Users/someone/clip.ass"


def test_backslashes_converted_to_forward_slashes():
    # на Windows Path сериализуется с '\', фильтр-граф ffmpeg должен получить '/'
    p = Path("C:\\Users\\someone\\Desktop\\clip.ass")
    escaped = _escape_filter_path(p)
    assert "\\Users\\someone" not in escaped
    assert "/Users/someone" in escaped
