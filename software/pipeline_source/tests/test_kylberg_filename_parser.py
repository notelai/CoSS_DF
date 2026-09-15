from cossdf.datasets.download import _parse_kylberg_name, KYLB_CLASSES


def test_kylberg_filename_parser_unrotated():
    assert _parse_kylberg_name("blanket1-a-p001.png") == ("blanket1", "a", 1, None)


def test_kylberg_filename_parser_rotated():
    assert _parse_kylberg_name("blanket1-d-p011-r180.png") == ("blanket1", "d", 11, 180)


def test_kylberg_class_contract():
    assert len(KYLB_CLASSES) == 28
    assert len(set(KYLB_CLASSES)) == 28


def test_kylberg_dot_separator_real_archive_anomaly():
    assert _parse_kylberg_name("rice1.b-p001.png") == ("rice1", "b", 1, None)
    assert _parse_kylberg_name("rice1.b-p040.png") == ("rice1", "b", 40, None)


def test_kylberg_dot_separator_rotated_tolerated():
    assert _parse_kylberg_name("rice1.b-p011-r180.png") == ("rice1", "b", 11, 180)


def test_kylberg_invalid_filename_rejected():
    assert _parse_kylberg_name("not_a_kylberg_filename.png") is None
