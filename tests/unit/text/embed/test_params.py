from tigerflow_ml.text.embed._base import _EmbedBase


def test_embed_defaults():
    p = _EmbedBase.Params()
    assert p.per_line is False
    assert p.batch_size == 32
    assert p.prompt is None
    assert p.prompt_name is None
    assert p.normalize is False
    assert p.truncate_dim is None
