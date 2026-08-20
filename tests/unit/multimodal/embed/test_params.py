from tigerflow_ml.multimodal.embed._base import _EmbedBase


def test_embed_defaults():
    p = _EmbedBase.Params()
    assert p.normalize is False
    assert p.truncate_dim is None
    assert p.encode_kwargs == "{}"
    assert p.use_encode_document is False
