"""Integration test for Embed with:
    --use-encode-document
    --normalize
    --truncate-dim

Note: --use-encode-query is not tested. Presumably, if --use-encode-document works,
it's counterpart also works
"""

import numpy as np
import pytest

from tigerflow_ml.text.embed._base import _EmbedBase

from .conftest import assert_or_update_array_snapshot


@pytest.fixture(scope="session")
def embed_dir(test_dir):
    return test_dir / "embed"


@pytest.fixture(scope="module")
def context(make_context):
    import gc

    import torch

    ctx = make_context(
        _EmbedBase.Params,
        "embed",
        use_encode_document=True,
        normalize=True,
        truncate_dim=200,
    )
    _EmbedBase.setup(ctx)
    yield ctx
    del ctx.embedder
    gc.collect()
    torch.cuda.empty_cache()


def test_setup(context):
    assert context.embedder is not None


def test_run_params(
    context,
    embed_dir,
    get_input_files,
    make_output_path,
    snapshot_dir,
    update_snapshots,
):
    for input_file in get_input_files(embed_dir):
        output_file = make_output_path(input_file, ".npy")
        _EmbedBase.run(context, input_file, output_file)

        embedding = np.load(output_file)
        assert_or_update_array_snapshot(
            embedding,
            f"embed/{input_file.stem}.params.npy",
            snapshot_dir,
            update_snapshots,
            threshold=0.99,
        )
