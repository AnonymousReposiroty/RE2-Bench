# Problem: sphinx@@sphinx_transforms_post_transforms_images.py@@setup_L294
# Module: sphinx.transforms.post.transforms.images
# Function: setup
# Line: 294

from sphinx.transforms.post.transforms.images import setup


def test_input(pred_input):
    assert "<SphinxTestApp buildername='html'>" == pred_input['args']['app'], 'Prediction failed!'