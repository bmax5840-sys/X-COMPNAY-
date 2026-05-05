import pytest

from cortexlab.inference._compat import dataframe
from cortexlab.inference.predictor import TribeModel


def test_from_pretrained_resolves_auto_device_to_runtime_string():
    model = TribeModel.from_pretrained("facebook/tribev2", device="auto")

    assert model.model_id == "facebook/tribev2"
    assert model.device in {"cpu", "cuda", "mps"}


def test_predict_returns_predictions_and_segments():
    model = TribeModel.from_pretrained("facebook/tribev2", device="cpu")
    events = dataframe({"timestamp": [0, 1, 2, 5, 6, 10]})

    preds, segments = model.predict(events)

    assert list(preds["prediction"]) == [
        "gathering",
        "gathering",
        "gathering",
        "transition",
        "transition",
        "dispersal",
    ]
    assert list(segments["label"]) == ["gathering", "transition", "dispersal"]
    assert list(segments["start"]) == [0.0, 5.0, 10.0]


def test_predict_accepts_iterable_records():
    model = TribeModel.from_pretrained("facebook/tribev2", device="cpu")

    preds, segments = model.predict([{"timestamp": 2}, {"timestamp": 3}])

    assert len(preds) == 2
    assert len(segments) == 1


def test_predict_requires_timestamp_column():
    model = TribeModel.from_pretrained("facebook/tribev2", device="cpu")

    with pytest.raises(ValueError, match="timestamp"):
        model.predict(dataframe({"frame": [1]}))
