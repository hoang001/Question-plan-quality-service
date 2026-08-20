from src import api


def test_repair_endpoint_is_visible_in_openapi_with_two_loop_limit():
    operation = api.app.openapi()["paths"]["/repair-generated-questions"]["post"]
    parameters = {item["name"]: item for item in operation["parameters"]}

    assert operation["summary"] == "Đánh giá và Repair generated questions"
    assert set(parameters) == {"strict_mode", "debug", "max_loop", "workers", "X-API-Key"}
    assert parameters["max_loop"]["schema"]["default"] == 2
    assert parameters["max_loop"]["schema"]["maximum"] == 2


def test_repair_endpoint_always_enables_auto_repair(monkeypatch):
    captured = {}

    def evaluate(payload, **kwargs):
        captured.update(payload=payload, **kwargs)
        return {"is_good": True, "results": []}

    monkeypatch.setattr(api, "evaluate_generated_questions", evaluate)
    payload = {"_id": "demo"}

    result = api.repair_generated_questions_api(
        payload,
        strict_mode=True,
        debug=True,
        max_loop=2,
        workers=1,
        _=None,
    )

    assert result == {"is_good": True, "results": []}
    assert captured == {
        "payload": payload,
        "strict_mode": True,
        "debug": True,
        "auto_repair": True,
        "max_loop": 2,
        "workers": 1,
    }
