from wmcp_jepa_service.schemas import RequestEnvelope


def test_minimal_score_request_contract() -> None:
    request = RequestEnvelope.model_validate(
        {
            "wmcp_version": "0.1",
            "request_id": "test-1",
            "operation": "score",
            "model": "lewm-pusht",
            "inputs": {
                "action_candidates": {
                    "space": "continuous",
                    "tensor": {
                        "kind": "tensor",
                        "encoding": "uri",
                        "dtype": "float32",
                        "shape": [1, 4, 4, 10],
                        "layout": "B,S,T,A",
                        "uri": "memory://actions.npy",
                    },
                }
            },
        }
    )
    assert request.operation == "score"
    assert request.model == "lewm-pusht"
