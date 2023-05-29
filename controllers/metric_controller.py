def metric_fn(items, config):
    metrics = [item["metrics"] for item in items]
    correct_predictions = sum([metric["correct_predictions"] for metric in metrics])
    total_ed = sum([metric["ed"] for metric in metrics])
    total = sum([metric["total"] for metric in metrics])
    accuracy = correct_predictions/total if total > 0 else 0
    ed = total_ed/total if total > 0 else 0
    loss = sum([metric["loss"] for metric in metrics])/len(metrics) if len(metrics) > 0 else 0

    composed_metric = {"loss": loss,
                       "ed": -ed,
                       "accuracy": accuracy*100}

    return composed_metric


def compose_dev_metric(metrics, config):
    total_metric = 0
    n = len(metrics)
    for key in metrics:
        total_metric += metrics[key][config["save_by"]]
    return config["metric_direction"] * total_metric / n
