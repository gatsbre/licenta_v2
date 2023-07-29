def precision_score(test_set, predictions, k=10):
    set_of_users = {user[0] for user in test_set}
    total_precision = 0

    for user in set_of_users:
        list_of_predictions = [prediction for prediction in predictions if prediction.uid == user]
        sorted_predictions = sorted(list_of_predictions, key=lambda x: x.est, reverse=True)
        top_k_predictions = sorted_predictions[:k]

        relevant_items_count = sum(1 for prediction in top_k_predictions if abs(prediction.r_ui - prediction.est) >= 0.5)
        precision = relevant_items_count / k
        total_precision += precision

    return total_precision / len(set_of_users)

def recall_score(test_set, predictions, k=10):
    set_of_users = {user[0] for user in test_set}
    total_recall = 0

    for user in set_of_users:
        list_of_predictions = [prediction for prediction in predictions if prediction.uid == user]
        sorted_predictions = sorted(list_of_predictions, key=lambda x: x.est, reverse=True)
        top_k_predictions = sorted_predictions[:k]

        relevant_items_count = sum(1 for prediction in top_k_predictions if abs(prediction.r_ui - prediction.est) >= 0.5)
        total_relevant_items = sum(1 for prediction in list_of_predictions if abs(prediction.r_ui - prediction.est) >= 0.5)

        if total_relevant_items > 0:
            recall = relevant_items_count / total_relevant_items
            total_recall += recall

    return total_recall / len(set_of_users)

def f1_score(test_set, predictions, k=10):
    precision = precision_score(test_set, predictions, k)
    recall = recall_score(test_set, predictions, k)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

def precision_recall_f1(test_set, predictions, k=10):
    precision = precision_score(test_set, predictions, k)
    recall = recall_score(test_set, predictions, k)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1