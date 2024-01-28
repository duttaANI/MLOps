# MLOps
repository for AWS App Runner tutorial for MLOPs

## curl command to run AWS console {as taken from postman}

curl --location --request POST 'http://192.168.0.109:8080/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
    "Pclass": "3",
    "Fare": "25",
    "topic_id": "1",
    "Parch": "1",
    "SibSp": "1",
    "retrain": "1"
}
