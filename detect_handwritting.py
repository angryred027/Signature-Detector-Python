from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="FmSF5c8hz4IG0uyDZiUd"
)

# infer on a local image
result = CLIENT.infer("test.jpeg", model_id="boston_globe_copyright_ocr-wa7ou/1")

print(result)