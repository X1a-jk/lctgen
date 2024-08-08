import vertexai
import openai

from google.auth import default, transport

# TODO(developer): Update and un-comment below lines
project_id = "peaceful-parity-431406-d6"
location = "asia-east2"

vertexai.init(project=project_id, location=location)

# Programmatically get an access token

credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
auth_request = transport.requests.Request()
credentials.refresh(auth_request)

_api_key = "AIzaSyDELhtRustU9xc7tjVmQHWTnfz3GGJqCWw"
print(f"{credentials.token=}")
# # OpenAI Client
client = openai.OpenAI(
            base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
                api_key = credentials.token,
                )

response = client.chat.completions.create(
            model="google/gemini-1.5-flash-001",
                messages=[{"role": "user", "content": "Why is the sky blue?"}],
                )

print(response)
