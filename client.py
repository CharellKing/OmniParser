import httpx
import json
import asyncio

BASE_URL = "http://127.0.0.1:8000"
SSE_ENDPOINT = f"{BASE_URL}/mcp"

async def main():
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("GET", SSE_ENDPOINT, headers={"Accept": "text/event-stream"}) as response:
                print(f"Connected to SSE endpoint with status code: {response.status_code}")
                response.raise_for_status()

                # After connection, the server should send an initialization message.
                # Then we can send our requests.

                # This is a simplified example. A real client would have a more robust
                # way to handle the bidirectional communication.

                # Let's try to listen for the initial messages from the server
                async for line in response.aiter_lines():
                    print(f"Server event: {line}")
                    # A proper client would parse these events.
                    # For this demo, we'll just print them.
                    # We are looking for a sign that the connection is established
                    # before we can send our own messages.

        except httpx.RequestError as e:
            print(f"Request error: {e}")
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e}")


if __name__ == "__main__":
    asyncio.run(main())