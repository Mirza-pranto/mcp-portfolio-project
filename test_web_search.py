import asyncio
from fastmcp import Client


async def main() -> None:
    # Use the simple filename transport like other local scripts
    client = Client("server.py")

    async with client:
        result = await client.call_tool(
            "web_search",
            {"query": "python fastmcp duckduckgo", "max_results": 3},
        )
        print("WEB_SEARCH_RESULT:\n", result.data)


if __name__ == "__main__":
    asyncio.run(main())
