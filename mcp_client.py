import asyncio
from fastmcp import Client


async def main() -> None:
    client = Client("server.py")

    async with client:
        result = await client.call_tool(
            "create_ticket",
            {"description": "Test ticket from async MCP client"},
        )
        print(result.data)


if __name__ == "__main__":
    asyncio.run(main())
