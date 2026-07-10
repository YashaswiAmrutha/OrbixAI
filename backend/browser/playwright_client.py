#playwright_client.py:
import asyncio
import traceback

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


async def main():
    try:
        server = StdioServerParameters(
            command="npx",
            args=["@playwright/mcp"]
        )

        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:

                await session.initialize()

                print("✅ Connected to Playwright MCP")

                tools = await session.list_tools()

                print("\nAvailable Tools:\n")

                for tool in tools.tools:
                    print(tool.name)

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

