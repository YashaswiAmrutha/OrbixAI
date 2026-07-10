#flights.py
import json
import re
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _js(value: Any) -> str:
    """Serialize a Python value as a safe JavaScript literal."""
    return json.dumps(value, ensure_ascii=False)


def _parse_tool_result(result) -> Any:
    if getattr(result, "isError", False):
        message = result.content[0].text if result.content else "Playwright MCP failed"
        raise RuntimeError(message)

    text = result.content[0].text if result.content else ""
    # browser_run_code wraps returned values in a Markdown result section.
    match = re.search(r"### Result\s*([\s\S]*?)(?:\s*###|\Z)", text)
    payload = match.group(1).strip() if match else text.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse Playwright MCP result: {text[:500]}") from exc


async def _run_browser_code(code: str) -> Any:
    server = StdioServerParameters(
        command="npx",
        args=["-y", "@playwright/mcp", "--browser", "chrome", "--isolated"],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            tool_name = (
                "browser_run_code"
                if "browser_run_code" in names
                else "browser_run_code_unsafe"
                if "browser_run_code_unsafe" in names
                else None
            )
            if not tool_name:
                raise RuntimeError(
                    "The installed Playwright MCP server has no browser_run_code tool"
                )
            result = await session.call_tool(tool_name, arguments={"code": code})
            return _parse_tool_result(result)


async def search_flights(
    from_city: str,
    to_city: str,
    departure_date: str,
    return_date: str | None = None,
):
    """Search Google Flights and return normalized flight cards."""
    query = (
        f"Round trip flights from {from_city} to {to_city} from {departure_date} to {return_date}"
        if return_date
        else f"One way flights from {from_city} to {to_city} on {departure_date}"
    )
    code = f"""
async (page) => {{
  const url = "https://www.google.com/travel/flights?q=" +
    encodeURIComponent({_js(query)});
  await page.goto(url, {{
    waitUntil: "domcontentloaded", timeout: 45000
  }});
  await page.waitForFunction(() => {{
    const text = document.body?.innerText || "";
    return /Top(?: departing)? flights|Other(?: departing)? flights/i.test(text)
      && /(?:₹|INR|Rs\\.?|\\$|€|£)\\s*[\\d,]+/.test(text);
  }}, {{ timeout: 15000 }}).catch(() => null);
  await page.waitForTimeout(750);

  const lines = (await page.locator("body").innerText())
    .split("\\n").map(value => value.trim()).filter(Boolean);
  const time = /\\d{{1,2}}:\\d{{2}}[\\s\\u202f]*(?:AM|PM)(?:\\+\\d)?/ig;
  const timeOnly = /^\\d{{1,2}}:\\d{{2}}[\\s\\u202f]*(?:AM|PM)(?:\\+\\d)?$/i;
  const dashOnly = /^[–—-]$/;
  const timeRange = /^\\s*(\\d{{1,2}}:\\d{{2}}[\\s\\u202f]*(?:AM|PM)(?:\\+\\d)?)\\s*[–—-]\\s*(\\d{{1,2}}:\\d{{2}}[\\s\\u202f]*(?:AM|PM)(?:\\+\\d)?)\\s*$/i;
  const money = /^(?:₹|INR|Rs\\.?|\\$|€|£)\\s*[\\d,]+$/i;
  const cleanTime = value => value.replace(/\\u202f/g, " ").replace(/\\s+/g, " ").trim();
  const flights = [];
  for (let i = 0; i < lines.length - 4 && flights.length < 10; i++) {{
    let departure = "";
    let arrival = "";
    let airline = "";
    let duration = "";
    let cursor = i;

    const range = lines[i].match(timeRange);
    if (range) {{
      departure = cleanTime(range[1]);
      arrival = cleanTime(range[2]);
      airline = lines[i + 1] || "";
      duration = lines[i + 2] || "";
      cursor = i + 3;
    }} else if (timeOnly.test(lines[i]) && dashOnly.test(lines[i + 1] || "") && timeOnly.test(lines[i + 2] || "")) {{
      departure = cleanTime(lines[i]);
      arrival = cleanTime(lines[i + 2]);
      airline = lines[i + 3] || "";
      duration = lines[i + 4] || "";
      cursor = i + 5;
    }} else {{
      const times = [...lines[i].matchAll(time)].map(match => match[0]);
      if (times.length >= 2) {{
        departure = cleanTime(times[0]);
        arrival = cleanTime(times[1]);
        airline = lines[i + 1] || "";
        duration = lines[i + 2] || "";
        cursor = i + 3;
      }} else {{
        continue;
      }}
    }}

    if (!/\\d+\\s*hr/i.test(duration)) continue;
    if (/^(best|cheapest|top flights|top departing flights|travel restricted)$/i.test(airline)) continue;

    let price = "";
    let stops = "";
    for (let j = cursor; j < Math.min(cursor + 12, lines.length); j++) {{
      if (/nonstop|\\d+ stop/i.test(lines[j])) stops = lines[j];
      if (money.test(lines[j])) {{ price = lines[j]; break; }}
    }}
    if (!price) continue;
    flights.push({{
      airline,
      price: price.replace(/[^\\d.,]/g, ""),
      currency: /₹|INR|Rs/i.test(price) ? "INR"
        : /€|EUR/i.test(price) ? "EUR"
        : /£|GBP/i.test(price) ? "GBP" : "USD",
      departure,
      arrival,
      duration,
      stops
    }});
  }}
  flights.sort((a, b) =>
    Number(String(a.price).replace(/,/g, "")) -
    Number(String(b.price).replace(/,/g, ""))
  );
  return flights;
}}
"""
    result = await _run_browser_code(code)
    return result if isinstance(result, list) else []
