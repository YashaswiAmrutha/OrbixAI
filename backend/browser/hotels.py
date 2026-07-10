#hotels.py

from browser.flights import _js, _run_browser_code


async def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    num_adults: int = 1,
):
    """Return hotel names, nightly prices, and ratings for the requested stay."""
    query = (
        f"hotels in {city} from {check_in} to {check_out} "
        f"for {num_adults} adult{'s' if num_adults != 1 else ''}"
    )
    code = f"""
async (page) => {{
  const url = "https://www.google.com/travel/search?q=" +
    encodeURIComponent({_js(query)});
  await page.goto(url, {{ waitUntil: "domcontentloaded", timeout: 60000 }});
  await page.waitForFunction(() => {{
    const text = document.body?.innerText || "";
    return /(?:₹|INR|Rs\\.?|\\$|€|£)\\s*[\\d,]+/.test(text)
      || /No results|Try changing|did not match/i.test(text);
  }}, {{ timeout: 25000 }}).catch(() => null);
  await page.waitForTimeout(1000);

  const lines = (await page.locator("body").innerText())
    .split("\\n").map(value => value.trim()).filter(Boolean);
  const money = /^(?:₹|INR|Rs\\.?|\\$|€|£)\\s*[\\d,]+$/i;
  const labels = /^(?:great deal|great price|deal|sponsored)$/i;
  const hotels = [];
  for (let i = 1; i < lines.length && hotels.length < 10; i++) {{
    if (!money.test(lines[i])) continue;
    let nameIndex = i - 1;
    while (nameIndex >= 0 && labels.test(lines[nameIndex])) nameIndex--;
    const name = lines[nameIndex] || "";
    if (!name || /price|under ₹|what you'll pay/i.test(name)) continue;
    const price = lines[i];
    const rating = /^\\d(?:\\.\\d)?$/.test(lines[i + 1] || "") ? lines[i + 1] : null;
    hotels.push({{
      name: name.slice(0, 120),
      price: price.replace(/[^\\d.,]/g, ""),
      currency: /₹|INR|Rs/i.test(price) ? "INR"
        : /€|EUR/i.test(price) ? "EUR"
        : /£|GBP/i.test(price) ? "GBP" : "USD",
      rating,
      room_type: null,
      check_in: {_js(check_in)},
      check_out: {_js(check_out)}
    }});
  }}
  return hotels;
}}
"""
    result = await _run_browser_code(code)
    return result if isinstance(result, list) else []
