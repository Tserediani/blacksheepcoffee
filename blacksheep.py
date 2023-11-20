import json
import os
import re
import ssl
import time
import requests
from pydantic import BaseModel
import pandas as pd
from datetime import date
import httpx
import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CURRENT_DATE = date.today().strftime("%Y-%m-%d")
POOL = 3
API_KEY = (
    "MJE03CF7ZZEGTIPRDZGIHD5CQ496X7HJ9TKOXFJWYQBSUL1KHOYMH17JM05JFVTW5RHJ71VHSN7CHTF1"
)
SCRAPINGBEE_URL = "https://app.scrapingbee.com/api/v1/"
VENUE_LIMIT = 150
DATA_DIRECTORY = "products"


VENUES_URL = "https://vmos2.vmos.io/tenant/v1/stores/tenant?offset=0&postcode=&limit={limit}&sortBy%5B%5D=name&sortBy%5B%5D=sortOrder&sortDir=ASC&status=1&weekday=5"
CATEGORIES_API_URL = "https://vmos2.vmos.io/catalog/v2/menu"
MENU_BUNDLES_URL = "https://vmos2.vmos.io/catalog/categories/{category_uuid}/bundles?forceStockStatus=0"
MENU_ITEM_URL = "https://vmos2.vmos.io/catalog/bundles/{menu_item_uuid}/item-types?forceStockStatus=0"
MENU_UPSELL_URL = "https://vmos2.vmos.io/catalog/bundles/{menu_item_uuid}/upsell-categories?forceStockStatus=0"


HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-GB,en;q=0.9",
    "Cache-Control": "no-store, max-age=0",
    "Connection": "keep-alive",
    "Host": "vmos2.vmos.io",
    "If-None-Match": 'W/"f158-daALAmDOcQtvypcavlPELhPGdB8"',
    "Origin": "https://blacksheepcoffee.vmos.io",
    "Pragma": "no-cache",
    "Referer": "https://blacksheepcoffee.vmos.io/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "tenant": "22e60728-df22-4876-a1fe-a8c03e930694",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "x-requested-from": "online",
}

os.makedirs(DATA_DIRECTORY, exist_ok=True)


class Venue(BaseModel):
    name: str
    address: str
    postcode: str
    uuid: str
    lat: str
    long: str


class MenuCategories(BaseModel):
    menu_name: str
    menu_uuid: str
    category_name: str
    category_uuid: str


class MenuBundle(BaseModel):
    item_name: str
    item_uuid: str
    item_desc: str
    item_cat: str
    item_subcat: str = ""


class MenuOptions(BaseModel):
    option_type: str
    option_name: str
    option_price: float
    size: str


class UpsellCategories(BaseModel):
    category_name: str
    item_name: str
    item_desc: str
    item_price: float


class MenuModel(BaseModel):
    ddate: str = CURRENT_DATE
    operator: str = "black sheep coffee"
    brand: str = "black sheep coffee"
    siteid: str
    menu: str = "Menu"
    cat: str = ""
    subcat: str = ""
    code: str = ""
    item_name: str
    port: str = ""
    stdserve: str = ""
    price: float
    incdrink: str = ""
    stock: str = ""
    desc: str = ""


def clean(string: str) -> str:
    if not isinstance(string, str):
        return string

    # Remove leading and trailing whitespaces
    string = string.strip()

    # Remove line breaks, tabs, and multiple spaces
    string = re.sub(r"[\r\n\t]+", " ", string)
    string = re.sub(r"\s+", " ", string)

    # Remove special characters
    string = re.sub(r"[|\"\\]", "", string)

    # Remove non-ASCII characters
    string = string.encode("ascii", "ignore").decode("ascii")

    # remove <p> tag
    string = string.removeprefix("<p>")
    string = string.removesuffix("</p>")

    return string


def filter_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Simple function that fiters dataframe."""

    # Dropping duplicate values
    dataframe.drop_duplicates(inplace=True)

    # Dropping where price is zero
    dataframe = dataframe[dataframe["_price"] != 0]

    # Drop stdserve values
    dataframe = dataframe[
        ~dataframe["_stdserve"].str.contains("Fancy a Waffle?".lower(), case=False)
    ]
    dataframe = dataframe[
        ~dataframe["_stdserve"].str.contains("Add a Snack".lower(), case=False)
    ]

    ## ADD YOUR RULES

    return dataframe


def write(data: pd.DataFrame, filename: str) -> None:
    try:
        data.to_csv(
            filename,
            header=not os.path.exists(filename),
            sep="|",
            encoding="UTF-8",
            index=False,
            mode="a",
        )
    except Exception as e:
        print(e)
        return


async def get_response(
    url: str,
    headers: dict[str, str] = None,
    max_retries: int = 3,
) -> httpx.Response:
    if headers is None:
        headers = {}
    spb_headers = {f"Spb-{key}": value for key, value in headers.items()}

    async def retry_request(url, retries):
        async with httpx.AsyncClient() as client:
            for _ in range(retries):
                try:
                    response = await client.get(
                        url=SCRAPINGBEE_URL,
                        params={
                            "api_key": API_KEY,
                            "url": url,
                            "premium_proxy": "false",
                            "render_js": "false",
                            "forward_headers": "true",
                        },
                        headers=spb_headers,
                    )
                    response.raise_for_status()
                    if response.status_code == 200:
                        return response
                except (
                    ssl.SSLError,
                    httpx.HTTPError,
                    httpx.RequestError,
                ) as error:
                    print(f"Invalid Request. URL: {url}. Error Details: {repr(error)}")
                    print("Retrying...")
                time.sleep(2**_)
            return None

    return await retry_request(url, max_retries)


async def parse_json_response(url: str, headers: dict, max_retries: int = 3) -> dict:
    response = await get_response(url, headers, max_retries)
    try:
        return response.json()
    except (requests.JSONDecodeError, AttributeError):
        return {}


def write_json(data: dict, filename: str) -> None:
    with open(filename, "w") as file:
        json.dump(data, file)


def read_json(filename: str) -> dict:
    with open(filename, "r") as file:
        json_data = json.load(file)
    return json_data


async def get_venues(limit: int) -> dict:
    url = VENUES_URL.format(limit=limit)
    return await parse_json_response(url, headers=HEADERS)


async def parse_address(address_postcode: str) -> tuple:
    address_postcode = address_postcode.replace(",", "").split(" ")
    postcode = " ".join(address_postcode[-2:])
    address = ", ".join(address_postcode[:-2])

    return address, postcode


async def parse_venues(venues_json: dict) -> Venue:
    venues: list[Venue] = []
    if not venues_json:
        return venues
    for venue in venues_json.get("payload", [{}]):
        name = venue.get("name", "")
        address, postcode = await parse_address(venue.get("address", ""))
        uuid = venue.get("uuid", "")
        lat = venue.get("lat", "")
        long = venue.get("long", "")
        venues.append(
            Venue(
                name=name,
                address=address,
                postcode=postcode,
                uuid=uuid,
                lat=lat,
                long=long,
            )
        )
    return venues


async def process_venues(limit: int) -> list[Venue]:
    venues_json = await get_venues(limit=limit)

    venues: list[Venue] = await parse_venues(venues_json=venues_json)
    venues_schema = list(Venue.__annotations__.keys())
    venues_data = [list(venue.dict().values()) for venue in venues]

    dataframe = pd.DataFrame(venues_data, columns=venues_schema)
    write(dataframe, filename=f"venues_{CURRENT_DATE}.txt")

    return venues


async def get_categories(url: str, venue_uuid: str) -> dict:
    headers = {
        **HEADERS,
        **{
            "store": venue_uuid,
            "menu": "21a7a281-1694-49e3-ab31-717707c8b774",
        },
    }
    return await parse_json_response(url, headers=headers)


async def parse_menu_categories(menu_json: dict):
    categories: list[MenuCategories] = []
    if not menu_json:
        return categories
    for menu in menu_json.get("payload", [{}]):
        menu_name = menu.get("name", "")
        menu_uuid = menu.get("uuid", "")
        for category in menu.get("categories", [{}]):
            category_name = category.get("name", "")
            category_uuid = category.get("uuid", "")
            categories.append(
                MenuCategories(
                    menu_name=menu_name,
                    menu_uuid=menu_uuid,
                    category_name=category_name,
                    category_uuid=category_uuid,
                )
            )
    return categories


async def process_menu_bundles(
    venue_uuid: str, categories: list[MenuCategories]
) -> list[MenuBundle]:
    bundles_dir = os.path.join("products", "bundles")
    os.makedirs(bundles_dir, exist_ok=True)

    headers = {
        **HEADERS,
        **{
            "store": venue_uuid,
            "menu": "21a7a281-1694-49e3-ab31-717707c8b774",
        },
    }

    menu_bundles: list[MenuBundle] = []
    for category in categories:
        url = MENU_BUNDLES_URL.format(category_uuid=category.category_uuid)
        response_json = await parse_json_response(url, headers=headers)
        write_json(
            response_json,
            os.path.join(bundles_dir, f"{venue_uuid}_{category.category_name}.json"),
        )
        menu_bundles.extend(
            await parse_menu_bundles(response_json, category.category_name)
        )
    return menu_bundles


async def parse_menu_bundles(
    menu_bundles_json: dict, category: str
) -> list[MenuBundle]:
    menu_bundles: list[MenuBundle] = []
    if not menu_bundles_json:
        return menu_bundles
    payload = menu_bundles_json.get("payload")
    if not payload:
        return
    bundles = payload.get("bundles", [])
    if bundles:
        for menu_item in bundles:
            item_name = menu_item.get("name", "")
            item_desc = clean(menu_item.get("description", ""))
            item_uuid = menu_item.get("uuid", "")
            menu_bundles.append(
                MenuBundle(
                    item_name=item_name,
                    item_uuid=item_uuid,
                    item_desc=item_desc,
                    item_cat=category,
                )
            )
    else:
        for subcat in payload.get("categories", []):
            item_subcat = subcat.get("name", "")
            for menu_item in subcat.get("bundles", []):
                item_name = menu_item.get("name", "")
                item_desc = clean(menu_item.get("description", ""))
                item_uuid = menu_item.get("uuid", "")
                menu_bundles.append(
                    MenuBundle(
                        item_name=item_name,
                        item_uuid=item_uuid,
                        item_desc=item_desc,
                        item_cat=category,
                        item_subcat=item_subcat,
                    )
                )
    return menu_bundles


async def get_upsell_categories(venue_uuid: str, menu_item: MenuBundle) -> dict:
    items_dir = os.path.join("products", "upsell_categories")
    os.makedirs(items_dir, exist_ok=True)
    headers = {
        **HEADERS,
        **{"store": venue_uuid, "menu": "21a7a281-1694-49e3-ab31-717707c8b774"},
    }
    url = MENU_UPSELL_URL.format(menu_item_uuid=menu_item.item_uuid)
    response_json = await parse_json_response(url=url, headers=headers)
    write_json(
        response_json,
        os.path.join(items_dir, f"{venue_uuid}_{menu_item.item_name}.json"),
    )
    return response_json


async def parse_upsell_categories(upsell_json: dict) -> list[UpsellCategories]:
    upsell_categories: list[UpsellCategories] = []
    if not upsell_json:
        return upsell_categories
    payload = upsell_json.get("payload")
    if not payload:
        return upsell_categories
    for payload in upsell_json.get("payload", []):
        category_name = payload.get("name")
        for bundle in payload.get("bundles", []):
            item_name = bundle.get("name", "")
            item_desc = clean(bundle.get("description", ""))
            for option in bundle.get("items", []):
                for customization in option.get("customizations", []):
                    for variation in customization.get("variations", []):
                        price = variation.get("price", "")
                        upsell_categories.append(
                            UpsellCategories(
                                category_name=category_name,
                                item_name=item_name,
                                item_desc=item_desc,
                                item_price=price,
                            )
                        )

    return upsell_categories


async def get_menu_item(url: str, venue_uuid: str, menu_item: MenuBundle) -> dict:
    items_dir = os.path.join("products", "items")
    os.makedirs(items_dir, exist_ok=True)
    headers = {
        **HEADERS,
        **{"store": venue_uuid, "menu": "21a7a281-1694-49e3-ab31-717707c8b774"},
    }
    url = MENU_ITEM_URL.format(menu_item_uuid=menu_item.item_uuid)
    response_json = await parse_json_response(url, headers)
    write_json(
        response_json,
        os.path.join(items_dir, f"{venue_uuid}_{menu_item.item_name}.json"),
    )
    return response_json


async def parse_menu_options(menu_item_json: dict) -> list[MenuOptions]:
    options: list[MenuOptions] = []
    if not menu_item_json:
        return options
    for item_type in menu_item_json.get("payload"):
        for item in item_type.get("items", []):
            customizations = item.get("customizations")
            if not customizations:
                continue
            for customization in customizations:
                for variation in customization.get("variations", []):
                    if not variation.get("unitsNumber"):
                        options.append(
                            MenuOptions(
                                option_type=item_type.get("name", ""),
                                option_name=item.get("name", ""),
                                option_price=variation.get("price", ""),
                                size=variation.get("name", ""),
                            )
                        )
                    else:
                        if variation.get("unitsNumber", 0) == 1:
                            options.append(
                                MenuOptions(
                                    option_type=item_type.get("name", ""),
                                    option_name=item.get("name", ""),
                                    option_price=variation.get("price", ""),
                                    size=variation.get("name", ""),
                                )
                            )
    return options


async def process_menu(venue: Venue) -> list[MenuModel]:
    categories_json: dict = await get_categories(CATEGORIES_API_URL, venue.uuid)
    categories: list[MenuCategories] = await parse_menu_categories(categories_json)
    menu_bundles: list[MenuBundle] = await process_menu_bundles(venue.uuid, categories)

    async def process_menu_bundle(menu_bundle: MenuBundle):
        print(
            f"Current: {venue.name} > {menu_bundle.item_cat} > {menu_bundle.item_subcat} > {menu_bundle.item_name}"
        )

        upsell_categories_json = await get_upsell_categories(venue.uuid, menu_bundle)
        upsell_categories = await parse_upsell_categories(upsell_categories_json)
        menu_item_json = await get_menu_item(MENU_ITEM_URL, venue.uuid, menu_bundle)
        menu_item = await parse_menu_options(menu_item_json)

        menu_items: list[MenuModel] = []

        for item in menu_item:
            menu_items.append(
                MenuModel(
                    siteid=f"black sheep coffee.{venue.postcode}",
                    cat=menu_bundle.item_cat,
                    subcat=menu_bundle.item_subcat,
                    item_name=menu_bundle.item_name,
                    stdserve=f"{item.option_type} - {item.option_name}"
                    if item.option_name.lower() != menu_bundle.item_name.lower()
                    else f"Size - {item.size}",
                    price=item.option_price,
                    desc=menu_bundle.item_desc,
                )
            )

        for item in upsell_categories:
            menu_items.append(
                MenuModel(
                    siteid=f"black sheep coffee.{venue.postcode}",
                    cat=menu_bundle.item_cat,
                    subcat=menu_bundle.item_subcat,
                    item_name=menu_bundle.item_name,
                    stdserve=f"{item.category_name} - {item.item_name}",
                    price=item.item_price,
                    desc=item.item_desc,
                )
            )

        return menu_items

    semaphore = asyncio.Semaphore(POOL)

    async def process_with_semaphore(menu_bundle):
        async with semaphore:
            return await process_menu_bundle(menu_bundle)

    tasks = [process_with_semaphore(menu_bundle) for menu_bundle in menu_bundles]
    results = await asyncio.gather(*tasks)
    return [item for result in results for item in result]


async def main():
    venues: list[Venue] = await process_venues(limit=VENUE_LIMIT)

    menu_items: list[MenuModel] = []
    for venue in venues:
        menu_items.extend(await process_menu(venue))

    menu_items_schema = [
        "_date",
        "_operator",
        "_brand",
        "_siteid",
        "_menu",
        "_cat",
        "_subcat",
        "_code",
        "_name",
        "_port",
        "_stdserve",
        "_price",
        "_incdrink",
        "_stock",
        "_desc",
    ]

    # items = [list(item.dict().values()) for item in menu_items]

    items = [list(item.dict(exclude_unset=True).values()) for item in menu_items]
    dataframe = filter_dataframe(pd.DataFrame(items, columns=menu_items_schema))
    write(dataframe, f"black_sheep_coffee_{CURRENT_DATE}.csv")


if __name__ == "__main__":
    asyncio.run(main())
