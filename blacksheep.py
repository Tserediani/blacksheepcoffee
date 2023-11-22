from enum import Enum
import json
import logging
import os
import random
import re
import ssl
import time
from typing import Callable, Optional
import requests
from pydantic import BaseModel
import pandas as pd
from datetime import date
import httpx
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CURRENT_DATE = date.today().strftime("%Y-%m-%d")

POOL = 3
VENUE_LIMIT = 150
DATA_DIRECTORY = "products"

VENUES_URL = "https://vmos2.vmos.io/tenant/v1/stores/tenant?offset=0&postcode=&limit={limit}&sortBy%5B%5D=name&sortBy%5B%5D=sortOrder&sortDir=ASC&status=1&weekday=5"
CATEGORIES_API_URL = "https://vmos2.vmos.io/catalog/v2/menu"
MENU_BUNDLES_URL = "https://vmos2.vmos.io/catalog/categories/{category_uuid}/bundles?forceStockStatus=0"
MENU_ITEM_URL = "https://vmos2.vmos.io/catalog/bundles/{menu_item_uuid}/item-types?forceStockStatus=0"
MENU_UPSELL_URL = "https://vmos2.vmos.io/catalog/bundles/{menu_item_uuid}/upsell-categories?forceStockStatus=0"


# SCRAPINGBEE CONFIGURATION
SCRAPINGBEE = True

SCRAPINGBEE_API_KEY = (
    "MJE03CF7ZZEGTIPRDZGIHD5CQ496X7HJ9TKOXFJWYQBSUL1KHOYMH17JM05JFVTW5RHJ71VHSN7CHTF1"
)

SCRAPINGBEE_URL = "https://app.scrapingbee.com/api/v1/"

SCRAPINGBEE_HEADERS = {
    "api_key": SCRAPINGBEE_API_KEY,
    "url": "{url}",
    "render_js": "false",
    "forward_headers": "true",
    "premium_proxy": "false",
}
###########################


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

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(WORKING_DIR, DATA_DIRECTORY), exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, "logs"), exist_ok=True)

logger = logging.getLogger("blacksheepcofee_logger")

if not logger.handlers:
    logger.setLevel(logging.DEBUG)

    logging_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Log file handler.
    log_filename = f"./logs/blacksheepcoffee_{CURRENT_DATE}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging_formatter)
    logger.addHandler(console_handler)


class Method(Enum):
    GET = "GET"
    POST = "POST"


class RequestError(Exception):
    ...


class MaxRetriesExceededError(RequestError):
    ...


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
    try:
        # Dropping where price is zero
        dataframe = dataframe[dataframe["_price"] != 0]
        # Drop stdserve values
        dataframe = dataframe[
            ~dataframe["_stdserve"].str.contains("Fancy a Waffle?".lower(), case=False)
        ]
        dataframe = dataframe[
            ~dataframe["_stdserve"].str.contains("Add a Snack".lower(), case=False)
        ]
        # ADD YOUR RULES

    except Exception:
        logger.error("Failed to use filter on dataframe.", exc_info=True)

    return dataframe


def write(
    data: list[BaseModel],
    scheme: list,
    filename: str,
    mode: str = "a",
    custom_filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
) -> None:
    items = [list(result.model_dump().values()) for result in data]
    dataframe = pd.DataFrame(items, columns=scheme)
    dataframe.drop_duplicates(inplace=True)
    if custom_filter:
        dataframe = filter_dataframe(dataframe)
    try:
        dataframe.to_csv(
            filename,
            header=not os.path.exists(filename) if mode == "a" else True,
            sep="|",
            encoding="UTF-8",
            index=False,
            mode=mode,
        )
    except pd.errors.EmptyDataError as e:
        logger.warning(f"Dataframe is empty. Error Details: {e}")
    except Exception as exc:
        logger.critical(f"Failed to write data. Error Details: {exc}", exc_info=True)


async def make_request(
    url: str,
    method: Method,
    headers: dict[str, str],
    data: str,
    params: dict[str, str],
) -> requests.Response:
    response = None
    async with httpx.AsyncClient(timeout=15) as client:
        if method == Method.GET:
            response = await client.get(url=url, headers=headers, params=params)
        elif method == Method.POST:
            response = await client.post(
                url=url, headers=headers, data=data, params=params
            )
        if response and response.status_code == 200:
            return response
        else:
            raise RequestError(f"Invalid response from {url}")


async def retry_request(
    url: str,
    max_retries: int,
    method: Method,
    headers: dict[str, str],
    params: dict[str, str],
    data: str,
    scrapingbee: bool = False,
    request_details: Optional[str] = "",
) -> requests.Response:
    for retry in range(max_retries):
        try:
            if scrapingbee:
                constructed_scrapingbee_api_params = {
                    **SCRAPINGBEE_HEADERS,
                    **{"url": url},
                }
                spb_headers = {f"Spb-{key}": value for key, value in headers.items()}
                response = await make_request(
                    url=SCRAPINGBEE_URL,
                    method=method,
                    headers=spb_headers,
                    params=constructed_scrapingbee_api_params,
                    data=data,
                )
            else:
                response = await make_request(
                    url=url, method=method, headers=headers, params=params, data=data
                )
            return response

        except (ssl.SSLError, httpx.HTTPError, httpx.RequestError) as error:
            logger.warning(
                f"Invalid Request. URL: {url}. Error Details: {repr(error)}. Tries: {retry + 1}/{max_retries}"
            )
        except RequestError as error:
            logger.warning(f"Request Error: {error}. Tries: {retry + 1}/{max_retries}")
        except Exception as e:
            logger.warning(
                f"Exception occurred while requesting {url}. Error Details: {repr(e)}. Tries: {retry + 1}/{max_retries}",
                exc_info=True,
            )

        time.sleep(random.uniform(2**retry - 1, 2 ** (retry + 1) - 1))

    logger.error(
        f"Failed to get a response from URL {url}. Request Details: {request_details}. Error Details: Script exceeded max retries count."
    )
    raise MaxRetriesExceededError(f"Max retries exceeded. {url}")


async def get_response(
    url: str,
    method: Method = Method.GET,
    headers: dict[str, str] = None,
    params: dict[str, str] = None,
    data: str = "",
    max_retries: int = 3,
    scrapingbee: bool = False,
    request_details: Optional[str] = "",
) -> requests.Response:
    if params is None:
        params = {}
    if headers is None:
        headers = {}

    return await retry_request(
        url=url,
        method=method,
        headers=headers,
        params=params,
        data=data,
        max_retries=max_retries,
        scrapingbee=scrapingbee,
        request_details=request_details,
    )


async def parse_json_response(
    url: str, headers: dict, max_retries: int = 3, request_details: Optional[str] = ""
) -> dict:
    try:
        response = await get_response(
            url=url,
            method=Method.GET,
            headers=headers,
            max_retries=max_retries,
            scrapingbee=SCRAPINGBEE,
            request_details=request_details,
        )
    except MaxRetriesExceededError:
        logger.error(f"Failed to scrape {url}. Max retries exceeded.")
        response = None
    try:
        return response.json()
    except (requests.JSONDecodeError, AttributeError):
        return {}


def write_json(data: dict, filename: str) -> None:
    with open(filename, "w") as file:
        json.dump(data, file)


def read_json(filename: str) -> dict:
    try:
        with open(filename, "r") as file:
            json_data = json.load(file)
        return json_data
    except json.JSONDecodeError:
        return {}


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

    write(
        venues,
        scheme=list(Venue.__annotations__),
        filename=f"venues_{CURRENT_DATE}.txt",
    )

    return venues


async def get_categories(url: str, venue: Venue) -> dict:

    response_json = None
    category_filename = f"{venue.name.lower()}_{venue.postcode}_categories.json"
    categories_filepath = os.path.join(DATA_DIRECTORY, category_filename)
    if os.path.exists(categories_filepath):
        logger.info(f'Using existing categories {categories_filepath}')
        response_json = read_json(categories_filepath)
    if not response_json:
        headers = {
        **HEADERS,
        **{
            "store": venue.uuid,
            "menu": "21a7a281-1694-49e3-ab31-717707c8b774",
        },
    }
        response_json = await parse_json_response(
            url,
            headers=headers,
            request_details=f"Getting categories for {venue.name} - {venue.postcode} - {venue.uuid}",
        )
        write_json(
            response_json,
            categories_filepath
        )

    return response_json


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
    venue_name: str, venue_uuid: str, categories: list[MenuCategories]
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
        bundle_filepath = os.path.join(
            bundles_dir, f"{venue_name.lower()}_{category.category_name.lower()}.json"
        )
        logger.info(
            f"Extracting products for {venue_name} - {category.menu_name} - {category.category_name}"
        )
        response_json = None
        if os.path.exists(bundle_filepath):
            logger.info(f"Using local JSON file {bundle_filepath}")
            response_json = read_json(bundle_filepath)
        if not response_json:
            url = MENU_BUNDLES_URL.format(category_uuid=category.category_uuid)
            response_json = await parse_json_response(
                url,
                headers=headers,
                request_details=f"Getting products for category {venue_name} - {category.menu_name} - {category.category_name}",
            )
            write_json(response_json, bundle_filepath)
        menu_bundle = await parse_menu_bundles(response_json, category.category_name)
        logger.info(f"Found total {len(menu_bundle)} products.")
        menu_bundles.extend(menu_bundle)
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


async def get_upsell_categories(
    venue_name: str, venue_uuid: str, menu_item: MenuBundle
) -> dict:
    items_dir = os.path.join("products", "upsell_categories")
    items_filepath = os.path.join(items_dir, f"{venue_name}_{menu_item.item_name}.json")
    os.makedirs(items_dir, exist_ok=True)
    response_json = None
    if os.path.exists(items_filepath):
        logger.info(f"Using local JSON file. {items_filepath}")
        response_json = read_json(items_filepath)

    if not response_json:
        headers = {
            **HEADERS,
            **{"store": venue_uuid, "menu": "21a7a281-1694-49e3-ab31-717707c8b774"},
        }
        url = MENU_UPSELL_URL.format(menu_item_uuid=menu_item.item_uuid)
        response_json = await parse_json_response(
            url=url,
            headers=headers,
            request_details=f"Getting options for {venue_name} - {menu_item.item_name}",
        )
        write_json(
            response_json,
            items_filepath,
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


async def get_menu_item(
    url: str, venue_name: str, venue_uuid: str, menu_item: MenuBundle
) -> dict:
    items_dir = os.path.join("products", "items")
    items_filepath = os.path.join(items_dir, f"{venue_name}_{menu_item.item_name}.json")
    os.makedirs(items_dir, exist_ok=True)
    response_json = None
    if os.path.exists(items_filepath):
        logger.info(f"Using local JSON file {items_filepath}")
        response_json = read_json(items_filepath)

    if not response_json:
        headers = {
            **HEADERS,
            **{"store": venue_uuid, "menu": "21a7a281-1694-49e3-ab31-717707c8b774"},
        }
        url = MENU_ITEM_URL.format(menu_item_uuid=menu_item.item_uuid)
        response_json = await parse_json_response(
            url,
            headers,
            request_details=f"Getting details for {venue_name} - {menu_item.item_name}",
        )
        write_json(
            response_json,
            items_filepath,
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
    categories_json: dict = await get_categories(CATEGORIES_API_URL, venue)
    categories: list[MenuCategories] = await parse_menu_categories(categories_json)
    logger.info(
        f"Total {len(categories)} categories will be extracted for {venue.name}"
    )
    menu_bundles: list[MenuBundle] = await process_menu_bundles(
        venue.name, venue.uuid, categories
    )

    async def process_menu_bundle(menu_bundle: MenuBundle):
        upsell_categories_json = await get_upsell_categories(
            venue_name=venue.name, venue_uuid=venue.uuid, menu_item=menu_bundle
        )
        upsell_categories = await parse_upsell_categories(upsell_categories_json)
        menu_item_json = await get_menu_item(
            MENU_ITEM_URL,
            venue_name=venue.name,
            venue_uuid=venue.uuid,
            menu_item=menu_bundle,
        )
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
        logger.info(
            f"Added {len(menu_items)} choices for {venue.name} > {menu_bundle.item_cat} > {menu_bundle.item_subcat} > {menu_bundle.item_name}"
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
    logger.info("Getting Venues.")
    venues: list[Venue] = await process_venues(limit=VENUE_LIMIT)

    logger.info(f"Found total {len(venues)} venue[s]")

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
    menu_items: list[MenuModel] = []

    for venue in venues:
        try:
            logger.info(f"Current Venue: {venue.name}")
            menu_items = await process_menu(venue)
            logger.info(
                f"Total {len(menu_items)} items and choices were found for venue: {venue.name}"
            )
            write(
                menu_items,
                scheme=menu_items_schema,
                filename=f"black_sheep_coffee_{CURRENT_DATE}.txt",
                custom_filter=filter_dataframe,
            )
        except Exception:
            logger.critical(f"Failed to scrape venue: {venue}", exc_info=True)


if __name__ == "__main__":
    logger.info("Start of the script.")
    asyncio.run(main())
    logger.info("End of the script.")
