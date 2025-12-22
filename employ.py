from __future__ import annotations

import argparse
import html
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://sousuo.www.gov.cn/search-gov/data"
DEFAULT_PARAMS: Dict[str, Any] = {
	"t": "zhengcelibrary",
	"q": "",
	"timetype": "",
	"mintime": "",
	"maxtime": "",
	"sort": "pubtime",
	"sortType": 1,
	"searchfield": "title",
	"pcodeJiguan": "",
	"childtype": "",
	"subchildtype": "",
	"tsbq": "",
	"pubtimeyear": "",
	"puborg": "",
	"pcodeYear": "",
	"pcodeNum": "",
	"filetype": "",
	"p": 1,
	"n": 100,
	"inpro": "",
	"bmfl": "",
	"dup": "",
	"orpro": "",
	"type": "gwyzcwjk",
}

HEADERS = {
	"User-Agent": (
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
		"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
	),
	"Accept": "application/json, text/plain, */*",
	"Referer": "https://sousuo.www.gov.cn/zcwjk/policyDocumentLibrary",
	"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT = 20

session = requests.Session()
session.headers.update(HEADERS)


@dataclass
class PolicyRecord:
	category: str
	title: str
	publish_date: Optional[str]
	publish_date_display: Optional[str]
	publish_timestamp: Optional[int]
	publisher: Optional[str]
	document_code: Optional[str]
	url: Optional[str]


def _parse_cli_date(value: str) -> date:
	try:
		return datetime.strptime(value, "%Y-%m-%d").date()
	except ValueError as exc:
		raise argparse.ArgumentTypeError(
			"Dates must follow the YYYY-MM-DD format"
		) from exc


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Scrape titles and publish dates from the State Council policy document library."
		)
	)
	parser.add_argument(
		"--query",
		default="",
		help="Keyword query to filter documents (defaults to all documents).",
	)
	parser.add_argument(
		"--page-size",
		type=int,
		default=100,
		help="Number of documents to request per category per page (max 100).",
	)
	parser.add_argument(
		"--max-pages",
		type=int,
		default=None,
		help="Limit the number of pages to fetch (per category). Default fetches all pages.",
	)
	parser.add_argument(
		"--sort",
		default="pubtime",
		choices=("pubtime", "score"),
		help="Sort order used by the API (default: pubtime for newest first).",
	)
	parser.add_argument(
		"--categories",
		nargs="*",
		default=None,
		help="Optional space-separated list of categories to include (e.g. gongwen gongbao).",
	)
	parser.add_argument(
		"--output",
		default="gov_policy_titles.csv",
		help="Path to the CSV output file.",
	)
	parser.add_argument(
		"--delay",
		type=float,
		default=0.0,
		help="Optional delay in seconds between page requests.",
	)
	parser.add_argument(
		"--start-date",
		type=_parse_cli_date,
		default=None,
		help="Inclusive lower bound (YYYY-MM-DD) for publish date filtering.",
	)
	parser.add_argument(
		"--end-date",
		type=_parse_cli_date,
		default=None,
		help="Inclusive upper bound (YYYY-MM-DD) for publish date filtering.",
	)
	args = parser.parse_args()
	if args.start_date and args.end_date and args.start_date > args.end_date:
		parser.error("--start-date must be earlier than or equal to --end-date")
	return args


def clean_text(text: Optional[str]) -> str:
	if not text:
		return ""
	no_tags = re.sub(r"<[^>]+>", " ", text)
	return html.unescape(no_tags).replace("\xa0", " ").strip()


def normalize_publish_date(pubtime: Any, pubtime_str: Optional[str]) -> tuple[Optional[str], Optional[str]]:
	if isinstance(pubtime, (int, float)) and pubtime > 0:
		try:
			dt = datetime.fromtimestamp(pubtime / 1000)
			return dt.strftime("%Y-%m-%d"), pubtime_str
		except (OSError, ValueError):
			pass

	if pubtime_str:
		for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"):
			try:
				dt = datetime.strptime(pubtime_str, fmt)
				return dt.strftime("%Y-%m-%d"), pubtime_str
			except ValueError:
				continue
	return None, pubtime_str


def fetch_page(params: Dict[str, Any]) -> Dict[str, Any]:
	response = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
	response.raise_for_status()
	payload = response.json()
	search_vo = payload.get("searchVO") or {}
	cat_map = search_vo.get("catMap")
	if not isinstance(cat_map, dict):
		raise ValueError("Unexpected response structure: missing category map")
	return cat_map


def collect_records(
	query: str,
	page_size: int,
	max_pages: Optional[int],
	categories: Optional[Iterable[str]],
	delay: float,
	start_date: Optional[date],
	end_date: Optional[date],
	sort_field: str,
) -> Dict[str, List[PolicyRecord]]:
	if page_size <= 0:
		raise ValueError("page_size must be positive")
	if page_size > 100:
		page_size = 100

	base_params = {**DEFAULT_PARAMS, "q": query or "", "n": page_size, "sort": sort_field}
	if start_date or end_date:
		base_params["timetype"] = "timeqb"
	if start_date:
		base_params["mintime"] = start_date.strftime("%Y-%m-%d")
	if end_date:
		base_params["maxtime"] = end_date.strftime("%Y-%m-%d")
	target_categories = set(categories) if categories else None

	records: Dict[str, List[PolicyRecord]] = {}
	totals: Dict[str, int] = {}
	seen_ids: Dict[str, set] = {}
	category_done: Dict[str, bool] = {}

	page = 1
	max_required_pages: Optional[int] = None

	while True:
		params = {**base_params, "p": page}
		cat_map = fetch_page(params)

		if not cat_map:
			break

		if page == 1:
			for cat, info in cat_map.items():
				if target_categories and cat not in target_categories:
					continue
				totals[cat] = info.get("totalCount", 0) or 0
				records[cat] = []
				seen_ids[cat] = set()
				category_done[cat] = False

			if target_categories:
				for cat in target_categories:
					totals.setdefault(cat, 0)
					records.setdefault(cat, [])
					seen_ids.setdefault(cat, set())
					category_done.setdefault(cat, False)

			if totals and not (start_date or end_date):
				max_required_pages = max(
					math.ceil((totals.get(cat, 0) or 0) / page_size) or 0
					for cat in (target_categories or totals.keys())
				) or 0

		empty_count = 0

		for cat, info in cat_map.items():
			if target_categories and cat not in target_categories:
				continue
			if category_done.get(cat):
				continue

			items = info.get("listVO") or []
			if not items:
				empty_count += 1
				if start_date and sort_field == "pubtime":
					category_done[cat] = True
				continue

			totals.setdefault(cat, info.get("totalCount", 0) or 0)
			records.setdefault(cat, [])
			seen_ids.setdefault(cat, set())
			category_done.setdefault(cat, False)

			all_older_than_start = bool(start_date)

			for item in items:
				record_id = item.get("id") or item.get("url")
				if record_id in seen_ids[cat]:
					continue

				seen_ids[cat].add(record_id)
				publish_date, display_date = normalize_publish_date(
					item.get("pubtime"), item.get("pubtimeStr")
				)
				record_dt: Optional[datetime] = None
				timestamp = item.get("pubtime")
				if isinstance(timestamp, (int, float)) and timestamp > 0:
					try:
						record_dt = datetime.fromtimestamp(timestamp / 1000)
					except (OSError, ValueError):
						record_dt = None
				elif publish_date:
					try:
						record_dt = datetime.strptime(publish_date, "%Y-%m-%d")
					except ValueError:
						record_dt = None

				record_date = record_dt.date() if record_dt else None
				include = True
				if start_date:
					if record_date is None or record_date < start_date:
						include = False
					else:
						all_older_than_start = False
				if end_date and (record_date is None or record_date > end_date):
					include = False

				if not include:
					continue
				records[cat].append(
					PolicyRecord(
						category=cat,
						title=clean_text(item.get("title")),
						publish_date=publish_date,
						publish_date_display=display_date,
						publish_timestamp=item.get("pubtime"),
						publisher=clean_text(item.get("puborg") or item.get("source")),
						document_code=clean_text(item.get("wenhao") or item.get("fwzh")),
						url=item.get("url"),
					)
				)

			if (
				start_date
				and sort_field == "pubtime"
				and all_older_than_start
				and items
			):
				category_done[cat] = True

		targets = target_categories or records.keys()
		if targets and all(
			(category_done.get(cat, False))
			or (totals.get(cat, 0) and len(records.get(cat, [])) >= totals.get(cat, 0))
			for cat in targets
		):
			break

		if max_pages is not None and page >= max_pages:
			break

		if (
			max_pages is None
			and max_required_pages is not None
			and page >= max_required_pages
		):
			break

		if empty_count == len(cat_map):
			break

		page += 1

		if delay:
			time.sleep(delay)

	return records


def records_to_dataframe(records: Dict[str, List[PolicyRecord]]) -> pd.DataFrame:
	rows = []
	for cat, items in records.items():
		for record in items:
			rows.append(
				{
					"category": cat,
					"title": record.title,
					"publish_date": record.publish_date,
					"publish_date_display": record.publish_date_display,
					"publish_timestamp": record.publish_timestamp,
					"publisher": record.publisher,
					"document_code": record.document_code,
					"url": record.url,
				}
			)

	if not rows:
		return pd.DataFrame(columns=[
			"category",
			"title",
			"publish_date",
			"publish_date_display",
			"publish_timestamp",
			"publisher",
			"document_code",
			"url",
		])

	df = pd.DataFrame(rows)
	if "publish_date" in df.columns:
		df = df.sort_values(
			by=["publish_date", "publish_timestamp"],
			ascending=[False, False],
			na_position="last",
		)
	return df.reset_index(drop=True)


def main() -> None:
	args = parse_arguments()
	records = collect_records(
		query=args.query,
		page_size=args.page_size,
		max_pages=args.max_pages,
		categories=args.categories,
		delay=args.delay,
		start_date=args.start_date,
		end_date=args.end_date,
		sort_field=args.sort,
	)

	df = records_to_dataframe(records)

	if df.empty:
		print("No records found for the given parameters.")
		return

	df.to_csv(args.output, index=False)
	print(f"Saved {len(df)} records to {args.output}.")


if __name__ == "__main__":
	main()
