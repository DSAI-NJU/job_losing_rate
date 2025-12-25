"""Scrape Baidu Index time series for employment-related keywords."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import requests


# Keywords grouped by thematic buckets defined in keywords.tex.
KEYWORD_GROUPS: Dict[str, List[str]] = {
	"job_search_activity": [
		"找工作",
		"求职",
		"招聘",
		"招聘信息",
		"校招",
		"春招",
		"秋招",
		"应届生就业",
		"面试",
		"简历",
	],
	"job_market_pressure": [
		"就业难",
		"就业形势",
		"应届生找工作难",
		"行业前景",
		"招聘网站",
		"招聘平台",
	],
	"unemployment_pressure": [
		"失业",
		"裁员",
		"裁员潮",
		"优化",
		"N+1",
		"失业金",
		"失业补助",
		"失业保险",
		"失业登记",
		"再就业",
		"低门槛工作",
		"临时工",
		"蓝领招聘",
	],
	"structural_factors": [
		"35岁就业",
		"35岁找工作",
		"公务员考试",
		"国考",
		"事业单位考试",
		"考研",
		"考研就业",
		"考研 vs 就业",
		"外卖骑手",
		"蓝领岗位",
		"底层劳动岗位",
	],
}


def flatten_keywords(groups: Dict[str, Iterable[str]]) -> Dict[str, str]:
	"""Return keyword -> category mapping without duplicates."""

	mapping: Dict[str, str] = {}
	for category, keywords in groups.items():
		for keyword in keywords:
			keyword = keyword.strip()
			if not keyword:
				continue
			mapping.setdefault(keyword, category)
	return mapping


def parse_date(value: str) -> dt.date:
	try:
		return dt.datetime.strptime(value, "%Y-%m-%d").date()
	except ValueError as exc:  # pragma: no cover - defensive guard
		raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
	days = (end - start).days + 1
	for offset in range(days):
		yield start + dt.timedelta(days=offset)


def decrypt_series(ptbk: str, data: str) -> str:
	half = len(ptbk) // 2
	lookup = {ptbk[i]: ptbk[i + half] for i in range(half)}
	return "".join(lookup.get(ch, ch) for ch in data)


@dataclass
class BaiduIndexPoint:
	keyword: str
	category: str
	date: dt.date
	value: Optional[float]


class BaiduIndexClient:
	BASE_URL = "https://index.baidu.com/api/SearchApi/index"
	PTBK_URL = "https://index.baidu.com/Interface/ptbk"

	def __init__(self, cookie: str, area: int = 0, session: Optional[requests.Session] = None) -> None:
		if not cookie:
			raise ValueError("Baidu cookie is required. Set BAIDU_COOKIE or pass --cookie.")
		self.area = area
		self.session = session or requests.Session()
		self.session.headers.update(
			{
				"User-Agent": (
					"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
					"AppleWebKit/537.36 (KHTML, like Gecko) "
					"Chrome/120.0.0.0 Safari/537.36"
				),
				"Referer": "https://index.baidu.com/v2/main/index.html",
				"Accept": "application/json, text/plain, */*",
				"Accept-Language": "zh-CN,zh;q=0.9",
				"Cookie": cookie,
			}
		)

	def fetch_keyword(self, keyword: str, start: dt.date, end: dt.date) -> List[BaiduIndexPoint]:
		params = {
			"area": str(self.area),
			"word": json.dumps([[{"name": keyword, "wordType": 1}]], ensure_ascii=False),
			"startDate": start.strftime("%Y-%m-%d"),
			"endDate": end.strftime("%Y-%m-%d"),
		}
		logging.debug("Query params for %s: %s", keyword, params)
		response = self.session.get(self.BASE_URL, params=params, timeout=15)
		response.raise_for_status()
		payload = response.json()
		if payload.get("status") != 0:
			message = payload.get("message", "Unknown error")
			raise RuntimeError(f"Baidu Index API error for {keyword}: {message}")

		data = payload.get("data", {})
		uniqid = data.get("uniqid")
		if not uniqid:
			raise RuntimeError(f"Missing uniqid for keyword {keyword}.")

		ptbk_response = self.session.get(self.PTBK_URL, params={"uniqid": uniqid}, timeout=15)
		ptbk_response.raise_for_status()
		ptbk = ptbk_response.json().get("data")
		if not ptbk:
			raise RuntimeError(f"Failed to obtain decryption key for {keyword}.")

		series = []
		user_indexes = data.get("userIndexes", [])
		if not user_indexes:
			logging.warning("No userIndexes returned for %s", keyword)
			return series

		encrypted = user_indexes[0].get("all", {}).get("data", "")
		if not encrypted:
			logging.warning("Empty series for %s", keyword)
			return series

		decrypted = decrypt_series(ptbk, encrypted)
		values = decrypted.split(",")
		total_days = (end - start).days + 1
		if len(values) != total_days:
			logging.warning(
				"Series length (%s) mismatch for %s between %s and %s",
				len(values),
				keyword,
				start,
				end,
			)

		for offset, day in enumerate(daterange(start, end)):
			raw = values[offset] if offset < len(values) else ""
			try:
				value = float(raw) if raw else None
			except ValueError:
				value = None
			series.append(
				BaiduIndexPoint(
					keyword=keyword,
					category="",
					date=day,
					value=value,
				)
			)
		return series


def write_csv(points: Iterable[BaiduIndexPoint], output_path: str) -> None:
	rows = list(points)
	if not rows:
		logging.warning("No data to write at %s", output_path)
		return
	with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
		fieldnames = ["category", "keyword", "date", "index"]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for point in rows:
			writer.writerow(
				{
					"category": point.category,
					"keyword": point.keyword,
					"date": point.date.strftime("%Y-%m-%d"),
					"index": "" if point.value is None else f"{point.value:.2f}",
				}
			)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"start",
		type=parse_date,
		nargs="?",
		default=dt.date(2024, 1, 1),
		help="Start date YYYY-MM-DD (default: 2024-01-01)",
	)
	parser.add_argument(
		"end",
		type=parse_date,
		nargs="?",
		default=dt.date(2025, 12, 31),
		help="End date YYYY-MM-DD (default: 2025-12-31)",
	)
	parser.add_argument(
		"--cookie",
		default=os.environ.get("BAIDU_COOKIE", ""),
		help="Baidu index cookie string (defaults to BAIDU_COOKIE env var).",
	)
	parser.add_argument(
		"--area",
		default=0,
		type=int,
		help="Baidu area code (0=全国).",
	)
	parser.add_argument(
		"--output",
		default="baidu_index.csv",
		help="Output CSV path (default: baidu_index.csv)",
	)
	parser.add_argument(
		"--categories",
		nargs="*",
		help="Optional subset of categories to collect (defaults to all)",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable debug logging",
	)
	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(level=logging.INFO)

	if args.start > args.end:
		parser.error("Start date must be earlier than end date.")

	groups = KEYWORD_GROUPS
	if args.categories:
		missing = [name for name in args.categories if name not in groups]
		if missing:
			parser.error(f"Unknown categories: {', '.join(missing)}")
		groups = {name: KEYWORD_GROUPS[name] for name in args.categories}

	keyword_to_category = flatten_keywords(groups)
	client = BaiduIndexClient(cookie=args.cookie, area=args.area)
	collected: List[BaiduIndexPoint] = []
	for keyword, category in keyword_to_category.items():
		logging.info("Fetching %s (%s)", keyword, category)
		series = client.fetch_keyword(keyword, args.start, args.end)
		for point in series:
			collected.append(
				BaiduIndexPoint(
					keyword=point.keyword,
					category=category,
					date=point.date,
					value=point.value,
				)
			)

	write_csv(collected, args.output)
	logging.info("Saved %d rows to %s", len(collected), args.output)


if __name__ == "__main__":
	main()
