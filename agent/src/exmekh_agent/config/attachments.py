"""Defaults and policy for attachments."""

from exmekh_agent.config.catalogs import settings

DEFAULTS = settings("attachments")

MAX_PDF_PAGES = DEFAULTS["max_pdf_pages"]
MAX_IMAGE_SIDE = DEFAULTS["max_image_side"]
IMAGE_SUFFIXES = set(DEFAULTS["image_suffixes"])

PASTE_TOKEN_THRESHOLD = DEFAULTS["paste_token_threshold"]
XML_TAG_NAME = DEFAULTS["xml_tag_name"]
