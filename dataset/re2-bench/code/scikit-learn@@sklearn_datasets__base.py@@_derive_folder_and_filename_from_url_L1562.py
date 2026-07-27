from urllib.parse import urlparse

def _derive_folder_and_filename_from_url(url):
    parsed_url = urlparse(url)
    if not parsed_url.hostname:
        raise ValueError(f"Invalid URL: {url}")
    folder_components = [_filter_filename(parsed_url.hostname, filter_dots=False)]
    path = parsed_url.path

    if "/" in path:
        base_folder, raw_filename = path.rsplit("/", 1)

        base_folder = _filter_filename(base_folder)
        if base_folder:
            folder_components.append(base_folder)
    else:
        raw_filename = path

    filename = _filter_filename(raw_filename, filter_dots=False)
    if not filename:
        filename = "downloaded_file"

    return "/".join(folder_components), filename
