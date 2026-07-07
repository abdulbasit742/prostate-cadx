"""
scripts/gdc_query.py
Query GDC API for open-access TCGA-PRAD diagnostic slide images.
Returns file metadata without downloading.
"""
import urllib.request, json, sys

def query_gdc():
    # Filter: TCGA-PRAD, Slide Image, open access, DX (diagnostic) slides
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {
                "field": "cases.project.project_id",
                "value": ["TCGA-PRAD"]
            }},
            {"op": "in", "content": {
                "field": "data_type",
                "value": ["Slide Image"]
            }},
            {"op": "in", "content": {
                "field": "access",
                "value": ["open"]
            }},
            {"op": "in", "content": {
                "field": "data_format",
                "value": ["SVS"]
            }}
        ]
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,access,cases.case_id",
        "size": "500",
        "format": "json"
    }
    query_str = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
    url = f"https://api.gdc.cancer.gov/files?{query_str}"

    try:
        req = urllib.request.urlopen(url, timeout=20)
        data = json.loads(req.read())
        hits = data["data"]["hits"]
        total = data["data"]["pagination"]["total"]
        return hits, total
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return [], 0


if __name__ == "__main__":
    hits, total = query_gdc()
    print(f"Total open-access TCGA-PRAD SVS: {total}")
    print(f"Returned in this query: {len(hits)}")
    print()
    for i, h in enumerate(hits[:10]):
        fname = h.get("file_name", "?")
        fsize = h.get("file_size", 0)
        fid   = h.get("file_id", "?")
        access = h.get("access", "?")
        print(f"  [{i+1}] {fname} | {fsize/1e6:.1f} MB | {access} | id={fid}")
