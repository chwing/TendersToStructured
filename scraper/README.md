# Scraper

Scrapes open tenders from public procurement portals and downloads their
documents into a folder ready for one of the extraction pipelines
(`full_llm/`, `first_staged_pipeline/`, `second_staged_pipeline/`) to consume.

```
Portal search → keyword-filtered listing → dedup against prior runs → download documents → tender_docs_new/
```

## Sources

| Source | Portal | Listing | Document download |
|---|---|---|---|
| `marchespublics` | Morocco — marchespublics.gov.ma | form POST (server-rendered) | Anonymous DCE download, unzipped automatically. **Best-effort**: the site appears to throttle/challenge repeated automated postbacks, so a scraping run can end with some (or all) documents undownloaded even though the flow itself works — metadata (with `detail_url`) is always saved regardless, for manual follow-up. |
| `tuneps` | Tunisia — tuneps.tn | public JSON REST API | **Not available** — TUNEPS requires a logged-in supplier account to download documents (public API returns 401). Document filenames are recorded in the metadata sidecar so a human can log in and fetch them manually via `detail_url`. |

Both sites are reverse-engineered integrations (neither publishes an API), so
selectors/endpoints may need adjustment if the sites change.

## Install

```bash
pip install -r scraper/requirements.txt
```

## Run

```bash
cd scraper
python run_scraper.py --output-dir ../tender_docs_new
```

**Common options:**
```bash
# one portal only
python run_scraper.py --source tuneps

# custom keyword filter (default is a French tech/IT keyword list)
python run_scraper.py --keywords informatique logiciel "intelligence artificielle"

# re-download everything, ignoring the seen-tenders index
python run_scraper.py --no-dedup

python run_scraper.py --max-pages 20 -v
```

## Output

Documents land flat in `--output-dir` (default `../tender_docs_new`) as
`<source>_<reference>__<filename>`, alongside one
`<source>_<reference>.metadata.json` sidecar per tender (title, buyer,
dates, detail URL). The `.json` sidecars are ignored by the extraction
pipelines' file glob (`*.pdf`/`*.docx`/`*.doc`/`*.txt`), so the output folder
can be pointed at directly:

```bash
python full_llm/run_extractor.py tender_docs_new
python second_staged_pipeline/run_pipeline.py tender_docs_new
```

A `.scraper_state/seen.json` index (also under `--output-dir`) tracks which
tender references have already been scraped, so re-running only fetches new
ones — pass `--no-dedup` to disable that.
