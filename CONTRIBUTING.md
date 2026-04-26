# Contributing

Thanks for helping improve `AI Care Prioritization Engine`.
This repository is maintained as a **professional, startup-style applied ML project**, so contributions should favor clarity, reproducibility, and stakeholder-readable outputs.

## Contribution workflow

1. Fork the repository or create a feature branch from `main`.
2. Keep changes focused and easy to review.
3. Add or update tests when behavior changes.
4. Update docs if the user workflow, app UI, or API contract changes.
5. Open a pull request using the provided template.

## Local setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the main interfaces locally:

```bash
streamlit run app.py
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Validation checklist

Before opening a PR, run:

```bash
black --check app src tests outputs
flake8 app src tests outputs
pytest -q
```

## Contribution standards

- Keep the product story clear: **risk prediction, care prioritization, and operational triage**.
- Prefer small, well-scoped commits over large mixed changes.
- Preserve reproducibility and demo reliability.
- Use language that is professional and understandable to both technical and healthcare stakeholders.
- Avoid introducing dependencies or UX complexity without clear value.

## Documentation expectations

Update relevant docs when needed:

- `README.md` for public GitHub-facing changes
- `DEPLOYMENT.md` for launch or hosting changes
- `README_STARTUP.md` / `STARTUP_GUIDE.md` for product and positioning updates
- `CHANGELOG.md` for notable unreleased changes
