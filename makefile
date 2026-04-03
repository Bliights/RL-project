# -------------- Install dependencies --------------
install:
	@echo Install the dependencies with uv and pre commit...
	uv sync
	uv run pre-commit install	

update:
	@echo Updating all dependencies of the environment...
	uv lock --upgrade
	uv sync
	uv run pre-commit autoupdate

# ------------------ Formating ---------------------
lint:
	@echo Check with ruff...
	uv run ruff check .

format:
	@echo Format with Ruff...
	uv run ruff format .

fix:
	@echo Fix with Ruff...
	uv run ruff check --fix .

# ------------------- Pre-commit -------------------
pre-commit:
	@echo Run pre-commit...
	uv run pre-commit run --all-files

# ------------------- Script -------------------
training:
	uv run python -m src.scripts.training.train \
		$(if $(SEED),--seed $(SEED)) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR)) \
		$(if $(HELP),--help)

evaluation:
	uv run python -m src.scripts.evaluation.evaluate \
		$(if $(MODEL_PATH),--model-path $(MODEL_PATH)) \
		$(if $(SEED),--seed $(SEED)) \
		$(if $(N),--n-episodes $(N)) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR)) \
		$(if $(HELP),--help)