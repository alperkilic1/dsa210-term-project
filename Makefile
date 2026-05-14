.PHONY: install run clean report help

PY := python3
VENV := .venv

help:
	@echo "Targets:"
	@echo "  install  - create venv and install requirements"
	@echo "  run      - execute eda_report.ipynb then ml_baseline.ipynb"
	@echo "  report   - regenerate final_report.html and final_report.pdf via pandoc"
	@echo "  clean    - remove .venv, __pycache__, .ipynb_checkpoints"

install:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

run:
	$(VENV)/bin/jupyter execute eda_report.ipynb
	$(VENV)/bin/jupyter execute ml_baseline.ipynb

report:
	pandoc final_report.md -o final_report.html --standalone --toc --metadata title="Incident Genome"
	pandoc final_report.md -o final_report.pdf --pdf-engine=xelatex --toc -V geometry:margin=1in

clean:
	rm -rf $(VENV) __pycache__ .ipynb_checkpoints
	find . -name '__pycache__' -type d -exec rm -rf {} +
