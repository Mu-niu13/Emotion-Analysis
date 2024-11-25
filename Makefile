.PHONY: all install train evaluate clean

# Define variables
PYTHON=python3.11
VENV=NLP
ACTIVATE=$(VENV)/bin/activate
PIP=$(VENV)/bin/pip
PYTHON_BIN=$(VENV)/bin/python

all: install train evaluate

install:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing requirements..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train:
	@echo "Running training script..."
	$(PYTHON_BIN) main.py --mode train

evaluate:
	@echo "Running evaluation script..."
	$(PYTHON_BIN) main.py --mode evaluate

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ */__pycache__ $(VENV)
