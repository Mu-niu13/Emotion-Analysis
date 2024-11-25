.PHONY: all install train evaluate clean

# Define variables
PYTHON=python
VENV=NLP
ACTIVATE=$(VENV)/bin/activate
PIP=$(VENV)/bin/pip
PYTHON_BIN=$(VENV)/bin/python

all: install train evaluate

install:
	@echo "Installing dependencies..."
	# Install python3-venv if not present (specific to Debian/Ubuntu systems like Colab)
	@sudo apt-get update -y && sudo apt-get install -y python3-venv
	@echo "Setting up virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Installing requirements..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

train:
	@echo "Running training script..."
	$(PYTHON_BIN) main.py --mode train

evaluate:
	@echo "Running evaluation script..."
	$(PYTHON_BIN) main.py --mode evaluate

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ */__pycache__ $(VENV)
