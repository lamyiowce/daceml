VENV_PATH ?= venv
PYTHON ?= python
PYTEST ?= pytest
PIP ?= pip
YAPF ?= yapf

TORCH_VERSION ?= torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

ifeq ($(VENV_PATH),)
ACTIVATE = 
else
ACTIVATE = . $(VENV_PATH)/bin/activate &&
endif

clean:
	! test -d $(VENV_PATH) || rm -r $(VENV_PATH)

reinstall: venv
	$(ACTIVATE) $(PIP) install --upgrade --force-reinstall -e .[testing]

venv: 
ifneq ($(VENV_PATH),)
	test -d $(VENV_PATH) || echo "Creating new venv" && $(PYTHON) -m venv ./$(VENV_PATH)
endif

install: venv
ifneq ($(VENV_PATH),)
	$(ACTIVATE) pip install --upgrade pip
endif
	$(ACTIVATE) $(PIP) install $(TORCH_VERSION) 
	$(ACTIVATE) $(PIP) install -e .[testing,debug]

test: 
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests

test-gpu: 
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests --gpu

codecov:
	curl -s https://codecov.io/bash | bash

check-formatting:
	$(ACTIVATE) $(YAPF) \
		--parallel \
		--diff \
		--recursive \
		daceml tests setup.py \
		--exclude daceml/onnx/shape_inference/symbolic_shape_infer.py
