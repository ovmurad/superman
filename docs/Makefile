# Makefile for Sphinx documentation

SPHINX_APIDOC = sphinx-apidoc
SPHINX_BUILD = sphinx-build

# Customize these paths
MODULE_DIR = ../src  # or ../your_package_name
SOURCE_DIR = source
BUILD_DIR = build
APIDOC_FLAGS = -f -o $(SOURCE_DIR)

.PHONY: apidoc html clean

apidoc:
	$(SPHINX_APIDOC) $(APIDOC_FLAGS) $(MODULE_DIR)

html: apidoc
	$(SPHINX_BUILD) -b html $(SOURCE_DIR) $(BUILD_DIR)/html

clean:
	rm -rf $(BUILD_DIR)/*
