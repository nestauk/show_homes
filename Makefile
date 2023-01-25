SHELL := /bin/bash

# Detect how to open things depending on our OS
OS = $(shell uname -s)
ifeq ($(OS),Linux)
	OPEN=xdg-open
else
	OPEN=open
endif

# Export all environment variables
export

# Import config variables
include .cookiecutter/config

# Ensure directory to track and log setup state exists
$(shell mkdir -p .cookiecutter/state)

.PHONY: test-setup
## Test that everything has been setup
test-setup:
	@echo Testing configuration

	@# Checking S3
	@$(call execute_in_env, (test '${IN_PYTEST}' && echo 'In test-suite: Skipping S3 checks') ||\
	 aws s3 ls s3://${BUCKET} > /dev/null 2>&1 || echo "ERROR: No bucket")

	@# Github
	@(test '${IN_PYTEST}' && echo 'In test-suite: Skipping Github checks') ||\
	 git ls-remote > /dev/null 2>&1 || echo "ERROR: No Git remote"

	@# Pre-commit valid
	@$(call execute_in_env, pre-commit run -a)

	#@# Metaflow
	#@$(call execute_in_env, python .cookiecutter/scripts/check_metaflow_aws.py || echo "ERROR: Metaflow+AWS configuration")

.PHONY: init
## Fully initialise a project: install; setup github repo; setup S3 bucket
init:  install .cookiecutter/state/setup-bucket .cookiecutter/state/setup-github
	@echo SETUP COMPLETE

.PHONY: install
## Install a project: create conda env; install local package; setup git hooks; setup metaflow+AWS
install: .cookiecutter/state/conda-create .cookiecutter/state/setup-git .cookiecutter/state/setup-metaflow
	@direnv reload  # Now the conda env exists, reload to activate it

.PHONY: inputs-pull
## Pull `inputs/` from S3
inputs-pull:
	$(call execute_in_env, aws s3 sync s3://${BUCKET}/inputs inputs)

.PHONY: inputs-push
## Push `inputs/` to S3 (WARNING: this may overwrite existing files!)
inputs-push:
	$(call execute_in_env, aws s3 sync inputs s3://${BUCKET}/inputs)

.PHONY: docs
## Build the API documentation
docs:
	$(call execute_in_env, sphinx-apidoc -o docs/api ${REPO_NAME})
	$(call execute_in_env, sphinx-build -b docs/ docs/_build)

.PHONY: docs-clean
## Clean the built API documentation
docs-clean:
	rm -r docs/source/api
	rm -r docs/_build

.PHONY: docs-open
## Open the docs in the browser
docs-open:
	$(OPEN) docs/_build/index.html

.PHONY: conda-update
## Update the conda-environment based on changes to `environment.yaml`
conda-update:
	conda env update -n ${REPO_NAME} -f environment.yaml
	$(MAKE) -s pip-install
	@direnv reload

.PHONY: clean
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: pre-commit
## Perform pre-commit actions
pre-commit:
	$(call execute_in_env, pre-commit)

.PHONY: lint
## Run flake8 linting on repository
lint:
	$(call execute_in_env, flake8)

.PHONY: pip-install
## Install our package and requirements in editable mode (including development dependencies)
pip-install:
	@$(call execute_in_env, pip install -e ".[dev]")

#################################################################################
# Helper Commands (no need to explicitly document)                              #
#################################################################################

define err
	(echo "$1, check $@ for more info" && exit 1)
endef

# Allow us to execute make commands from within our project's conda env
# and with bash utilities available
define execute_in_env
	source .cookiecutter/scripts/import.sh && conda_activate ${REPO_NAME} && $1
endef

.cookiecutter/state/conda-create:
	@echo -n "Creating environment ${REPO_NAME} and installing all dependencies"
	@(conda env create -q -n ${REPO_NAME} -f environment.yaml\
	  && $(call execute_in_env, pip install -e ".[dev]"))\
	 > $@.log 2>&1\
	 || $(call err,Python environment setup failed)
	@touch $@
	@echo " DONE"

.cookiecutter/state/setup-git:
	@echo -n "Installing and configuring git pre-commit hooks"
	@$(call execute_in_env, \
	 pre-commit install --install-hooks\
	 > $@.log 2>&1\
	 || $(call err,Git pre-commit setup failed)\
	)
	@touch $@
	@echo " DONE"

#.cookiecutter/state/setup-metaflow:
#	@echo -n "Configuring Metaflow + AWS"
#	@$(call execute_in_env, \
#	 get_metaflow_config\
#	 > $@.log 2>&1\
#	 || $(call err,AWS + Metaflow setup failed)\
#	)
#	@touch $@
#	@echo " DONE"

.cookiecutter/state/setup-github:
	@echo -n "Creating and configuring Github repo '${GITHUB_ACCOUNT}/${REPO_NAME}'"
	@$(call execute_in_env, \
	 create_gh_repo ${GITHUB_ACCOUNT} ${REPO_NAME} ${PROJECT_OPENNESS}\
	 > $@.log 2>&1\
	 || $(call err,Github repo creation failed)\
	)
	@touch $@
	@echo " DONE"

.cookiecutter/state/setup-bucket:
	@echo -n "Creating S3 bucket '${BUCKET}'"
	@$(call execute_in_env, \
	 (create_bucket ${BUCKET} && make_bucket_private ${BUCKET})\
	 > $@.log 2>&1\
	 || $(call err,S3 Bucket creation failed)\
	)
	@touch $@
	@echo " DONE"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
