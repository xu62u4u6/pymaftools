.PHONY: test test-core test-plot test-model test-fast test-slow test-coverage clean-test install-test-deps

# Test commands
test:
	python run_tests.py --type all

test-core:
	python run_tests.py --type core

test-plot:
	python run_tests.py --type plot

test-model:
	python run_tests.py --type model

test-fast:
	python run_tests.py --type fast

test-slow:
	python run_tests.py --type slow

test-coverage:
	python run_tests.py --type all --coverage

test-quiet:
	python run_tests.py --type all --quiet

# Install test dependencies
install-test-deps:
	pip install -e .[test]

# Clean test artifacts
clean-test:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage 2>/dev/null || true

# Help
help:
	@echo "Available test commands:"
	@echo "  test           - Run all tests"
	@echo "  test-core      - Run core functionality tests"
	@echo "  test-plot      - Run plotting tests"
	@echo "  test-model     - Run model tests"
	@echo "  test-fast      - Run fast tests only"
	@echo "  test-slow      - Run slow tests only"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-quiet     - Run tests in quiet mode"
	@echo "  install-test-deps - Install test dependencies"
	@echo "  clean-test     - Clean test artifacts"